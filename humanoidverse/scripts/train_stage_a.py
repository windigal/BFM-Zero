from __future__ import annotations

import copy
import contextlib
import glob
import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
import tyro
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from torch.utils.tensorboard import SummaryWriter

from humanoidverse.language.stage_a.dataset import (
    ClipTextSource,
    StageALatentDataset,
    collate_stage_a,
    load_latent_store,
    load_manifest_records,
)
from humanoidverse.language.stage_a.model import StageAModelConfig, StageATextToLatentModel, stage_a_losses


DEFAULT_MANIFEST_PATH = Path("artifacts/stage_a/seed_manifest.jsonl")
DEFAULT_LATENT_GLOB = "artifacts/stage_a/latents/*.joblib"
DEFAULT_OUTPUT_DIR = Path("results/stage_a")
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_WORKERS = 4
DEFAULT_DEVICE = "cuda"

StageAPreset = Literal["seed_default", "lafan1_smoke", "custom"]


@dataclass
class Args:
    preset: StageAPreset = "seed_default"
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    latent_paths: list[Path] | None = None
    latent_glob: str = DEFAULT_LATENT_GLOB
    output_dir: Path = DEFAULT_OUTPUT_DIR
    clip_model_name: str = "openai/clip-vit-base-patch32"
    device: str = DEFAULT_DEVICE
    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = DEFAULT_NUM_WORKERS
    num_epochs: int = 10
    lr: float = 5e-5
    text_encoder_lr_scale: float = 0.2
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.03
    weight_decay: float = 1e-4
    hidden_dim: int = 768
    num_chunks: int = 8
    num_decoder_layers: int = 4
    num_decoder_heads: int = 12
    dropout: float = 0.1
    freeze_text_encoder: bool = False
    use_query_latent_decoder: bool = True
    train_text_mode: Literal["expand_all", "sample_one"] = "sample_one"
    val_text_mode: Literal["expand_all", "sample_one"] = "expand_all"
    clip_text_source_train: ClipTextSource = "all"
    clip_text_source_val: ClipTextSource = "all"
    clip_loss_weight: float = 1.0
    chunk_loss_weight: float = 1.0
    info_nce_weight: float = 0.1
    smooth_weight: float = 0.05
    clip_cosine_weight: float = 0.25
    chunk_cosine_weight: float = 0.25
    grad_clip_norm: float = 1.0
    amp_dtype: Literal["none", "fp16", "bf16"] = "bf16"
    ema_decay: float = 0.999
    log_every_steps: int = 10
    save_every_epochs: int = 1
    keep_last_n_checkpoints: int = 5
    prefetch_factor: int = 4
    persistent_workers: bool = True
    use_timestamp_subdir: bool = True
    resume_from: Path | None = None
    seed: int = 42


PRESET_CONFIGS: dict[str, dict[str, object]] = {
    "seed_default": {
        "manifest_path": DEFAULT_MANIFEST_PATH,
        "latent_glob": DEFAULT_LATENT_GLOB,
        "output_dir": DEFAULT_OUTPUT_DIR,
        "device": "cuda",
        "batch_size": 128,
        "num_workers": 4,
    },
    "lafan1_smoke": {
        "manifest_path": Path("artifacts/lafan1_stage_a/lafan1_stage_a_manifest.jsonl"),
        "latent_paths": [Path("artifacts/lafan1_stage_a/lafan1_stage_a_latents.joblib")],
        "output_dir": DEFAULT_OUTPUT_DIR,
        "device": "cuda",
        "batch_size": 8,
        "num_workers": 0,
    },
}


def apply_preset(args: Args) -> Args:
    if args.preset == "custom":
        return args

    preset = PRESET_CONFIGS[args.preset]
    if args.manifest_path == DEFAULT_MANIFEST_PATH and "manifest_path" in preset:
        args.manifest_path = preset["manifest_path"]  # type: ignore[assignment]
    if args.latent_paths is None and args.latent_glob == DEFAULT_LATENT_GLOB:
        if "latent_paths" in preset:
            args.latent_paths = preset["latent_paths"]  # type: ignore[assignment]
        elif "latent_glob" in preset:
            args.latent_glob = preset["latent_glob"]  # type: ignore[assignment]
    if args.output_dir == DEFAULT_OUTPUT_DIR and "output_dir" in preset:
        args.output_dir = preset["output_dir"]  # type: ignore[assignment]
    if args.device == DEFAULT_DEVICE and "device" in preset:
        args.device = preset["device"]  # type: ignore[assignment]
    if args.batch_size == DEFAULT_BATCH_SIZE and "batch_size" in preset:
        args.batch_size = preset["batch_size"]  # type: ignore[assignment]
    if args.num_workers == DEFAULT_NUM_WORKERS and "num_workers" in preset:
        args.num_workers = preset["num_workers"]  # type: ignore[assignment]
    return args


def resolve_latent_paths(args: Args) -> list[Path]:
    if args.latent_paths:
        return args.latent_paths
    return sorted(Path(path) for path in glob.glob(args.latent_glob))


def resolve_output_dir(args: Args) -> Path:
    if args.resume_from is not None:
        resume_dir = args.resume_from.resolve().parent
        if resume_dir.name == "checkpoints":
            resume_dir = resume_dir.parent
        return resume_dir
    if args.use_timestamp_subdir:
        return args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    return args.output_dir


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("stage_a_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


def get_trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def build_optimizer(model: StageATextToLatentModel, args: Args) -> torch.optim.Optimizer:
    text_encoder_params: list[torch.nn.Parameter] = []
    other_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("text_encoder."):
            text_encoder_params.append(param)
        else:
            other_params.append(param)

    param_groups: list[dict[str, object]] = []
    if other_params:
        param_groups.append(
            {
                "params": other_params,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            }
        )
    if text_encoder_params:
        param_groups.append(
            {
                "params": text_encoder_params,
                "lr": args.lr * args.text_encoder_lr_scale,
                "weight_decay": args.weight_decay,
            }
        )
    if not param_groups:
        raise RuntimeError("No trainable parameters found for optimizer construction.")
    return torch.optim.AdamW(param_groups)


def to_jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def prune_old_checkpoints(checkpoint_dir: Path, keep_last_n: int) -> None:
    if keep_last_n <= 0:
        return
    ckpts = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if len(ckpts) <= keep_last_n:
        return
    for stale in ckpts[: len(ckpts) - keep_last_n]:
        stale.unlink(missing_ok=True)


def build_lr_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float, min_lr_ratio: float) -> LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return min_lr_ratio
        progress = (step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters(), strict=True):
            ema_param.lerp_(param.detach(), 1.0 - decay)
        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers(), strict=True):
            ema_buffer.copy_(buffer)


def get_grad_scaler(device: torch.device, amp_dtype: str) -> torch.cuda.amp.GradScaler | None:
    if device.type != "cuda" or amp_dtype == "none":
        return None
    if amp_dtype == "fp16":
        return torch.cuda.amp.GradScaler()
    return None


def get_autocast_context(device: torch.device, amp_dtype: str) -> contextlib.AbstractContextManager[None]:
    if device.type != "cuda" or amp_dtype == "none":
        return contextlib.nullcontext()
    if amp_dtype == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if amp_dtype == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"Unsupported amp_dtype={amp_dtype!r}")


def evaluate(
    model: StageATextToLatentModel,
    loader: DataLoader,
    device: torch.device,
    args: Args,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "clip_loss": 0.0,
        "chunk_loss": 0.0,
        "info_nce_loss": 0.0,
        "smooth_loss": 0.0,
        "clip_cosine_loss": 0.0,
        "chunk_cosine_loss": 0.0,
    }
    count = 0
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["text"], device=device)
            losses = stage_a_losses(
                outputs=outputs,
                target_clip=batch["z_clip"].to(device),
                target_chunks=batch["z_chunks"].to(device),
                temperature=model.cfg.info_nce_temperature,
                clip_weight=args.clip_loss_weight,
                chunk_weight=args.chunk_loss_weight,
                info_nce_weight=args.info_nce_weight,
                smooth_weight=args.smooth_weight,
                clip_cosine_weight=args.clip_cosine_weight,
                chunk_cosine_weight=args.chunk_cosine_weight,
            )
            batch_size = batch["z_clip"].shape[0]
            for key in totals:
                totals[key] += float(losses[key]) * batch_size
            count += batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def load_resume_state(
    checkpoint_path: Path,
    model: StageATextToLatentModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int, float, list[dict[str, float | int]], dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    history = checkpoint.get("history", [])
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    global_step = int(checkpoint.get("global_step", 0))
    best_val = float(checkpoint.get("best_val", float("inf")))
    return start_epoch, global_step, best_val, history, checkpoint


def main(args: Args) -> None:
    args = apply_preset(args)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    args.output_dir = resolve_output_dir(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output_dir)
    writer = SummaryWriter(log_dir=str(args.output_dir / "tensorboard")) if SummaryWriter is not None else None

    latent_paths = resolve_latent_paths(args)
    if not latent_paths:
        raise FileNotFoundError("No latent files found. Run extract_stage_a_teacher_latents.py first.")

    manifest_records = load_manifest_records(args.manifest_path)
    latent_store = load_latent_store(latent_paths)
    z_dim = next(iter(latent_store.values()))["z_clip"].shape[-1]
    device = torch.device(args.device)

    train_dataset = StageALatentDataset(
        manifest_records,
        latent_store,
        split="train",
        text_mode=args.train_text_mode,
        clip_text_source=args.clip_text_source_train,
        seed=args.seed,
    )
    val_dataset = StageALatentDataset(
        manifest_records,
        latent_store,
        split="val",
        text_mode=args.val_text_mode,
        clip_text_source=args.clip_text_source_val,
        seed=args.seed,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("Stage A train dataset is empty after joining manifest and latent store.")

    if len(val_dataset) == 0:
        logger.warning("Validation split is empty. Best checkpoint will track the fallback loss used in training summary.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_stage_a,
        pin_memory=device.type == "cuda",
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_stage_a,
        pin_memory=device.type == "cuda",
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    model = StageATextToLatentModel(
        StageAModelConfig(
            clip_model_name=args.clip_model_name,
            z_dim=z_dim,
            hidden_dim=args.hidden_dim,
            num_chunks=args.num_chunks,
            num_decoder_layers=args.num_decoder_layers,
            num_decoder_heads=args.num_decoder_heads,
            dropout=args.dropout,
            freeze_text_encoder=args.freeze_text_encoder,
            use_query_latent_decoder=args.use_query_latent_decoder,
        )
    ).to(device)

    trainable_parameters = get_trainable_parameters(model)
    optimizer = build_optimizer(model, args)
    total_steps = len(train_loader) * max(args.num_epochs, 1)
    scheduler = build_lr_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
    )
    grad_scaler = get_grad_scaler(device, args.amp_dtype)
    ema_model = copy.deepcopy(model).eval() if args.ema_decay > 0 else None
    if ema_model is not None:
        ema_model.requires_grad_(False)
    checkpoint_dir = args.output_dir / "checkpoints"
    trainable_params, frozen_params = count_parameters(model)

    dataset_info = {
        "manifest_path": str(args.manifest_path.resolve()),
        "latent_paths": [str(path.resolve()) for path in latent_paths],
        "manifest_records_total": len(manifest_records),
        "latent_keys_total": len(latent_store),
        "train_items": len(train_dataset),
        "val_items": len(val_dataset),
        "train_text_mode": args.train_text_mode,
        "val_text_mode": args.val_text_mode,
        "clip_text_source_train": args.clip_text_source_train,
        "clip_text_source_val": args.clip_text_source_val,
        "z_dim": int(z_dim),
        "device": str(device),
        "use_query_latent_decoder": args.use_query_latent_decoder,
        "freeze_text_encoder": args.freeze_text_encoder,
        "text_encoder_lr_scale": args.text_encoder_lr_scale,
    }
    train_args_dict = to_jsonable(asdict(args))
    write_json(args.output_dir / "train_args.json", train_args_dict)
    write_json(args.output_dir / "dataset_info.json", dataset_info)
    if writer is not None:
        writer.add_text("config/train_args", json.dumps(train_args_dict, indent=2, ensure_ascii=False), 0)
        writer.add_text("config/dataset_info", json.dumps(dataset_info, indent=2, ensure_ascii=False), 0)

    logger.info(
        "Stage A training setup | preset=%s | train_items=%d | val_items=%d | z_dim=%d | trainable_params=%d | frozen_params=%d | query_decoder=%s | freeze_text_encoder=%s",
        args.preset,
        len(train_dataset),
        len(val_dataset),
        z_dim,
        trainable_params,
        frozen_params,
        args.use_query_latent_decoder,
        args.freeze_text_encoder,
    )
    if len(optimizer.param_groups) > 1:
        logger.info(
            "Optimizer parameter groups | body_lr=%.6e | text_encoder_lr=%.6e",
            optimizer.param_groups[0]["lr"],
            optimizer.param_groups[1]["lr"],
        )
    else:
        logger.info("Optimizer parameter groups | lr=%.6e", optimizer.param_groups[0]["lr"])
    logger.info("TensorBoard directory: %s", args.output_dir / "tensorboard")
    logger.info("Checkpoint directory: %s", checkpoint_dir)

    best_val = float("inf")
    history: list[dict[str, float | int]] = []
    global_step = 0
    start_epoch = 1

    if args.resume_from is not None:
        if not args.resume_from.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_from}")
        start_epoch, global_step, best_val, history, checkpoint = load_resume_state(
            checkpoint_path=args.resume_from,
            model=model,
            optimizer=optimizer,
            device=device,
        )
        logger.info(
            "Resumed Stage A training from %s | next_epoch=%d | global_step=%d | best_val=%.6f | loaded_history=%d",
            args.resume_from,
            start_epoch,
            global_step,
            best_val,
            len(history),
        )
        if writer is not None:
            writer.add_text("resume/checkpoint", str(args.resume_from.resolve()), global_step)
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if ema_model is not None and checkpoint.get("ema_state_dict") is not None:
            ema_model.load_state_dict(checkpoint["ema_state_dict"])

    if start_epoch > args.num_epochs:
        logger.warning(
            "Resume checkpoint is already at epoch %d, which is beyond requested num_epochs=%d. Nothing to do.",
            start_epoch - 1,
            args.num_epochs,
        )
        if writer is not None:
            writer.close()
        return

    for epoch in range(start_epoch, args.num_epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        running = 0.0
        running_metrics = {
            "loss": 0.0,
            "clip_loss": 0.0,
            "chunk_loss": 0.0,
            "info_nce_loss": 0.0,
            "smooth_loss": 0.0,
            "clip_cosine_loss": 0.0,
            "chunk_cosine_loss": 0.0,
        }
        seen = 0
        progress = tqdm(train_loader, desc=f"Stage A Epoch {epoch}/{args.num_epochs}", leave=False)
        for step_idx, batch in enumerate(progress, start=1):
            optimizer.zero_grad(set_to_none=True)
            target_clip = batch["z_clip"].to(device, non_blocking=True)
            target_chunks = batch["z_chunks"].to(device, non_blocking=True)
            with get_autocast_context(device, args.amp_dtype):
                outputs = model(batch["text"], device=device)
                losses = stage_a_losses(
                    outputs=outputs,
                    target_clip=target_clip,
                    target_chunks=target_chunks,
                    temperature=model.cfg.info_nce_temperature,
                    clip_weight=args.clip_loss_weight,
                    chunk_weight=args.chunk_loss_weight,
                    info_nce_weight=args.info_nce_weight,
                    smooth_weight=args.smooth_weight,
                    clip_cosine_weight=args.clip_cosine_weight,
                    chunk_cosine_weight=args.chunk_cosine_weight,
                )
            if grad_scaler is not None:
                grad_scaler.scale(losses["loss"]).backward()
            else:
                losses["loss"].backward()
            grad_norm = None
            if args.grad_clip_norm > 0:
                if grad_scaler is not None:
                    grad_scaler.unscale_(optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_parameters, args.grad_clip_norm))
            if grad_scaler is not None:
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            if ema_model is not None:
                update_ema(ema_model, model, args.ema_decay)
            batch_size = batch["z_clip"].shape[0]
            batch_metrics = {key: float(value.detach()) for key, value in losses.items()}
            running += batch_metrics["loss"] * batch_size
            for key in running_metrics:
                running_metrics[key] += batch_metrics[key] * batch_size
            seen += batch_size
            global_step += 1

            if writer is not None:
                for key, value in batch_metrics.items():
                    writer.add_scalar(f"train_step/{key}", value, global_step)
                writer.add_scalar("train_step/lr", optimizer.param_groups[0]["lr"], global_step)
                if "logit_scale" in outputs:
                    writer.add_scalar("train_step/logit_scale", float(outputs["logit_scale"].detach().cpu()), global_step)
                if grad_norm is not None:
                    writer.add_scalar("train_step/grad_norm", grad_norm, global_step)

            if step_idx % max(args.log_every_steps, 1) == 0 or step_idx == len(train_loader):
                progress.set_postfix(
                    loss=f"{batch_metrics['loss']:.4f}",
                    clip=f"{batch_metrics['clip_loss']:.4f}",
                    chunk=f"{batch_metrics['chunk_loss']:.4f}",
                    nce=f"{batch_metrics['info_nce_loss']:.4f}",
                )

        train_loss = running / max(seen, 1)
        train_metrics = {f"train_{key}": value / max(seen, 1) for key, value in running_metrics.items()}
        eval_model = ema_model if ema_model is not None else model
        val_metrics = evaluate(eval_model, val_loader, device, args) if len(val_dataset) > 0 else {"loss": train_loss}
        summary = {"epoch": epoch, "global_step": global_step, "train_loss": train_loss, **train_metrics, **val_metrics}
        history.append(summary)
        logger.info(
            "Epoch %d/%d | train_loss=%.6f | val_loss=%.6f | clip=%.6f | chunk=%.6f | nce=%.6f | smooth=%.6f | lr=%.6e",
            epoch,
            args.num_epochs,
            train_loss,
            float(val_metrics["loss"]),
            train_metrics["train_clip_loss"],
            train_metrics["train_chunk_loss"],
            train_metrics["train_info_nce_loss"],
            train_metrics["train_smooth_loss"],
            optimizer.param_groups[0]["lr"],
        )

        if writer is not None:
            writer.add_scalar("train_epoch/loss", train_loss, epoch)
            for key, value in train_metrics.items():
                writer.add_scalar(f"train_epoch/{key.removeprefix('train_')}", value, epoch)
            for key, value in val_metrics.items():
                writer.add_scalar(f"val_epoch/{key}", float(value), epoch)
            writer.flush()

        current_val = float(val_metrics["loss"])
        improved = current_val < best_val
        if improved:
            best_val = current_val

        checkpoint_payload = {
            "epoch": epoch,
            "global_step": global_step,
            "best_val": best_val,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_model.state_dict() if ema_model is not None else None,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": asdict(model.cfg),
            "train_args": train_args_dict,
            "history": history,
            "dataset_info": dataset_info,
        }
        save_checkpoint(args.output_dir / "stage_a_last.pt", checkpoint_payload)
        if args.save_every_epochs > 0 and epoch % args.save_every_epochs == 0:
            save_checkpoint(checkpoint_dir / f"epoch_{epoch:04d}.pt", checkpoint_payload)
            prune_old_checkpoints(checkpoint_dir, args.keep_last_n_checkpoints)

        if improved:
            save_checkpoint(args.output_dir / "stage_a_best.pt", checkpoint_payload)
            logger.info("Updated best checkpoint at epoch %d with loss %.6f", epoch, best_val)

        write_json(args.output_dir / "history.json", history)

    if writer is not None:
        writer.close()
    logger.info("Stage A training finished. Final artifacts saved to %s", args.output_dir)


if __name__ == "__main__":
    main(tyro.cli(Args))
