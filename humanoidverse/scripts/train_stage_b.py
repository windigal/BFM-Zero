from __future__ import annotations

import copy
import glob
import gzip
import inspect
import json
import logging
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from humanoidverse.language.stage_a.dataset import load_latent_store, load_manifest_records
from humanoidverse.language.stage_b.dataset import (
    BFMTextOpPrimitiveSequenceIterableConfig,
    BFMTextOpPrimitiveSequenceIterableDataset,
    BFMTextOpSequenceDataset,
    BFMTextOpSequenceDatasetConfig,
    collate_bfm_textop_sequences,
)
from humanoidverse.language.stage_b.model import (
    BFMTextOpLossConfig,
    BFMTextOpModel,
    BFMTextOpModelConfig,
    bfm_textop_losses,
)


DEFAULT_MANIFEST_PATH = Path("artifacts/stage_a/seed_manifest.jsonl")
DEFAULT_LATENT_GLOB = "artifacts/stage_a/latents/*.joblib"
DEFAULT_OUTPUT_DIR = Path("results/stage_b")


@dataclass
class Args:
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    latent_paths: list[Path] | None = None
    latent_glob: str = DEFAULT_LATENT_GLOB
    primitive_dataset_dir: Path | None = None
    primitive_train_glob: str | None = None
    primitive_val_glob: str | None = None
    output_dir: Path = DEFAULT_OUTPUT_DIR
    device: str = "cuda"
    batch_size: int = 64
    num_workers: int = 4
    num_epochs: int = 10
    lr: float = 2e-4
    text_encoder_lr_scale: float = 0.2
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.03
    weight_decay: float = 1e-4
    history_len: int = 4
    future_len: int = 16
    num_primitives: int = 8
    window_stride: int = 16
    sequence_stride_primitives: int = 1
    primitive_shuffle_buffer_size: int = 2048
    sample_types: tuple[str, ...] = ("clip",)
    clip_model_name: str = "openai/clip-vit-base-patch32"
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    diffusion_steps: int = 5
    objective_type: str = "diffusion"
    sampling_steps: int | None = None
    cond_mask_prob: float = 0.1
    freeze_text_encoder: bool = True
    target_representation: str = "dct"
    dct_keep_coeffs: int | None = 4
    flow_loss_weight: float = 1.0
    flow_sigma_min: float = 0.0
    flow_time_sampling: str = "logit_normal"
    flow_logit_mean: float = 0.0
    flow_logit_std: float = 1.0
    flow_sampler: str = "euler"
    target_loss_weight: float = 1.0
    future_loss_weight: float = 1.0
    cosine_loss_weight: float = 0.25
    boundary_loss_weight: float = 0.05
    smooth_loss_weight: float = 0.05
    grad_clip_norm: float = 1.0
    amp_dtype: str = "bf16"
    allow_tf32: bool = True
    fused_adamw: bool = True
    max_rollout_prob: float = 0.5
    rollout_start_epoch: int = 1
    rollout_use_sampling: bool = False
    rollout_guidance_scale: float = 1.0
    val_rollout_probability: float = 0.0
    log_every_steps: int = 10
    save_every_epochs: int = 1
    keep_last_n_checkpoints: int = 5
    persistent_workers: bool = True
    prefetch_factor: int = 4
    use_timestamp_subdir: bool = True
    resume_from: Path | None = None
    seed: int = 42


def resolve_latent_paths(args: Args) -> list[Path]:
    if args.latent_paths:
        return args.latent_paths
    return sorted(Path(path) for path in glob.glob(args.latent_glob))


def _list_primitive_shards(split_dir: Path) -> list[Path]:
    shard_paths: list[Path] = []
    for path in sorted(split_dir.iterdir()):
        if not path.is_file() or path.name == "_index.jsonl":
            continue
        if path.suffix in {".parquet", ".jsonl", ".gz"}:
            shard_paths.append(path)
    return shard_paths


def resolve_primitive_shard_paths(args: Args, split: str) -> list[Path]:
    explicit_glob = args.primitive_train_glob if split == "train" else args.primitive_val_glob
    if explicit_glob:
        return sorted(Path(path) for path in glob.glob(explicit_glob))
    if args.primitive_dataset_dir is None:
        return []
    split_dir = args.primitive_dataset_dir / split
    if not split_dir.exists():
        return []
    return _list_primitive_shards(split_dir)


def infer_prompt_dim_from_primitive_shards(dataset_dir: Path | None, shard_paths: list[Path]) -> int:
    if dataset_dir is not None:
        meta_path = dataset_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            prompt_dim = meta.get("prompt_dim")
            if prompt_dim is not None:
                return int(prompt_dim)
    for path in shard_paths:
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            batch_iter = pq.ParquetFile(path).iter_batches(batch_size=1)
            first_batch = next(batch_iter, None)
            if first_batch is None:
                continue
            batch_dict = first_batch.to_pydict()
            if batch_dict.get("prompt_dim"):
                return int(batch_dict["prompt_dim"][0])
            continue
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                return int(row["prompt_dim"])
    raise RuntimeError("Could not infer prompt_dim from primitive shards.")


def load_primitive_dataset_meta(dataset_dir: Path | None) -> dict[str, Any] | None:
    if dataset_dir is None:
        return None
    meta_path = dataset_dir / "meta.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def validate_primitive_dataset_compatibility(
    *,
    args: Args,
    dataset_meta: dict[str, Any] | None,
    prompt_dim: int,
) -> None:
    if dataset_meta is None:
        return
    mismatches: list[str] = []
    for field_name, expected_value in (
        ("history_len", args.history_len),
        ("future_len", args.future_len),
        ("prompt_dim", prompt_dim),
    ):
        actual_value = dataset_meta.get(field_name)
        if actual_value is None:
            continue
        if int(actual_value) != int(expected_value):
            mismatches.append(f"{field_name}: dataset={actual_value}, train_arg={expected_value}")
    if mismatches:
        raise ValueError(
            "Primitive dataset is incompatible with the requested Stage B config. "
            "Regenerate shards with matching settings or point training at the correct dataset.\n"
            + "\n".join(mismatches)
        )


def using_primitive_dataset(args: Args) -> bool:
    return args.primitive_dataset_dir is not None or args.primitive_train_glob is not None or args.primitive_val_glob is not None


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
    logger = logging.getLogger("stage_b_train")
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


def configure_tf32(*, allow_tf32: bool, device: torch.device, logger: logging.Logger) -> None:
    enabled = allow_tf32 and device.type == "cuda" and torch.cuda.is_available()
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = enabled
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = enabled
    if enabled:
        torch.set_float32_matmul_precision("high")
    logger.info("TF32 matmul/cudnn enabled=%s", enabled)


def build_optimizer(
    model: BFMTextOpModel,
    args: Args,
    *,
    device: torch.device,
    logger: logging.Logger,
) -> torch.optim.Optimizer:
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
        param_groups.append({"params": other_params, "lr": args.lr, "weight_decay": args.weight_decay})
    if text_encoder_params:
        param_groups.append(
            {
                "params": text_encoder_params,
                "lr": args.lr * args.text_encoder_lr_scale,
                "weight_decay": args.weight_decay,
            }
        )
    adamw_kwargs: dict[str, object] = {}
    fused_requested = args.fused_adamw and device.type == "cuda"
    fused_supported = "fused" in inspect.signature(torch.optim.AdamW).parameters
    if fused_requested and fused_supported:
        adamw_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(param_groups, **adamw_kwargs)
    logger.info(
        "Optimizer=AdamW fused=%s",
        bool(adamw_kwargs.get("fused", False)),
    )
    return optimizer


def build_lr_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float, min_lr_ratio: float) -> LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, schedule)


def prune_old_checkpoints(checkpoint_dir: Path, keep_last_n: int) -> None:
    if keep_last_n <= 0:
        return
    ckpts = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if len(ckpts) <= keep_last_n:
        return
    for stale in ckpts[: len(ckpts) - keep_last_n]:
        stale.unlink(missing_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_amp_dtype(name: str) -> torch.dtype | None:
    lowered = name.lower()
    if lowered == "none":
        return None
    if lowered == "fp16":
        return torch.float16
    if lowered == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported amp_dtype={name!r}")


def get_rollout_probability(epoch: int, args: Args) -> float:
    if epoch < args.rollout_start_epoch:
        return 0.0
    horizon = max(args.num_epochs - args.rollout_start_epoch - 1, 1)
    progress = min(max(epoch - args.rollout_start_epoch, 0) / horizon, 1.0)
    return progress * args.max_rollout_prob


def _primitive_text_at(batch_texts: list[list[str]], primitive_idx: int) -> list[str]:
    return [texts[primitive_idx] for texts in batch_texts]


def run_epoch(
    *,
    model: BFMTextOpModel,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer | None,
    lr_scheduler: LambdaLR | None,
    loss_cfg: BFMTextOpLossConfig,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    rollout_probability: float,
    rollout_use_sampling: bool,
    rollout_guidance_scale: float,
    grad_clip_norm: float,
    log_every_steps: int,
    logger: logging.Logger,
    writer: SummaryWriter,
    epoch: int,
    global_step: int,
    split: str,
) -> tuple[dict[str, float], int]:
    is_train = optimizer is not None
    model.train(is_train)
    metrics_running: dict[str, float] = {}
    metrics_count = 0

    progress = tqdm(loader, desc=f"{split} epoch {epoch}", leave=False)
    for batch_idx, batch in enumerate(progress):
        z_hist = batch["z_hist"].to(device)
        z_fut = batch["z_fut"].to(device)
        z_fut_target = batch["z_fut_target"].to(device)
        primitive_texts = batch["primitive_texts"]

        total_loss = torch.zeros((), device=device)
        aggregate_terms: dict[str, torch.Tensor] = {}
        prev_future: torch.Tensor | None = None

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        autocast_enabled = amp_dtype is not None and device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
            for primitive_idx in range(z_hist.shape[1]):
                gt_history = z_hist[:, primitive_idx]
                target_future = z_fut[:, primitive_idx]
                target_future_repr = z_fut_target[:, primitive_idx]
                texts = _primitive_text_at(primitive_texts, primitive_idx)

                use_rollout_history = primitive_idx > 0 and prev_future is not None and random.random() < rollout_probability
                history = prev_future[:, -model.cfg.history_len :] if use_rollout_history else gt_history

                objective_inputs = model.prepare_training_inputs(target_future_repr)
                x_t = objective_inputs["x_t"]
                timesteps = objective_inputs["timesteps"]
                source_noise = objective_inputs["source_noise"]
                objective_target = objective_inputs["objective_target"]

                raw_pred = model(
                    x_t=x_t,  # type: ignore[arg-type]
                    timesteps=timesteps,  # type: ignore[arg-type]
                    texts=texts,
                    history=history,
                    force_uncond=False,
                )
                pred_target = model.reconstruct_target_from_model_output(
                    raw_pred,
                    source_noise=source_noise,  # type: ignore[arg-type]
                )
                pred_future = model.decode_future_target(pred_target)
                losses = bfm_textop_losses(
                    pred_target=pred_target,
                    target_target=target_future_repr,
                    pred_future=pred_future,
                    target_future=target_future,
                    history=history,
                    target_history=gt_history,
                    cfg=loss_cfg,
                    use_reconstruction_loss=model.target_representation != "raw",
                )
                objective_loss = torch.zeros((), device=device)
                if model.objective_type == "flow":
                    objective_loss = torch.nn.functional.mse_loss(
                        raw_pred,
                        objective_target,  # type: ignore[arg-type]
                    )
                total_loss = total_loss + losses["loss"] + loss_cfg.flow_weight * objective_loss
                aggregate_terms.setdefault("rollout_history_ratio", torch.zeros((), device=device))
                aggregate_terms["rollout_history_ratio"] = aggregate_terms["rollout_history_ratio"] + torch.tensor(
                    float(use_rollout_history),
                    device=device,
                )
                aggregate_terms["objective_loss"] = aggregate_terms.get(
                    "objective_loss",
                    torch.zeros_like(objective_loss),
                ) + objective_loss.detach()
                for key, value in losses.items():
                    aggregate_terms[key] = aggregate_terms.get(key, torch.zeros_like(value)) + value

                if primitive_idx + 1 < z_hist.shape[1]:
                    if rollout_use_sampling:
                        with torch.no_grad():
                            sampled_target = model.sample(
                                texts=texts,
                                history=history.detach(),
                                guidance_scale=rollout_guidance_scale,
                            )
                            prev_future = model.decode_future_target(sampled_target)
                    else:
                        prev_future = pred_future.detach()

        total_loss = total_loss / z_hist.shape[1]
        aggregate_terms = {key: value / z_hist.shape[1] for key, value in aggregate_terms.items()}

        if is_train:
            total_loss.backward()
            if optimizer is None:
                raise RuntimeError("optimizer unexpectedly None in training mode")
            if math.isfinite(float(total_loss.detach().cpu())) and any(
                param.grad is not None for param in model.parameters() if param.requires_grad
            ):
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
            global_step += 1

        scalar_metrics = {
            "loss": float(total_loss.detach().cpu()),
            **{key: float(value.detach().cpu()) for key, value in aggregate_terms.items()},
        }
        for key, value in scalar_metrics.items():
            metrics_running[key] = metrics_running.get(key, 0.0) + value
            if is_train:
                writer.add_scalar(f"{split}/{key}", value, global_step)
        metrics_count += 1

        mean_loss = metrics_running["loss"] / max(metrics_count, 1)
        progress.set_postfix(loss=f"{mean_loss:.4f}", rollout=f"{rollout_probability:.2f}")
        # if batch_idx % max(log_every_steps, 1) == 0:
        #     logger.info(
        #         "%s epoch=%d step=%d batch=%d loss=%.4f rollout_prob=%.3f",
        #         split,
        #         epoch,
        #         global_step,
        #         batch_idx,
        #         scalar_metrics["loss"],
        #         rollout_probability,
        #     )

    mean_metrics = {key: value / max(metrics_count, 1) for key, value in metrics_running.items()}
    for key, value in mean_metrics.items():
        writer.add_scalar(f"{split}_epoch/{key}", value, epoch)
    return mean_metrics, global_step


def main(args: Args) -> None:
    set_seed(args.seed)

    args.output_dir = resolve_output_dir(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output_dir)
    writer = SummaryWriter(args.output_dir / "tb")

    dataset_cfg: Any
    data_source: str
    latent_paths: list[Path] = []
    if using_primitive_dataset(args):
        train_shards = resolve_primitive_shard_paths(args, "train")
        val_shards = resolve_primitive_shard_paths(args, "val")
        if not train_shards:
            raise FileNotFoundError(
                "No primitive train shards found. Run humanoidverse.scripts.build_stage_b_primitive_dataset first."
            )
        prompt_dim = infer_prompt_dim_from_primitive_shards(args.primitive_dataset_dir, train_shards)
        validate_primitive_dataset_compatibility(
            args=args,
            dataset_meta=load_primitive_dataset_meta(args.primitive_dataset_dir),
            prompt_dim=prompt_dim,
        )
        dataset_cfg = BFMTextOpPrimitiveSequenceIterableConfig(
            history_len=args.history_len,
            future_len=args.future_len,
            prompt_dim=prompt_dim,
            num_primitives=args.num_primitives,
            sequence_stride_primitives=args.sequence_stride_primitives,
            shuffle=True,
            shuffle_buffer_size=args.primitive_shuffle_buffer_size,
            seed=args.seed,
            target_representation=args.target_representation,
            dct_keep_coeffs=args.dct_keep_coeffs,
        )
        train_dataset = BFMTextOpPrimitiveSequenceIterableDataset(train_shards, split="train", cfg=dataset_cfg)
        val_dataset = (
            BFMTextOpPrimitiveSequenceIterableDataset(
                val_shards,
                split="val",
                cfg=BFMTextOpPrimitiveSequenceIterableConfig(
                    history_len=args.history_len,
                    future_len=args.future_len,
                    prompt_dim=prompt_dim,
                    num_primitives=args.num_primitives,
                    sequence_stride_primitives=args.sequence_stride_primitives,
                    shuffle=False,
                    shuffle_buffer_size=args.primitive_shuffle_buffer_size,
                    seed=args.seed,
                    target_representation=args.target_representation,
                    dct_keep_coeffs=args.dct_keep_coeffs,
                ),
            )
            if val_shards
            else None
        )
        data_source = "primitive_shards"
        train_sequences = len(train_dataset)
        val_sequences = len(val_dataset) if val_dataset is not None else 0
    else:
        latent_paths = resolve_latent_paths(args)
        if not latent_paths:
            raise FileNotFoundError(
                "No latent files found. Run humanoidverse.scripts.extract_stage_a_teacher_latents with --save-z-seq."
            )

        manifest_records = load_manifest_records(args.manifest_path)
        latent_store = load_latent_store(latent_paths)
        first_latent = next((value for key, value in latent_store.items() if key != "__meta__"), None)
        if first_latent is None:
            raise RuntimeError("Latent store is empty.")
        if "z_seq" not in first_latent:
            raise RuntimeError(
                "Latent store does not contain z_seq. Re-extract teacher latents with --save-z-seq or train from primitive shards."
            )
        prompt_dim = int(first_latent["z_seq"].shape[-1])

        dataset_cfg = BFMTextOpSequenceDatasetConfig(
            history_len=args.history_len,
            future_len=args.future_len,
            num_primitives=args.num_primitives,
            window_stride=args.window_stride,
            sample_types=args.sample_types,
            target_representation=args.target_representation,
            dct_keep_coeffs=args.dct_keep_coeffs,
        )
        train_dataset = BFMTextOpSequenceDataset(
            manifest_records=manifest_records,
            latent_store=latent_store,
            split="train",
            cfg=dataset_cfg,
        )
        val_dataset = BFMTextOpSequenceDataset(
            manifest_records=manifest_records,
            latent_store=latent_store,
            split="val",
            cfg=dataset_cfg,
        )
        data_source = "z_seq_latents"
        train_sequences = len(train_dataset)
        val_sequences = len(val_dataset)

    if train_sequences == 0:
        raise RuntimeError("Stage B train dataset is empty.")
    if val_sequences == 0:
        logger.warning("Stage B val dataset is empty; validation metrics will be skipped.")

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_bfm_textop_sequences,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    if data_source == "primitive_shards":
        train_loader = DataLoader(train_dataset, shuffle=False, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs) if val_dataset is not None else None
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs) if val_sequences > 0 else None

    model_cfg = BFMTextOpModelConfig(
        clip_model_name=args.clip_model_name,
        prompt_dim=prompt_dim,
        history_len=args.history_len,
        future_len=args.future_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        diffusion_steps=args.diffusion_steps,
        objective_type=args.objective_type,
        sampling_steps=args.sampling_steps,
        freeze_text_encoder=args.freeze_text_encoder,
        cond_mask_prob=args.cond_mask_prob,
        target_representation=args.target_representation,
        dct_keep_coeffs=args.dct_keep_coeffs,
        flow_sigma_min=args.flow_sigma_min,
        flow_time_sampling=args.flow_time_sampling,
        flow_logit_mean=args.flow_logit_mean,
        flow_logit_std=args.flow_logit_std,
        flow_sampler=args.flow_sampler,
    )
    loss_cfg = BFMTextOpLossConfig(
        target_weight=args.target_loss_weight,
        future_weight=args.future_loss_weight,
        cosine_weight=args.cosine_loss_weight,
        boundary_weight=args.boundary_loss_weight,
        smooth_weight=args.smooth_loss_weight,
        flow_weight=args.flow_loss_weight,
    )

    device = torch.device(args.device)
    configure_tf32(allow_tf32=args.allow_tf32, device=device, logger=logger)
    model = BFMTextOpModel(model_cfg).to(device)
    optimizer = build_optimizer(model, args, device=device, logger=logger)
    total_steps = max(len(train_loader) * args.num_epochs, 1)
    lr_scheduler = build_lr_scheduler(optimizer, total_steps, args.warmup_ratio, args.min_lr_ratio)
    amp_dtype = resolve_amp_dtype(args.amp_dtype)

    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")
    best_checkpoint_payload: dict[str, Any] | None = None

    if args.resume_from is not None:
        payload = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        lr_scheduler.load_state_dict(payload["scheduler_state"])
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload.get("global_step", 0))
        best_val_loss = float(payload.get("best_val_loss", best_val_loss))
        logger.info("Resumed from %s at epoch %d", args.resume_from, start_epoch)

    trainable, frozen = count_parameters(model)
    config_payload = {
        "args": asdict(args),
        "model_config": asdict(model_cfg),
        "loss_config": asdict(loss_cfg),
        "dataset_config": asdict(dataset_cfg),
        "data_source": data_source,
        "latent_paths": [str(path.resolve()) for path in latent_paths],
        "train_sequences": train_sequences,
        "val_sequences": val_sequences,
        "prompt_dim": prompt_dim,
        "trainable_params": trainable,
        "frozen_params": frozen,
    }
    write_json(args.output_dir / "config.json", config_payload)
    logger.info(
        "Stage B config written to %s | source=%s train_sequences=%d val_sequences=%d prompt_dim=%d trainable=%d frozen=%d",
        args.output_dir / "config.json",
        data_source,
        train_sequences,
        val_sequences,
        prompt_dim,
        trainable,
        frozen,
    )

    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.num_epochs + 1):
        rollout_prob = get_rollout_probability(epoch, args)
        train_metrics, global_step = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss_cfg=loss_cfg,
            device=device,
            amp_dtype=amp_dtype,
            rollout_probability=rollout_prob,
            rollout_use_sampling=args.rollout_use_sampling,
            rollout_guidance_scale=args.rollout_guidance_scale,
            grad_clip_norm=args.grad_clip_norm,
            log_every_steps=args.log_every_steps,
            logger=logger,
            writer=writer,
            epoch=epoch,
            global_step=global_step,
            split="train",
        )
        logger.info("train epoch=%d metrics=%s", epoch, {k: round(v, 6) for k, v in train_metrics.items()})

        val_metrics = None
        if val_loader is not None:
            val_metrics, global_step = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                lr_scheduler=None,
                loss_cfg=loss_cfg,
                device=device,
                amp_dtype=amp_dtype,
                rollout_probability=max(args.val_rollout_probability, 0.0),
                rollout_use_sampling=args.rollout_use_sampling,
                rollout_guidance_scale=args.rollout_guidance_scale,
                grad_clip_norm=args.grad_clip_norm,
                log_every_steps=max(args.log_every_steps, 50),
                logger=logger,
                writer=writer,
                epoch=epoch,
                global_step=global_step,
                split="val",
            )
            logger.info("val epoch=%d metrics=%s", epoch, {k: round(v, 6) for k, v in val_metrics.items()})

        checkpoint_payload = {
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": lr_scheduler.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "args": asdict(args),
            "dataset_config": asdict(dataset_cfg),
            "loss_config": asdict(loss_cfg),
        }
        checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        model.save_checkpoint(checkpoint_path, checkpoint_payload)
        if epoch % max(args.save_every_epochs, 1) == 0:
            prune_old_checkpoints(checkpoint_dir, args.keep_last_n_checkpoints)

        current_val_loss = float(val_metrics["loss"]) if val_metrics is not None else float(train_metrics["loss"])
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_checkpoint_payload = copy.deepcopy(checkpoint_payload)
            best_checkpoint_payload["best_val_loss"] = best_val_loss
            model.save_checkpoint(args.output_dir / "stage_b_best.pt", best_checkpoint_payload)

        latest_payload = copy.deepcopy(checkpoint_payload)
        latest_payload["best_val_loss"] = best_val_loss
        model.save_checkpoint(args.output_dir / "stage_b_last.pt", latest_payload)

    if best_checkpoint_payload is None:
        best_checkpoint_payload = {
            "epoch": 0,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "args": asdict(args),
        }
    write_json(
        args.output_dir / "summary.json",
        {
            "best_val_loss": best_val_loss,
            "last_epoch": args.num_epochs,
            "best_checkpoint": str((args.output_dir / "stage_b_best.pt").resolve()),
            "last_checkpoint": str((args.output_dir / "stage_b_last.pt").resolve()),
        },
    )
    writer.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
