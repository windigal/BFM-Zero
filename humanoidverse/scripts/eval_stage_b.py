from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from torch.utils.data import DataLoader

from humanoidverse.language.stage_a.dataset import load_latent_store, load_manifest_records
from humanoidverse.language.stage_b.dataset import (
    BFMTextOpPrimitiveSequenceIterableConfig,
    BFMTextOpPrimitiveSequenceIterableDataset,
    BFMTextOpSequenceDataset,
    BFMTextOpSequenceDatasetConfig,
    collate_bfm_textop_sequences,
)
from humanoidverse.language.stage_b.controller import load_bfm_textop_checkpoint
from humanoidverse.language.stage_b.model import BFMTextOpLossConfig, bfm_textop_losses


@dataclass
class Args:
    checkpoint_path: Path = Path("results/stage_b/stage_b_best.pt")
    manifest_path: Path = Path("artifacts/stage_a/seed_manifest.jsonl")
    latent_paths: list[Path] | None = None
    latent_glob: str = "artifacts/stage_a/latents/*.joblib"
    primitive_dataset_dir: Path | None = None
    primitive_glob: str | None = None
    split: str = "val"
    batch_size: int = 64
    num_workers: int = 2
    device: str = "cuda"
    output_json: Path | None = None


def _resolve_latent_paths(args: Args) -> list[Path]:
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


def _resolve_primitive_paths(args: Args) -> list[Path]:
    if args.primitive_glob is not None:
        return sorted(Path(path) for path in glob.glob(args.primitive_glob))
    if args.primitive_dataset_dir is None:
        return []
    split_dir = args.primitive_dataset_dir / args.split
    if not split_dir.exists():
        return []
    return _list_primitive_shards(split_dir)


def _load_primitive_dataset_meta(dataset_dir: Path | None) -> dict | None:
    if dataset_dir is None:
        return None
    meta_path = dataset_dir / "meta.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _validate_primitive_dataset_meta(
    *,
    dataset_meta: dict | None,
    dataset_cfg: BFMTextOpPrimitiveSequenceIterableConfig,
) -> None:
    if dataset_meta is None:
        return
    mismatches: list[str] = []
    for field_name, expected_value in (
        ("history_len", dataset_cfg.history_len),
        ("future_len", dataset_cfg.future_len),
        ("prompt_dim", dataset_cfg.prompt_dim),
    ):
        actual_value = dataset_meta.get(field_name)
        if actual_value is None:
            continue
        if int(actual_value) != int(expected_value):
            mismatches.append(f"{field_name}: dataset={actual_value}, checkpoint={expected_value}")
    if mismatches:
        raise ValueError(
            "Primitive dataset is incompatible with the Stage B checkpoint configuration.\n" + "\n".join(mismatches)
        )


def main(args: Args) -> None:
    model, payload = load_bfm_textop_checkpoint(args.checkpoint_path, device=args.device)
    loss_cfg = BFMTextOpLossConfig(**payload["loss_config"])
    data_source = payload.get("data_source", "z_seq_latents")

    if data_source == "primitive_shards" or args.primitive_dataset_dir is not None or args.primitive_glob is not None:
        shard_paths = _resolve_primitive_paths(args)
        if not shard_paths:
            raise FileNotFoundError("No primitive shards found for evaluation.")
        dataset_cfg = BFMTextOpPrimitiveSequenceIterableConfig(**payload["dataset_config"])
        _validate_primitive_dataset_meta(
            dataset_meta=_load_primitive_dataset_meta(args.primitive_dataset_dir),
            dataset_cfg=dataset_cfg,
        )
        dataset = BFMTextOpPrimitiveSequenceIterableDataset(
            shard_paths=shard_paths,
            split=args.split,
            cfg=BFMTextOpPrimitiveSequenceIterableConfig(
                history_len=dataset_cfg.history_len,
                future_len=dataset_cfg.future_len,
                prompt_dim=dataset_cfg.prompt_dim,
                num_primitives=dataset_cfg.num_primitives,
                sequence_stride_primitives=dataset_cfg.sequence_stride_primitives,
                shuffle=False,
                shuffle_buffer_size=dataset_cfg.shuffle_buffer_size,
                seed=dataset_cfg.seed,
                target_representation=dataset_cfg.target_representation,
                dct_keep_coeffs=dataset_cfg.dct_keep_coeffs,
            ),
        )
    else:
        dataset_cfg = BFMTextOpSequenceDatasetConfig(**payload["dataset_config"])
        manifest_records = load_manifest_records(args.manifest_path)
        latent_store = load_latent_store(_resolve_latent_paths(args))
        dataset = BFMTextOpSequenceDataset(
            manifest_records=manifest_records,
            latent_store=latent_store,
            split=args.split,
            cfg=dataset_cfg,
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_bfm_textop_sequences,
    )

    device = torch.device(args.device)
    aggregate = {
        "loss": 0.0,
        "target_loss": 0.0,
        "future_loss": 0.0,
        "cosine_loss": 0.0,
        "boundary_loss": 0.0,
        "smooth_loss": 0.0,
        "teacher_forced_loss": 0.0,
    }
    num_batches = 0

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            z_hist = batch["z_hist"].to(device)
            z_fut = batch["z_fut"].to(device)
            z_fut_target = batch["z_fut_target"].to(device)
            primitive_texts = batch["primitive_texts"]
            prev_future = None
            seq_loss = torch.zeros((), device=device)
            teacher_loss = torch.zeros((), device=device)
            for primitive_idx in range(z_hist.shape[1]):
                gt_history = z_hist[:, primitive_idx]
                target_future = z_fut[:, primitive_idx]
                target_future_repr = z_fut_target[:, primitive_idx]
                texts = [texts[primitive_idx] for texts in primitive_texts]
                sampled_target = model.sample(
                    texts=texts,
                    history=(prev_future[:, -model.cfg.history_len :] if prev_future is not None else gt_history),
                )
                sampled = model.decode_future_target(sampled_target)
                rollout_losses = bfm_textop_losses(
                    pred_target=sampled_target,
                    target_target=target_future_repr,
                    pred_future=sampled,
                    target_future=target_future,
                    history=(prev_future[:, -model.cfg.history_len :] if prev_future is not None else gt_history),
                    target_history=gt_history,
                    cfg=loss_cfg,
                    use_reconstruction_loss=model.target_representation != "raw",
                )
                seq_loss = seq_loss + rollout_losses["loss"]
                prev_future = sampled

                if model.objective_type == "flow":
                    timesteps = torch.ones((target_future_repr.shape[0],), device=device, dtype=target_future_repr.dtype)
                    teacher_source_noise = torch.zeros_like(target_future_repr)
                else:
                    timesteps = torch.zeros((target_future_repr.shape[0],), device=device, dtype=torch.long)
                    teacher_source_noise = None
                teacher_raw = model(
                    x_t=target_future_repr,
                    timesteps=timesteps,
                    texts=texts,
                    history=gt_history,
                    force_uncond=False,
                )
                teacher_pred_target = model.reconstruct_target_from_model_output(
                    teacher_raw,
                    source_noise=teacher_source_noise,
                )
                teacher_pred = model.decode_future_target(teacher_pred_target)
                teacher_losses = bfm_textop_losses(
                    pred_target=teacher_pred_target,
                    target_target=target_future_repr,
                    pred_future=teacher_pred,
                    target_future=target_future,
                    history=gt_history,
                    target_history=gt_history,
                    cfg=loss_cfg,
                    use_reconstruction_loss=model.target_representation != "raw",
                )
                teacher_loss = teacher_loss + teacher_losses["loss"]
                for key in ("target_loss", "future_loss", "cosine_loss", "boundary_loss", "smooth_loss"):
                    aggregate[key] += float((rollout_losses[key] / z_hist.shape[1]).cpu())

            aggregate["loss"] += float((seq_loss / z_hist.shape[1]).cpu())
            aggregate["teacher_forced_loss"] += float((teacher_loss / z_hist.shape[1]).cpu())
            num_batches += 1

    if num_batches == 0:
        raise RuntimeError(f"No batches available for split={args.split!r}")
    metrics = {key: value / num_batches for key, value in aggregate.items()}
    metrics["num_sequences"] = len(dataset)
    metrics["checkpoint_path"] = str(args.checkpoint_path.resolve())
    metrics["split"] = args.split

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(tyro.cli(Args))
