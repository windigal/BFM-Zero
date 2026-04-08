from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import tyro
from torch.utils.data import DataLoader

from humanoidverse.language.stage_a.dataset import (
    ClipTextSource,
    StageAItem,
    collate_stage_a,
    load_latent_store,
    load_manifest_records,
    resolve_record_texts,
)
from humanoidverse.language.stage_a.model import StageAModelConfig, StageATextToLatentModel, stage_a_losses


SplitName = Literal["train", "val", "all"]


@dataclass
class Args:
    checkpoint_path: Path
    manifest_path: Path | None = None
    latent_paths: list[Path] | None = None
    latent_glob: str | None = None
    device: str = "cuda"
    split: SplitName = "train"
    clip_text_source: ClipTextSource = "all"
    batch_size: int = 64
    num_workers: int = 0
    clip_loss_weight: float | None = None
    chunk_loss_weight: float | None = None
    info_nce_weight: float | None = None
    smooth_weight: float | None = None
    clip_cosine_weight: float | None = None
    chunk_cosine_weight: float | None = None
    output_json: Path | None = None
    print_examples: int = 5
    retrieval_subset_size: int = 4096
    retrieval_seed: int = 42


def _resolve_manifest_and_latents(args: Args, checkpoint_payload: dict[str, Any]) -> tuple[Path, list[Path]]:
    train_args = checkpoint_payload.get("train_args", {})
    manifest_path = args.manifest_path or Path(train_args["manifest_path"])
    if args.latent_paths is not None:
        latent_paths = args.latent_paths
    elif args.latent_glob is not None:
        latent_paths = [Path(path) for path in sorted(glob.glob(args.latent_glob))]
    else:
        saved_latent_paths = train_args.get("latent_paths")
        if saved_latent_paths:
            latent_paths = [Path(path) for path in saved_latent_paths]
        else:
            latent_glob = train_args.get("latent_glob")
            if latent_glob is None:
                raise ValueError("Could not resolve latent paths from args or checkpoint.")
            latent_paths = [Path(path) for path in sorted(glob.glob(latent_glob))]
    return manifest_path, latent_paths


def _filter_records(records: list[dict[str, Any]], split: SplitName) -> list[dict[str, Any]]:
    if split == "all":
        return records
    return [record for record in records if record["split"] == split]


def _build_eval_items(
    records: list[dict[str, Any]],
    latent_store: dict[str, Any],
    clip_text_source: ClipTextSource,
) -> list[StageAItem]:
    items: list[StageAItem] = []
    for record in records:
        latent = latent_store.get(record["sample_id"])
        if latent is None:
            continue
        for text in resolve_record_texts(record, clip_text_source):
            items.append(
                StageAItem(
                    sample_id=record["sample_id"],
                    text=text,
                    z_clip=torch.from_numpy(latent["z_clip"]).float(),
                    z_chunks=torch.from_numpy(latent["z_chunks"]).float(),
                    sample_type=record["sample_type"],
                    split=record["split"],
                )
            )
    return items


class _EvalDataset(torch.utils.data.Dataset[StageAItem]):
    def __init__(self, items: list[StageAItem]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> StageAItem:
        return self.items[idx]


def _compute_retrieval_metrics(
    text_features: torch.Tensor,
    target_clips: torch.Tensor,
    sample_ids: list[str],
    texts: list[str],
    subset_size: int,
    seed: int,
    topk: tuple[int, ...] = (1, 5),
) -> dict[str, Any]:
    if text_features.shape[0] > subset_size:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        subset_indices = torch.randperm(text_features.shape[0], generator=generator)[:subset_size]
        text_features = text_features[subset_indices]
        target_clips = target_clips[subset_indices]
        sample_ids = [sample_ids[idx] for idx in subset_indices.tolist()]
        texts = [texts[idx] for idx in subset_indices.tolist()]
    logits = text_features @ torch.nn.functional.normalize(target_clips, dim=-1).T
    order = torch.argsort(logits, dim=-1, descending=True)
    labels = torch.arange(order.shape[0], device=order.device)

    metrics: dict[str, Any] = {}
    for k in topk:
        hits = (order[:, : min(k, order.shape[1])] == labels[:, None]).any(dim=-1).float().mean().item()
        metrics[f"retrieval_top{k}"] = hits

    example_rows: list[dict[str, Any]] = []
    for row_idx in range(min(len(sample_ids), 5)):
        ranked = order[row_idx, : min(5, order.shape[1])].tolist()
        example_rows.append(
            {
                "query_sample_id": sample_ids[row_idx],
                "query_text": texts[row_idx],
                "top_matches": [sample_ids[col_idx] for col_idx in ranked],
                "top_scores": [float(logits[row_idx, col_idx]) for col_idx in ranked],
            }
        )
    metrics["retrieval_examples"] = example_rows
    metrics["retrieval_num_items"] = logits.shape[0]
    return metrics


def main(args: Args) -> None:
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
    manifest_path, latent_paths = _resolve_manifest_and_latents(args, checkpoint)
    train_args = checkpoint.get("train_args", {})

    manifest_records = load_manifest_records(manifest_path)
    manifest_records = _filter_records(manifest_records, args.split)
    latent_store = load_latent_store(latent_paths)
    items = _build_eval_items(manifest_records, latent_store, args.clip_text_source)
    if not items:
        raise RuntimeError(f"No eval items found for split={args.split!r}.")

    dataset = _EvalDataset(items)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_stage_a,
        pin_memory=args.device.startswith("cuda"),
    )

    model = StageATextToLatentModel(StageAModelConfig(**checkpoint["model_config"])).to(args.device)
    model_state_dict = checkpoint.get("ema_state_dict") or checkpoint["model_state_dict"]
    model.load_state_dict(model_state_dict)
    model.eval()

    totals = {"loss": 0.0, "clip_loss": 0.0, "chunk_loss": 0.0, "info_nce_loss": 0.0, "smooth_loss": 0.0}
    num_items = 0
    clip_cosines: list[torch.Tensor] = []
    chunk_cosines: list[torch.Tensor] = []
    all_text_features: list[torch.Tensor] = []
    all_target_clips: list[torch.Tensor] = []
    all_sample_ids: list[str] = []
    all_texts: list[str] = []
    example_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["text"], device=torch.device(args.device))
            target_clip = batch["z_clip"].to(args.device)
            target_chunks = batch["z_chunks"].to(args.device)
            losses = stage_a_losses(
                outputs=outputs,
                target_clip=target_clip,
                target_chunks=target_chunks,
                temperature=model.cfg.info_nce_temperature,
                clip_weight=args.clip_loss_weight if args.clip_loss_weight is not None else float(train_args.get("clip_loss_weight", 1.0)),
                chunk_weight=args.chunk_loss_weight if args.chunk_loss_weight is not None else float(train_args.get("chunk_loss_weight", 1.0)),
                info_nce_weight=args.info_nce_weight if args.info_nce_weight is not None else float(train_args.get("info_nce_weight", 0.1)),
                smooth_weight=args.smooth_weight if args.smooth_weight is not None else float(train_args.get("smooth_weight", 0.05)),
                clip_cosine_weight=args.clip_cosine_weight if args.clip_cosine_weight is not None else float(train_args.get("clip_cosine_weight", 0.0)),
                chunk_cosine_weight=args.chunk_cosine_weight if args.chunk_cosine_weight is not None else float(train_args.get("chunk_cosine_weight", 0.0)),
            )

            batch_size = target_clip.shape[0]
            num_items += batch_size
            for key in totals:
                totals[key] += float(losses[key]) * batch_size

            clip_cos = torch.nn.functional.cosine_similarity(outputs["clip_pred"], target_clip, dim=-1)
            pred_chunks = outputs["chunk_pred"].reshape(-1, outputs["chunk_pred"].shape[-1])
            tgt_chunks = target_chunks.reshape(-1, target_chunks.shape[-1])
            chunk_cos = torch.nn.functional.cosine_similarity(pred_chunks, tgt_chunks, dim=-1)
            clip_cosines.append(clip_cos.cpu())
            chunk_cosines.append(chunk_cos.cpu())
            all_text_features.append(outputs["contrastive_text"].cpu())
            all_target_clips.append(target_clip.cpu())
            all_sample_ids.extend(batch["sample_id"])
            all_texts.extend(batch["text"])

            if len(example_rows) < args.print_examples:
                for row_idx in range(min(args.print_examples - len(example_rows), batch_size)):
                    example_rows.append(
                        {
                            "sample_id": batch["sample_id"][row_idx],
                            "text": batch["text"][row_idx],
                            "clip_cosine": float(clip_cos[row_idx].cpu()),
                            "clip_l2": float(torch.norm(outputs["clip_pred"][row_idx] - target_clip[row_idx]).cpu()),
                            "chunk_cosine_mean": float(
                                torch.nn.functional.cosine_similarity(
                                    outputs["chunk_pred"][row_idx], target_chunks[row_idx], dim=-1
                                )
                                .mean()
                                .cpu()
                            ),
                        }
                    )

    metrics = {key: value / max(num_items, 1) for key, value in totals.items()}
    metrics["num_items"] = num_items
    metrics["split"] = args.split
    metrics["clip_cosine_mean"] = float(torch.cat(clip_cosines).mean())
    metrics["chunk_cosine_mean"] = float(torch.cat(chunk_cosines).mean())
    retrieval = _compute_retrieval_metrics(
        text_features=torch.cat(all_text_features, dim=0),
        target_clips=torch.cat(all_target_clips, dim=0),
        sample_ids=all_sample_ids,
        texts=all_texts,
        subset_size=args.retrieval_subset_size,
        seed=args.retrieval_seed,
    )
    metrics.update(retrieval)
    metrics["examples"] = example_rows

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main(tyro.cli(Args))
