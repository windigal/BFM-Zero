from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import tyro

from humanoidverse.language.stage_a.seed import (
    SeedTrainFilterConfig,
    build_seed_training_records,
    write_jsonl_records,
)


@dataclass
class Args:
    dataset_root: Path = Path("~/dataset/seed").expanduser()
    metadata_csv: Path = Path("~/dataset/seed/metadata/seed_metadata_v003.csv").expanduser()
    output_manifest: Path = Path("artifacts/seed_train/seed_train_manifest.jsonl")
    output_report: Path = Path("artifacts/seed_train/seed_train_manifest_report.json")
    include_mirrors: bool = True
    min_clip_duration_s: float = 1.0
    max_duration_s: float = 120.0


def main(args: Args) -> None:
    filter_cfg = SeedTrainFilterConfig(
        include_mirrors=args.include_mirrors,
        min_clip_duration_s=args.min_clip_duration_s,
        max_duration_s=args.max_duration_s,
    )
    records, stats = build_seed_training_records(
        dataset_root=args.dataset_root,
        metadata_csv=args.metadata_csv,
        cfg=filter_cfg,
    )
    write_jsonl_records(records, args.output_manifest)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved filtered training manifest with {len(records)} clips to {args.output_manifest}")
    print(f"Saved filter report to {args.output_report}")


if __name__ == "__main__":
    main(tyro.cli(Args))
