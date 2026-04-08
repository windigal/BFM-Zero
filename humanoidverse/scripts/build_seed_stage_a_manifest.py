from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import tyro

from humanoidverse.language.stage_a.seed import (
    StageAFilterConfig,
    build_stage_a_samples,
    write_manifest,
)


@dataclass
class Args:
    dataset_root: Path = Path("~/dataset/seed").expanduser()
    metadata_csv: Path = Path("~/dataset/seed/metadata/seed_metadata_v003.csv").expanduser()
    temporal_jsonl: Path = Path("~/dataset/seed/metadata/seed_metadata_v002_temporal_labels.jsonl").expanduser()
    output_manifest: Path = Path("artifacts/stage_a/seed_manifest.jsonl")
    output_report: Path = Path("artifacts/stage_a/seed_manifest_report.json")
    include_clips: bool = True
    include_events: bool = True
    include_mirrors: bool = True
    require_empty_props: bool = True
    min_clip_duration_s: float = 1.0
    min_event_duration_s: float = 0.6
    max_duration_s: float = 30.0


def main(args: Args) -> None:
    filter_cfg = StageAFilterConfig(
        include_clips=args.include_clips,
        include_events=args.include_events,
        include_mirrors=args.include_mirrors,
        require_empty_props=args.require_empty_props,
        min_clip_duration_s=args.min_clip_duration_s,
        min_event_duration_s=args.min_event_duration_s,
        max_duration_s=args.max_duration_s,
    )
    samples, stats = build_stage_a_samples(
        dataset_root=args.dataset_root,
        metadata_csv=args.metadata_csv,
        temporal_jsonl=args.temporal_jsonl,
        cfg=filter_cfg,
    )
    write_manifest(samples, args.output_manifest)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    with args.output_report.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Saved manifest with {len(samples)} samples to {args.output_manifest}")
    print(f"Saved filter report to {args.output_report}")


if __name__ == "__main__":
    main(tyro.cli(Args))

