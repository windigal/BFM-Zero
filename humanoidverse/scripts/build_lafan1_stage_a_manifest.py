from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import tyro

from humanoidverse.language.stage_a.seed import stable_split_from_filename, write_jsonl_records


@dataclass
class Args:
    dataset_dir: Path = Path("~/dataset/LAFAN1/g1").expanduser()
    output_manifest: Path = Path("artifacts/lafan1_stage_a/lafan1_stage_a_manifest.jsonl")
    output_report: Path = Path("artifacts/lafan1_stage_a/lafan1_stage_a_report.json")
    pattern: str = "*.csv"


def main(args: Args) -> None:
    csv_paths = sorted(args.dataset_dir.glob(args.pattern))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files matched {args.pattern!r} under {args.dataset_dir}")

    records: list[dict[str, object]] = []
    split_counts: dict[str, int] = {}
    for csv_path in csv_paths:
        motion_name = csv_path.stem
        split = stable_split_from_filename(motion_name)
        split_counts[split] = split_counts.get(split, 0) + 1
        records.append(
            {
                "sample_id": motion_name,
                "sample_type": "clip",
                "split": split,
                "filename": motion_name,
                "motion_name": motion_name,
                "primary_text": motion_name,
                "texts": [motion_name],
                "motion_csv_path": str(csv_path.resolve()),
                "source_dataset": "lafan1_g1",
            }
        )

    write_jsonl_records(records, args.output_manifest)
    report = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "pattern": args.pattern,
        "num_records": len(records),
        "split_counts": split_counts,
        "sample_types": {"clip": len(records)},
        "text_source": "motion_name",
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved {len(records)} LAFAN1 Stage A manifest records to {args.output_manifest}")
    print(f"Saved report to {args.output_report}")


if __name__ == "__main__":
    main(tyro.cli(Args))
