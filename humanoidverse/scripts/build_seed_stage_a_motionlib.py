from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import tyro

from humanoidverse.language.stage_a.seed import manifest_records_to_motion_dict, read_manifest


@dataclass
class Args:
    manifest_path: Path = Path("artifacts/stage_a/seed_manifest.jsonl")
    output_dir: Path = Path("artifacts/stage_a/motionlib")
    mjcf_path: Path = Path("humanoidverse/data/robots/g1/g1_29dof.xml")
    target_fps: int = 30
    shard_size: int = 2000
    root_euler_order: str = "xyz"
    output_report: Path = Path("artifacts/stage_a/motionlib_report.json")


def main(args: Args) -> None:
    records = read_manifest(args.manifest_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_written = 0
    total_shards = 0
    shard_reports: list[dict[str, object]] = []
    for shard_idx in range(0, len(records), args.shard_size):
        shard_records = records[shard_idx : shard_idx + args.shard_size]
        motion_dict = manifest_records_to_motion_dict(
            records=shard_records,
            mjcf_path=args.mjcf_path,
            target_fps=args.target_fps,
            root_euler_order=args.root_euler_order,
        )
        shard_name = f"seed_stage_a_{shard_idx // args.shard_size:05d}"
        motion_path = args.output_dir / f"{shard_name}.pkl"
        index_path = args.output_dir / f"{shard_name}.index.json"
        joblib.dump(motion_dict, motion_path)
        shard_report = {
            "motion_file": str(motion_path),
            "num_records_input": len(shard_records),
            "num_records_written": len(motion_dict),
            "sample_ids": list(motion_dict.keys()),
        }
        with index_path.open("w", encoding="utf-8") as f:
            json.dump(shard_report, f, indent=2)
        total_written += len(motion_dict)
        total_shards += 1
        shard_reports.append(shard_report)
        print(f"Saved {len(motion_dict)} motions to {motion_path}")
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(
            {
                "manifest_path": str(args.manifest_path.resolve()),
                "output_dir": str(args.output_dir.resolve()),
                "target_fps": args.target_fps,
                "shard_size": args.shard_size,
                "root_euler_order": args.root_euler_order,
                "total_records_input": len(records),
                "total_records_written": total_written,
                "total_shards": total_shards,
                "shards": shard_reports,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Finished building motionlib shards. Total written motions: {total_written}")
    print(f"Saved motionlib report to {args.output_report}")


if __name__ == "__main__":
    main(tyro.cli(Args))
