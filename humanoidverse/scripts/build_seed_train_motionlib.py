from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import tyro

from humanoidverse.language.stage_a.seed import (
    build_fixed_length_clip_records,
    manifest_records_to_motion_dict,
    read_manifest,
    write_jsonl_records,
)


@dataclass
class Args:
    manifest_path: Path = Path("artifacts/seed_train/seed_train_manifest.jsonl")
    output_motionlib: Path = Path("artifacts/seed_train/seed_train_10s_clipped.pkl")
    output_clipped_manifest: Path = Path("artifacts/seed_train/seed_train_10s_clipped_manifest.jsonl")
    output_report: Path = Path("artifacts/seed_train/seed_train_10s_clipped_report.json")
    mjcf_path: Path = Path("humanoidverse/data/robots/g1/g1_29dof.xml")
    target_fps: int = 30
    clip_length_s: float = 10.0
    clip_step_s: float | None = None
    root_euler_order: str = "xyz"
    manifest_is_clipped: bool = False
    include_foot_contact_binary: bool = False
    contact_dataset_root: Path = Path("/home/hanwei/dataset/LAFAN1/g1_seed")


def main(args: Args) -> None:
    records = read_manifest(args.manifest_path)
    if args.manifest_is_clipped:
        clipped_records = records
        clip_stats = {
            "source_records_total": len(records),
            "records_with_at_least_one_window": len(records),
            "records_dropped_too_short": 0,
            "windows_total": len(records),
            "target_fps": args.target_fps,
            "target_clip_frames": int(round(args.clip_length_s * args.target_fps)),
            "clip_length_s": args.clip_length_s,
            "clip_step_s": args.clip_length_s if args.clip_step_s is None else args.clip_step_s,
            "windows_by_package": {},
            "windows_by_category": {},
        }
    else:
        clipped_records, clip_stats = build_fixed_length_clip_records(
            records=records,
            clip_length_s=args.clip_length_s,
            clip_step_s=args.clip_step_s,
            target_fps=args.target_fps,
        )
    motion_dict = manifest_records_to_motion_dict(
        records=clipped_records,
        mjcf_path=args.mjcf_path,
        target_fps=args.target_fps,
        root_euler_order=args.root_euler_order,
        include_foot_contact_binary=args.include_foot_contact_binary,
        contact_dataset_root=args.contact_dataset_root,
    )

    args.output_motionlib.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(motion_dict, args.output_motionlib)
    write_jsonl_records(clipped_records, args.output_clipped_manifest)

    report = {
        "input_manifest": str(args.manifest_path.resolve()),
        "output_motionlib": str(args.output_motionlib.resolve()),
        "output_clipped_manifest": str(args.output_clipped_manifest.resolve()),
        "num_input_records": len(records),
        "num_clipped_records": len(clipped_records),
        "num_motion_entries_written": len(motion_dict),
        "include_foot_contact_binary": args.include_foot_contact_binary,
        "contact_dataset_root": str(args.contact_dataset_root.resolve()),
        "clip_stats": clip_stats,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved {len(motion_dict)} clipped motions to {args.output_motionlib}")
    print(f"Saved clipped manifest to {args.output_clipped_manifest}")
    print(f"Saved report to {args.output_report}")


if __name__ == "__main__":
    main(tyro.cli(Args))
