from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import tyro

from humanoidverse.language.stage_a.seed import read_manifest
from humanoidverse.scripts.convert_lafan1_csv_to_motion import build_motion_entry_from_lafan1_g1


@dataclass
class Args:
    manifest_path: Path = Path("artifacts/lafan1_stage_a/lafan1_stage_a_manifest.jsonl")
    output_motionlib: Path = Path("artifacts/lafan1_stage_a/lafan1_stage_a_motionlib.pkl")
    output_report: Path = Path("artifacts/lafan1_stage_a/lafan1_stage_a_motionlib_report.json")
    mjcf_path: Path = Path("humanoidverse/data/robots/g1/g1_29dof.xml")


def main(args: Args) -> None:
    records = read_manifest(args.manifest_path)
    if not records:
        raise RuntimeError(f"No records found in {args.manifest_path}")

    motion_dict: dict[str, dict[str, object]] = {}
    frame_counts: list[int] = []
    for record in records:
        motion_name = str(record["sample_id"])
        csv_path = Path(str(record["motion_csv_path"])).expanduser()
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing CSV for {motion_name}: {csv_path}")
        motion_entry = build_motion_entry_from_lafan1_g1(
            csv_path=csv_path,
            mjcf_path=args.mjcf_path,
            motion_name=motion_name,
        )
        motion_dict[motion_name] = motion_entry
        frame_counts.append(int(np.asarray(motion_entry["root_trans_offset"]).shape[0]))

    args.output_motionlib.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(motion_dict, args.output_motionlib)

    report = {
        "manifest_path": str(args.manifest_path.resolve()),
        "output_motionlib": str(args.output_motionlib.resolve()),
        "num_motion_entries": len(motion_dict),
        "frame_count_min": int(min(frame_counts)),
        "frame_count_max": int(max(frame_counts)),
        "frame_count_mean": float(np.mean(frame_counts)),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved {len(motion_dict)} LAFAN1 motions to {args.output_motionlib}")
    print(f"Saved report to {args.output_report}")


if __name__ == "__main__":
    main(tyro.cli(Args))
