from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import tyro

from humanoidverse.language.stage_a.seed import read_manifest, write_jsonl_records


@dataclass
class Args:
    input_manifest: Path = Path("artifacts/babel_train/babel_train_seed2k_matched_50fps.jsonl")
    input_motionlib: Path = Path("humanoidverse/data/babel_train_seed2k_matched_50fps.pkl")
    output_manifest: Path = Path("artifacts/babel_train/babel_train_seed2k_matched_10s_50fps.jsonl")
    output_motionlib: Path = Path("humanoidverse/data/babel_train_seed2k_matched_10s_50fps.pkl")
    output_report: Path = Path("artifacts/babel_train/babel_train_seed2k_matched_10s_50fps.report.json")
    clip_length_s: float = 10.0
    clip_step_s: float | None = None
    target_fps: int = 50


def _slice_motion_entry(entry: dict[str, Any], start: int, end: int, motion_name: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in entry.items():
        if hasattr(value, "shape") and len(getattr(value, "shape", ())) > 0 and value.shape[0] >= end:
            out[key] = value[start:end].copy()
        else:
            out[key] = value
    out["motion_name"] = motion_name
    out["fps"] = int(entry.get("fps", 50))
    return out


def main(args: Args) -> None:
    if args.clip_length_s <= 0:
        raise ValueError("clip_length_s must be > 0")
    if args.clip_step_s is not None and args.clip_step_s <= 0:
        raise ValueError("clip_step_s must be > 0")

    records = read_manifest(args.input_manifest)
    motion_dict = joblib.load(args.input_motionlib)
    clip_frames = int(round(args.clip_length_s * args.target_fps))
    step_frames = clip_frames if args.clip_step_s is None else int(round(args.clip_step_s * args.target_fps))
    if clip_frames <= 0 or step_frames <= 0:
        raise ValueError("Computed non-positive clip/window frames")

    clipped_records: list[dict[str, Any]] = []
    clipped_motion_dict: dict[str, Any] = {}
    windows_by_category: dict[str, int] = {}
    windows_by_package: dict[str, int] = {}
    dropped_too_short = 0
    records_with_windows = 0

    for record in records:
        motion = motion_dict.get(record["sample_id"])
        if motion is None:
            raise KeyError(f"Missing motion entry for sample_id={record['sample_id']}")
        total_frames = int(motion["root_trans_offset"].shape[0])
        if total_frames < clip_frames:
            dropped_too_short += 1
            continue

        parent_id = str(record["sample_id"])
        clip_idx = 0
        max_offset = total_frames - clip_frames
        offset = 0
        while offset <= max_offset:
            sample_id = f"{parent_id}_clip{clip_idx}"
            clip_record = dict(record)
            clip_record["parent_sample_id"] = parent_id
            clip_record["sample_id"] = sample_id
            clip_record["clip_index"] = int(clip_idx)
            clip_record["start_frame"] = int(offset)
            clip_record["end_frame"] = int(offset + clip_frames)
            clip_record["duration_s"] = float(args.clip_length_s)
            clip_record["fps"] = int(args.target_fps)
            clipped_records.append(clip_record)
            clipped_motion_dict[sample_id] = _slice_motion_entry(motion, offset, offset + clip_frames, sample_id)

            category = str(clip_record.get("category", ""))
            package = str(clip_record.get("package", ""))
            windows_by_category[category] = windows_by_category.get(category, 0) + 1
            windows_by_package[package] = windows_by_package.get(package, 0) + 1
            clip_idx += 1
            offset += step_frames

        if clip_idx > 0:
            records_with_windows += 1

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_motionlib.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_records(clipped_records, args.output_manifest)
    joblib.dump(clipped_motion_dict, args.output_motionlib)

    report = {
        "input_manifest": str(args.input_manifest.resolve()),
        "input_motionlib": str(args.input_motionlib.resolve()),
        "output_manifest": str(args.output_manifest.resolve()),
        "output_motionlib": str(args.output_motionlib.resolve()),
        "num_input_records": len(records),
        "num_input_motion_entries": len(motion_dict),
        "num_clipped_records": len(clipped_records),
        "num_clipped_motion_entries": len(clipped_motion_dict),
        "records_with_at_least_one_window": records_with_windows,
        "records_dropped_too_short": dropped_too_short,
        "clip_length_s": float(args.clip_length_s),
        "clip_step_s": float(args.clip_length_s if args.clip_step_s is None else args.clip_step_s),
        "target_fps": int(args.target_fps),
        "target_clip_frames": int(clip_frames),
        "windows_by_category": windows_by_category,
        "windows_by_package": windows_by_package,
    }
    args.output_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved fixed-length manifest to {args.output_manifest}")
    print(f"Saved fixed-length motionlib to {args.output_motionlib}")
    print(f"Saved report to {args.output_report}")
    print(f"Created {len(clipped_records)} windows at {args.target_fps}Hz")


if __name__ == "__main__":
    main(tyro.cli(Args))
