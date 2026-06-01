from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import tyro
from scipy.spatial.transform import Rotation

from humanoidverse.language.stage_a.seed import read_manifest, write_jsonl_records
from humanoidverse.scripts.build_babel_seed2k_matched_motionlib import (
    CATEGORY_TO_PACKAGE,
    SEED_CATEGORY_ORDER,
    _act_cats,
    _category_match_score,
    _classify_category,
    _motion_stats,
)
from humanoidverse.scripts.convert_pico_to_motion import parse_joint_axes


@dataclass
class Args:
    input_dir: Path = Path("artifacts/textop_babel_h2_f8_50fps")
    reference_manifest: Path = Path("artifacts/seed_train/tiers/seed_train_10s_2000.jsonl")
    output_manifest: Path = Path("artifacts/babel_train/babel_train_seed2k_windowed_2000_10s_50fps.jsonl")
    output_motionlib: Path = Path("humanoidverse/data/babel_train_seed2k_windowed_2000_10s_50fps.pkl")
    output_report: Path = Path("artifacts/babel_train/babel_train_seed2k_windowed_2000_10s_50fps.report.json")
    mjcf_path: Path = Path("humanoidverse/data/robots/g1/g1_29dof.xml")
    clip_length_s: float = 10.0
    clip_step_s: float | None = None
    target_fps: int = 50
    seed: int = 42


def _joined_texts(entry: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for ann in entry.get("frame_ann") or []:
        text = str(ann[2]).strip()
        if text and text not in texts:
            texts.append(text)
    return texts


def _reference_category_counts(reference_manifest: Path) -> Counter[str]:
    records = read_manifest(reference_manifest)
    return Counter(record["category"] for record in records)


def _slice_motion_window(motion: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    root_trans = np.asarray(motion["root_trans_offset"], dtype=np.float32)[start:end].copy()
    root_rot = np.asarray(motion["root_rot"], dtype=np.float32)[start:end].copy()
    dof = np.asarray(motion["dof"], dtype=np.float32)[start:end].copy()
    fps = int(motion.get("fps", 50))
    return {
        "root_trans_offset": root_trans,
        "root_rot": root_rot,
        "dof": dof,
        "fps": fps,
    }


def _build_window_motion_entry(window_motion: dict[str, Any], joint_axes_29: np.ndarray, motion_name: str) -> dict[str, Any]:
    root_trans = window_motion["root_trans_offset"]
    root_rot = window_motion["root_rot"]
    dof = window_motion["dof"]
    root_rotvec = Rotation.from_quat(root_rot).as_rotvec().astype(np.float32)
    local_joint_rotvec = dof[..., None] * joint_axes_29[None, ...]
    pose_aa = np.concatenate([root_rotvec[:, None, :], local_joint_rotvec], axis=1).astype(np.float32)
    return {
        "root_trans_offset": root_trans,
        "root_rot": root_rot,
        "dof": dof,
        "pose_aa": pose_aa,
        "fps": int(window_motion["fps"]),
        "motion_name": motion_name,
    }


def _balanced_take(records: list[dict[str, Any]], quota: int, rng: random.Random) -> list[dict[str, Any]]:
    if quota <= 0 or not records:
        return []
    if quota >= len(records):
        return list(records)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["feat_p"])].append(record)

    for feat_p, items in groups.items():
        groups[feat_p] = sorted(
            items,
            key=lambda item: (
                -_category_match_score(item),
                item["motion_stats"]["height_span"],
                -item["motion_stats"]["disp"],
                item["sample_id"],
            ),
        )

    parent_keys = list(groups.keys())
    rng.shuffle(parent_keys)
    selected: list[dict[str, Any]] = []
    indices = {key: 0 for key in parent_keys}
    active_keys = list(parent_keys)

    while active_keys and len(selected) < quota:
        next_active: list[str] = []
        for key in active_keys:
            idx = indices[key]
            if idx < len(groups[key]):
                selected.append(groups[key][idx])
                indices[key] += 1
                if indices[key] < len(groups[key]):
                    next_active.append(key)
                if len(selected) >= quota:
                    break
        active_keys = next_active

    return selected


def main(args: Args) -> None:
    if args.clip_length_s <= 0:
        raise ValueError("clip_length_s must be > 0")
    if args.clip_step_s is not None and args.clip_step_s <= 0:
        raise ValueError("clip_step_s must be > 0")

    rng = random.Random(args.seed)
    target_counts = _reference_category_counts(args.reference_manifest)
    joint_axes_29 = parse_joint_axes(args.mjcf_path)

    clip_frames = int(round(args.clip_length_s * args.target_fps))
    step_frames = clip_frames if args.clip_step_s is None else int(round(args.clip_step_s * args.target_fps))
    if clip_frames <= 0 or step_frames <= 0:
        raise ValueError("Computed non-positive clip window size")

    split_payloads: dict[str, list[dict[str, Any]]] = {}
    all_windows: list[dict[str, Any]] = []
    for split in ("train", "val"):
        split_path = args.input_dir / f"{split}.pkl"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing BABEL split file: {split_path}")
        payload = joblib.load(split_path)
        split_payloads[split] = payload
        for index, entry in enumerate(payload):
            category = _classify_category(entry)
            texts = _joined_texts(entry)
            act_cats = sorted(_act_cats(entry))
            feat_p = str(entry.get("feat_p", ""))
            motion = entry["motion"]
            total_frames = int(np.asarray(motion["root_trans_offset"]).shape[0])
            if total_frames < clip_frames:
                continue
            max_offset = total_frames - clip_frames
            offset = 0
            clip_idx = 0
            while offset <= max_offset:
                sample_id = f"babel::{split}::{index:06d}::{Path(feat_p).stem}_clip{clip_idx}"
                window_motion = _slice_motion_window(motion, offset, offset + clip_frames)
                all_windows.append(
                    {
                        "sample_id": sample_id,
                        "parent_sample_id": f"babel::{split}::{index:06d}::{Path(feat_p).stem}",
                        "split": split,
                        "sample_index": index,
                        "clip_index": clip_idx,
                        "feat_p": feat_p,
                        "babel_sid": int(entry.get("babel_sid", -1)),
                        "texts": texts,
                        "act_cats": act_cats,
                        "duration_s": float(args.clip_length_s),
                        "length": int(clip_frames),
                        "fps": int(args.target_fps),
                        "category": category,
                        "package": CATEGORY_TO_PACKAGE[category],
                        "start_frame": int(offset),
                        "end_frame": int(offset + clip_frames),
                        "motion_stats": _motion_stats({"motion": window_motion}),
                    }
                )
                clip_idx += 1
                offset += step_frames

    available_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in all_windows:
        available_by_category[record["category"]].append(record)

    final_targets = {category: min(int(target_counts.get(category, 0)), len(available_by_category.get(category, []))) for category in SEED_CATEGORY_ORDER}
    remaining = sum(target_counts.values()) - sum(final_targets.values())
    if remaining > 0:
        fallback_order = [
            "Basic Locomotion Neutral",
            "Baseline",
            "Complex Actions",
            "Advanced Locomotion",
            "Sports",
            "Communication",
            "Dancing",
            "Other",
            "Gestures",
            "Household",
            "Looking and Pointing",
            "Magic",
        ]
        for category in fallback_order:
            if remaining <= 0:
                break
            spare = len(available_by_category.get(category, [])) - final_targets.get(category, 0)
            if spare <= 0:
                continue
            take = min(spare, remaining)
            final_targets[category] += take
            remaining -= take
    if remaining != 0:
        raise RuntimeError(f"Unable to allocate full 2000 windows, remaining={remaining}")

    selected: list[dict[str, Any]] = []
    for category in SEED_CATEGORY_ORDER:
        quota = final_targets.get(category, 0)
        selected.extend(_balanced_take(available_by_category.get(category, []), quota, rng))

    if len(selected) != sum(target_counts.values()):
        raise RuntimeError(f"Expected {sum(target_counts.values())} selected windows, got {len(selected)}")

    selected = sorted(selected, key=lambda item: item["sample_id"])
    manifest_records = []
    motion_dict = {}
    for item in selected:
        manifest_records.append(
            {
                "sample_id": item["sample_id"],
                "parent_sample_id": item["parent_sample_id"],
                "split": item["split"],
                "sample_index": item["sample_index"],
                "clip_index": item["clip_index"],
                "feat_p": item["feat_p"],
                "babel_sid": item["babel_sid"],
                "texts": item["texts"],
                "act_cats": item["act_cats"],
                "duration_s": item["duration_s"],
                "length": item["length"],
                "fps": item["fps"],
                "category": item["category"],
                "package": item["package"],
                "start_frame": item["start_frame"],
                "end_frame": item["end_frame"],
                "motion_stats": item["motion_stats"],
            }
        )
        src_motion = split_payloads[item["split"]][item["sample_index"]]["motion"]
        window_motion = _slice_motion_window(src_motion, item["start_frame"], item["end_frame"])
        motion_dict[item["sample_id"]] = _build_window_motion_entry(window_motion, joint_axes_29, item["sample_id"])

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_motionlib.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_records(manifest_records, args.output_manifest)
    joblib.dump(motion_dict, args.output_motionlib)

    report = {
        "input_dir": str(args.input_dir.resolve()),
        "reference_manifest": str(args.reference_manifest.resolve()),
        "output_manifest": str(args.output_manifest.resolve()),
        "output_motionlib": str(args.output_motionlib.resolve()),
        "num_input_windows": len(all_windows),
        "num_selected": len(selected),
        "requested_category_counts": {cat: int(target_counts.get(cat, 0)) for cat in SEED_CATEGORY_ORDER},
        "available_category_counts": {cat: int(len(available_by_category.get(cat, []))) for cat in SEED_CATEGORY_ORDER},
        "final_target_category_counts": final_targets,
        "selected_by_category": dict(Counter(item["category"] for item in selected)),
        "selected_by_package": dict(Counter(item["package"] for item in selected)),
        "selected_by_split": dict(Counter(item["split"] for item in selected)),
        "motion_stats_mean": {
            key: float(np.mean([item["motion_stats"][key] for item in selected]))
            for key in ["disp", "height_span", "dof_abs_mean", "root_speed_mean"]
        },
        "examples_by_category": {
            category: [
                {
                    "sample_id": item["sample_id"],
                    "feat_p": item["feat_p"],
                    "texts": item["texts"][:3],
                    "start_frame": item["start_frame"],
                    "end_frame": item["end_frame"],
                }
                for item in selected
                if item["category"] == category
            ][:5]
            for category in SEED_CATEGORY_ORDER
        },
    }
    args.output_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved windowed manifest to {args.output_manifest}")
    print(f"Saved windowed motionlib to {args.output_motionlib}")
    print(f"Saved report to {args.output_report}")
    print(f"Selected {len(selected)} windows of {args.clip_length_s}s at {args.target_fps}Hz")


if __name__ == "__main__":
    main(tyro.cli(Args))
