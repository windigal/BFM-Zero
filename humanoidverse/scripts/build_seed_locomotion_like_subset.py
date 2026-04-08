from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import tyro

from humanoidverse.language.stage_a.seed import (
    SEED_SOURCE_FPS,
    build_fixed_length_clip_records,
    load_seed_g1_rows,
    manifest_records_to_motion_dict,
    parse_joint_axes,
    read_manifest,
    seed_rows_to_motion_entry,
    write_jsonl_records,
)


LAFAN_LOCOMOTION_FAMILIES = ("walk1", "walk2", "walk3", "walk4", "run1", "run2", "sprint1")
DEFAULT_ALLOWED_CATEGORIES = (
    "Basic Locomotion Neutral",
    "Basic Locomotion Styles",
    "Baseline",
)
DEFAULT_BLOCKED_KEYWORDS = (
    "dance",
    "body_check",
    "come_up",
    "box",
    "jump",
    "frog",
    "hop",
    "crawl",
    "crouch",
    "kneel",
    "lying",
    "lie down",
    "faint",
    "puke",
    "patches",
    "edge",
)
ACTOR_ID_PATTERN = re.compile(r"__(A\d+)(?:_M)?$")
LAFAN_KEY_PATTERN = re.compile(r"^(?P<family>.+?)_subject(?P<subject>\d+)_clip\d+$")


def _extract_actor_id(filename: str) -> str:
    match = ACTOR_ID_PATTERN.search(filename)
    return match.group(1) if match is not None else "unknown"


def _combined_text(record: dict[str, Any]) -> str:
    parts: list[str] = [str(record.get("filename", "")), str(record.get("primary_text", ""))]
    parts.extend(str(text) for text in record.get("texts", []))
    return " ".join(parts).lower()


def _compute_lafan_locomotion_thresholds(
    lafan_motionlib_path: Path,
    lower_quantile: float,
    upper_quantile: float,
) -> dict[str, dict[str, float]]:
    data = joblib.load(lafan_motionlib_path)
    stats: dict[str, list[float]] = defaultdict(list)
    for key, motion in data.items():
        match = LAFAN_KEY_PATTERN.match(key)
        family = match.group("family") if match is not None else key
        if family not in LAFAN_LOCOMOTION_FAMILIES:
            continue

        root_trans = motion["root_trans_offset"]
        dof = motion["dof"]
        stats["disp"].append(float(np.linalg.norm(root_trans[-1, :2] - root_trans[0, :2])))
        stats["height_span"].append(float(root_trans[:, 2].max() - root_trans[:, 2].min()))
        stats["dof_abs"].append(float(np.abs(dof).mean()))
        stats["root_speed"].append(float(np.linalg.norm(np.diff(root_trans[:, :2], axis=0), axis=1).mean() * 30.0))

    thresholds: dict[str, dict[str, float]] = {}
    for name, values in stats.items():
        arr = np.asarray(values, dtype=np.float32)
        thresholds[name] = {
            "lower": float(np.percentile(arr, lower_quantile)),
            "upper": float(np.percentile(arr, upper_quantile)),
            "mean": float(arr.mean()),
        }
    return thresholds


def _seed_motion_stats(record: dict[str, Any], joint_axes: np.ndarray, target_fps: int, root_euler_order: str) -> dict[str, float]:
    stride = max(1, round(SEED_SOURCE_FPS / target_fps))
    rows = load_seed_g1_rows(
        csv_path=Path(record["motion_csv_path"]),
        start_frame=int(record["start_frame"]),
        end_frame=int(record["end_frame"]),
        stride=stride,
    )
    motion = seed_rows_to_motion_entry(
        rows,
        joint_axes=joint_axes,
        target_fps=int(SEED_SOURCE_FPS / stride),
        root_euler_order=root_euler_order,
    )
    root_trans = motion["root_trans_offset"]
    dof = motion["dof"]
    return {
        "disp": float(np.linalg.norm(root_trans[-1, :2] - root_trans[0, :2])),
        "height_span": float(root_trans[:, 2].max() - root_trans[:, 2].min()),
        "dof_abs": float(np.abs(dof).mean()),
        "root_speed": float(np.linalg.norm(np.diff(root_trans[:, :2], axis=0), axis=1).mean() * motion["fps"]),
    }


def _record_passes_filters(
    record: dict[str, Any],
    allowed_categories: set[str],
    blocked_keywords: tuple[str, ...],
    thresholds: dict[str, dict[str, float]],
    joint_axes: np.ndarray,
    target_fps: int,
    root_euler_order: str,
) -> tuple[bool, list[str], dict[str, float] | None]:
    reasons: list[str] = []
    if record.get("category") not in allowed_categories:
        reasons.append("category_blocked")

    text = _combined_text(record)
    for keyword in blocked_keywords:
        if keyword in text:
            reasons.append(f"keyword:{keyword}")
            break

    if reasons:
        return False, reasons, None

    stats = _seed_motion_stats(
        record=record,
        joint_axes=joint_axes,
        target_fps=target_fps,
        root_euler_order=root_euler_order,
    )

    # Keep clips within locomotion-like ranges. Lower bounds remove mostly static routines.
    for stat_name in ("disp", "height_span", "dof_abs", "root_speed"):
        lower = thresholds[stat_name]["lower"]
        upper = thresholds[stat_name]["upper"]
        value = stats[stat_name]
        if value < lower:
            reasons.append(f"{stat_name}_below")
        elif value > upper:
            reasons.append(f"{stat_name}_above")

    return len(reasons) == 0, reasons, stats


def _allocate_quotas_by_category(records: list[dict[str, Any]], target_size: int) -> dict[str, int]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_category[record["category"]].append(record)

    total = len(records)
    if target_size >= total:
        return {category: len(items) for category, items in by_category.items()}

    quotas = {category: 0 for category in by_category}
    categories = list(by_category.keys())
    if target_size >= len(categories):
        for category in categories:
            quotas[category] = 1

    remaining = target_size - sum(quotas.values())
    fractional_parts: list[tuple[float, str]] = []
    for category, items in by_category.items():
        raw_quota = remaining * (len(items) / total)
        alloc = min(int(raw_quota), len(items) - quotas[category])
        quotas[category] += alloc
        fractional_parts.append((raw_quota - int(raw_quota), category))

    left = target_size - sum(quotas.values())
    for _, category in sorted(fractional_parts, reverse=True):
        if left <= 0:
            break
        if quotas[category] >= len(by_category[category]):
            continue
        quotas[category] += 1
        left -= 1

    return quotas


def _balanced_take(
    records: list[dict[str, Any]],
    quota: int,
    rng: random.Random,
    key_fn: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    if quota <= 0 or not records:
        return []
    if quota >= len(records):
        return list(records)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[key_fn(record)].append(record)

    keys = list(groups.keys())
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(groups[key])

    selected: list[dict[str, Any]] = []
    indices = {key: 0 for key in keys}
    active = list(keys)

    while active and len(selected) < quota:
        next_active: list[str] = []
        for key in active:
            idx = indices[key]
            if idx < len(groups[key]):
                selected.append(groups[key][idx])
                indices[key] += 1
                if indices[key] < len(groups[key]):
                    next_active.append(key)
                if len(selected) >= quota:
                    break
        active = next_active

    return selected


@dataclass
class Args:
    manifest_path: Path = Path("artifacts/seed_train/seed_train_manifest.jsonl")
    lafan_motionlib_path: Path = Path("humanoidverse/data/lafan_29dof_10s-clipped.pkl")
    output_manifest: Path = Path("artifacts/seed_train/filtered/seed_train_locomotion_like_10s_1000.jsonl")
    output_motionlib: Path = Path("humanoidverse/data/seed_train_locomotion_like_10s_1000.pkl")
    output_report: Path = Path("artifacts/seed_train/filtered/seed_train_locomotion_like_10s_1000.report.json")
    target_size: int = 1000
    target_fps: int = 30
    clip_length_s: float = 10.0
    clip_step_s: float | None = None
    mjcf_path: Path = Path("humanoidverse/data/robots/g1/g1_29dof.xml")
    root_euler_order: str = "xyz"
    seed: int = 42
    lower_quantile: float = 5.0
    upper_quantile: float = 95.0
    allowed_categories: tuple[str, ...] = DEFAULT_ALLOWED_CATEGORIES
    blocked_keywords: tuple[str, ...] = DEFAULT_BLOCKED_KEYWORDS


def main(args: Args) -> None:
    source_records = read_manifest(args.manifest_path)
    clipped_records, clip_stats = build_fixed_length_clip_records(
        records=source_records,
        clip_length_s=args.clip_length_s,
        clip_step_s=args.clip_step_s,
        target_fps=args.target_fps,
    )

    thresholds = _compute_lafan_locomotion_thresholds(
        lafan_motionlib_path=args.lafan_motionlib_path,
        lower_quantile=args.lower_quantile,
        upper_quantile=args.upper_quantile,
    )
    joint_axes = parse_joint_axes(args.mjcf_path)
    allowed_categories = set(args.allowed_categories)
    rng = random.Random(args.seed)

    candidates: list[dict[str, Any]] = []
    stats_by_reason: Counter[str] = Counter()
    stats_by_category: Counter[str] = Counter()
    stats_by_actor: Counter[str] = Counter()

    for idx, record in enumerate(clipped_records, start=1):
        keep, reasons, motion_stats = _record_passes_filters(
            record=record,
            allowed_categories=allowed_categories,
            blocked_keywords=args.blocked_keywords,
            thresholds=thresholds,
            joint_axes=joint_axes,
            target_fps=args.target_fps,
            root_euler_order=args.root_euler_order,
        )
        if not keep:
            stats_by_reason.update(reasons)
            continue

        enriched = dict(record)
        assert motion_stats is not None
        enriched["motion_stats"] = motion_stats
        enriched["actor_id"] = _extract_actor_id(record["filename"])
        candidates.append(enriched)
        stats_by_category[enriched["category"]] += 1
        stats_by_actor[enriched["actor_id"]] += 1

        if idx % 2000 == 0:
            print(f"Scored {idx}/{len(clipped_records)} clipped records, kept {len(candidates)}")

    if len(candidates) < args.target_size:
        raise ValueError(f"Only {len(candidates)} candidates remain after filtering, need {args.target_size}")

    quotas = _allocate_quotas_by_category(candidates, args.target_size)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        by_category[record["category"]].append(record)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for category in sorted(by_category):
        picks = _balanced_take(
            by_category[category],
            quotas.get(category, 0),
            rng,
            key_fn=lambda r: f"{r['actor_id']}::{r['filename']}",
        )
        for record in picks:
            selected.append(record)
            selected_ids.add(record["sample_id"])

    if len(selected) < args.target_size:
        remaining = [record for record in candidates if record["sample_id"] not in selected_ids]
        selected.extend(
            _balanced_take(
                remaining,
                args.target_size - len(selected),
                rng,
                key_fn=lambda r: f"{r['actor_id']}::{r['filename']}",
            )
        )

    selected = selected[: args.target_size]
    selected_for_manifest = [{k: v for k, v in record.items() if k != "motion_stats"} for record in selected]
    motion_dict = manifest_records_to_motion_dict(
        records=selected_for_manifest,
        mjcf_path=args.mjcf_path,
        target_fps=args.target_fps,
        root_euler_order=args.root_euler_order,
    )

    write_jsonl_records(selected_for_manifest, args.output_manifest)
    args.output_motionlib.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(motion_dict, args.output_motionlib)

    report = {
        "source_manifest": str(args.manifest_path.resolve()),
        "num_source_records": len(source_records),
        "num_clipped_records": len(clipped_records),
        "num_candidates_after_filtering": len(candidates),
        "target_size": args.target_size,
        "num_selected": len(selected_for_manifest),
        "thresholds": thresholds,
        "allowed_categories": list(args.allowed_categories),
        "blocked_keywords": list(args.blocked_keywords),
        "clip_stats": clip_stats,
        "candidate_by_category": dict(stats_by_category),
        "candidate_by_actor_top20": dict(stats_by_actor.most_common(20)),
        "filtered_reasons": dict(stats_by_reason),
        "selected_by_category": dict(Counter(record["category"] for record in selected_for_manifest)),
        "selected_by_package": dict(Counter(record["package"] for record in selected_for_manifest)),
        "selected_actor_top20": dict(Counter(record["actor_id"] for record in selected).most_common(20)),
        "selected_motion_stats_mean": {
            name: float(np.mean([record["motion_stats"][name] for record in selected]))
            for name in ("disp", "height_span", "dof_abs", "root_speed")
        },
        "output_manifest": str(args.output_manifest.resolve()),
        "output_motionlib": str(args.output_motionlib.resolve()),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved filtered manifest to {args.output_manifest}")
    print(f"Saved filtered motionlib to {args.output_motionlib}")
    print(f"Saved report to {args.output_report}")


if __name__ == "__main__":
    main(tyro.cli(Args))
