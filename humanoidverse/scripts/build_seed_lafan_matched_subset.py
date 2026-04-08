from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


LAFAN_KEY_PATTERN = re.compile(r"^(?P<family>.+?)_subject(?P<subject>\d+)_clip\d+$")
ACTOR_ID_PATTERN = re.compile(r"__(A\d+)(?:_M)?$")

LAFAN_FAMILY_TO_GROUP = {
    "walk1": "walk",
    "walk2": "walk",
    "walk3": "walk",
    "walk4": "walk",
    "run1": "run",
    "run2": "run",
    "sprint1": "run",
    "dance1": "dance",
    "dance2": "dance",
    "jumps1": "jump",
    "fight1": "fight",
    "fightAndSports1": "fight",
    "fallAndGetUp1": "fall_getup",
    "fallAndGetUp2": "fall_getup",
    "fallAndGetUp3": "fall_getup",
}

GROUP_KEYWORDS = {
    "walk": (
        "walk",
        "walking",
        "sideway",
        "sideways",
        "diagonal",
        "crisscross",
        "forward",
        "backward",
        "stride",
    ),
    "run": (
        "run",
        "running",
        "jog",
        "jogging",
        "sprint",
        "hurry",
        "fast",
        "quickly",
    ),
    "dance": (
        "dance",
        "dancing",
        "vogue",
        "salsa",
        "latino",
        "freedom",
        "pivot walk",
    ),
    "jump": (
        "jump",
        "jumping",
        "hop",
        "hopping",
        "frog jump",
        "leap",
    ),
    "fight": (
        "fight",
        "fighting",
        "punch",
        "kick",
        "boxing",
        "martial",
        "combat",
        "spar",
        "strike",
    ),
    "fall_getup": (
        "fall",
        "falling",
        "faint",
        "collapse",
        "ground",
        "get up",
        "stand up",
        "lying",
        "lie down",
        "puke",
    ),
}

GROUP_PACKAGE_PRIORS = {
    "walk": {"Locomotion"},
    "run": {"Locomotion"},
    "dance": {"Dances"},
    "jump": {"Sport", "Locomotion"},
    "fight": {"Sport", "Other"},
    "fall_getup": {"Other", "Sport", "Locomotion"},
}

GROUP_CATEGORY_PRIORS = {
    "walk": {"Basic Locomotion Neutral", "Basic Locomotion Styles", "Baseline", "Advanced Locomotion"},
    "run": {"Basic Locomotion Neutral", "Basic Locomotion Styles", "Baseline", "Complex Actions"},
    "dance": {"Dancing", "Baseline"},
    "jump": {"Sports", "Complex Actions", "Other"},
    "fight": {"Sports", "Other", "Communication"},
    "fall_getup": {"Complex Actions", "Other", "Sports"},
}

GROUP_BLOCKED_KEYWORDS = {
    "walk": ("dance", "jump", "fight", "faint", "crawl", "crouch", "kneel"),
    "run": ("dance", "fight", "faint", "crawl", "crouch", "kneel"),
    "dance": ("fight", "faint", "crawl"),
    "jump": ("dance", "fight", "faint", "crawl"),
    "fight": ("dance", "faint"),
    "fall_getup": ("dance",),
}


def _extract_actor_id(filename: str) -> str:
    match = ACTOR_ID_PATTERN.search(filename)
    return match.group(1) if match is not None else "unknown"


def _combined_text(record: dict[str, Any]) -> str:
    parts: list[str] = [str(record.get("filename", "")), str(record.get("primary_text", ""))]
    parts.extend(str(text) for text in record.get("texts", []))
    return " ".join(parts).lower()


def _lafan_family_stats(lafan_motionlib_path: Path) -> dict[str, dict[str, float]]:
    data = joblib.load(lafan_motionlib_path)
    per_family: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for key, motion in data.items():
        match = LAFAN_KEY_PATTERN.match(key)
        family = match.group("family") if match is not None else key
        root_trans = motion["root_trans_offset"]
        dof = motion["dof"]
        per_family[family]["disp"].append(float(np.linalg.norm(root_trans[-1, :2] - root_trans[0, :2])))
        per_family[family]["height_span"].append(float(root_trans[:, 2].max() - root_trans[:, 2].min()))
        per_family[family]["dof_abs"].append(float(np.abs(dof).mean()))
        per_family[family]["root_speed"].append(float(np.linalg.norm(np.diff(root_trans[:, :2], axis=0), axis=1).mean() * 30.0))

    out: dict[str, dict[str, float]] = {}
    for family, stats in per_family.items():
        out[family] = {
            "count": len(stats["disp"]),
            "disp_mean": float(np.mean(stats["disp"])),
            "disp_std": float(np.std(stats["disp"]) + 1e-6),
            "height_span_mean": float(np.mean(stats["height_span"])),
            "height_span_std": float(np.std(stats["height_span"]) + 1e-6),
            "dof_abs_mean": float(np.mean(stats["dof_abs"])),
            "dof_abs_std": float(np.std(stats["dof_abs"]) + 1e-6),
            "root_speed_mean": float(np.mean(stats["root_speed"])),
            "root_speed_std": float(np.std(stats["root_speed"]) + 1e-6),
        }
    return out


def _family_quotas(family_stats: dict[str, dict[str, float]], target_size: int) -> dict[str, int]:
    total = sum(int(v["count"]) for v in family_stats.values())
    quotas: dict[str, int] = {}
    fracs: list[tuple[float, str]] = []
    for family, stats in family_stats.items():
        raw = target_size * (stats["count"] / total)
        q = int(math.floor(raw))
        quotas[family] = q
        fracs.append((raw - q, family))
    remaining = target_size - sum(quotas.values())
    for _, family in sorted(fracs, reverse=True):
        if remaining <= 0:
            break
        quotas[family] += 1
        remaining -= 1
    return quotas


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


def _group_text_score(record: dict[str, Any], group: str) -> float:
    text = _combined_text(record)
    score = 0.0
    for blocked in GROUP_BLOCKED_KEYWORDS.get(group, ()):
        if blocked in text:
            return -1.0

    package = str(record.get("package", ""))
    category = str(record.get("category", ""))
    if package in GROUP_PACKAGE_PRIORS.get(group, set()):
        score += 1.0
    if category in GROUP_CATEGORY_PRIORS.get(group, set()):
        score += 1.0

    for keyword in GROUP_KEYWORDS[group]:
        if keyword in text:
            score += 1.0
    return score


def _family_match_score(record: dict[str, Any], family: str, family_stat: dict[str, float]) -> tuple[float, float, float]:
    group = LAFAN_FAMILY_TO_GROUP[family]
    text_score = _group_text_score(record, group)
    if text_score < 0:
        return -1e9, text_score, 1e9

    stats = record["motion_stats"]
    z_dist = 0.0
    z_dist += abs((stats["disp"] - family_stat["disp_mean"]) / family_stat["disp_std"])
    z_dist += abs((stats["height_span"] - family_stat["height_span_mean"]) / family_stat["height_span_std"])
    z_dist += abs((stats["dof_abs"] - family_stat["dof_abs_mean"]) / family_stat["dof_abs_std"])
    z_dist += abs((stats["root_speed"] - family_stat["root_speed_mean"]) / family_stat["root_speed_std"])

    score = 3.0 * text_score - z_dist
    return score, text_score, z_dist


def _balanced_pick_ranked(records: list[dict[str, Any]], quota: int, seed: int) -> list[dict[str, Any]]:
    if quota <= 0:
        return []
    rng = random.Random(seed)
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = f"{record['actor_id']}::{record['filename']}"
        by_key[key].append(record)
    keys = list(by_key.keys())
    rng.shuffle(keys)
    for key in keys:
        by_key[key] = sorted(by_key[key], key=lambda r: (-r["_match_score"], r["_match_distance"]))

    selected: list[dict[str, Any]] = []
    indices = {key: 0 for key in keys}
    active = list(keys)
    while active and len(selected) < quota:
        next_active: list[str] = []
        for key in active:
            idx = indices[key]
            if idx < len(by_key[key]):
                selected.append(by_key[key][idx])
                indices[key] += 1
                if indices[key] < len(by_key[key]):
                    next_active.append(key)
                if len(selected) >= quota:
                    break
        active = next_active
    return selected


@dataclass
class Args:
    source_manifest: Path = Path("artifacts/seed_train/seed_train_manifest.jsonl")
    lafan_motionlib_path: Path = Path("humanoidverse/data/lafan_29dof_10s-clipped.pkl")
    output_manifest: Path = Path("artifacts/seed_train/lafan_matched/seed_train_lafan_matched_10s_1000.jsonl")
    output_motionlib: Path = Path("humanoidverse/data/seed_train_lafan_matched_10s_1000.pkl")
    output_report: Path = Path("artifacts/seed_train/lafan_matched/seed_train_lafan_matched_10s_1000.report.json")
    target_size: int = 1000
    target_fps: int = 30
    clip_length_s: float = 10.0
    clip_step_s: float | None = None
    mjcf_path: Path = Path("humanoidverse/data/robots/g1/g1_29dof.xml")
    root_euler_order: str = "xyz"
    seed: int = 42


def main(args: Args) -> None:
    family_stats = _lafan_family_stats(args.lafan_motionlib_path)
    quotas = _family_quotas(family_stats, args.target_size)
    source_records = read_manifest(args.source_manifest)
    clipped_records, clip_stats = build_fixed_length_clip_records(
        records=source_records,
        clip_length_s=args.clip_length_s,
        clip_step_s=args.clip_step_s,
        target_fps=args.target_fps,
    )
    joint_axes = parse_joint_axes(args.mjcf_path)

    enriched_records: list[dict[str, Any]] = []
    for idx, record in enumerate(clipped_records, start=1):
        motion_stats = _seed_motion_stats(
            record=record,
            joint_axes=joint_axes,
            target_fps=args.target_fps,
            root_euler_order=args.root_euler_order,
        )
        enriched = dict(record)
        enriched["motion_stats"] = motion_stats
        enriched["actor_id"] = _extract_actor_id(record["filename"])
        enriched_records.append(enriched)
        if idx % 2000 == 0:
            print(f"Scored {idx}/{len(clipped_records)} clipped records")

    ranked_by_family: dict[str, list[dict[str, Any]]] = {}
    report_family_candidates: dict[str, dict[str, Any]] = {}

    for family, family_stat in family_stats.items():
        ranked: list[dict[str, Any]] = []
        positive_text_count = 0
        for record in enriched_records:
            score, text_score, z_dist = _family_match_score(record, family, family_stat)
            if text_score <= 0:
                continue
            positive_text_count += 1
            candidate = dict(record)
            candidate["_family"] = family
            candidate["_match_score"] = float(score)
            candidate["_match_text_score"] = float(text_score)
            candidate["_match_distance"] = float(z_dist)
            ranked.append(candidate)
        ranked.sort(key=lambda r: (-r["_match_score"], r["_match_distance"]))
        ranked_by_family[family] = ranked
        report_family_candidates[family] = {
            "group": LAFAN_FAMILY_TO_GROUP[family],
            "lafan_count": int(family_stat["count"]),
            "quota": int(quotas[family]),
            "candidates": len(ranked),
            "positive_text_candidates": positive_text_count,
        }

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    family_shortage: dict[str, int] = {}

    for family in sorted(quotas.keys(), key=lambda f: quotas[f], reverse=True):
        quota = quotas[family]
        ranked = [r for r in ranked_by_family[family] if r["sample_id"] not in selected_ids]
        top_pool = ranked[: max(quota * 8, quota)]
        picks = _balanced_pick_ranked(top_pool, quota, seed=args.seed + hash(family) % 10000)
        for record in picks:
            selected.append(record)
            selected_ids.add(record["sample_id"])
        if len(picks) < quota:
            family_shortage[family] = quota - len(picks)

    if len(selected) < args.target_size:
        global_ranked: list[dict[str, Any]] = []
        for family, ranked in ranked_by_family.items():
            for record in ranked:
                if record["sample_id"] in selected_ids:
                    continue
                global_ranked.append(record)
        global_ranked.sort(key=lambda r: (-r["_match_score"], r["_match_distance"]))
        filler = _balanced_pick_ranked(global_ranked[: max((args.target_size - len(selected)) * 10, 1000)], args.target_size - len(selected), seed=args.seed + 999)
        for record in filler:
            if record["sample_id"] in selected_ids:
                continue
            selected.append(record)
            selected_ids.add(record["sample_id"])
            if len(selected) >= args.target_size:
                break

    if len(selected) < args.target_size:
        raise ValueError(f"Selected only {len(selected)} clips, need {args.target_size}")

    selected = selected[: args.target_size]
    selected_for_manifest = []
    for record in selected:
        manifest_record = {k: v for k, v in record.items() if not k.startswith("_")}
        manifest_record["lafan_family_match"] = record["_family"]
        manifest_record["lafan_match_score"] = float(record["_match_score"])
        manifest_record["lafan_match_distance"] = float(record["_match_distance"])
        manifest_record["lafan_match_text_score"] = float(record["_match_text_score"])
        selected_for_manifest.append(manifest_record)
    motion_dict = manifest_records_to_motion_dict(
        records=selected_for_manifest,
        mjcf_path=args.mjcf_path,
        target_fps=args.target_fps,
        root_euler_order=args.root_euler_order,
    )

    write_jsonl_records(selected_for_manifest, args.output_manifest)
    args.output_motionlib.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(motion_dict, args.output_motionlib)

    selected_by_family = Counter(record["_family"] for record in selected)
    report = {
        "source_manifest": str(args.source_manifest.resolve()),
        "lafan_motionlib_path": str(args.lafan_motionlib_path.resolve()),
        "num_source_records": len(source_records),
        "num_clipped_records": len(clipped_records),
        "target_size": args.target_size,
        "num_selected": len(selected_for_manifest),
        "clip_stats": clip_stats,
        "lafan_family_stats": family_stats,
        "lafan_quotas": quotas,
        "family_candidates": report_family_candidates,
        "family_shortage": family_shortage,
        "selected_by_family": dict(selected_by_family),
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

    print(f"Saved matched manifest to {args.output_manifest}")
    print(f"Saved matched motionlib to {args.output_motionlib}")
    print(f"Saved report to {args.output_report}")


if __name__ == "__main__":
    main(tyro.cli(Args))
