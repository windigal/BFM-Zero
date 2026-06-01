from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from humanoidverse.language.stage_a.seed import (
    build_fixed_length_clip_records,
    manifest_records_to_motion_dict,
    read_manifest,
    write_jsonl_records,
)


POSITIVE_JUMP_PREFIXES = (
    "jump",
    "high_jump",
    "reach_jump",
    "scissors_jump",
    "hop",
)
BLOCKED_JUMP_PREFIXES = (
    "painful_stand_on",
)


@dataclass(slots=True)
class MirrorUnit:
    key: tuple[str, int]
    base_filename: str
    category: str
    package: str
    records: list[dict[str, Any]]
    is_jump: bool

    @property
    def size(self) -> int:
        return len(self.records)


def _mirror_base_filename(filename: str) -> str:
    return filename[:-2] if filename.endswith("_M") else filename


def _is_jump_record(record: dict[str, Any]) -> bool:
    filename = str(record["filename"]).lower()
    if filename.startswith(BLOCKED_JUMP_PREFIXES):
        return False
    if filename.startswith(POSITIVE_JUMP_PREFIXES):
        return True

    text = " ".join([record.get("primary_text", "")] + list(record.get("texts", [])))
    text = text.lower()
    if ("jump" not in text and "hop" not in text) or "pain" in text or "limp" in text:
        return False
    return ("both legs" in text) or ("jumping in place" in text) or ("high jump" in text)


def _group_records_into_units(records: list[dict[str, Any]]) -> list[MirrorUnit]:
    units: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        clip_index = int(record.get("clip_index", 0))
        key = (_mirror_base_filename(str(record["filename"])), clip_index)
        units[key].append(record)

    grouped_units: list[MirrorUnit] = []
    for key, items in units.items():
        items = sorted(items, key=lambda rec: (str(rec["filename"]).endswith("_M"), str(rec["sample_id"])))
        grouped_units.append(
            MirrorUnit(
                key=key,
                base_filename=key[0],
                category=str(items[0]["category"]),
                package=str(items[0]["package"]),
                records=items,
                is_jump=any(_is_jump_record(record) for record in items),
            )
        )
    return grouped_units


def _balanced_take_units(units: list[MirrorUnit], target_records: int, rng: random.Random) -> list[MirrorUnit]:
    if target_records <= 0 or not units:
        return []

    by_parent: dict[str, list[MirrorUnit]] = defaultdict(list)
    for unit in units:
        by_parent[unit.base_filename].append(unit)

    parent_keys = list(by_parent.keys())
    rng.shuffle(parent_keys)
    for parent_key in parent_keys:
        rng.shuffle(by_parent[parent_key])

    selected: list[MirrorUnit] = []
    indices = {key: 0 for key in parent_keys}
    active_keys = list(parent_keys)
    total_records = 0

    while active_keys and total_records < target_records:
        next_active: list[str] = []
        progress = False
        for key in active_keys:
            idx = indices[key]
            if idx >= len(by_parent[key]):
                continue

            unit = by_parent[key][idx]
            indices[key] += 1
            if total_records + unit.size <= target_records:
                selected.append(unit)
                total_records += unit.size
                progress = True

            if indices[key] < len(by_parent[key]):
                next_active.append(key)
            if total_records >= target_records:
                break

        if not progress:
            break
        active_keys = next_active

    return selected


def _allocate_unit_quotas(capacity_by_category: dict[str, int], target_units: int) -> dict[str, int]:
    quotas = {category: 0 for category in capacity_by_category}
    total_capacity = sum(capacity_by_category.values())
    if target_units <= 0 or total_capacity <= 0:
        return quotas

    fractional_parts: list[tuple[float, str]] = []
    for category, capacity in capacity_by_category.items():
        raw_quota = target_units * capacity / total_capacity
        quota = min(capacity, int(raw_quota))
        quotas[category] = quota
        fractional_parts.append((raw_quota - int(raw_quota), category))

    remaining = target_units - sum(quotas.values())
    for _, category in sorted(fractional_parts, reverse=True):
        if remaining <= 0:
            break
        if quotas[category] >= capacity_by_category[category]:
            continue
        quotas[category] += 1
        remaining -= 1
    return quotas


def _collect_records(units: list[MirrorUnit]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for unit in units:
        records.extend(unit.records)
    return records


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_package: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    jump_by_category: Counter[str] = Counter()
    mirror_counts: Counter[str] = Counter()
    jump_records = 0

    for record in records:
        category = str(record["category"])
        package = str(record["package"])
        by_category[category] += 1
        by_package[package] += 1
        if str(record["filename"]).endswith("_M"):
            mirror_counts["mirror_records"] += 1
        else:
            mirror_counts["non_mirror_records"] += 1
        if _is_jump_record(record):
            jump_records += 1
            jump_by_category[category] += 1

    return {
        "num_records": len(records),
        "jump_records": jump_records,
        "jump_ratio": (jump_records / len(records)) if records else 0.0,
        "by_package": dict(sorted(by_package.items(), key=lambda kv: kv[1], reverse=True)),
        "by_category": dict(sorted(by_category.items(), key=lambda kv: kv[1], reverse=True)),
        "jump_by_category": dict(sorted(jump_by_category.items(), key=lambda kv: kv[1], reverse=True)),
        "mirror_counts": dict(mirror_counts),
    }


def _build_subset(
    *,
    source_records: list[dict[str, Any]],
    reference_records: list[dict[str, Any]],
    target_size: int,
    target_jump_ratio: float,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if target_size <= 0:
        raise ValueError("target_size must be > 0")

    clipped_records, clip_stats = build_fixed_length_clip_records(
        records=source_records,
        clip_length_s=10.0,
        clip_step_s=None,
        target_fps=30,
    )
    grouped_units = _group_records_into_units(clipped_records)

    units_by_category: dict[str, list[MirrorUnit]] = defaultdict(list)
    for unit in grouped_units:
        units_by_category[unit.category].append(unit)

    jump_units_by_category = {category: [unit for unit in units if unit.is_jump] for category, units in units_by_category.items()}
    nonjump_units_by_category = {
        category: [unit for unit in units if not unit.is_jump] for category, units in units_by_category.items()
    }

    reference_by_category: Counter[str] = Counter(str(record["category"]) for record in reference_records)
    target_jump_records = int(target_size * target_jump_ratio)
    target_jump_records -= target_jump_records % 2
    target_jump_units = target_jump_records // 2

    jump_capacity_by_category = {
        category: min(len(jump_units_by_category.get(category, [])), reference_by_category.get(category, 0) // 2)
        for category in reference_by_category
    }
    jump_quota_by_category = _allocate_unit_quotas(jump_capacity_by_category, target_jump_units)

    selected_units: list[MirrorUnit] = []
    selected_keys: set[tuple[str, int]] = set()
    selected_by_category: Counter[str] = Counter()
    jump_record_count = 0

    for category in sorted(reference_by_category):
        picks = _balanced_take_units(
            jump_units_by_category.get(category, []),
            jump_quota_by_category.get(category, 0) * 2,
            rng,
        )
        for unit in picks:
            selected_units.append(unit)
            selected_keys.add(unit.key)
            selected_by_category[category] += unit.size
            jump_record_count += unit.size

    for category in sorted(reference_by_category):
        target_category_records = reference_by_category[category]
        remaining_records = max(0, target_category_records - selected_by_category[category])

        nonjump_pool = [unit for unit in nonjump_units_by_category.get(category, []) if unit.key not in selected_keys]
        picks = _balanced_take_units(nonjump_pool, remaining_records, rng)
        for unit in picks:
            selected_units.append(unit)
            selected_keys.add(unit.key)
            selected_by_category[category] += unit.size

        remaining_records = max(0, target_category_records - selected_by_category[category])
        if remaining_records <= 0:
            continue

        jump_pool = [unit for unit in jump_units_by_category.get(category, []) if unit.key not in selected_keys]
        picks = _balanced_take_units(jump_pool, remaining_records, rng)
        for unit in picks:
            selected_units.append(unit)
            selected_keys.add(unit.key)
            selected_by_category[category] += unit.size
            jump_record_count += unit.size

    total_records = sum(unit.size for unit in selected_units)
    remaining_units = [unit for unit in grouped_units if unit.key not in selected_keys]
    for category in sorted(reference_by_category, key=lambda cat: reference_by_category[cat] - selected_by_category[cat], reverse=True):
        if total_records >= target_size:
            break
        category_pool = [unit for unit in remaining_units if unit.category == category]
        picks = _balanced_take_units(category_pool, target_size - total_records, rng)
        for unit in picks:
            selected_units.append(unit)
            selected_keys.add(unit.key)
            selected_by_category[category] += unit.size
            total_records += unit.size
            if unit.is_jump:
                jump_record_count += unit.size
        remaining_units = [unit for unit in remaining_units if unit.key not in selected_keys]

    if total_records < target_size:
        picks = _balanced_take_units(remaining_units, target_size - total_records, rng)
        for unit in picks:
            selected_units.append(unit)
            selected_keys.add(unit.key)
            selected_by_category[unit.category] += unit.size
            total_records += unit.size
            if unit.is_jump:
                jump_record_count += unit.size

    selected_records = _collect_records(selected_units)
    pair_size_counts = Counter(unit.size for unit in selected_units)
    category_delta = {
        category: int(selected_by_category.get(category, 0) - reference_by_category.get(category, 0))
        for category in sorted(reference_by_category)
    }

    report = {
        "clip_stats": clip_stats,
        "target_size": int(target_size),
        "target_jump_ratio": float(target_jump_ratio),
        "target_jump_records": int(target_jump_records),
        "selected_jump_records": int(jump_record_count),
        "selected_jump_ratio": (jump_record_count / len(selected_records)) if selected_records else 0.0,
        "pair_size_counts": dict(pair_size_counts),
        "category_delta_vs_reference": category_delta,
        "summary": _summarize_records(selected_records),
    }
    return selected_records, report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a mirror-paired SEED train subset with a bounded jump ratio.")
    parser.add_argument("--source-manifest", type=Path, default=Path("artifacts/seed_train/seed_train_manifest.jsonl"))
    parser.add_argument("--reference-manifest", type=Path, default=Path("artifacts/seed_train/tiers/seed_train_10s_2000.jsonl"))
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("artifacts/seed_train/ablations/seed_train_10s_2000_jump08_pair.jsonl"),
    )
    parser.add_argument(
        "--output-motionlib",
        type=Path,
        default=Path("humanoidverse/data/seed_train_10s_2000_jump08_pair_with_contact.pkl"),
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("artifacts/seed_train/ablations/seed_train_10s_2000_jump08_pair.report.json"),
    )
    parser.add_argument("--mjcf-path", type=Path, default=Path("humanoidverse/data/robots/g1/g1_29dof.xml"))
    parser.add_argument("--target-size", type=int, default=2000)
    parser.add_argument("--target-jump-ratio", type=float, default=0.08)
    parser.add_argument("--max-jump-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root-euler-order", type=str, default="xyz")
    parser.add_argument("--target-fps", type=int, default=30)
    parser.add_argument("--include-foot-contact-binary", action="store_true", default=True)
    parser.add_argument("--no-include-foot-contact-binary", dest="include_foot_contact_binary", action="store_false")
    parser.add_argument("--contact-dataset-root", type=Path, default=Path("/home/hanwei/dataset/LAFAN1/g1_seed"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.target_jump_ratio < 0 or args.target_jump_ratio > args.max_jump_ratio:
        raise ValueError(
            f"target_jump_ratio must be in [0, max_jump_ratio], got {args.target_jump_ratio} > {args.max_jump_ratio}"
        )

    rng = random.Random(args.seed)
    source_records = read_manifest(args.source_manifest)
    reference_records = read_manifest(args.reference_manifest)

    selected_records, report = _build_subset(
        source_records=source_records,
        reference_records=reference_records,
        target_size=int(args.target_size),
        target_jump_ratio=float(args.target_jump_ratio),
        rng=rng,
    )
    if len(selected_records) != int(args.target_size):
        raise RuntimeError(f"Expected {args.target_size} selected records, got {len(selected_records)}")

    motion_dict = manifest_records_to_motion_dict(
        records=selected_records,
        mjcf_path=args.mjcf_path,
        target_fps=int(args.target_fps),
        root_euler_order=args.root_euler_order,
        include_foot_contact_binary=bool(args.include_foot_contact_binary),
        contact_dataset_root=args.contact_dataset_root,
    )
    if len(motion_dict) != len(selected_records):
        raise RuntimeError(f"Expected {len(selected_records)} motion entries, got {len(motion_dict)}")

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_motionlib.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)

    write_jsonl_records(selected_records, args.output_manifest)
    joblib.dump(motion_dict, args.output_motionlib)

    report.update(
        {
            "source_manifest": str(args.source_manifest.resolve()),
            "reference_manifest": str(args.reference_manifest.resolve()),
            "output_manifest": str(args.output_manifest.resolve()),
            "output_motionlib": str(args.output_motionlib.resolve()),
            "target_fps": int(args.target_fps),
            "include_foot_contact_binary": bool(args.include_foot_contact_binary),
            "contact_dataset_root": str(args.contact_dataset_root.resolve()),
            "jump_classifier": {
                "positive_prefixes": list(POSITIVE_JUMP_PREFIXES),
                "blocked_prefixes": list(BLOCKED_JUMP_PREFIXES),
            },
        }
    )
    args.output_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved manifest to {args.output_manifest}")
    print(f"Saved motionlib to {args.output_motionlib}")
    print(f"Saved report to {args.output_report}")
    print(
        "Selected "
        f"{report['summary']['num_records']} records with jump_ratio={report['summary']['jump_ratio']:.4f} "
        f"and pair_sizes={report['pair_size_counts']}"
    )


if __name__ == "__main__":
    main()
