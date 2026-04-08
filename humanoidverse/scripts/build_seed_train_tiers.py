from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro

from humanoidverse.language.stage_a.seed import (
    build_fixed_length_clip_records,
    read_manifest,
    write_jsonl_records,
)


def _allocate_quotas_by_category(records: list[dict[str, Any]], target_size: int) -> dict[str, int]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_category[record["category"]].append(record)

    total = len(records)
    if target_size >= total:
        return {category: len(items) for category, items in by_category.items()}

    categories = list(by_category.keys())
    quotas = {category: 0 for category in categories}
    fractional_parts: list[tuple[float, str]] = []

    # Give each non-empty category at least one slot when possible.
    if target_size >= len(categories):
        for category in categories:
            quotas[category] = 1
        remaining = target_size - len(categories)
    else:
        remaining = target_size

    for category, items in by_category.items():
        raw_quota = target_size * (len(items) / total)
        extra = int(raw_quota)
        capacity = len(items) - quotas[category]
        alloc = min(extra, max(capacity, 0))
        quotas[category] += alloc
        fractional_parts.append((raw_quota - int(raw_quota), category))

    assigned = sum(quotas.values())
    remaining = target_size - assigned
    if remaining > 0:
        for _, category in sorted(fractional_parts, reverse=True):
            if remaining <= 0:
                break
            if quotas[category] >= len(by_category[category]):
                continue
            quotas[category] += 1
            remaining -= 1

    return quotas


def _balanced_take_within_category(
    records: list[dict[str, Any]],
    quota: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if quota <= 0 or not records:
        return []
    if quota >= len(records):
        return list(records)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        parent_key = str(record.get("parent_sample_id") or record.get("filename") or record["sample_id"])
        groups[parent_key].append(record)

    parent_keys = list(groups.keys())
    rng.shuffle(parent_keys)
    for parent_key in parent_keys:
        rng.shuffle(groups[parent_key])

    selected: list[dict[str, Any]] = []
    group_indices = {key: 0 for key in parent_keys}
    active_keys = list(parent_keys)

    while active_keys and len(selected) < quota:
        next_active: list[str] = []
        for key in active_keys:
            idx = group_indices[key]
            if idx < len(groups[key]):
                selected.append(groups[key][idx])
                group_indices[key] += 1
                if group_indices[key] < len(groups[key]):
                    next_active.append(key)
                if len(selected) >= quota:
                    break
        active_keys = next_active

    return selected


def _sample_diverse_subset(
    records: list[dict[str, Any]],
    target_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if target_size >= len(records):
        return list(records)

    rng = random.Random(seed)
    quotas = _allocate_quotas_by_category(records, target_size)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_category[record["category"]].append(record)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    for category in sorted(by_category.keys()):
        quota = quotas.get(category, 0)
        picks = _balanced_take_within_category(by_category[category], quota, rng)
        for record in picks:
            selected.append(record)
            selected_ids.add(record["sample_id"])

    if len(selected) < target_size:
        remaining_records = [record for record in records if record["sample_id"] not in selected_ids]
        extra = _balanced_take_within_category(remaining_records, target_size - len(selected), rng)
        selected.extend(extra)

    return selected[:target_size]


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_package: dict[str, int] = defaultdict(int)
    by_category: dict[str, int] = defaultdict(int)
    by_filename: dict[str, int] = defaultdict(int)
    for record in records:
        by_package[record["package"]] += 1
        by_category[record["category"]] += 1
        by_filename[record["filename"]] += 1

    return {
        "num_records": len(records),
        "unique_filenames": len(by_filename),
        "by_package": dict(sorted(by_package.items(), key=lambda kv: kv[1], reverse=True)),
        "by_category": dict(sorted(by_category.items(), key=lambda kv: kv[1], reverse=True)),
    }


@dataclass
class Args:
    manifest_path: Path = Path("artifacts/seed_train/seed_train_manifest.jsonl")
    output_dir: Path = Path("artifacts/seed_train/tiers")
    target_fps: int = 30
    clip_length_s: float = 10.0
    clip_step_s: float | None = None
    tiers: tuple[int, ...] = (20000, 10000, 5000, 2000, 1000)
    seed: int = 42


def main(args: Args) -> None:
    source_records = read_manifest(args.manifest_path)
    clipped_records, clipped_stats = build_fixed_length_clip_records(
        records=source_records,
        clip_length_s=args.clip_length_s,
        clip_step_s=args.clip_step_s,
        target_fps=args.target_fps,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_manifest": str(args.manifest_path.resolve()),
        "num_source_records": len(source_records),
        "num_clipped_records": len(clipped_records),
        "clipped_stats": clipped_stats,
        "tiers": {},
    }

    for tier_size in args.tiers:
        actual_size = min(int(tier_size), len(clipped_records))
        sampled = _sample_diverse_subset(
            records=clipped_records,
            target_size=actual_size,
            seed=args.seed + int(tier_size),
        )
        manifest_path = args.output_dir / f"seed_train_10s_{actual_size}.jsonl"
        report_path = args.output_dir / f"seed_train_10s_{actual_size}.report.json"
        write_jsonl_records(sampled, manifest_path)
        report = _summarize_records(sampled)
        report["target_size"] = int(tier_size)
        report["actual_size"] = actual_size
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        summary["tiers"][str(actual_size)] = {
            "manifest": str(manifest_path.resolve()),
            "report": str(report_path.resolve()),
            **report,
        }
        print(f"Saved tier {actual_size} to {manifest_path}")

    summary_path = args.output_dir / "seed_train_tiers_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved tier summary to {summary_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
