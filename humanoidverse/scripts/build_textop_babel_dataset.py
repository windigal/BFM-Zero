from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tyro
import yaml


FPS = 50
BABEL_SPLITS = ("train", "val")
MOTION_KEYS = ("root_trans_offset", "root_rot", "dof", "contact_mask")


@dataclass
class Args:
    amass_robot_dir: Path = Path("~/dataset/amass_robot_50fps").expanduser()
    babel_dir: Path = Path("~/dataset/babel_v1-0_release/babel_v1.0_release").expanduser()
    output_dir: Path = Path("artifacts/textop_babel_h2_f8_50fps")
    history_len: int = 2
    future_len: int = 8
    standing_history_mode: str = "zero"
    standing_history_prob: float = 0.7
    rows_per_shard: int = 5000
    overwrite_output: bool = False
    seed: int = 42


def _deterministic_int(key: str) -> int:
    return int.from_bytes(hashlib.md5(key.encode("utf-8")).digest()[:8], "big")


def _deterministic_float(key: str) -> float:
    return _deterministic_int(key) / float(2**64 - 1)


def canonicalize_feat_path(path_like: str) -> str:
    path_str = str(path_like).replace(".npz", ".pkl")
    parts = Path(path_str).as_posix().split("/")
    if len(parts) >= 2 and parts[0] == parts[1]:
        parts = parts[1:]
    else:
        parts = parts[1:]
    return "/".join(parts)


def process_babel_json(babel_json_path: Path) -> dict[str, dict[str, Any]]:
    with babel_json_path.open("r", encoding="utf-8") as f:
        babel_json = json.load(f)
    result: dict[str, dict[str, Any]] = {}

    for key, value in babel_json.items():
        try:
            babel_id = int(key)
            duration = float(value.get("dur", -1))
            assert duration >= 0

            feat_p = value.get("feat_p")
            assert feat_p and feat_p.endswith(".npz"), f"Invalid feat_p for {babel_id}"
            feat_p = canonicalize_feat_path(feat_p)

            frame_ann_raw = value.get("frame_ann")
            has_frame_ann = frame_ann_raw is not None and isinstance(frame_ann_raw, dict)
            if not has_frame_ann:
                seq_ann = value.get("seq_ann")
                assert seq_ann and isinstance(seq_ann, dict) and "labels" in seq_ann, f"Missing seq_ann for {babel_id}"
                seq_labels = seq_ann["labels"]
                for label in seq_labels:
                    label["start_t"] = 0.0
                    label["end_t"] = duration
                frame_ann_raw = {"labels": seq_labels}

            frame_ann_list = []
            for label in frame_ann_raw["labels"]:
                start_t = label.get("start_t")
                end_t = label.get("end_t")
                proc_label = label.get("proc_label")
                act_cat = label.get("act_cat")
                assert start_t is not None and end_t is not None, f"Missing time range for {babel_id}"
                assert proc_label, f"Missing proc_label for {babel_id}"
                assert act_cat and isinstance(act_cat, list), f"Missing act_cat for {babel_id}"
                frame_ann_list.append((float(start_t), float(end_t), str(proc_label), list(act_cat)))

            result[feat_p] = {
                "babel_sid": babel_id,
                "frame_ann": frame_ann_list,
                "duration": duration,
            }
        except AssertionError:
            continue
    return result


def load_babel(babel_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    all_splits: dict[str, dict[str, dict[str, Any]]] = {}
    for split in BABEL_SPLITS:
        json_path = babel_dir / f"{split}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Missing BABEL split file: {json_path}")
        all_splits[split] = process_babel_json(json_path)
    return all_splits


def _motion_length(motion: dict[str, Any]) -> int:
    if "motion_len" in motion:
        return int(motion["motion_len"])
    for key in MOTION_KEYS:
        if key in motion:
            return int(np.asarray(motion[key]).shape[0])
    raise KeyError(f"Could not infer motion_len from keys={list(motion.keys())}")


def _motion_dof_dim(motion: dict[str, Any]) -> int:
    if "dof" not in motion:
        raise KeyError(f"Could not infer dof_dim because motion has no 'dof' field: keys={list(motion.keys())}")
    dof = np.asarray(motion["dof"])
    if dof.ndim != 2:
        raise ValueError(f"Expected motion['dof'] to have shape (T, D), got {dof.shape}")
    return int(dof.shape[1])


def build_amass_robot_index(amass_robot_dir: Path) -> dict[str, Path]:
    if not amass_robot_dir.exists():
        raise FileNotFoundError(f"AMASS robot dir does not exist: {amass_robot_dir}")
    index: dict[str, Path] = {}
    for pkl_path in sorted(amass_robot_dir.rglob("*.pkl")):
        rel_path = canonicalize_feat_path(pkl_path.relative_to(amass_robot_dir).as_posix())
        index[rel_path] = pkl_path
    if not index:
        raise FileNotFoundError(
            f"No robot motion pkl files found under {amass_robot_dir}. "
            "This script expects the output of TextOp's retarget + 50Hz interpolation pipeline."
        )
    return index


def load_motion_from_pkl(path: Path) -> dict[str, Any]:
    payload = joblib.load(path)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Unexpected motion payload in {path}")
    if all(key in payload for key in MOTION_KEYS):
        motion = payload
    else:
        motion = next(iter(payload.values()))
    motion = dict(motion)
    motion["motion_len"] = _motion_length(motion)
    return motion


def merge_textop_style_dataset(
    *,
    amass_robot_index: dict[str, Path],
    babel_by_split: dict[str, dict[str, dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    merged: dict[str, list[dict[str, Any]]] = {}
    stats = {
        "fps": FPS,
        "babel_count": {split: len(entries) for split, entries in babel_by_split.items()},
        "amass_robot_count": len(amass_robot_index),
        "merged_count": {},
        "skipped_missing_motion": 0,
        "skipped_short_motion": 0,
        "dof_dims": [],
    }
    dof_dims_seen: set[int] = set()
    for split in BABEL_SPLITS:
        split_rows: list[dict[str, Any]] = []
        for feat_p, babel_entry in babel_by_split[split].items():
            motion_path = amass_robot_index.get(feat_p)
            if motion_path is None:
                stats["skipped_missing_motion"] += 1
                continue
            motion = load_motion_from_pkl(motion_path)
            if int(motion["motion_len"]) <= 67:
                stats["skipped_short_motion"] += 1
                continue
            motion["fps"] = FPS
            dof_dims_seen.add(_motion_dof_dim(motion))
            split_rows.append(
                {
                    "feat_p": feat_p,
                    "babel_sid": int(babel_entry["babel_sid"]),
                    "frame_ann": babel_entry["frame_ann"],
                    "duration": float(babel_entry["duration"]),
                    "length": int(np.ceil(float(babel_entry["duration"]) * FPS)),
                    "motion": motion,
                }
            )
        merged[split] = split_rows
        stats["merged_count"][split] = len(split_rows)
    stats["dof_dims"] = sorted(dof_dims_seen)
    return merged, stats


def _get_overlap(seg1: tuple[float, float], seg2: tuple[float, float]) -> float:
    return max(0.0, min(seg1[1], seg2[1]) - max(seg1[0], seg2[0]))


def _overlapping_annotations(frame_ann: list[tuple[float, float, str, list[str]]], future_start: int, future_end: int) -> list[dict[str, Any]]:
    overlaps: list[dict[str, Any]] = []
    for start_t, end_t, proc_label, act_cat in frame_ann:
        overlap = _get_overlap((start_t * FPS, end_t * FPS), (future_start, future_end))
        if overlap > 0:
            overlaps.append(
                {
                    "label": proc_label,
                    "act_cat": act_cat,
                    "overlap": overlap,
                    "start_t": start_t,
                    "end_t": end_t,
                }
            )
    overlaps.sort(key=lambda item: (-item["overlap"], item["label"]))
    return overlaps


def _slice_motion(motion: dict[str, Any], start: int, end: int) -> dict[str, np.ndarray]:
    sliced = {}
    for key in MOTION_KEYS:
        sliced[key] = np.asarray(motion[key][start:end], dtype=np.float32)
    sliced["motion_len"] = int(end - start)
    sliced["fps"] = FPS
    return sliced


def _build_zero_standing_history(history_len: int, dof_dim: int) -> dict[str, np.ndarray]:
    frames = history_len + 1
    root_trans_offset = np.repeat(np.array([[0.0, 0.0, 0.77]], dtype=np.float32), frames, axis=0)
    root_rot = np.repeat(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), frames, axis=0)
    dof = np.zeros((frames, dof_dim), dtype=np.float32)
    contact_mask = np.ones((frames, 2), dtype=np.float32)
    return {
        "root_trans_offset": root_trans_offset,
        "root_rot": root_rot,
        "dof": dof,
        "contact_mask": contact_mask,
        "motion_len": frames,
        "fps": FPS,
    }


def _flatten_motion_field(
    motion: dict[str, np.ndarray],
    key: str,
    *,
    expected_frames: int | None = None,
    expected_dim: int | None = None,
    fill_value: float = 0.0,
) -> list[float]:
    if key not in motion:
        if expected_frames is None or expected_dim is None:
            raise KeyError(f"Missing field {key!r} and no fallback shape provided.")
        arr = np.full((expected_frames, expected_dim), fill_value, dtype=np.float32)
        return arr.reshape(-1).astype(np.float16).tolist()
    arr = np.asarray(motion[key], dtype=np.float32)
    if expected_frames is not None and expected_dim is not None:
        if arr.size == 0:
            arr = np.full((expected_frames, expected_dim), fill_value, dtype=np.float32)
        else:
            arr = arr.reshape(arr.shape[0], -1)
            if arr.shape[0] != expected_frames or arr.shape[1] != expected_dim:
                arr_fixed = np.full((expected_frames, expected_dim), fill_value, dtype=np.float32)
                copy_frames = min(expected_frames, arr.shape[0])
                copy_dim = min(expected_dim, arr.shape[1])
                arr_fixed[:copy_frames, :copy_dim] = arr[:copy_frames, :copy_dim]
                arr = arr_fixed
    return arr.reshape(-1).astype(np.float16).tolist()


class HistoryPool:
    def __init__(self, max_size: int, seed: int) -> None:
        self.max_size = max_size
        self.rng = random.Random(seed)
        self.items: list[dict[str, np.ndarray]] = []
        self.seen = 0

    def add(self, motion_hist: dict[str, np.ndarray]) -> None:
        self.seen += 1
        payload = {
            key: np.asarray(value, dtype=np.float32).astype(np.float16)
            for key, value in motion_hist.items()
            if key in MOTION_KEYS
        }
        payload["motion_len"] = motion_hist["motion_len"]
        payload["fps"] = motion_hist["fps"]
        if len(self.items) < self.max_size:
            self.items.append(payload)
            return
        idx = self.rng.randrange(self.seen)
        if idx < self.max_size:
            self.items[idx] = payload

    def sample(self, key: str) -> dict[str, np.ndarray] | None:
        if not self.items:
            return None
        idx = _deterministic_int(key) % len(self.items)
        cached = self.items[idx]
        restored = {
            field: np.asarray(cached[field], dtype=np.float16).astype(np.float32)
            for field in MOTION_KEYS
        }
        restored["motion_len"] = cached["motion_len"]
        restored["fps"] = cached["fps"]
        return restored


class PrimitiveWriter:
    def __init__(self, split_dir: Path, rows_per_shard: int, dof_dim: int) -> None:
        self.split_dir = split_dir
        self.rows_per_shard = rows_per_shard
        self.dof_dim = dof_dim
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self.current_rows: list[dict[str, Any]] = []
        self.shard_idx = 0
        self.total_rows = 0
        self.index_handle = (self.split_dir / "_index.jsonl").open("w", encoding="utf-8")

    def _flush(self) -> None:
        if not self.current_rows:
            return
        first = self.current_rows[0]
        hist_frames = first["history_len"] + 1
        fut_frames = first["future_len"] + 1
        arrays = [
            pa.array([row["sample_id"] for row in self.current_rows], type=pa.string()),
            pa.array([row["split"] for row in self.current_rows], type=pa.string()),
            pa.array([row["feat_p"] for row in self.current_rows], type=pa.string()),
            pa.array([row["babel_sid"] for row in self.current_rows], type=pa.int32()),
            pa.array([row["text_chunk"] for row in self.current_rows], type=pa.string()),
            pa.array([json.dumps(row["text_candidates"], ensure_ascii=False) for row in self.current_rows], type=pa.string()),
            pa.array([json.dumps(row["act_cat_candidates"], ensure_ascii=False) for row in self.current_rows], type=pa.string()),
            pa.array([row["history_source"] for row in self.current_rows], type=pa.string()),
            pa.array([row["future_start"] for row in self.current_rows], type=pa.int32()),
            pa.array([row["future_end"] for row in self.current_rows], type=pa.int32()),
            pa.array([row["history_root_trans_offset"] for row in self.current_rows], type=pa.list_(pa.float16(), hist_frames * 3)),
            pa.array([row["history_root_rot"] for row in self.current_rows], type=pa.list_(pa.float16(), hist_frames * 4)),
            pa.array([row["history_dof"] for row in self.current_rows], type=pa.list_(pa.float16(), hist_frames * self.dof_dim)),
            pa.array([row["history_contact_mask"] for row in self.current_rows], type=pa.list_(pa.float16(), hist_frames * 2)),
            pa.array([row["future_root_trans_offset"] for row in self.current_rows], type=pa.list_(pa.float16(), fut_frames * 3)),
            pa.array([row["future_root_rot"] for row in self.current_rows], type=pa.list_(pa.float16(), fut_frames * 4)),
            pa.array([row["future_dof"] for row in self.current_rows], type=pa.list_(pa.float16(), fut_frames * self.dof_dim)),
            pa.array([row["future_contact_mask"] for row in self.current_rows], type=pa.list_(pa.float16(), fut_frames * 2)),
        ]
        names = [
            "sample_id",
            "split",
            "feat_p",
            "babel_sid",
            "text_chunk",
            "text_candidates",
            "act_cat_candidates",
            "history_source",
            "future_start",
            "future_end",
            "history_root_trans_offset",
            "history_root_rot",
            "history_dof",
            "history_contact_mask",
            "future_root_trans_offset",
            "future_root_rot",
            "future_dof",
            "future_contact_mask",
        ]
        shard_name = f"part-{self.shard_idx:05d}.parquet"
        pq.write_table(pa.Table.from_arrays(arrays, names=names), self.split_dir / shard_name, compression="zstd")
        self.index_handle.write(json.dumps({"split": first["split"], "num_rows": len(self.current_rows), "shard": shard_name}, ensure_ascii=False) + "\n")
        self.total_rows += len(self.current_rows)
        self.current_rows.clear()
        self.shard_idx += 1

    def add(self, row: dict[str, Any]) -> None:
        self.current_rows.append(row)
        if len(self.current_rows) >= self.rows_per_shard:
            self._flush()

    def close(self) -> None:
        self._flush()
        self.index_handle.close()


def build_primitive_dataset(
    *,
    merged: dict[str, list[dict[str, Any]]],
    args: Args,
    dof_dim: int,
) -> dict[str, Any]:
    standing_hist = _build_zero_standing_history(args.history_len, dof_dim)
    output_primitives = args.output_dir / "primitives"
    if output_primitives.exists() and args.overwrite_output:
        shutil.rmtree(output_primitives)
    output_primitives.mkdir(parents=True, exist_ok=True)

    history_pools: dict[str, HistoryPool] = {}
    history_pool_counts: Counter = Counter()
    for split, entries in merged.items():
        for entry in entries:
            motion = entry["motion"]
            motion_len = int(motion["motion_len"])
            for future_start in range(args.future_len, motion_len - args.future_len, args.future_len):
                hist = _slice_motion(motion, future_start - args.history_len, future_start + 1)
                overlaps = _overlapping_annotations(entry["frame_ann"], future_start, future_start + args.future_len)
                if not overlaps:
                    continue
                key = overlaps[0]["label"]
                history_pools.setdefault(key, HistoryPool(max_size=256, seed=args.seed + len(history_pools))).add(hist)
                history_pool_counts[key] += 1

    writers = {split: PrimitiveWriter(output_primitives / split, args.rows_per_shard, dof_dim=dof_dim) for split in merged.keys()}
    summary = {
        "splits": {},
        "history_source_counts": Counter(),
        "text_candidate_count_hist": Counter(),
        "history_pool_labels": len(history_pools),
    }
    for split, entries in merged.items():
        rows = 0
        for entry in entries:
            motion = entry["motion"]
            motion_len = int(motion["motion_len"])
            for future_start in range(0, motion_len - args.future_len, args.future_len):
                future_end = future_start + args.future_len
                overlaps = _overlapping_annotations(entry["frame_ann"], future_start, future_end)
                if not overlaps:
                    continue
                text_candidates = [item["label"] for item in overlaps]
                act_cat_candidates = [item["act_cat"] for item in overlaps]
                text_chunk = text_candidates[0]

                if future_start == 0:
                    if _deterministic_float(f"{split}:{entry['feat_p']}:{future_start}") < args.standing_history_prob:
                        history = standing_hist
                        history_source = "standing"
                    else:
                        history = history_pools.get(text_chunk, HistoryPool(1, args.seed)).sample(f"{entry['feat_p']}:{future_start}")
                        if history is None:
                            history = standing_hist
                            history_source = "standing_fallback"
                        else:
                            history_source = "similar_moving"
                else:
                    history = _slice_motion(motion, future_start - args.history_len, future_start + 1)
                    history_source = "previous_frames"

                future_motion = _slice_motion(motion, future_start, future_end + 1)
                row = {
                    "sample_id": f"{entry['feat_p']}::{future_start:06d}",
                    "split": split,
                    "feat_p": entry["feat_p"],
                    "babel_sid": int(entry["babel_sid"]),
                    "text_chunk": text_chunk,
                    "text_candidates": text_candidates,
                    "act_cat_candidates": act_cat_candidates,
                    "history_source": history_source,
                    "future_start": future_start,
                    "future_end": future_end,
                    "history_root_trans_offset": _flatten_motion_field(history, "root_trans_offset", expected_frames=args.history_len + 1, expected_dim=3),
                    "history_root_rot": _flatten_motion_field(history, "root_rot", expected_frames=args.history_len + 1, expected_dim=4),
                    "history_dof": _flatten_motion_field(history, "dof", expected_frames=args.history_len + 1, expected_dim=dof_dim),
                    "history_contact_mask": _flatten_motion_field(history, "contact_mask", expected_frames=args.history_len + 1, expected_dim=2, fill_value=1.0),
                    "future_root_trans_offset": _flatten_motion_field(future_motion, "root_trans_offset", expected_frames=args.future_len + 1, expected_dim=3),
                    "future_root_rot": _flatten_motion_field(future_motion, "root_rot", expected_frames=args.future_len + 1, expected_dim=4),
                    "future_dof": _flatten_motion_field(future_motion, "dof", expected_frames=args.future_len + 1, expected_dim=dof_dim),
                    "future_contact_mask": _flatten_motion_field(future_motion, "contact_mask", expected_frames=args.future_len + 1, expected_dim=2, fill_value=0.0),
                    "history_len": args.history_len,
                    "future_len": args.future_len,
                }
                writers[split].add(row)
                rows += 1
                summary["history_source_counts"][history_source] += 1
                summary["text_candidate_count_hist"][str(len(text_candidates))] += 1
        writers[split].close()
        summary["splits"][split] = {"rows": writers[split].total_rows}
    summary["history_source_counts"] = dict(summary["history_source_counts"])
    summary["text_candidate_count_hist"] = dict(summary["text_candidate_count_hist"])
    return summary


def main(args: Args) -> None:
    if args.output_dir.exists():
        has_existing = any(args.output_dir.iterdir())
        if has_existing and not args.overwrite_output:
            raise FileExistsError(f"{args.output_dir} is not empty. Pass --overwrite-output.")
        if has_existing and args.overwrite_output:
            shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    babel_by_split = load_babel(args.babel_dir)
    amass_robot_index = build_amass_robot_index(args.amass_robot_dir)
    merged, merge_stats = merge_textop_style_dataset(
        amass_robot_index=amass_robot_index,
        babel_by_split=babel_by_split,
    )
    dof_dims = list(merge_stats.get("dof_dims") or [])
    if not dof_dims:
        raise RuntimeError("Could not infer dof_dim from merged AMASS robot motions.")
    if len(dof_dims) != 1:
        raise ValueError(f"Expected a single dof_dim in amass_robot_dir, got {dof_dims}")
    dof_dim = int(dof_dims[0])

    for split, rows in merged.items():
        split_path = args.output_dir / f"{split}.pkl"
        joblib.dump(rows, split_path)

    primitive_stats = build_primitive_dataset(merged=merged, args=args, dof_dim=dof_dim)
    stats = {
        "fps": FPS,
        "dof_dim": dof_dim,
        "babel_dir": str(args.babel_dir.resolve()),
        "amass_robot_dir": str(args.amass_robot_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "history_len": args.history_len,
        "future_len": args.future_len,
        "standing_history_prob": args.standing_history_prob,
        "merge_stats": merge_stats,
        "primitive_stats": primitive_stats,
    }
    with (args.output_dir / "statistics.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(stats, f, sort_keys=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(tyro.cli(Args))
