from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import tyro

from humanoidverse.language.stage_b.frequency import DCTFutureCodec


MOTION_ALLOWED_PACKAGES = (
    "Locomotion",
    "Dances",
    "Sport",
    "Gaming",
)

MOTION_ALLOWED_CATEGORIES = (
    "Baseline",
    "Basic Locomotion Neutral",
    "Basic Locomotion Styles",
    "Dancing",
    "Advanced Locomotion",
    "Unusual Locomotion",
    "Sports",
    "Stunts",
    "Martial Arts",
    "Magic",
)

SHORT_LABEL_TEMPORAL_PATTERNS = (
    r"\bthen\b",
    r"\bafter\b",
    r"\bbefore\b",
    r"\bwhile\b",
    r"\bduring\b",
    r"\bfollowed by\b",
    r"\bsubsequently\b",
    r"\bnext\b",
    r"\bfinally\b",
    r"\bstart(?:s|ing|ed| to)?\b",
    r"\bbegin(?:s|ning|ning to)?\b",
    r"\bstop(?:s|ping|ped)?\b",
    r"\bidle\b",
    r"\bwarm up\b",
    r"\bwarming up\b",
    r"\btransition(?:s|ing)?\b",
    r"\breturn(?:s|ing|ed)?\b",
    r"\bcome(?:s)? to a stop\b",
    r"\btwice\b",
    r"\bmultiple times\b",
    r"\bin a row\b",
)

SHORT_LABEL_OBJECT_PATTERNS = (
    r"\bchair\b",
    r"\bstool\b",
    r"\bbench\b",
    r"\bdoor\b",
    r"\bbox\b",
    r"\bball\b",
    r"\bphone\b",
    r"\bbook\b",
    r"\bcup\b",
    r"\btable\b",
    r"\bviolin\b",
    r"\bguitar\b",
    r"\bfridge\b",
    r"\bcrutch(?:es)?\b",
)

SUBJECT_PREFIX_PATTERNS = (
    r"^a person\b",
    r"^the person\b",
    r"^person\b",
    r"^someone\b",
    r"^character\b",
    r"^a character\b",
    r"^talent\b",
    r"^individual\b",
    r"^performer\b",
)

SHORT_LABEL_LEMMA_MAP = {
    "walks": "walk",
    "walking": "walk",
    "runs": "run",
    "running": "run",
    "jogs": "jog",
    "jogging": "jog",
    "turns": "turn",
    "turning": "turn",
    "jumps": "jump",
    "jumping": "jump",
    "hops": "hop",
    "hopping": "hop",
    "steps": "step",
    "stepping": "step",
    "marches": "march",
    "marching": "march",
    "shuffles": "shuffle",
    "shuffling": "shuffle",
    "dances": "dance",
    "dancing": "dance",
    "squats": "squat",
    "squatting": "squat",
    "lunges": "lunge",
    "lunging": "lunge",
    "backwards": "backward",
    "forwards": "forward",
    "sideways": "sideways",
    "moonwalking": "moonwalk",
}

SHORT_LABEL_DROP_WORDS = {
    "person",
    "someone",
    "character",
    "performer",
    "simple",
    "complex",
    "more",
    "routine",
    "light",
    "whole",
    "body",
    "slight",
    "slightly",
    "fast",
    "slow",
    "slowly",
    "quick",
    "quickly",
    "small",
    "large",
}

SHORT_LABEL_ALLOWED_HEADS = {
    "walk",
    "run",
    "jog",
    "turn",
    "jump",
    "hop",
    "step",
    "march",
    "shuffle",
    "dance",
    "moonwalk",
    "squat",
    "lunge",
}

SHORT_LABEL_ALLOWED_MODIFIERS = {
    "forward",
    "backward",
    "left",
    "right",
    "around",
    "sideways",
    "high",
    "low",
    "up",
    "down",
}


@dataclass
class Args:
    input_dir: Path = Path("artifacts/stage_b/primitives_seed_full_parquet")
    manifest_path: Path = Path("artifacts/seed_train/seed_train_manifest.jsonl")
    output_dir: Path = Path("artifacts/stage_b/primitives_seed_clip_shortlabel_h2_f8_k3")
    init_history_path: Path = Path("/home/hanwei/code/BFM-zero-deploy/model/tracking_seed_2k_s/zs_seed2k_init_stand_latest.pkl")
    history_len: int = 2
    future_len: int = 8
    prompt_dim: int = 256
    dct_keep_coeffs: int = 3
    rows_per_shard: int = 5000
    standing_history_prob: float = 0.7
    static_step_threshold: float = 0.08
    moving_history_threshold: float = 0.12
    min_chunks_per_clip: int = 1
    overwrite_output: bool = False
    seed: int = 42


def _deterministic_int(key: str) -> int:
    return int.from_bytes(hashlib.md5(key.encode("utf-8")).digest()[:8], "big")


def _deterministic_float(key: str) -> float:
    return _deterministic_int(key) / float(2**64 - 1)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _is_mirror(filename: str) -> bool:
    return filename.endswith("_M")


def _load_manifest_clips(path: Path) -> dict[str, dict[str, Any]]:
    clips: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            if not sample_id.startswith("clip::"):
                continue
            clips[sample_id] = row
    return clips


def _load_standing_history(path: Path, history_len: int, prompt_dim: int) -> np.ndarray:
    arr = np.asarray(joblib.load(path), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] != prompt_dim:
        raise ValueError(f"Expected standing history prompt_dim={prompt_dim}, got {arr.shape}")
    if arr.shape[0] < history_len:
        arr = np.repeat(arr[:1], history_len, axis=0)
    return arr[:history_len].astype(np.float32)


def _clean_short_label(raw_text: str) -> str:
    text = raw_text.strip().lower()
    if not text:
        return ""
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[,:;.!?()\[\]{}]+", " ", text)
    for pattern in SUBJECT_PREFIX_PATTERNS:
        text = re.sub(pattern, "", text)
    text = text.replace("their ", "").replace("the ", "").replace("a ", "").replace("an ", "")
    return _normalize_whitespace(text)


def _lemmatize_short_label_token(token: str) -> str:
    return SHORT_LABEL_LEMMA_MAP.get(token, token)


def _normalize_motion_short_label(raw_text: str) -> tuple[str | None, str | None]:
    text = _clean_short_label(raw_text)
    if not text:
        return None, "empty_short_label"
    if "macarena" in text:
        return "macarena dance", None
    for pattern in SHORT_LABEL_TEMPORAL_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return None, "temporal_short_label"
    for pattern in SHORT_LABEL_OBJECT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return None, "object_short_label"

    tokens = [_lemmatize_short_label_token(token) for token in text.split()]
    tokens = [token for token in tokens if token and token not in SHORT_LABEL_DROP_WORDS]
    if not tokens:
        return None, "empty_short_label"

    if tokens[0] == "moonwalk":
        return "moonwalk dance", None
    if tokens[0] == "dance":
        return "dance", None

    head_idx = next((idx for idx, token in enumerate(tokens) if token in SHORT_LABEL_ALLOWED_HEADS), None)
    if head_idx is None:
        return None, "non_motion_short_label"
    if head_idx > 0:
        return None, "leading_noise_short_label"

    head = tokens[0]
    kept = [head]
    seen = {head}
    for token in tokens[1:]:
        if token in SHORT_LABEL_ALLOWED_MODIFIERS and token not in seen:
            kept.append(token)
            seen.add(token)
            continue
        if token in SHORT_LABEL_DROP_WORDS:
            continue
        return None, "noisy_short_label"
    return " ".join(kept), None


def _choose_short_label(record: dict[str, Any]) -> tuple[str | None, str | None]:
    text_fields = record.get("text_fields") or {}
    candidates = [
        str(text_fields.get("content_short_description") or "").strip(),
        str(text_fields.get("content_short_description_2") or "").strip(),
    ]
    raw_choice = next((candidate for candidate in candidates if candidate), "")
    if not raw_choice:
        return None, "missing_short_label"
    return _normalize_motion_short_label(raw_choice)


def _history_motion(history: np.ndarray) -> float:
    return float(np.linalg.norm(history[1] - history[0]))


def _trim_static_edges(z_seq: np.ndarray, static_step_threshold: float) -> tuple[np.ndarray | None, int, int]:
    if z_seq.shape[0] < 2:
        return None, 0, 0
    step_deltas = np.linalg.norm(np.diff(z_seq, axis=0), axis=-1)
    moving_idx = np.flatnonzero(step_deltas >= static_step_threshold)
    if moving_idx.size == 0:
        return None, 0, 0
    start_idx = int(moving_idx[0])
    end_idx = int(moving_idx[-1] + 2)
    start_idx = max(0, min(start_idx, z_seq.shape[0] - 1))
    end_idx = max(start_idx + 1, min(end_idx, z_seq.shape[0]))
    return z_seq[start_idx:end_idx].astype(np.float32, copy=False), start_idx, z_seq.shape[0] - end_idx


def _reconstruct_clip_sequence(clip_rows: list[dict[str, Any]], history_len: int, future_len: int) -> np.ndarray:
    if not clip_rows:
        raise ValueError("clip_rows must not be empty")
    clip_rows = sorted(clip_rows, key=lambda row: int(row["chunk_id"]))
    first_hist = clip_rows[0]["z_hist_np"]
    first_fut = clip_rows[0]["z_fut_np"]
    if first_hist.shape[0] != history_len or first_fut.shape[0] != future_len:
        raise ValueError(
            f"Unexpected primitive shapes for clip {clip_rows[0]['clip_id']}: "
            f"hist={first_hist.shape}, fut={first_fut.shape}"
        )
    parts = [first_hist, first_fut]
    prev_tail = first_fut[-history_len:]
    for row in clip_rows[1:]:
        hist = row["z_hist_np"]
        fut = row["z_fut_np"]
        if hist.shape[0] != history_len or fut.shape[0] != future_len:
            raise ValueError(
                f"Unexpected primitive shapes for clip {row['clip_id']}: hist={hist.shape}, fut={fut.shape}"
            )
        if not np.allclose(hist, prev_tail, atol=5e-3):
            raise ValueError(
                f"Primitive overlap mismatch for clip {row['clip_id']} chunk={row['chunk_id']}"
            )
        parts.append(fut)
        prev_tail = fut[-history_len:]
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


def _iter_parquet_batches(path: Path, columns: list[str], batch_size: int = 1024) -> Iterator[pa.RecordBatch]:
    parquet = pq.ParquetFile(path)
    yield from parquet.iter_batches(batch_size=batch_size, columns=columns)


def _batch_to_numpy(batch: pa.RecordBatch, field: str, rows: int, frame_len: int, prompt_dim: int) -> np.ndarray:
    array = batch.column(batch.schema.get_field_index(field))
    flat = array.values.to_numpy(zero_copy_only=False)
    return flat.reshape(rows, frame_len, prompt_dim).astype(np.float32)


class ReservoirPool:
    def __init__(self, max_size: int, seed: int) -> None:
        self.max_size = max_size
        self.rng = random.Random(seed)
        self.items: list[np.ndarray] = []
        self.seen = 0

    def add(self, value: np.ndarray) -> None:
        self.seen += 1
        payload = value.astype(np.float16, copy=True)
        if len(self.items) < self.max_size:
            self.items.append(payload)
            return
        idx = self.rng.randrange(self.seen)
        if idx < self.max_size:
            self.items[idx] = payload

    def sample(self, key: str) -> np.ndarray | None:
        if not self.items:
            return None
        idx = _deterministic_int(key) % len(self.items)
        return self.items[idx].astype(np.float32, copy=True)


class ClipSplitWriter:
    def __init__(self, split_dir: Path, rows_per_shard: int) -> None:
        self.split_dir = split_dir
        self.rows_per_shard = rows_per_shard
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self.current_rows: list[dict[str, Any]] = []
        self.current_shard_idx = 0
        self.index_handle = (self.split_dir / "_index.jsonl").open("w", encoding="utf-8")
        self.total_rows = 0
        self.total_clips = 0

    def _flush(self) -> None:
        if not self.current_rows:
            return
        shard_path = self.split_dir / f"part-{self.current_shard_idx:05d}.parquet"
        hist_size = len(self.current_rows[0]["z_hist_raw"])
        fut_size = len(self.current_rows[0]["z_fut_raw"])
        dct_size = len(self.current_rows[0]["z_fut_dct"])
        value_type = pa.float16()
        arrays = [
            pa.array([row["clip_id"] for row in self.current_rows], type=pa.string()),
            pa.array([row["sample_type"] for row in self.current_rows], type=pa.string()),
            pa.array([row["split"] for row in self.current_rows], type=pa.string()),
            pa.array([row["chunk_id"] for row in self.current_rows], type=pa.int32()),
            pa.array([row["text_chunk"] for row in self.current_rows], type=pa.string()),
            pa.array([row["clip_text"] for row in self.current_rows], type=pa.string()),
            pa.array([row["z_hist_raw"] for row in self.current_rows], type=pa.list_(value_type, hist_size)),
            pa.array([row["z_fut_raw"] for row in self.current_rows], type=pa.list_(value_type, fut_size)),
            pa.array([row["z_fut_dct"] for row in self.current_rows], type=pa.list_(value_type, dct_size)),
            pa.array([row["history_len"] for row in self.current_rows], type=pa.int16()),
            pa.array([row["future_len"] for row in self.current_rows], type=pa.int16()),
            pa.array([row["prompt_dim"] for row in self.current_rows], type=pa.int16()),
            pa.array([row["target_representation"] for row in self.current_rows], type=pa.string()),
            pa.array([row["dct_keep_coeffs"] for row in self.current_rows], type=pa.int16()),
            pa.array([row["window_start"] for row in self.current_rows], type=pa.int32()),
            pa.array([row["t_start"] for row in self.current_rows], type=pa.int32()),
            pa.array([row["t_end"] for row in self.current_rows], type=pa.int32()),
            pa.array([row["source_clip_id"] for row in self.current_rows], type=pa.string()),
            pa.array([row["source_chunk_id"] for row in self.current_rows], type=pa.int32()),
            pa.array([row["source_text_chunk"] for row in self.current_rows], type=pa.string()),
            pa.array([row["history_source"] for row in self.current_rows], type=pa.string()),
            pa.array([row["trim_start_frames"] for row in self.current_rows], type=pa.int16()),
            pa.array([row["trim_end_frames"] for row in self.current_rows], type=pa.int16()),
        ]
        names = [
            "clip_id",
            "sample_type",
            "split",
            "chunk_id",
            "text_chunk",
            "clip_text",
            "z_hist_raw",
            "z_fut_raw",
            "z_fut_dct",
            "history_len",
            "future_len",
            "prompt_dim",
            "target_representation",
            "dct_keep_coeffs",
            "window_start",
            "t_start",
            "t_end",
            "source_clip_id",
            "source_chunk_id",
            "source_text_chunk",
            "history_source",
            "trim_start_frames",
            "trim_end_frames",
        ]
        pq.write_table(pa.Table.from_arrays(arrays, names=names), shard_path, compression="zstd")
        self.current_rows.clear()
        self.current_shard_idx += 1

    def add_clip_rows(self, split: str, sample_id: str, sample_type: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        if self.current_rows and len(self.current_rows) + len(rows) > self.rows_per_shard:
            self._flush()
        shard_name = f"part-{self.current_shard_idx:05d}.parquet"
        self.index_handle.write(
            json.dumps(
                {
                    "split": split,
                    "sample_id": sample_id,
                    "sample_type": sample_type,
                    "num_rows": len(rows),
                    "shard": shard_name,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        self.total_rows += len(rows)
        self.total_clips += 1
        self.current_rows.extend(rows)
        if len(self.current_rows) >= self.rows_per_shard:
            self._flush()

    def close(self) -> None:
        self._flush()
        self.index_handle.close()


def _select_history(
    *,
    label: str,
    history_key: str,
    standing_history: np.ndarray,
    standing_history_prob: float,
    label_pools: dict[str, ReservoirPool],
    head_pools: dict[str, ReservoirPool],
) -> tuple[np.ndarray, str]:
    use_standing = _deterministic_float(history_key) < standing_history_prob
    if not use_standing:
        label_pool = label_pools.get(label)
        if label_pool is not None:
            sampled = label_pool.sample(history_key)
            if sampled is not None:
                return sampled, "similar_moving"
        head = label.split()[0]
        head_pool = head_pools.get(head)
        if head_pool is not None:
            sampled = head_pool.sample(history_key)
            if sampled is not None:
                return sampled, "similar_moving_head"
    return standing_history.copy(), "standing"


def _iter_clip_rows(split_dir: Path, history_len: int, future_len: int, prompt_dim: int) -> Iterator[list[dict[str, Any]]]:
    columns = ["clip_id", "sample_type", "split", "chunk_id", "z_hist_raw", "z_fut_raw"]
    current_clip_id: str | None = None
    current_rows: list[dict[str, Any]] = []
    for shard_path in sorted(split_dir.glob("*.parquet")):
        for batch in _iter_parquet_batches(shard_path, columns=columns):
            rows = batch.num_rows
            hist_np = _batch_to_numpy(batch, "z_hist_raw", rows, history_len, prompt_dim)
            fut_np = _batch_to_numpy(batch, "z_fut_raw", rows, future_len, prompt_dim)
            clip_ids = batch.column(batch.schema.get_field_index("clip_id")).to_pylist()
            sample_types = batch.column(batch.schema.get_field_index("sample_type")).to_pylist()
            splits = batch.column(batch.schema.get_field_index("split")).to_pylist()
            chunk_ids = batch.column(batch.schema.get_field_index("chunk_id")).to_pylist()
            for row_idx in range(rows):
                clip_id = str(clip_ids[row_idx])
                if current_clip_id is not None and clip_id != current_clip_id:
                    yield current_rows
                    current_rows = []
                current_clip_id = clip_id
                current_rows.append(
                    {
                        "clip_id": clip_id,
                        "sample_type": str(sample_types[row_idx]),
                        "split": str(splits[row_idx]),
                        "chunk_id": int(chunk_ids[row_idx]),
                        "z_hist_np": hist_np[row_idx],
                        "z_fut_np": fut_np[row_idx],
                    }
                )
    if current_rows:
        yield current_rows


def _filter_clip_record(record: dict[str, Any]) -> tuple[str | None, str | None]:
    filename = str(record.get("filename") or "")
    if not filename:
        return None, "missing_filename"
    if _is_mirror(filename):
        return None, "mirror_removed"
    if record.get("package") not in MOTION_ALLOWED_PACKAGES:
        return None, f"package_not_allowed:{record.get('package')}"
    if record.get("category") not in MOTION_ALLOWED_CATEGORIES:
        return None, f"category_not_allowed:{record.get('category')}"
    props_value = str(record.get("content_props") or "").strip().lower()
    if props_value not in {"", "0", "none"}:
        return None, "props_not_empty"
    body_position = str(record.get("content_body_position") or "").strip().lower()
    if body_position != "standing":
        return None, f"body_position_not_standing:{body_position or 'missing'}"
    return _choose_short_label(record)


def _build_history_pools(
    *,
    args: Args,
    manifest_clips: dict[str, dict[str, Any]],
) -> tuple[dict[str, ReservoirPool], dict[str, ReservoirPool], Counter, dict[str, int]]:
    label_pools: dict[str, ReservoirPool] = {}
    head_pools: dict[str, ReservoirPool] = {}
    filter_counts: Counter = Counter()
    label_counts: dict[str, int] = defaultdict(int)
    for split in ("train", "val"):
        split_dir = args.input_dir / split
        if not split_dir.exists():
            continue
        for clip_rows in _iter_clip_rows(split_dir, args.history_len, args.future_len, args.prompt_dim):
            clip_id = str(clip_rows[0]["clip_id"])
            record = manifest_clips.get(clip_id)
            if record is None:
                filter_counts["missing_manifest_record"] += 1
                continue
            label, reason = _filter_clip_record(record)
            if label is None:
                filter_counts[reason or "clip_rejected"] += 1
                continue
            z_seq = _reconstruct_clip_sequence(clip_rows, args.history_len, args.future_len)
            trimmed, _, _ = _trim_static_edges(z_seq, args.static_step_threshold)
            if trimmed is None:
                filter_counts["fully_static_clip"] += 1
                continue
            num_chunks = int(trimmed.shape[0] // args.future_len)
            if num_chunks < args.min_chunks_per_clip:
                filter_counts["too_short_after_trim"] += 1
                continue
            trimmed = trimmed[: num_chunks * args.future_len]
            label_counts[label] += 1
            for chunk_idx in range(1, num_chunks):
                hist = trimmed[chunk_idx * args.future_len - args.history_len : chunk_idx * args.future_len]
                if hist.shape[0] != args.history_len:
                    continue
                if _history_motion(hist) < args.moving_history_threshold:
                    continue
                label_pool = label_pools.setdefault(label, ReservoirPool(256, args.seed + len(label_pools)))
                label_pool.add(hist)
                head = label.split()[0]
                head_pool = head_pools.setdefault(head, ReservoirPool(512, args.seed + 10_000 + len(head_pools)))
                head_pool.add(hist)
    return label_pools, head_pools, filter_counts, label_counts


def build_dataset(args: Args) -> dict[str, Any]:
    if args.output_dir.exists():
        has_existing = any(args.output_dir.iterdir())
        if has_existing and not args.overwrite_output:
            raise FileExistsError(f"{args.output_dir} is not empty. Pass --overwrite-output.")
        if has_existing and args.overwrite_output:
            shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_clips = _load_manifest_clips(args.manifest_path)
    standing_history = _load_standing_history(args.init_history_path, args.history_len, args.prompt_dim)
    dct_codec = DCTFutureCodec(future_len=args.future_len, keep_coeffs=args.dct_keep_coeffs)

    label_pools, head_pools, prefilter_counts, pool_label_counts = _build_history_pools(
        args=args,
        manifest_clips=manifest_clips,
    )

    split_writers = {
        split: ClipSplitWriter(args.output_dir / split, args.rows_per_shard)
        for split in ("train", "val")
        if (args.input_dir / split).exists()
    }

    stats: dict[str, Any] = {
        "filter_counts": Counter(prefilter_counts),
        "history_source_counts": Counter(),
        "split_clip_counts": Counter(),
        "split_row_counts": Counter(),
        "package_counts": Counter(),
        "category_counts": Counter(),
        "label_counts": Counter(),
        "trim_start_hist": Counter(),
        "trim_end_hist": Counter(),
        "pool_label_count": len(label_pools),
        "pool_head_count": len(head_pools),
        "pool_clip_label_counts": dict(sorted(pool_label_counts.items())),
    }

    for split in ("train", "val"):
        split_dir = args.input_dir / split
        if not split_dir.exists():
            continue
        writer = split_writers[split]
        for clip_rows in _iter_clip_rows(split_dir, args.history_len, args.future_len, args.prompt_dim):
            clip_id = str(clip_rows[0]["clip_id"])
            record = manifest_clips.get(clip_id)
            if record is None:
                stats["filter_counts"]["missing_manifest_record"] += 1
                continue
            label, reason = _filter_clip_record(record)
            if label is None:
                stats["filter_counts"][reason or "clip_rejected"] += 1
                continue
            z_seq = _reconstruct_clip_sequence(clip_rows, args.history_len, args.future_len)
            trimmed, trim_start, trim_end = _trim_static_edges(z_seq, args.static_step_threshold)
            if trimmed is None:
                stats["filter_counts"]["fully_static_clip"] += 1
                continue
            num_chunks = int(trimmed.shape[0] // args.future_len)
            if num_chunks < args.min_chunks_per_clip:
                stats["filter_counts"]["too_short_after_trim"] += 1
                continue
            trimmed = trimmed[: num_chunks * args.future_len]
            output_rows: list[dict[str, Any]] = []
            for chunk_idx in range(num_chunks):
                fut_start = chunk_idx * args.future_len
                fut_end = fut_start + args.future_len
                future = trimmed[fut_start:fut_end]
                if future.shape[0] != args.future_len:
                    continue
                if chunk_idx == 0:
                    hist, history_source = _select_history(
                        label=label,
                        history_key=f"{clip_id}:{chunk_idx}",
                        standing_history=standing_history,
                        standing_history_prob=args.standing_history_prob,
                        label_pools=label_pools,
                        head_pools=head_pools,
                    )
                else:
                    hist = trimmed[fut_start - args.history_len : fut_start]
                    history_source = "clip_previous_frames"
                output_rows.append(
                    {
                        "clip_id": clip_id,
                        "sample_type": "clip",
                        "split": split,
                        "chunk_id": chunk_idx,
                        "text_chunk": label,
                        "clip_text": label,
                        "z_hist_raw": hist.reshape(-1).astype(np.float16).tolist(),
                        "z_fut_raw": future.reshape(-1).astype(np.float16).tolist(),
                        "z_fut_dct": dct_codec.encode(torch.from_numpy(future)).reshape(-1).cpu().numpy().astype(np.float16).tolist(),
                        "history_len": args.history_len,
                        "future_len": args.future_len,
                        "prompt_dim": args.prompt_dim,
                        "target_representation": "dct",
                        "dct_keep_coeffs": args.dct_keep_coeffs,
                        "window_start": fut_start,
                        "t_start": fut_start,
                        "t_end": fut_end,
                        "source_clip_id": clip_id,
                        "source_chunk_id": chunk_idx,
                        "source_text_chunk": label,
                        "history_source": history_source,
                        "trim_start_frames": trim_start,
                        "trim_end_frames": trim_end,
                    }
                )
                stats["history_source_counts"][history_source] += 1
            if not output_rows:
                stats["filter_counts"]["empty_after_chunking"] += 1
                continue
            writer.add_clip_rows(split=split, sample_id=clip_id, sample_type="clip", rows=output_rows)
            stats["split_clip_counts"][split] += 1
            stats["split_row_counts"][split] += len(output_rows)
            stats["package_counts"][str(record.get("package") or "")] += 1
            stats["category_counts"][str(record.get("category") or "")] += 1
            stats["label_counts"][label] += 1
            stats["trim_start_hist"][str(trim_start)] += 1
            stats["trim_end_hist"][str(trim_end)] += 1

    for writer in split_writers.values():
        writer.close()

    summary = {
        "input_dir": str(args.input_dir.resolve()),
        "manifest_path": str(args.manifest_path.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "init_history_path": str(args.init_history_path.resolve()),
        "history_len": args.history_len,
        "future_len": args.future_len,
        "prompt_dim": args.prompt_dim,
        "target_representation": "dct",
        "available_target_representations": ["raw", "dct"],
        "dct_keep_coeffs": args.dct_keep_coeffs,
        "rows_per_shard": args.rows_per_shard,
        "standing_history_prob": args.standing_history_prob,
        "similar_history_prob_requested": 1.0 - args.standing_history_prob,
        "static_step_threshold": args.static_step_threshold,
        "moving_history_threshold": args.moving_history_threshold,
        "total_samples": int(sum(stats["split_clip_counts"].values())),
        "total_rows": int(sum(stats["split_row_counts"].values())),
        "split_clip_counts": dict(stats["split_clip_counts"]),
        "split_row_counts": dict(stats["split_row_counts"]),
        "sample_type_counts": {"clip": int(sum(stats["split_clip_counts"].values()))},
        "package_counts": dict(stats["package_counts"]),
        "category_counts": dict(stats["category_counts"]),
        "label_counts_top50": dict(stats["label_counts"].most_common(50)),
        "filter_counts": dict(stats["filter_counts"]),
        "history_source_counts": dict(stats["history_source_counts"]),
        "trim_start_hist": dict(stats["trim_start_hist"]),
        "trim_end_hist": dict(stats["trim_end_hist"]),
        "pool_label_count": stats["pool_label_count"],
        "pool_head_count": stats["pool_head_count"],
    }

    (args.output_dir / "meta.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "\n".join(
            [
                "# SEED Clip-Level Short-Label Dataset",
                "",
                "This dataset is rebuilt from full SEED clip primitives with these rules:",
                "",
                "- mirrored clips are removed",
                "- only motion-heavy packages/categories are kept",
                "- clip-level short semantic labels are used for every primitive row",
                "- obvious static head/tail latent frames are trimmed per clip",
                "- each trimmed clip is split into consecutive future chunks of length 8",
                "- the first chunk uses 70% standing history and 30% similar moving history when available",
                "- later chunks use the previous 2 frames from the same clip as history",
                "",
                f"- total clips: {summary['total_samples']}",
                f"- total rows: {summary['total_rows']}",
                f"- history_len: {args.history_len}",
                f"- future_len: {args.future_len}",
                f"- dct_keep_coeffs: {args.dct_keep_coeffs}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main(args: Args) -> None:
    summary = build_dataset(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(tyro.cli(Args))