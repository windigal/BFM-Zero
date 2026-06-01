from __future__ import annotations

import hashlib
import json
import random
import re
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


SUBJECT_PATTERNS = [
    r"^a person\b",
    r"^the person\b",
    r"^person\b",
    r"^a standing person\b",
    r"^standing person\b",
    r"^lone person\b",
    r"^scared person\b",
    r"^someone\b",
    r"^character\b",
    r"^talent\b",
]

SCENE_NOISE_PATTERNS = [
    r"\bstanding upright\b",
    r"\bstand upright\b",
    r"\bstands upright\b",
    r"\bwhile standing\b",
    r"\bwhile stood\b",
    r"\bstanding\b",
    r"\bstands\b",
    r"\bupright\b",
    r"\bfacing [^,.]+",
    r"\blooks? [^,.]+",
    r"\blooking [^,.]+",
    r"\bto (?:the )?(?:left|right)(?: front| back)? diagonal\b",
    r"\btoward(?:s)? (?:the )?(?:left|right)(?: front| back)? diagonal\b",
    r"\bfrom (?:the )?(?:left|right)(?: front| back)? diagonal\b",
    r"\bdiagonally\b",
    r"\bto chest level\b",
    r"\bat chest height\b",
    r"\bto shoulder height\b",
    r"\bfrom shoulder height\b",
    r"\bto their side\b",
    r"\bfrom their side\b",
    r"\bmomentarily\b",
    r"\bplayfully\b",
    r"\bcomically\b",
    r"\bsteadily\b",
    r"\bslightly\b",
    r"\bneutral(?:ly)?\b",
    r"\bmedium energy\b",
    r"\blight\b",
    r"\bruthlessly\b",
    r"\bswiftly\b",
    r"\ball the time\b",
    r"\bto face [^,.]+",
    r"\bover (?:their |the )?(?:left|right) shoulder\b",
]

TEMPORAL_REJECT_PATTERNS = [
    r"\bthen\b",
    r"\bbefore\b",
    r"\bafter\b",
    r"\bbegin(?:s|ning|ning to)?\b",
    r"\bstart(?:s|ing|ed)?\b",
    r"\bstop(?:s|ping|ped)?\b",
    r"\bcomes? to a stop\b",
    r"\bslows down\b",
    r"\breturns? to\b",
    r"\btransitions? into\b",
    r"\bfrom start to finish\b",
    r"\brises?\b",
    r"\bstraightens? up\b",
    r"\buntil\b",
    r"\blands\b",
]

OBJECT_REJECT_PATTERNS = [
    r"\blasso\b",
    r"\bchair\b",
    r"\bsofa\b",
    r"\bbench\b",
    r"\bstool\b",
    r"\bbox\b",
    r"\bdoor\b",
    r"\bfridge\b",
    r"\bnewspaper\b",
    r"\bphone\b",
    r"\bguitar\b",
    r"\bviolin\b",
    r"\bcrutch(?:es)?\b",
]

ADVERB_DROP_WORDS = {
    "momentarily",
    "playfully",
    "comically",
    "steadily",
    "slightly",
    "neutrally",
    "lightly",
    "swiftly",
    "ruthlessly",
    "still",
    "gently",
    "subtly",
    "heartily",
    "shakily",
    "rapidly",
    "loosely",
    "smooth",
    "sassy",
    "quick",
    "quickly",
    "relaxed",
    "upright",
    "lone",
    "scared",
    "talent",
}

LEADING_NOISE_WORDS = {
    "is",
    "are",
    "was",
    "were",
    "with",
    "in",
    "at",
    "on",
    "by",
    "towards",
    "toward",
    "forward",
    "backward",
    "left",
    "right",
    "head",
    "arms",
    "arm",
    "hands",
    "hand",
    "body",
    "person",
}

LOCOMOTION_WORDS = {
    "walk",
    "run",
    "jog",
    "march",
    "step",
    "move",
    "skip",
    "shuffle",
    "turn",
    "jump",
    "hop",
    "moonwalk",
}

UPPER_BODY_WORDS = {
    "wave",
    "raise",
    "lower",
    "point",
    "laugh",
    "cry",
    "clap",
    "gesture",
    "nod",
    "shake",
    "cross",
    "scratch",
    "touch",
    "punch",
    "bend",
    "bow",
    "extend",
    "swing",
    "cover",
    "clear",
}

ACTION_HEAD_WORDS = LOCOMOTION_WORDS | UPPER_BODY_WORDS | {
    "drop",
    "lean",
    "moonwalk",
    "dance",
    "sway",
    "stretch",
    "hold",
    "rest",
    "tilt",
    "cover",
    "punch",
    "scratch",
    "clear",
    "cry",
    "laugh",
    "jump",
    "turn",
    "walk",
    "run",
    "gesture",
    "macarena",
    "basket",
    "bow",
}

LEMMA_MAP = {
    "raises": "raise",
    "raising": "raise",
    "raised": "raise",
    "lowers": "lower",
    "lowering": "lower",
    "lowered": "lower",
    "laughs": "laugh",
    "laughing": "laugh",
    "walks": "walk",
    "walking": "walk",
    "turns": "turn",
    "turning": "turn",
    "jumps": "jump",
    "jumping": "jump",
    "waves": "wave",
    "waving": "wave",
    "points": "point",
    "pointing": "point",
    "drops": "drop",
    "dropping": "drop",
    "extends": "extend",
    "extended": "extend",
    "extending": "extend",
    "bends": "bend",
    "bending": "bend",
    "touches": "touch",
    "touching": "touch",
    "runs": "run",
    "running": "run",
    "cries": "cry",
    "crying": "cry",
    "claps": "clap",
    "clapping": "clap",
    "nods": "nod",
    "nodding": "nod",
    "shakes": "shake",
    "shaking": "shake",
    "scratches": "scratch",
    "scratching": "scratch",
    "crosses": "cross",
    "crossing": "cross",
    "crossed": "cross",
    "clears": "clear",
    "clearing": "clear",
    "punches": "punch",
    "punching": "punch",
    "leans": "lean",
    "leaning": "lean",
    "bows": "bow",
    "bowing": "bow",
    "swings": "swing",
    "swinging": "swing",
    "marches": "march",
    "marching": "march",
    "moves": "move",
    "moving": "move",
    "moonwalking": "moonwalk",
    "performs": "perform",
    "performing": "perform",
}


@dataclass
class Args:
    input_dir: Path = Path("artifacts/stage_b/primitives_seed_full_dct_f8_k3")
    manifest_path: Path = Path("artifacts/seed_train/seed_train_manifest.jsonl")
    output_dir: Path = Path("artifacts/stage_b/primitives_seed_curated_v2_h2_f8_k3")
    init_history_path: Path = Path("/home/hanwei/code/BFM-zero-deploy/model/tracking_seed_2k_s/zs_seed2k_init_stand_latest.pkl")
    history_len: int = 2
    future_len: int = 8
    prompt_dim: int = 256
    dct_keep_coeffs: int = 3
    rows_per_shard: int = 5000
    min_future_motion: float = 0.2
    static_step_threshold: float = 0.08
    max_edge_static_steps: int = 1
    max_static_run: int = 1
    moving_history_threshold: float = 0.12
    min_run_primitives: int = 8
    max_pool_per_label: int = 128
    max_pool_per_head: int = 256
    standing_history_prob: float = 0.7
    upper_body_only_keep_prob: float = 0.1
    max_normalized_words: int = 10
    max_action_heads: int = 2
    max_while_count: int = 1
    overwrite_output: bool = False
    seed: int = 42


def _deterministic_int(key: str) -> int:
    return int.from_bytes(hashlib.md5(key.encode("utf-8")).digest()[:8], "big")


def _deterministic_float(key: str) -> float:
    return _deterministic_int(key) / float(2**64 - 1)


def _load_manifest_clips(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        fallback = Path("artifacts/seed_train/seed_train_manifest.jsonl")
        if fallback.exists():
            path = fallback
        else:
            raise FileNotFoundError(f"Could not find manifest jsonl: {path}")
    clips: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            sample_type = row.get("sample_type")
            if sample_type not in (None, "clip") and not sample_id.startswith("clip::"):
                continue
            clips[sample_id] = row
    return clips


def _load_standing_history(path: Path, history_len: int, prompt_dim: int) -> np.ndarray:
    arr = joblib.load(path)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] != prompt_dim:
        raise ValueError(f"Expected standing history prompt_dim={prompt_dim}, got {arr.shape}")
    if arr.shape[0] < history_len:
        arr = np.repeat(arr[:1], history_len, axis=0)
    return arr[:history_len].astype(np.float32)


def _lemmatize_token(token: str) -> str:
    return LEMMA_MAP.get(token, token)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _convert_with_clause(text: str) -> str:
    patterns = [
        (r"\bwith (?:their |the )?(right|left|both)? ?(arm|arms|hand|hands|leg|legs) extended\b", "while extend {0} {1}"),
        (r"\bwith (?:their |the )?(right|left|both)? ?(arm|arms|hand|hands|leg|legs) raised\b", "while raise {0} {1}"),
        (r"\bwith (?:their |the )?(right|left|both)? ?(arm|arms|hand|hands|leg|legs) lowered\b", "while lower {0} {1}"),
        (r"\bwith (?:their |the )?(right|left|both)? ?(arm|arms|hand|hands|leg|legs) crossed\b", "while cross {0} {1}"),
    ]
    out = text
    for pattern, template in patterns:
        match = re.search(pattern, out)
        if match is None:
            continue
        side = (match.group(1) or "").strip()
        body = match.group(2).strip()
        phrase = template.format(side, body).replace("  ", " ").strip()
        out = re.sub(pattern, phrase, out)
    return out


def _extract_action_heads(tokens: list[str]) -> list[str]:
    heads: list[str] = []
    for token in tokens:
        if token in ACTION_HEAD_WORDS and (not heads or heads[-1] != token):
            heads.append(token)
    return heads


def classify_action_group(label: str) -> str:
    tokens = set(label.split())
    has_locomotion = any(token in LOCOMOTION_WORDS for token in tokens)
    has_upper_body = any(token in UPPER_BODY_WORDS for token in tokens)
    if has_locomotion and has_upper_body:
        return "locomotion_combo"
    if has_locomotion:
        return "locomotion_only"
    if has_upper_body:
        return "upper_body_only"
    return "other_dynamic"


def normalize_action_label(raw_text: str, *, args: Args) -> tuple[str | None, str | None]:
    text = raw_text.strip().lower()
    if not text:
        return None, "empty_text"
    for pattern in OBJECT_REJECT_PATTERNS:
        if re.search(pattern, text):
            return None, "object_reference"
    for pattern in TEMPORAL_REJECT_PATTERNS:
        if re.search(pattern, text):
            return None, "temporal_multistage"

    text = text.replace("-", " ").replace(",", " ").replace(".", " ").replace(";", " ")
    for pattern in SUBJECT_PATTERNS:
        text = re.sub(pattern, "", text)
    text = _convert_with_clause(text)
    for pattern in SCENE_NOISE_PATTERNS:
        text = re.sub(pattern, " ", text)
    text = text.replace("their ", "").replace("the ", "").replace("a ", "").replace("an ", "").replace("they ", "")
    text = _normalize_whitespace(text)
    if not text:
        return None, "empty_after_cleanup"

    tokens = [_lemmatize_token(token) for token in text.split() if token not in ADVERB_DROP_WORDS]
    tokens = [token for token in tokens if token not in {"person", "someone", "character"}]
    while tokens and tokens[0] in LEADING_NOISE_WORDS:
        tokens.pop(0)
    if tokens and tokens[0] not in ACTION_HEAD_WORDS:
        for idx, token in enumerate(tokens):
            if token in ACTION_HEAD_WORDS:
                tokens = tokens[idx:]
                break
    text = " ".join(tokens)
    text = re.sub(r"\bto forward\b", "forward", text)
    text = re.sub(r"\bto backward\b", "backward", text)
    text = re.sub(r"\bto left\b", "left", text)
    text = re.sub(r"\bto right\b", "right", text)
    text = _normalize_whitespace(text)
    if not text:
        return None, "empty_after_lemmatize"

    if " and " in text and " while " not in text:
        has_locomotion = any(word in text.split() for word in LOCOMOTION_WORDS)
        has_upper_body = any(word in text.split() for word in UPPER_BODY_WORDS)
        if has_locomotion and has_upper_body:
            text = text.replace(" and ", " while ", 1)
        else:
            return None, "and_multistage"

    text = re.sub(r"\bwhile while\b", "while", text)
    text = re.sub(r"\band\b", " ", text)
    text = _normalize_whitespace(text)
    if not text:
        return None, "empty_after_and"
    if text.startswith("while "):
        return None, "bad_prefix"
    if text.count(" while ") > args.max_while_count:
        return None, "too_many_while"

    tokens = text.split()
    if len(tokens) > args.max_normalized_words:
        return None, "too_many_words"
    action_heads = _extract_action_heads(tokens)
    if not action_heads:
        return None, "no_action_head"
    if len(action_heads) > args.max_action_heads:
        return None, "too_many_action_heads"

    head = tokens[0]
    if head not in ACTION_HEAD_WORDS:
        return None, "bad_head"
    return text, head


def _max_consecutive_true(flags: np.ndarray) -> int:
    best = 0
    cur = 0
    for flag in flags.tolist():
        if flag:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _edge_static_run(flags: np.ndarray, reverse: bool = False) -> int:
    seq = flags[::-1] if reverse else flags
    count = 0
    for flag in seq.tolist():
        if not flag:
            break
        count += 1
    return count


def _future_motion_stats(future: np.ndarray, static_step_threshold: float) -> tuple[float, int, int, int]:
    step_deltas = np.linalg.norm(np.diff(future, axis=0), axis=-1)
    static_flags = step_deltas < static_step_threshold
    return (
        float(step_deltas.mean()),
        _max_consecutive_true(static_flags),
        _edge_static_run(static_flags, reverse=False),
        _edge_static_run(static_flags, reverse=True),
    )


def _history_motion(history: np.ndarray) -> float:
    return float(np.linalg.norm(history[1] - history[0]))


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


class CuratedSplitWriter:
    def __init__(self, split_dir: Path, rows_per_shard: int) -> None:
        self.split_dir = split_dir
        self.rows_per_shard = rows_per_shard
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self.current_rows: list[dict[str, Any]] = []
        self.current_shard_idx = 0
        self.index_handle = (self.split_dir / "_index.jsonl").open("w", encoding="utf-8")
        self.total_rows = 0
        self.total_runs = 0

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
            pa.array([row["future_motion_mean"] for row in self.current_rows], type=pa.float32()),
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
            "future_motion_mean",
        ]
        pq.write_table(pa.Table.from_arrays(arrays, names=names), shard_path, compression="zstd")
        self.current_rows.clear()
        self.current_shard_idx += 1

    def add_run(self, sample_id: str, split: str, sample_type: str, rows: list[dict[str, Any]]) -> None:
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
        self.total_runs += 1
        self.total_rows += len(rows)
        self.current_rows.extend(rows)
        if len(self.current_rows) >= self.rows_per_shard:
            self._flush()

    def close(self) -> None:
        self._flush()
        self.index_handle.close()


def _iter_parquet_batches(path: Path, columns: list[str], batch_size: int = 1024):
    parquet = pq.ParquetFile(path)
    yield from parquet.iter_batches(batch_size=batch_size, columns=columns)


def _batch_to_numpy(batch: pa.RecordBatch, field: str, rows: int, frame_len: int, prompt_dim: int) -> np.ndarray:
    array = batch.column(batch.schema.get_field_index(field))
    flat = array.values.to_numpy(zero_copy_only=False)
    return flat.reshape(rows, frame_len, prompt_dim).astype(np.float32)


def _collect_candidate_pools(
    args: Args,
    shard_paths: list[Path],
) -> tuple[dict[str, ReservoirPool], dict[str, ReservoirPool], Counter, list[dict[str, str]]]:
    label_pools: dict[str, ReservoirPool] = {}
    head_pools: dict[str, ReservoirPool] = {}
    filter_counts: Counter = Counter()
    examples: list[dict[str, str]] = []
    seen_examples: set[tuple[str, str]] = set()
    columns = [
        "clip_id",
        "text_chunk",
        "z_hist_raw",
        "z_fut_raw",
    ]
    for shard_path in shard_paths:
        for batch in _iter_parquet_batches(shard_path, columns=columns):
            rows = batch.num_rows
            hist_np = _batch_to_numpy(batch, "z_hist_raw", rows, args.history_len, args.prompt_dim)
            fut_np = _batch_to_numpy(batch, "z_fut_raw", rows, args.future_len, args.prompt_dim)
            text_list = batch.column(batch.schema.get_field_index("text_chunk")).to_pylist()
            for row_idx in range(rows):
                mean_delta, max_static_run, head_run, tail_run = _future_motion_stats(
                    fut_np[row_idx],
                    static_step_threshold=args.static_step_threshold,
                )
                if mean_delta < args.min_future_motion:
                    filter_counts["low_motion"] += 1
                    continue
                if max_static_run > args.max_static_run or head_run > args.max_edge_static_steps or tail_run > args.max_edge_static_steps:
                    filter_counts["static_frames"] += 1
                    continue
                label, reason = normalize_action_label(str(text_list[row_idx]), args=args)
                if label is None:
                    filter_counts[reason or "label_rejected"] += 1
                    continue
                if classify_action_group(label) == "upper_body_only":
                    keep_value = _deterministic_float(f"pool:{shard_path.name}:{row_idx}:{label}")
                    if keep_value > args.upper_body_only_keep_prob:
                        filter_counts["upper_body_only_downsampled"] += 1
                        continue
                if _history_motion(hist_np[row_idx]) < args.moving_history_threshold:
                    continue
                head = label.split()[0]
                label_pools.setdefault(label, ReservoirPool(args.max_pool_per_label, args.seed + len(label_pools))).add(
                    hist_np[row_idx]
                )
                head_pools.setdefault(head, ReservoirPool(args.max_pool_per_head, args.seed + 10_000 + len(head_pools))).add(
                    hist_np[row_idx]
                )
                example_key = (str(text_list[row_idx]), label)
                if example_key not in seen_examples and len(examples) < 64:
                    seen_examples.add(example_key)
                    examples.append({"raw": example_key[0], "normalized": label})
    return label_pools, head_pools, filter_counts, examples


def _choose_history(
    *,
    source_chunk_id: int,
    source_history: np.ndarray,
    label: str,
    head: str,
    history_key: str,
    standing_history: np.ndarray,
    label_pools: dict[str, ReservoirPool],
    head_pools: dict[str, ReservoirPool],
    standing_history_prob: float,
) -> tuple[np.ndarray, str]:
    if source_chunk_id > 0:
        return source_history.copy(), "source_history"

    use_stand = _deterministic_float(history_key) < standing_history_prob
    if not use_stand:
        candidate = label_pools.get(label)
        if candidate is None or not candidate.items:
            candidate = head_pools.get(head)
        if candidate is not None:
            sampled = candidate.sample(history_key)
            if sampled is not None:
                return sampled, "similar_moving"
    return standing_history.copy(), "standing"


def _finalize_clip_rows(
    *,
    clip_rows: list[dict[str, Any]],
    clip_record: dict[str, Any] | None,
    split_writers: dict[str, CuratedSplitWriter],
    standing_history: np.ndarray,
    label_pools: dict[str, ReservoirPool],
    head_pools: dict[str, ReservoirPool],
    stats: dict[str, Any],
    args: Args,
) -> None:
    if not clip_rows:
        return
    split = str(clip_rows[0]["split"])
    sample_type = str(clip_rows[0]["sample_type"])
    package = str(clip_record.get("package", "")) if clip_record else ""
    category = str(clip_record.get("category", "")) if clip_record else ""

    eligible_rows: list[dict[str, Any]] = []
    for row in clip_rows:
        mean_delta, max_static_run, head_run, tail_run = _future_motion_stats(
            row["z_fut_np"],
            static_step_threshold=args.static_step_threshold,
        )
        if mean_delta < args.min_future_motion:
            stats["filter_counts"]["low_motion"] += 1
            continue
        if max_static_run > args.max_static_run or head_run > args.max_edge_static_steps or tail_run > args.max_edge_static_steps:
            stats["filter_counts"]["static_frames"] += 1
            continue
        label, reason = normalize_action_label(str(row["text_chunk"]), args=args)
        if label is None:
            stats["filter_counts"][reason or "label_rejected"] += 1
            continue
        action_group = classify_action_group(label)
        if action_group == "upper_body_only":
            keep_value = _deterministic_float(f"{row['clip_id']}:{row['chunk_id']}:{label}")
            if keep_value > args.upper_body_only_keep_prob:
                stats["filter_counts"]["upper_body_only_downsampled"] += 1
                continue
        row["normalized_label"] = label
        row["normalized_head"] = label.split()[0]
        row["action_group"] = action_group
        row["future_motion_mean"] = mean_delta
        eligible_rows.append(row)

    if not eligible_rows:
        return

    runs: list[list[dict[str, Any]]] = []
    current_run: list[dict[str, Any]] = []
    previous_chunk_id: int | None = None
    for row in eligible_rows:
        chunk_id = int(row["chunk_id"])
        if previous_chunk_id is None or chunk_id == previous_chunk_id + 1:
            current_run.append(row)
        else:
            runs.append(current_run)
            current_run = [row]
        previous_chunk_id = chunk_id
    if current_run:
        runs.append(current_run)

    for run_idx, run_rows in enumerate(runs):
        if len(run_rows) < args.min_run_primitives:
            stats["filter_counts"]["short_run"] += len(run_rows)
            continue
        sample_id = f"{run_rows[0]['clip_id']}::run_{run_idx:03d}"
        run_output_rows: list[dict[str, Any]] = []
        for local_idx, row in enumerate(run_rows):
            if local_idx == 0:
                hist_np, history_source = _choose_history(
                    source_chunk_id=int(row["chunk_id"]),
                    source_history=row["z_hist_np"],
                    label=str(row["normalized_label"]),
                    head=str(row["normalized_head"]),
                    history_key=f"{sample_id}:{local_idx}",
                    standing_history=standing_history,
                    label_pools=label_pools,
                    head_pools=head_pools,
                    standing_history_prob=args.standing_history_prob,
                )
            else:
                hist_np = run_rows[local_idx - 1]["z_fut_np"][-args.history_len :].copy()
                history_source = "previous_chunk_tail"
            fut_np = row["z_fut_np"].astype(np.float32, copy=False)
            dct_np = row["z_fut_dct_np"].astype(np.float32, copy=False)
            run_output_rows.append(
                {
                    "clip_id": sample_id,
                    "sample_type": sample_type,
                    "split": split,
                    "chunk_id": local_idx,
                    "text_chunk": str(row["text_chunk"]),
                    "clip_text": str(row["clip_text"]),
                    "z_hist_raw": hist_np.reshape(-1).astype(np.float16).tolist(),
                    "z_fut_raw": fut_np.reshape(-1).astype(np.float16).tolist(),
                    "z_fut_dct": dct_np.reshape(-1).astype(np.float16).tolist(),
                    "history_len": args.history_len,
                    "future_len": args.future_len,
                    "prompt_dim": args.prompt_dim,
                    "target_representation": "dct",
                    "dct_keep_coeffs": args.dct_keep_coeffs,
                    "window_start": local_idx * args.future_len,
                    "t_start": local_idx * args.future_len,
                    "t_end": (local_idx + 1) * args.future_len,
                    "source_clip_id": str(row["clip_id"]),
                    "source_chunk_id": int(row["chunk_id"]),
                    "source_text_chunk": str(row["text_chunk"]),
                    "history_source": history_source,
                    "future_motion_mean": float(row["future_motion_mean"]),
                }
            )
            stats["history_source_counts"][history_source] += 1
        split_writers[split].add_run(sample_id=sample_id, split=split, sample_type=sample_type, rows=run_output_rows)
        stats["split_run_counts"][split] += 1
        stats["package_counts"][package] += 1
        stats["category_counts"][category] += 1
        for row in run_rows:
            stats["action_group_counts"][str(row["action_group"])] += 1


def build_curated_dataset(args: Args) -> dict[str, Any]:
    input_meta = json.loads((args.input_dir / "meta.json").read_text(encoding="utf-8"))
    if int(input_meta["history_len"]) != args.history_len or int(input_meta["future_len"]) != args.future_len:
        raise ValueError(
            f"Input dataset history/future mismatch: got {input_meta['history_len']}/{input_meta['future_len']} "
            f"expected {args.history_len}/{args.future_len}"
        )
    if args.output_dir.exists():
        has_existing = any(args.output_dir.iterdir())
        if has_existing and not args.overwrite_output:
            raise FileExistsError(f"{args.output_dir} is not empty. Pass --overwrite-output.")
        if has_existing and args.overwrite_output:
            shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clip_manifest = _load_manifest_clips(args.manifest_path)
    standing_history = _load_standing_history(args.init_history_path, args.history_len, args.prompt_dim)

    split_shard_paths = {
        split: sorted((args.input_dir / split).glob("*.parquet"))
        for split in ("train", "val")
        if (args.input_dir / split).exists()
    }
    all_shards = [path for paths in split_shard_paths.values() for path in paths]
    label_pools, head_pools, pool_filter_counts, normalization_examples = _collect_candidate_pools(args, all_shards)

    split_writers = {
        split: CuratedSplitWriter(args.output_dir / split, args.rows_per_shard)
        for split in split_shard_paths.keys()
    }
    stats: dict[str, Any] = {
        "filter_counts": Counter(),
        "pool_filter_counts": Counter(pool_filter_counts),
        "history_source_counts": Counter(),
        "action_group_counts": Counter(),
        "split_run_counts": Counter(),
        "package_counts": Counter(),
        "category_counts": Counter(),
        "normalization_examples": normalization_examples,
        "pool_label_count": len(label_pools),
        "pool_head_count": len(head_pools),
    }

    columns = [
        "clip_id",
        "sample_type",
        "split",
        "chunk_id",
        "text_chunk",
        "clip_text",
        "z_hist_raw",
        "z_fut_raw",
        "z_fut_dct",
        "window_start",
        "t_start",
        "t_end",
    ]
    for split, shard_paths in split_shard_paths.items():
        current_clip_id: str | None = None
        current_clip_rows: list[dict[str, Any]] = []
        for shard_idx, shard_path in enumerate(shard_paths, start=1):
            for batch in _iter_parquet_batches(shard_path, columns=columns):
                rows = batch.num_rows
                hist_np = _batch_to_numpy(batch, "z_hist_raw", rows, args.history_len, args.prompt_dim)
                fut_np = _batch_to_numpy(batch, "z_fut_raw", rows, args.future_len, args.prompt_dim)
                dct_array = batch.column(batch.schema.get_field_index("z_fut_dct"))
                dct_np = dct_array.values.to_numpy(zero_copy_only=False).reshape(rows, args.dct_keep_coeffs, args.prompt_dim).astype(np.float32)
                clip_ids = batch.column(batch.schema.get_field_index("clip_id")).to_pylist()
                sample_types = batch.column(batch.schema.get_field_index("sample_type")).to_pylist()
                splits = batch.column(batch.schema.get_field_index("split")).to_pylist()
                chunk_ids = batch.column(batch.schema.get_field_index("chunk_id")).to_pylist()
                text_chunks = batch.column(batch.schema.get_field_index("text_chunk")).to_pylist()
                clip_texts = batch.column(batch.schema.get_field_index("clip_text")).to_pylist()
                window_starts = batch.column(batch.schema.get_field_index("window_start")).to_pylist()
                t_starts = batch.column(batch.schema.get_field_index("t_start")).to_pylist()
                t_ends = batch.column(batch.schema.get_field_index("t_end")).to_pylist()
                for row_idx in range(rows):
                    clip_id = str(clip_ids[row_idx])
                    if current_clip_id is not None and clip_id != current_clip_id:
                        _finalize_clip_rows(
                            clip_rows=current_clip_rows,
                            clip_record=clip_manifest.get(current_clip_id),
                            split_writers=split_writers,
                            standing_history=standing_history,
                            label_pools=label_pools,
                            head_pools=head_pools,
                            stats=stats,
                            args=args,
                        )
                        current_clip_rows = []
                    current_clip_id = clip_id
                    current_clip_rows.append(
                        {
                            "clip_id": clip_id,
                            "sample_type": str(sample_types[row_idx]),
                            "split": str(splits[row_idx]),
                            "chunk_id": int(chunk_ids[row_idx]),
                            "text_chunk": str(text_chunks[row_idx]),
                            "clip_text": str(clip_texts[row_idx]),
                            "window_start": int(window_starts[row_idx]),
                            "t_start": int(t_starts[row_idx]),
                            "t_end": int(t_ends[row_idx]),
                            "z_hist_np": hist_np[row_idx],
                            "z_fut_np": fut_np[row_idx],
                            "z_fut_dct_np": dct_np[row_idx],
                        }
                    )
            print(f"[{split}] processed shard {shard_idx}/{len(shard_paths)}: {shard_path.name}", flush=True)
        if current_clip_rows:
            _finalize_clip_rows(
                clip_rows=current_clip_rows,
                clip_record=clip_manifest.get(current_clip_id or ""),
                split_writers=split_writers,
                standing_history=standing_history,
                label_pools=label_pools,
                head_pools=head_pools,
                stats=stats,
                args=args,
            )
            current_clip_rows = []
            current_clip_id = None

    for writer in split_writers.values():
        writer.close()

    split_row_counts = {split: writer.total_rows for split, writer in split_writers.items()}
    split_sample_counts = {split: writer.total_runs for split, writer in split_writers.items()}
    summary = {
        "input_dir": str(args.input_dir.resolve()),
        "manifest_path": str(args.manifest_path.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "init_history_path": str(args.init_history_path.resolve()),
        "history_len": args.history_len,
        "future_len": args.future_len,
        "prompt_dim": args.prompt_dim,
        "target_representation": "dct",
        "dct_keep_coeffs": args.dct_keep_coeffs,
        "rows_per_shard": args.rows_per_shard,
        "filter_hyperparams": {
            "min_future_motion": args.min_future_motion,
            "static_step_threshold": args.static_step_threshold,
            "max_edge_static_steps": args.max_edge_static_steps,
            "max_static_run": args.max_static_run,
            "moving_history_threshold": args.moving_history_threshold,
            "min_run_primitives": args.min_run_primitives,
            "standing_history_prob": args.standing_history_prob,
            "upper_body_only_keep_prob": args.upper_body_only_keep_prob,
            "max_normalized_words": args.max_normalized_words,
            "max_action_heads": args.max_action_heads,
            "max_while_count": args.max_while_count,
        },
        "total_samples": int(sum(split_sample_counts.values())),
        "total_rows": int(sum(split_row_counts.values())),
        "split_sample_counts": split_sample_counts,
        "split_row_counts": split_row_counts,
        "split_run_counts": dict(stats["split_run_counts"]),
        "sample_type_counts": {"clip": int(sum(split_sample_counts.values()))},
        "package_counts": dict(stats["package_counts"]),
        "category_counts": dict(stats["category_counts"]),
        "action_group_counts": dict(stats["action_group_counts"]),
        "filter_counts": dict(stats["filter_counts"]),
        "pool_filter_counts": dict(stats["pool_filter_counts"]),
        "history_source_counts": dict(stats["history_source_counts"]),
        "normalization_examples": stats["normalization_examples"],
        "pool_label_count": stats["pool_label_count"],
        "pool_head_count": stats["pool_head_count"],
    }
    (args.output_dir / "meta.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Stage B Curated Dataset",
                "",
                f"- source: `{args.input_dir.resolve()}`",
                f"- history_len: `{args.history_len}`",
                f"- future_len: `{args.future_len}`",
                f"- dct_keep_coeffs: `{args.dct_keep_coeffs}`",
                f"- total_rows: `{summary['total_rows']}`",
                "",
                "This dataset was rebuilt with stricter motion, text, and static-frame filtering.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main(args: Args) -> None:
    summary = build_curated_dataset(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(tyro.cli(Args))
