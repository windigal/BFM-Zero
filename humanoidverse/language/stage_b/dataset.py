from __future__ import annotations

import gzip
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .frequency import DCTFutureCodec


@dataclass(slots=True)
class BFMTextOpSequenceItem:
    sample_id: str
    split: str
    clip_text: str
    primitive_texts: list[str]
    z_hist: torch.Tensor
    z_fut: torch.Tensor
    z_fut_target: torch.Tensor
    window_start: int


@dataclass(slots=True)
class BFMTextOpSequenceDatasetConfig:
    history_len: int = 4
    future_len: int = 16
    num_primitives: int = 8
    window_stride: int = 16
    sample_types: tuple[str, ...] = ("clip",)
    require_z_seq: bool = True
    min_overlap_s: float = 1e-6
    target_representation: str = "dct"
    dct_keep_coeffs: int | None = 4


@dataclass(slots=True)
class BFMTextOpPrimitiveSequenceIterableConfig:
    history_len: int = 4
    future_len: int = 16
    prompt_dim: int = 256
    num_primitives: int = 8
    sequence_stride_primitives: int = 1
    shuffle: bool = False
    shuffle_buffer_size: int = 1024
    seed: int = 42
    target_representation: str = "dct"
    dct_keep_coeffs: int | None = 4


def _normalize_target_representation(value: str) -> str:
    normalized = value.lower().strip()
    if normalized not in {"raw", "dct"}:
        raise ValueError(f"Unsupported target_representation={value!r}. Expected 'raw' or 'dct'.")
    return normalized


def _maybe_build_dct_codec(
    *,
    target_representation: str,
    future_len: int,
    dct_keep_coeffs: int | None,
) -> DCTFutureCodec | None:
    normalized = _normalize_target_representation(target_representation)
    if normalized == "raw":
        return None
    if dct_keep_coeffs is None:
        raise ValueError("dct_keep_coeffs must be provided when target_representation='dct'.")
    return DCTFutureCodec(future_len=future_len, keep_coeffs=int(dct_keep_coeffs))


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def select_primitive_text(
    record: dict[str, Any],
    future_start_idx: int,
    future_end_idx: int,
    seq_len: int,
    min_overlap_s: float,
) -> str:
    fallback = str(record.get("primary_text") or "").strip()
    events = record.get("events") or []
    duration_s = _to_float(record.get("duration_s"), 0.0)
    sample_start_s = _to_float(record.get("start_time_s"), 0.0)

    if not events or duration_s <= 0 or seq_len <= 0:
        return fallback

    latent_fps = seq_len / duration_s
    future_start_s = sample_start_s + future_start_idx / latent_fps
    future_end_s = sample_start_s + future_end_idx / latent_fps

    best_text = fallback
    best_overlap = 0.0
    for event in events:
        description = str(event.get("description") or "").strip()
        if not description:
            continue
        event_start_s = _to_float(event.get("start_time"), sample_start_s) - sample_start_s
        event_end_s = _to_float(event.get("end_time"), sample_start_s) - sample_start_s
        overlap = max(
            0.0,
            min(future_end_s - sample_start_s, event_end_s) - max(future_start_s - sample_start_s, event_start_s),
        )
        if overlap > best_overlap and overlap >= min_overlap_s:
            best_text = description
            best_overlap = overlap
    return best_text or fallback


class BFMTextOpSequenceDataset(Dataset[BFMTextOpSequenceItem]):
    def __init__(
        self,
        manifest_records: list[dict[str, Any]],
        latent_store: dict[str, Any],
        split: str,
        cfg: BFMTextOpSequenceDatasetConfig,
    ) -> None:
        self.split = split
        self.cfg = cfg
        self.target_representation = _normalize_target_representation(cfg.target_representation)
        self.dct_codec = _maybe_build_dct_codec(
            target_representation=self.target_representation,
            future_len=cfg.future_len,
            dct_keep_coeffs=cfg.dct_keep_coeffs,
        )
        self.samples: dict[str, dict[str, Any]] = {}
        self.index: list[tuple[str, int]] = []

        if cfg.history_len <= 0 or cfg.future_len <= 0 or cfg.num_primitives <= 0:
            raise ValueError("history_len, future_len, and num_primitives must all be > 0")

        segment_len = cfg.history_len + cfg.future_len * cfg.num_primitives

        for record in manifest_records:
            if record.get("split") != split:
                continue
            if cfg.sample_types and record.get("sample_type") not in cfg.sample_types:
                continue

            sample_id = str(record["sample_id"])
            latent = latent_store.get(sample_id)
            if latent is None:
                continue
            if cfg.require_z_seq and "z_seq" not in latent:
                continue

            z_seq = latent.get("z_seq")
            if z_seq is None:
                continue
            z_seq_t = torch.as_tensor(z_seq, dtype=torch.float32)
            if z_seq_t.ndim != 2:
                continue
            if z_seq_t.shape[0] < segment_len:
                continue

            self.samples[sample_id] = {
                "record": record,
                "z_seq": z_seq_t,
                "clip_text": str(record.get("primary_text") or "").strip(),
            }
            max_start = z_seq_t.shape[0] - segment_len
            for start in range(0, max_start + 1, max(int(cfg.window_stride), 1)):
                self.index.append((sample_id, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> BFMTextOpSequenceItem:
        sample_id, window_start = self.index[idx]
        sample = self.samples[sample_id]
        record = sample["record"]
        z_seq: torch.Tensor = sample["z_seq"]

        segment_len = self.cfg.history_len + self.cfg.future_len * self.cfg.num_primitives
        window = z_seq[window_start : window_start + segment_len]

        z_hist: list[torch.Tensor] = []
        z_fut: list[torch.Tensor] = []
        z_fut_target: list[torch.Tensor] = []
        primitive_texts: list[str] = []
        for primitive_idx in range(self.cfg.num_primitives):
            primitive_start = primitive_idx * self.cfg.future_len
            hist = window[primitive_start : primitive_start + self.cfg.history_len]
            fut = window[
                primitive_start + self.cfg.history_len : primitive_start + self.cfg.history_len + self.cfg.future_len
            ]
            if hist.shape[0] != self.cfg.history_len or fut.shape[0] != self.cfg.future_len:
                raise RuntimeError(
                    f"Invalid primitive slice for sample_id={sample_id}, window_start={window_start}, primitive={primitive_idx}"
                )
            future_start_idx = window_start + primitive_start + self.cfg.history_len
            future_end_idx = future_start_idx + self.cfg.future_len
            primitive_texts.append(
                select_primitive_text(
                    record=record,
                    future_start_idx=future_start_idx,
                    future_end_idx=future_end_idx,
                    seq_len=int(z_seq.shape[0]),
                    min_overlap_s=self.cfg.min_overlap_s,
                )
            )
            z_hist.append(hist)
            z_fut.append(fut)
            z_fut_target.append(self.dct_codec.encode(fut) if self.dct_codec is not None else fut)

        return BFMTextOpSequenceItem(
            sample_id=sample_id,
            split=self.split,
            clip_text=str(sample["clip_text"]),
            primitive_texts=primitive_texts,
            z_hist=torch.stack(z_hist, dim=0),
            z_fut=torch.stack(z_fut, dim=0),
            z_fut_target=torch.stack(z_fut_target, dim=0),
            window_start=window_start,
        )


def _open_text_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _resolve_index_path(shard_paths: list[Path]) -> Path | None:
    if not shard_paths:
        return None
    candidate = shard_paths[0].parent / "_index.jsonl"
    if candidate.exists():
        return candidate
    return None


class BFMTextOpPrimitiveSequenceIterableDataset(IterableDataset[BFMTextOpSequenceItem]):
    def __init__(
        self,
        shard_paths: list[Path],
        split: str,
        cfg: BFMTextOpPrimitiveSequenceIterableConfig,
    ) -> None:
        if cfg.history_len <= 0 or cfg.future_len <= 0 or cfg.num_primitives <= 0:
            raise ValueError("history_len, future_len, and num_primitives must all be > 0")
        if cfg.sequence_stride_primitives <= 0:
            raise ValueError("sequence_stride_primitives must be > 0")
        self.shard_paths = sorted(Path(path) for path in shard_paths)
        self.split = split
        self.cfg = cfg
        self.target_representation = _normalize_target_representation(cfg.target_representation)
        self.dct_codec = _maybe_build_dct_codec(
            target_representation=self.target_representation,
            future_len=cfg.future_len,
            dct_keep_coeffs=cfg.dct_keep_coeffs,
        )
        self._length = self._compute_length()

    def _compute_length(self) -> int | None:
        index_path = _resolve_index_path(self.shard_paths)
        if index_path is None:
            return None
        total = 0
        with index_path.open("r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("split") != self.split:
                    continue
                num_rows = int(entry.get("num_rows", 0))
                if num_rows < self.cfg.num_primitives:
                    continue
                total += 1 + (num_rows - self.cfg.num_primitives) // self.cfg.sequence_stride_primitives
        return total

    def __len__(self) -> int:
        if self._length is None:
            raise TypeError("Length is unavailable because _index.jsonl was not found for primitive shards.")
        return self._length

    def _iter_rows(self, path: Path) -> Iterator[dict[str, Any]]:
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=1024):
                batch_dict = batch.to_pydict()
                if not batch_dict:
                    continue
                num_rows = len(next(iter(batch_dict.values())))
                for row_idx in range(num_rows):
                    yield {key: values[row_idx] for key, values in batch_dict.items()}
            return

        with _open_text_maybe_gzip(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def _window_to_item(self, window_rows: list[dict[str, Any]]) -> BFMTextOpSequenceItem:
        z_hist = torch.stack(
            [
                torch.tensor(row["z_hist_raw"], dtype=torch.float32).view(self.cfg.history_len, self.cfg.prompt_dim)
                for row in window_rows
            ],
            dim=0,
        )
        z_fut = torch.stack(
            [
                torch.tensor(row["z_fut_raw"], dtype=torch.float32).view(self.cfg.future_len, self.cfg.prompt_dim)
                for row in window_rows
            ],
            dim=0,
        )
        if self.target_representation == "dct":
            expected_keep = self.dct_codec.keep_coeffs if self.dct_codec is not None else self.cfg.future_len
            z_fut_target = torch.stack(
                [
                    (
                        torch.tensor(row["z_fut_dct"], dtype=torch.float32).view(
                            expected_keep,
                            self.cfg.prompt_dim,
                        )
                        if row.get("z_fut_dct") is not None
                        and int(row.get("dct_keep_coeffs", expected_keep)) == expected_keep
                        else self.dct_codec.encode(z_fut[row_idx])
                    )
                    for row_idx, row in enumerate(window_rows)
                ],
                dim=0,
            )
        else:
            z_fut_target = z_fut
        return BFMTextOpSequenceItem(
            sample_id=str(window_rows[0]["clip_id"]),
            split=self.split,
            clip_text=str(window_rows[0].get("clip_text", "")),
            primitive_texts=[str(row.get("text_chunk", "")) for row in window_rows],
            z_hist=z_hist,
            z_fut=z_fut,
            z_fut_target=z_fut_target,
            window_start=int(window_rows[0].get("window_start", window_rows[0].get("chunk_id", 0))),
        )

    def _yield_with_buffer(
        self,
        item: BFMTextOpSequenceItem,
        rng: random.Random,
        buffer: list[BFMTextOpSequenceItem],
    ) -> Iterator[BFMTextOpSequenceItem]:
        if not self.cfg.shuffle:
            yield item
            return
        buffer.append(item)
        if len(buffer) >= self.cfg.shuffle_buffer_size:
            idx = rng.randrange(len(buffer))
            yield buffer.pop(idx)

    def __iter__(self) -> Iterator[BFMTextOpSequenceItem]:
        worker_info = get_worker_info()
        if worker_info is None:
            assigned_paths = list(self.shard_paths)
            worker_id = 0
        else:
            assigned_paths = self.shard_paths[worker_info.id :: worker_info.num_workers]
            worker_id = worker_info.id

        rng = random.Random(self.cfg.seed + worker_id)
        if self.cfg.shuffle:
            rng.shuffle(assigned_paths)

        buffer: list[BFMTextOpSequenceItem] = []
        current_clip_id: str | None = None
        current_rows: deque[dict[str, Any]] = deque()
        current_idx = 0

        for path in assigned_paths:
            for row in self._iter_rows(path):
                clip_id = str(row["clip_id"])
                if clip_id != current_clip_id:
                    current_clip_id = clip_id
                    current_rows.clear()
                    current_idx = 0
                current_rows.append(row)
                if len(current_rows) < self.cfg.num_primitives:
                    continue
                if len(current_rows) > self.cfg.num_primitives:
                    current_rows.popleft()
                if current_idx % self.cfg.sequence_stride_primitives == 0:
                    item = self._window_to_item(list(current_rows))
                    yield from self._yield_with_buffer(item, rng, buffer)
                current_idx += 1

        if self.cfg.shuffle and buffer:
            rng.shuffle(buffer)
        for item in buffer:
            yield item


def collate_bfm_textop_sequences(items: list[BFMTextOpSequenceItem]) -> dict[str, Any]:
    return {
        "sample_id": [item.sample_id for item in items],
        "split": [item.split for item in items],
        "clip_text": [item.clip_text for item in items],
        "primitive_texts": [item.primitive_texts for item in items],
        "z_hist": torch.stack([item.z_hist for item in items], dim=0),
        "z_fut": torch.stack([item.z_fut for item in items], dim=0),
        "z_fut_target": torch.stack([item.z_fut_target for item in items], dim=0),
        "window_start": torch.tensor([item.window_start for item in items], dtype=torch.long),
    }
