from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import torch
from torch.utils.data import Dataset


ClipTextSource = Literal[
    "all",
    "primary_text",
    "natural_123",
    "content_natural_desc_1",
    "content_natural_desc_2",
    "content_natural_desc_3",
    "content_natural_desc_4",
    "content_technical_description",
    "content_short_description",
    "content_short_description_2",
]

@dataclass(slots=True)
class StageAItem:
    sample_id: str
    text: str
    z_clip: torch.Tensor
    z_chunks: torch.Tensor
    sample_type: str
    split: str


@dataclass(slots=True)
class StageASample:
    sample_id: str
    texts: list[str]
    z_clip: torch.Tensor
    z_chunks: torch.Tensor
    sample_type: str
    split: str


def load_manifest_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_latent_store(paths: list[Path]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in paths:
        store = joblib.load(path)
        for key, value in store.items():
            if key == "__meta__":
                continue
            merged[key] = value
    return merged


def resolve_record_texts(record: dict[str, Any], clip_text_source: ClipTextSource) -> list[str]:
    if record.get("sample_type") != "clip" or clip_text_source == "all":
        return [text for text in (record.get("texts") or []) if text]

    if clip_text_source == "primary_text":
        text = record.get("primary_text")
        return [text] if text else []

    if clip_text_source == "natural_123":
        text_fields = record.get("text_fields") or {}
        texts: list[str] = []
        for field_name in (
            "content_natural_desc_1",
            "content_natural_desc_2",
            "content_natural_desc_3",
        ):
            text = text_fields.get(field_name)
            if text and text not in texts:
                texts.append(text)
        if texts:
            return texts
        fallback = record.get("primary_text")
        if fallback:
            return [fallback]
        return [text for text in (record.get("texts") or []) if text]

    text_fields = record.get("text_fields") or {}
    text = text_fields.get(clip_text_source)
    if text:
        return [text]

    fallback = record.get("primary_text")
    if fallback:
        return [fallback]
    return [text for text in (record.get("texts") or []) if text]


class StageALatentDataset(Dataset[StageAItem]):
    def __init__(
        self,
        manifest_records: list[dict[str, Any]],
        latent_store: dict[str, Any],
        split: str,
        text_mode: Literal["expand_all", "sample_one"] = "expand_all",
        clip_text_source: ClipTextSource = "all",
        seed: int = 42,
    ) -> None:
        self.split = split
        self.text_mode = text_mode
        self.clip_text_source = clip_text_source
        self.seed = seed
        self.epoch = 0
        self.samples: list[StageASample] = []
        self.index_map: list[tuple[int, int]] = []
        for record in manifest_records:
            if record["split"] != split:
                continue
            sample_id = record["sample_id"]
            latent = latent_store.get(sample_id)
            if latent is None:
                continue
            texts = resolve_record_texts(record, clip_text_source)
            if not texts:
                continue
            sample = StageASample(
                sample_id=sample_id,
                texts=texts,
                z_clip=torch.from_numpy(latent["z_clip"]).float(),
                z_chunks=torch.from_numpy(latent["z_chunks"]).float(),
                sample_type=record["sample_type"],
                split=record["split"],
            )
            sample_idx = len(self.samples)
            self.samples.append(sample)
            if self.text_mode == "expand_all":
                self.index_map.extend((sample_idx, text_idx) for text_idx in range(len(texts)))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _sample_text_index(self, sample: StageASample) -> int:
        if len(sample.texts) == 1:
            return 0
        digest = hashlib.sha1(
            f"{self.seed}:{self.epoch}:{self.split}:{sample.sample_id}".encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], "big") % len(sample.texts)

    def _make_item(self, sample: StageASample, text_idx: int) -> StageAItem:
        return StageAItem(
            sample_id=sample.sample_id,
            text=sample.texts[text_idx],
            z_clip=sample.z_clip,
            z_chunks=sample.z_chunks,
            sample_type=sample.sample_type,
            split=sample.split,
        )

    def __len__(self) -> int:
        if self.text_mode == "sample_one":
            return len(self.samples)
        return len(self.index_map)

    def __getitem__(self, idx: int) -> StageAItem:
        if self.text_mode == "sample_one":
            sample = self.samples[idx]
            text_idx = self._sample_text_index(sample)
            return self._make_item(sample, text_idx)
        sample_idx, text_idx = self.index_map[idx]
        return self._make_item(self.samples[sample_idx], text_idx)


def collate_stage_a(items: list[StageAItem]) -> dict[str, Any]:
    return {
        "sample_id": [item.sample_id for item in items],
        "text": [item.text for item in items],
        "z_clip": torch.stack([item.z_clip for item in items], dim=0),
        "z_chunks": torch.stack([item.z_chunks for item in items], dim=0),
        "sample_type": [item.sample_type for item in items],
        "split": [item.split for item in items],
    }
