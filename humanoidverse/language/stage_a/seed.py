from __future__ import annotations

import csv
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


SEED_TEXT_FIELDS = (
    "content_natural_desc_1",
    "content_natural_desc_2",
    "content_natural_desc_3",
    "content_natural_desc_4",
    "content_technical_description",
    "content_short_description",
    "content_short_description_2",
)

SEED_G1_COLUMNS = (
    "root_translateX",
    "root_translateY",
    "root_translateZ",
    "root_rotateX",
    "root_rotateY",
    "root_rotateZ",
    "left_hip_pitch_joint_dof",
    "left_hip_roll_joint_dof",
    "left_hip_yaw_joint_dof",
    "left_knee_joint_dof",
    "left_ankle_pitch_joint_dof",
    "left_ankle_roll_joint_dof",
    "right_hip_pitch_joint_dof",
    "right_hip_roll_joint_dof",
    "right_hip_yaw_joint_dof",
    "right_knee_joint_dof",
    "right_ankle_pitch_joint_dof",
    "right_ankle_roll_joint_dof",
    "waist_yaw_joint_dof",
    "waist_roll_joint_dof",
    "waist_pitch_joint_dof",
    "left_shoulder_pitch_joint_dof",
    "left_shoulder_roll_joint_dof",
    "left_shoulder_yaw_joint_dof",
    "left_elbow_joint_dof",
    "left_wrist_roll_joint_dof",
    "left_wrist_pitch_joint_dof",
    "left_wrist_yaw_joint_dof",
    "right_shoulder_pitch_joint_dof",
    "right_shoulder_roll_joint_dof",
    "right_shoulder_yaw_joint_dof",
    "right_elbow_joint_dof",
    "right_wrist_roll_joint_dof",
    "right_wrist_pitch_joint_dof",
    "right_wrist_yaw_joint_dof",
)

SEED_G1_JOINT_COLUMNS = SEED_G1_COLUMNS[6:]
SEED_SOURCE_FPS = 120
VALID_ROOT_EULER_ORDERS = ("xyz", "xzy", "yxz", "yzx", "zxy", "zyx")

DEFAULT_ALLOWED_PACKAGES = (
    "Locomotion",
    "Communication",
    "Dances",
    "Sport",
    "Gaming",
    "Other",
)

DEFAULT_BLOCKED_PACKAGES = (
    "Interactions",
    "Everyday",
)

DEFAULT_BLOCKED_CATEGORIES = (
    "Object Manipulation",
    "Object Interaction",
    "Household",
    "Consuming",
    "Environments",
    "Complex Actions",
)

# Conservative defaults: keep standing/free-space motions first.
DEFAULT_BLOCKED_KEYWORDS = (
    "sit",
    "sitting",
    "seated",
    "chair",
    "sofa",
    "bench",
    "stool",
    "kneel",
    "kneeling",
    "crawl",
    "crawling",
    "lie down",
    "lying",
    "supine",
    "prone",
    "stairs",
    "stair",
    "ladder",
    "climb",
    "climbing",
    "door",
    "knob",
    "book",
    "newspaper",
    "box",
    "object",
    "carry",
    "carrying",
    "pick up",
    "put down",
    "shelf",
    "table",
    "cup",
    "drink",
    "eat",
    "phone",
    "computer",
    "keyboard",
    "violin",
    "guitar",
    "swim",
    "swimming",
)

TRAIN_ALLOWED_PACKAGES = (
    "Locomotion",
    "Communication",
    "Dances",
    "Sport",
    "Gaming",
    "Everyday",
    "Other",
)

TRAIN_BLOCKED_PACKAGES = (
    "Interactions",
)

TRAIN_BLOCKED_CATEGORIES = (
    "Object Interaction",
    "Environments",
    "Consuming",
)

TRAIN_BLOCKED_BODY_POSITION_KEYWORDS = (
    "sit",
    "sitting",
    "kneel",
    "kneeling",
    "crawl",
    "crawling",
    "on all fours",
    "lying",
    "prone",
    "supine",
    "ladder",
    "handstanding",
)

TRAIN_BLOCKED_TEXT_KEYWORDS = (
    "stairs",
    "stair",
    "ladder",
    "climb",
    "climbing",
    "chair",
    "sofa",
    "bench",
    "stool",
    "sit down",
    "sits down",
    "lie down",
    "lies down",
    "door handle",
    "open door",
    "close door",
    "opening door",
    "closing door",
    "open fridge",
    "close fridge",
    "refrigerator door",
    "wall-supported",
    "wall supported",
    "handstand",
)

TRAIN_BLOCKED_PROP_KEYWORDS = (
    "chair",
    "stool",
    "bench",
    "sofa",
    "ladder",
    "stairs",
    "stair",
    "wall",
    "door",
    "fridge",
    "shelves",
    "scaffolding",
    "lever",
    "valve",
    "obstacle",
    "crutch",
    "crutches",
)


@dataclass(slots=True)
class StageAFilterConfig:
    include_clips: bool = True
    include_events: bool = True
    include_mirrors: bool = True
    require_empty_props: bool = True
    min_clip_duration_s: float = 1.0
    min_event_duration_s: float = 0.6
    max_duration_s: float = 30.0
    allowed_packages: tuple[str, ...] = DEFAULT_ALLOWED_PACKAGES
    blocked_packages: tuple[str, ...] = DEFAULT_BLOCKED_PACKAGES
    blocked_categories: tuple[str, ...] = DEFAULT_BLOCKED_CATEGORIES
    blocked_keywords: tuple[str, ...] = DEFAULT_BLOCKED_KEYWORDS


@dataclass(slots=True)
class SeedTrainFilterConfig:
    include_mirrors: bool = True
    min_clip_duration_s: float = 1.0
    max_duration_s: float = 120.0
    allowed_packages: tuple[str, ...] = TRAIN_ALLOWED_PACKAGES
    blocked_packages: tuple[str, ...] = TRAIN_BLOCKED_PACKAGES
    blocked_categories: tuple[str, ...] = TRAIN_BLOCKED_CATEGORIES
    blocked_body_position_keywords: tuple[str, ...] = TRAIN_BLOCKED_BODY_POSITION_KEYWORDS
    blocked_text_keywords: tuple[str, ...] = TRAIN_BLOCKED_TEXT_KEYWORDS
    blocked_prop_keywords: tuple[str, ...] = TRAIN_BLOCKED_PROP_KEYWORDS


@dataclass(slots=True)
class SeedSample:
    sample_id: str
    sample_type: str
    filename: str
    motion_csv_path: str
    package: str
    category: str
    start_time_s: float
    end_time_s: float
    start_frame: int
    end_frame: int
    duration_s: float
    split: str
    primary_text: str
    texts: list[str]
    parent_clip_text: str | None = None
    text_fields: dict[str, str] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    filter_notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def dedupe_texts(texts: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for text in texts:
        normalized = re.sub(r"\s+", " ", text or "").strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        deduped.append(normalized)
        seen.add(key)
    return deduped


def stable_split_from_filename(filename: str, val_ratio: float = 0.1) -> str:
    bucket = sum(filename.encode("utf-8")) % 1000
    return "val" if bucket < int(val_ratio * 1000) else "train"


def load_seed_temporal_labels(path: Path) -> dict[str, list[dict[str, Any]]]:
    labels: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            labels[entry["filename"]] = entry.get("events", [])
    return labels


def load_seed_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def resolve_g1_csv_path(dataset_root: Path, row: dict[str, str]) -> Path:
    candidate = (dataset_root / row["move_g1_path"]).resolve() if row.get("move_g1_path") else None
    if candidate is not None and candidate.exists():
        return candidate
    fallback = dataset_root / "g1" / "csv" / str(row["take_date"]) / f"{row['filename']}.csv"
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(f"Could not locate G1 CSV for {row['filename']}")


def collect_clip_texts(row: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    text_fields = {field: (row.get(field) or "").strip() for field in SEED_TEXT_FIELDS}
    texts = dedupe_texts(list(text_fields.values()))
    return texts, text_fields


def _contains_blocked_keyword(text: str, blocked_keywords: tuple[str, ...]) -> str | None:
    lowered = f" {text.lower()} "
    for keyword in blocked_keywords:
        if keyword.lower() in lowered:
            return keyword
    return None


def _body_position_indicates_nonstanding(row: dict[str, str]) -> str | None:
    body_position = (row.get("content_body_position") or "").strip().lower()
    if not body_position:
        return None
    for keyword in ("sit", "seat", "chair", "kneel", "crawl", "lie", "prone", "supine"):
        if keyword in body_position:
            return body_position
    return None


def _has_meaningful_props(row: dict[str, str]) -> bool:
    value = (row.get("content_props") or "").strip().lower()
    return value not in {"", "0", "none", "null", "nan"}


def filter_row(
    row: dict[str, str],
    clip_texts: list[str],
    cfg: StageAFilterConfig,
) -> list[str]:
    reasons: list[str] = []
    package = (row.get("package") or "").strip()
    category = (row.get("category") or "").strip()
    if cfg.allowed_packages and package not in cfg.allowed_packages:
        reasons.append(f"package_not_allowed:{package}")
    if package in cfg.blocked_packages:
        reasons.append(f"package_blocked:{package}")
    if category in cfg.blocked_categories:
        reasons.append(f"category_blocked:{category}")
    if not cfg.include_mirrors and (row.get("is_mirror") or "").strip().lower() in {"1", "true"}:
        reasons.append("mirror_blocked")
    if cfg.require_empty_props and _has_meaningful_props(row):
        reasons.append("has_props")
    body_position = _body_position_indicates_nonstanding(row)
    if body_position is not None:
        reasons.append(f"body_position_blocked:{body_position}")
    joined_text = " ".join(clip_texts + [row.get("content_name", "") or ""])
    blocked_keyword = _contains_blocked_keyword(joined_text, cfg.blocked_keywords)
    if blocked_keyword is not None:
        reasons.append(f"keyword_blocked:{blocked_keyword}")
    return reasons


def filter_event_description(description: str, cfg: StageAFilterConfig) -> list[str]:
    reasons: list[str] = []
    blocked_keyword = _contains_blocked_keyword(description, cfg.blocked_keywords)
    if blocked_keyword is not None:
        reasons.append(f"keyword_blocked:{blocked_keyword}")
    return reasons


def build_stage_a_samples(
    dataset_root: Path,
    metadata_csv: Path,
    temporal_jsonl: Path,
    cfg: StageAFilterConfig,
) -> tuple[list[SeedSample], dict[str, Any]]:
    rows = load_seed_metadata(metadata_csv)
    temporal_labels = load_seed_temporal_labels(temporal_jsonl)

    samples: list[SeedSample] = []
    stats = {
        "rows_total": len(rows),
        "clips_kept": 0,
        "clips_filtered": 0,
        "events_kept": 0,
        "events_filtered": 0,
        "filter_reasons": {},
    }

    def count_reasons(reasons: list[str]) -> None:
        for reason in reasons:
            stats["filter_reasons"][reason] = stats["filter_reasons"].get(reason, 0) + 1

    for row in rows:
        texts, text_fields = collect_clip_texts(row)
        if not texts:
            stats["clips_filtered"] += 1
            count_reasons(["missing_clip_text"])
            continue

        try:
            motion_csv_path = resolve_g1_csv_path(dataset_root, row)
        except FileNotFoundError:
            stats["clips_filtered"] += 1
            count_reasons(["missing_g1_csv"])
            continue

        duration_s = float(row["move_duration_frames"]) / float(SEED_SOURCE_FPS)
        if duration_s < cfg.min_clip_duration_s or duration_s > cfg.max_duration_s:
            stats["clips_filtered"] += 1
            count_reasons(["clip_duration_out_of_range"])
            continue

        clip_reasons = filter_row(row, texts, cfg)
        if clip_reasons:
            stats["clips_filtered"] += 1
            count_reasons(clip_reasons)
            continue

        split = stable_split_from_filename(row["filename"])
        clip_events = temporal_labels.get(row["filename"], [])
        primary_text = (
            text_fields.get("content_natural_desc_4")
            or text_fields.get("content_natural_desc_1")
            or text_fields.get("content_short_description")
            or texts[0]
        )

        if cfg.include_clips:
            clip_sample = SeedSample(
                sample_id=f"clip::{row['filename']}",
                sample_type="clip",
                filename=row["filename"],
                motion_csv_path=str(motion_csv_path),
                package=row["package"],
                category=row["category"],
                start_time_s=0.0,
                end_time_s=duration_s,
                start_frame=0,
                end_frame=int(row["move_duration_frames"]),
                duration_s=duration_s,
                split=split,
                primary_text=primary_text,
                texts=texts,
                parent_clip_text=primary_text,
                text_fields=text_fields,
                events=clip_events,
            )
            samples.append(clip_sample)
            stats["clips_kept"] += 1

        if not cfg.include_events:
            continue

        for event_idx, event in enumerate(clip_events):
            start_time_s = float(event["start_time"])
            end_time_s = float(event["end_time"])
            event_duration_s = end_time_s - start_time_s
            if event_duration_s < cfg.min_event_duration_s or event_duration_s > cfg.max_duration_s:
                stats["events_filtered"] += 1
                count_reasons(["event_duration_out_of_range"])
                continue

            description = (event.get("description") or "").strip()
            if not description:
                stats["events_filtered"] += 1
                count_reasons(["missing_event_text"])
                continue

            event_reasons = filter_event_description(description, cfg)
            if event_reasons:
                stats["events_filtered"] += 1
                count_reasons(event_reasons)
                continue

            start_frame = int(math.floor(start_time_s * SEED_SOURCE_FPS))
            end_frame = int(math.ceil(end_time_s * SEED_SOURCE_FPS))
            if end_frame - start_frame < 2:
                stats["events_filtered"] += 1
                count_reasons(["event_too_short_in_frames"])
                continue

            event_sample = SeedSample(
                sample_id=f"event::{row['filename']}::{event_idx:03d}",
                sample_type="event",
                filename=row["filename"],
                motion_csv_path=str(motion_csv_path),
                package=row["package"],
                category=row["category"],
                start_time_s=start_time_s,
                end_time_s=end_time_s,
                start_frame=start_frame,
                end_frame=end_frame,
                duration_s=event_duration_s,
                split=split,
                primary_text=description,
                texts=[description],
                parent_clip_text=primary_text,
                text_fields={"event_description": description},
                events=[event],
            )
            samples.append(event_sample)
            stats["events_kept"] += 1

    return samples, stats


def _contains_any_keyword(text: str, keywords: tuple[str, ...]) -> str | None:
    lowered = text.lower()
    for keyword in keywords:
        if keyword.lower() in lowered:
            return keyword
    return None


def filter_row_for_training(
    row: dict[str, str],
    clip_texts: list[str],
    cfg: SeedTrainFilterConfig,
) -> list[str]:
    reasons: list[str] = []
    package = (row.get("package") or "").strip()
    category = (row.get("category") or "").strip()
    if cfg.allowed_packages and package not in cfg.allowed_packages:
        reasons.append(f"package_not_allowed:{package}")
    if package in cfg.blocked_packages:
        reasons.append(f"package_blocked:{package}")
    if category in cfg.blocked_categories:
        reasons.append(f"category_blocked:{category}")
    if not cfg.include_mirrors and (row.get("is_mirror") or "").strip().lower() in {"1", "true"}:
        reasons.append("mirror_blocked")

    body_position = (row.get("content_body_position") or "").strip().lower()
    blocked_body_keyword = _contains_any_keyword(body_position, cfg.blocked_body_position_keywords)
    if blocked_body_keyword is not None:
        reasons.append(f"body_position_blocked:{blocked_body_keyword}")

    joined_text = " ".join(
        clip_texts
        + [
            row.get("content_name", "") or "",
            row.get("content_short_description", "") or "",
            row.get("content_technical_description", "") or "",
        ]
    )
    blocked_text_keyword = _contains_any_keyword(joined_text, cfg.blocked_text_keywords)
    if blocked_text_keyword is not None:
        reasons.append(f"text_blocked:{blocked_text_keyword}")

    props = (row.get("content_props") or "").strip().lower()
    if props not in {"", "0", "none", "null", "nan"}:
        blocked_prop_keyword = _contains_any_keyword(props, cfg.blocked_prop_keywords)
        if blocked_prop_keyword is not None:
            reasons.append(f"props_blocked:{blocked_prop_keyword}")

    return reasons


def build_seed_training_records(
    dataset_root: Path,
    metadata_csv: Path,
    cfg: SeedTrainFilterConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_seed_metadata(metadata_csv)
    records: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "rows_total": len(rows),
        "clips_kept": 0,
        "clips_filtered": 0,
        "filter_reasons": {},
        "kept_by_package": {},
        "kept_by_category": {},
        "kept_duration_seconds": {
            "min": None,
            "max": None,
            "mean": 0.0,
        },
    }
    kept_durations: list[float] = []

    def count_reasons(reasons: list[str]) -> None:
        for reason in reasons:
            stats["filter_reasons"][reason] = stats["filter_reasons"].get(reason, 0) + 1

    for row in rows:
        texts, text_fields = collect_clip_texts(row)
        if not texts:
            stats["clips_filtered"] += 1
            count_reasons(["missing_clip_text"])
            continue

        try:
            motion_csv_path = resolve_g1_csv_path(dataset_root, row)
        except FileNotFoundError:
            stats["clips_filtered"] += 1
            count_reasons(["missing_g1_csv"])
            continue

        duration_s = float(row["move_duration_frames"]) / float(SEED_SOURCE_FPS)
        if duration_s < cfg.min_clip_duration_s or duration_s > cfg.max_duration_s:
            stats["clips_filtered"] += 1
            count_reasons(["clip_duration_out_of_range"])
            continue

        reasons = filter_row_for_training(row, texts, cfg)
        if reasons:
            stats["clips_filtered"] += 1
            count_reasons(reasons)
            continue

        split = stable_split_from_filename(row["filename"])
        primary_text = (
            text_fields.get("content_natural_desc_4")
            or text_fields.get("content_natural_desc_1")
            or text_fields.get("content_short_description")
            or texts[0]
        )
        record = {
            "sample_id": f"clip::{row['filename']}",
            "filename": row["filename"],
            "motion_csv_path": str(motion_csv_path),
            "package": row.get("package", ""),
            "category": row.get("category", ""),
            "content_props": row.get("content_props", ""),
            "content_body_position": row.get("content_body_position", ""),
            "start_frame": 0,
            "end_frame": int(row["move_duration_frames"]),
            "duration_s": duration_s,
            "split": split,
            "primary_text": primary_text,
            "texts": texts,
            "text_fields": text_fields,
        }
        records.append(record)
        kept_durations.append(duration_s)
        stats["clips_kept"] += 1
        stats["kept_by_package"][record["package"]] = stats["kept_by_package"].get(record["package"], 0) + 1
        stats["kept_by_category"][record["category"]] = stats["kept_by_category"].get(record["category"], 0) + 1

    if kept_durations:
        kept_np = np.asarray(kept_durations, dtype=np.float32)
        stats["kept_duration_seconds"] = {
            "min": float(kept_np.min()),
            "max": float(kept_np.max()),
            "mean": float(kept_np.mean()),
        }

    return records, stats


def build_fixed_length_clip_records(
    records: list[dict[str, Any]],
    clip_length_s: float = 10.0,
    clip_step_s: float | None = None,
    source_fps: int = SEED_SOURCE_FPS,
    target_fps: int = 30,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if clip_length_s <= 0:
        raise ValueError("clip_length_s must be > 0")
    if clip_step_s is not None and clip_step_s <= 0:
        raise ValueError("clip_step_s must be > 0")

    target_clip_frames = int(round(clip_length_s * target_fps))
    source_clip_frames = int(round(clip_length_s * source_fps))
    source_step_frames = source_clip_frames if clip_step_s is None else int(round(clip_step_s * source_fps))
    if source_step_frames <= 0:
        raise ValueError("Computed non-positive source_step_frames")

    clipped_records: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "source_records_total": len(records),
        "records_with_at_least_one_window": 0,
        "records_dropped_too_short": 0,
        "windows_total": 0,
        "target_fps": target_fps,
        "target_clip_frames": target_clip_frames,
        "clip_length_s": clip_length_s,
        "clip_step_s": clip_length_s if clip_step_s is None else clip_step_s,
        "windows_by_package": {},
        "windows_by_category": {},
    }

    for record in records:
        total_source_frames = int(record["end_frame"]) - int(record["start_frame"])
        if total_source_frames < source_clip_frames:
            stats["records_dropped_too_short"] += 1
            continue

        clip_count = 0
        max_offset = total_source_frames - source_clip_frames
        offset = 0
        while offset <= max_offset:
            clip_index = clip_count
            clip_record = dict(record)
            clip_record["parent_sample_id"] = record["sample_id"]
            clip_record["sample_id"] = f"{record['filename']}_clip{clip_index}"
            clip_record["clip_index"] = clip_index
            clip_record["start_frame"] = int(record["start_frame"]) + offset
            clip_record["end_frame"] = clip_record["start_frame"] + source_clip_frames
            clip_record["duration_s"] = float(clip_length_s)
            clip_record["target_fps"] = target_fps
            clipped_records.append(clip_record)
            stats["windows_total"] += 1
            stats["windows_by_package"][clip_record["package"]] = (
                stats["windows_by_package"].get(clip_record["package"], 0) + 1
            )
            stats["windows_by_category"][clip_record["category"]] = (
                stats["windows_by_category"].get(clip_record["category"], 0) + 1
            )
            clip_count += 1
            offset += source_step_frames

        if clip_count > 0:
            stats["records_with_at_least_one_window"] += 1

    return clipped_records, stats


def write_manifest(samples: list[SeedSample], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(sample.to_json())
            f.write("\n")


def read_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def write_jsonl_records(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def parse_joint_axes(mjcf_path: Path) -> np.ndarray:
    root = ET.parse(mjcf_path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Could not find worldbody in {mjcf_path}")
    axes: list[np.ndarray] = []
    for joint in worldbody.findall(".//joint"):
        if joint.attrib.get("type") == "free":
            continue
        axis_str = joint.attrib.get("axis")
        if axis_str is None:
            raise ValueError(f"Joint {joint.attrib.get('name')} has no axis in {mjcf_path}")
        axis = np.fromstring(axis_str, sep=" ", dtype=np.float32)
        if axis.shape != (3,):
            raise ValueError(f"Joint {joint.attrib.get('name')} axis has invalid shape {axis.shape}")
        axes.append(axis)
    return np.stack(axes, axis=0)


def euler_degrees_to_quaternion_xyzw(rx_deg: float, ry_deg: float, rz_deg: float, order: str = "xyz") -> np.ndarray:
    order = order.lower()
    if order not in VALID_ROOT_EULER_ORDERS:
        raise ValueError(f"Unsupported Euler order {order!r}. Expected one of {VALID_ROOT_EULER_ORDERS}.")
    quat = Rotation.from_euler(order, [rx_deg, ry_deg, rz_deg], degrees=True).as_quat().astype(np.float32)
    quat /= np.linalg.norm(quat) + 1e-8
    return quat


def euler_xyz_degrees_to_quaternion_xyzw(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    return euler_degrees_to_quaternion_xyzw(rx_deg, ry_deg, rz_deg, order="xyz")


def quaternion_xyzw_to_rotvec(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = quat_xyzw.astype(np.float64, copy=False)
    quat = quat / (np.linalg.norm(quat) + 1e-12)
    xyz = quat[:3]
    w = float(np.clip(quat[3], -1.0, 1.0))
    xyz_norm = float(np.linalg.norm(xyz))
    if xyz_norm < 1e-8:
        return (2.0 * xyz).astype(np.float32)
    angle = 2.0 * math.atan2(xyz_norm, w)
    axis = xyz / xyz_norm
    return (axis * angle).astype(np.float32)


def load_seed_g1_rows(
    csv_path: Path,
    start_frame: int,
    end_frame: int,
    stride: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx < start_frame:
                continue
            if idx >= end_frame:
                break
            if (idx - start_frame) % stride != 0:
                continue
            rows.append(row)
    return rows


def _compute_resampled_frame_count(num_source_frames: int, source_fps: int, target_fps: int) -> int:
    if num_source_frames <= 0:
        raise ValueError(f"num_source_frames must be > 0, got {num_source_frames}")
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError(f"source_fps and target_fps must be > 0, got {source_fps}, {target_fps}")
    return max(1, int(round(num_source_frames * float(target_fps) / float(source_fps))))


def _resample_linear(values: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    flat_values = values.reshape(values.shape[0], -1).astype(np.float64, copy=False)
    resampled = np.empty((target_times.shape[0], flat_values.shape[1]), dtype=np.float32)
    for col_idx in range(flat_values.shape[1]):
        resampled[:, col_idx] = np.interp(target_times, source_times, flat_values[:, col_idx]).astype(np.float32)
    return resampled.reshape((target_times.shape[0],) + values.shape[1:])


def _resample_quaternions_xyzw(quaternions_xyzw: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    rotations = Rotation.from_quat(quaternions_xyzw.astype(np.float64, copy=False))
    slerp = Slerp(source_times, rotations)
    resampled = slerp(target_times).as_quat().astype(np.float32)
    resampled /= np.linalg.norm(resampled, axis=-1, keepdims=True) + 1e-8
    return resampled


def _resample_binary_nearest(values: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    right_idx = np.searchsorted(source_times, target_times, side="left")
    right_idx = np.clip(right_idx, 0, len(source_times) - 1)
    left_idx = np.clip(right_idx - 1, 0, len(source_times) - 1)
    choose_right = np.abs(source_times[right_idx] - target_times) < np.abs(target_times - source_times[left_idx])
    nearest_idx = np.where(choose_right, right_idx, left_idx)
    return values[nearest_idx]


def resolve_seed_g1_contact_csv_path(motion_csv_path: Path, contact_dataset_root: Path) -> Path:
    contact_csv_path = contact_dataset_root / motion_csv_path.parent.name / motion_csv_path.name
    if not contact_csv_path.exists():
        raise FileNotFoundError(f"Could not locate contact-annotated G1 CSV at {contact_csv_path}")
    return contact_csv_path


def load_seed_g1_contact_binary(
    csv_path: Path,
    start_frame: int,
    end_frame: int,
    stride: int,
) -> np.ndarray:
    contacts: list[np.ndarray] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx < start_frame:
                continue
            if idx >= end_frame:
                break
            if (idx - start_frame) % stride != 0:
                continue
            if len(row) < 2:
                raise ValueError(f"Expected at least 2 columns in contact CSV {csv_path}, got row with {len(row)} columns")
            # g1_seed files store appended left/right binary contacts as 1=contact, 0=non-contact.
            contacts.append(np.asarray([float(row[-2]), float(row[-1])], dtype=np.float32))
    if not contacts:
        return np.zeros((0, 2), dtype=np.float32)
    return (np.stack(contacts, axis=0) > 0.5).astype(np.float32)


def seed_rows_to_motion_entry(
    rows: list[dict[str, str]],
    joint_axes: np.ndarray,
    target_fps: int,
    source_fps: int = SEED_SOURCE_FPS,
    root_euler_order: str = "xyz",
    foot_contact_binary: np.ndarray | None = None,
) -> dict[str, np.ndarray | int]:
    if not rows:
        raise ValueError("Received empty SEED row list for motion conversion")
    if joint_axes.shape != (len(SEED_G1_JOINT_COLUMNS), 3):
        raise ValueError(f"Expected joint_axes {(len(SEED_G1_JOINT_COLUMNS), 3)}, got {joint_axes.shape}")

    root_positions: list[np.ndarray] = []
    root_quats: list[np.ndarray] = []
    dof_values: list[np.ndarray] = []

    for row in rows:
        root_positions.append(
            np.asarray(
                [
                    float(row["root_translateX"]) * 0.01,
                    float(row["root_translateY"]) * 0.01,
                    float(row["root_translateZ"]) * 0.01,
                ],
                dtype=np.float32,
            )
        )
        root_quats.append(
            euler_degrees_to_quaternion_xyzw(
                float(row["root_rotateX"]),
                float(row["root_rotateY"]),
                float(row["root_rotateZ"]),
                order=root_euler_order,
            )
        )
        dof_values.append(
            np.asarray([math.radians(float(row[name])) for name in SEED_G1_JOINT_COLUMNS], dtype=np.float32)
        )

    root_positions_np = np.stack(root_positions, axis=0)
    root_quats_np = np.stack(root_quats, axis=0)
    dof_values_np = np.stack(dof_values, axis=0)

    if source_fps != target_fps:
        num_target_frames = _compute_resampled_frame_count(
            num_source_frames=root_positions_np.shape[0],
            source_fps=source_fps,
            target_fps=target_fps,
        )
        source_times = np.arange(root_positions_np.shape[0], dtype=np.float64) / float(source_fps)
        target_times = np.arange(num_target_frames, dtype=np.float64) / float(target_fps)
        root_positions_np = _resample_linear(root_positions_np, source_times, target_times).astype(np.float32)
        dof_values_np = _resample_linear(dof_values_np, source_times, target_times).astype(np.float32)
        root_quats_np = _resample_quaternions_xyzw(root_quats_np, source_times, target_times).astype(np.float32)
        if foot_contact_binary is not None:
            foot_contact_binary = _resample_binary_nearest(
                foot_contact_binary.astype(np.float32, copy=False),
                source_times,
                target_times,
            ).astype(np.float32)

    root_rotvec_np = np.stack([quaternion_xyzw_to_rotvec(q) for q in root_quats_np], axis=0)
    local_joint_rotvec = dof_values_np[..., None] * joint_axes[None, ...]
    pose_aa = np.concatenate([root_rotvec_np[:, None, :], local_joint_rotvec], axis=1).astype(np.float32)
    motion_entry = {
        "root_trans_offset": root_positions_np.astype(np.float32),
        "pose_aa": pose_aa.astype(np.float32),
        "dof": dof_values_np.astype(np.float32),
        "root_rot": root_quats_np.astype(np.float32),
        "fps": int(target_fps),
    }
    if foot_contact_binary is not None:
        if foot_contact_binary.shape[0] != root_positions_np.shape[0]:
            raise ValueError(
                f"foot_contact_binary length {foot_contact_binary.shape[0]} does not match motion length {root_positions_np.shape[0]}"
            )
        motion_entry["foot_contact_binary"] = foot_contact_binary.astype(np.float32)
    return motion_entry


def manifest_records_to_motion_dict(
    records: list[dict[str, Any]],
    mjcf_path: Path,
    target_fps: int = 30,
    root_euler_order: str = "xyz",
    include_foot_contact_binary: bool = False,
    contact_dataset_root: Path | None = None,
) -> dict[str, dict[str, np.ndarray | int]]:
    if target_fps <= 0:
        raise ValueError(f"target_fps must be > 0, got {target_fps}")
    exact_stride = 1
    source_rows_fps = SEED_SOURCE_FPS
    if SEED_SOURCE_FPS % int(target_fps) == 0:
        exact_stride = SEED_SOURCE_FPS // int(target_fps)
        source_rows_fps = int(target_fps)
    joint_axes = parse_joint_axes(mjcf_path)
    motion_dict: dict[str, dict[str, np.ndarray | int]] = {}

    for record in records:
        motion_csv_path = Path(record["motion_csv_path"])
        rows = load_seed_g1_rows(
            csv_path=motion_csv_path,
            start_frame=int(record["start_frame"]),
            end_frame=int(record["end_frame"]),
            stride=exact_stride,
        )
        if len(rows) < 2:
            continue
        foot_contact_binary = None
        if include_foot_contact_binary:
            if contact_dataset_root is None:
                raise ValueError("contact_dataset_root must be provided when include_foot_contact_binary=True")
            contact_csv_path = resolve_seed_g1_contact_csv_path(motion_csv_path, contact_dataset_root)
            foot_contact_binary = load_seed_g1_contact_binary(
                csv_path=contact_csv_path,
                start_frame=int(record["start_frame"]),
                end_frame=int(record["end_frame"]),
                stride=exact_stride,
            )
        motion_entry = seed_rows_to_motion_entry(
            rows,
            joint_axes=joint_axes,
            target_fps=target_fps,
            source_fps=source_rows_fps,
            root_euler_order=root_euler_order,
            foot_contact_binary=foot_contact_binary,
        )
        motion_entry["motion_name"] = str(record["sample_id"])
        motion_dict[record["sample_id"]] = motion_entry
    return motion_dict
