from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import tyro
from scipy.spatial.transform import Rotation

from humanoidverse.language.stage_a.seed import read_manifest, write_jsonl_records
from humanoidverse.scripts.convert_pico_to_motion import parse_joint_axes


SEED_CATEGORY_ORDER = [
    "Basic Locomotion Neutral",
    "Baseline",
    "Basic Locomotion Styles",
    "Dancing",
    "Gestures",
    "Other",
    "Complex Actions",
    "Communication",
    "Sports",
    "Advanced Locomotion",
    "Household",
    "Looking and Pointing",
    "Magic",
]


CATEGORY_TO_PACKAGE = {
    "Basic Locomotion Neutral": "Locomotion",
    "Baseline": "Locomotion",
    "Basic Locomotion Styles": "Locomotion",
    "Complex Actions": "Locomotion",
    "Advanced Locomotion": "Locomotion",
    "Dancing": "Dances",
    "Gestures": "Communication",
    "Communication": "Communication",
    "Looking and Pointing": "Communication",
    "Magic": "Gaming",
    "Sports": "Sport",
    "Household": "Everyday",
    "Other": "Other",
}


@dataclass
class Args:
    input_dir: Path = Path("artifacts/textop_babel_h2_f8_50fps")
    reference_manifest: Path = Path("artifacts/seed_train/tiers/seed_train_10s_2000.jsonl")
    output_manifest: Path = Path("artifacts/babel_train/babel_train_seed2k_matched_50fps.jsonl")
    output_motionlib: Path = Path("humanoidverse/data/babel_train_seed2k_matched_50fps.pkl")
    output_report: Path = Path("artifacts/babel_train/babel_train_seed2k_matched_50fps.report.json")
    mjcf_path: Path = Path("humanoidverse/data/robots/g1/g1_29dof.xml")
    seed: int = 42


def _joined_texts(entry: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for ann in entry.get("frame_ann") or []:
        text = str(ann[2]).strip()
        if text and text not in texts:
            texts.append(text)
    return texts


def _joined_text_lower(entry: dict[str, Any]) -> str:
    return " | ".join(text.lower() for text in _joined_texts(entry))


def _act_cats(entry: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for ann in entry.get("frame_ann") or []:
        for cat in ann[3]:
            out.add(str(cat).strip().lower())
    return out


def _motion_stats(entry: dict[str, Any]) -> dict[str, float]:
    motion = entry["motion"]
    root_trans = np.asarray(motion["root_trans_offset"], dtype=np.float32)
    dof = np.asarray(motion["dof"], dtype=np.float32)
    fps = float(motion.get("fps", 50))
    xy = root_trans[:, :2]
    disp = float(np.linalg.norm(xy[-1] - xy[0])) if len(xy) else 0.0
    z = root_trans[:, 2]
    height_span = float(z.max() - z.min()) if len(z) else 0.0
    dof_abs_mean = float(np.mean(np.abs(dof))) if dof.size else 0.0
    if len(root_trans) > 1:
        vel = np.diff(xy, axis=0) * fps
        root_speed_mean = float(np.linalg.norm(vel, axis=1).mean())
    else:
        root_speed_mean = 0.0
    return {
        "disp": disp,
        "height_span": height_span,
        "dof_abs_mean": dof_abs_mean,
        "root_speed_mean": root_speed_mean,
    }


def _classify_category(entry: dict[str, Any]) -> str:
    text = _joined_text_lower(entry)
    act_cats = _act_cats(entry)

    def text_has(*terms: str) -> bool:
        return any(term in text for term in terms)

    def cat_has(*terms: str) -> bool:
        return any(term in act_cats for term in terms)

    if text_has("dance", "waltz", "salsa", "cha cha", "ballroom", "ballet") or cat_has("dance"):
        return "Dancing"

    if (
        cat_has("play sport", "sports move", "action with ball", "martial art", "exercise/training")
        or text_has("basketball", "soccer", "tennis", "boxing", "martial", "kickbox")
    ):
        return "Sports"

    if cat_has("point") or text_has("point"):
        return "Looking and Pointing"

    if (
        cat_has("gesture", "greet", "wave", "clap", "communicate (vocalise)")
        or text_has("wave", "clap", "salute", "gesture", "greet", "applaud")
    ):
        return "Communication"

    if (
        cat_has(
            "interact with/use object",
            "take/pick something up",
            "place something",
            "clean something",
            "telephone call",
            "drink",
            "press something",
            "knock",
            "move something",
            "lift something",
        )
        or text_has("phone", "telephone", "knock", "clean", "wash", "cook", "drink", "lift", "carry", "pick", "place", "open", "close", "door")
    ):
        return "Household"

    locomotion_tags = {
        "walk",
        "run",
        "jog",
        "forward movement",
        "backwards movement",
        "sideways movement",
        "turn",
        "step",
    }
    is_locomotion = cat_has(*locomotion_tags) or text_has("walk", "run", "jog", "backward", "backwards", "side step", "turn", "step")
    if is_locomotion:
        is_advanced = (
            cat_has("balance", "stumble", "evade", "hop", "jump", "move up/down incline", "sneak", "limp", "skip", "slide")
            or text_has("limp", "sneak", "jump over", "hop", "skip", "stumble")
        )
        is_style = (
            cat_has("bend", "lean")
            or text_has("bend walk", "hunch", "limp", "sneak")
            or cat_has("backwards movement", "sideways movement")
        )
        is_complex = (
            (len(act_cats) >= 6 and cat_has("interact with/use object", "take/pick something up", "place something", "gesture", "hand movements"))
            or ("transition" in act_cats and len(act_cats) >= 8)
        )
        if is_complex:
            return "Complex Actions"
        if is_advanced:
            return "Advanced Locomotion"
        if is_style:
            return "Basic Locomotion Styles"
        if len(act_cats) <= 3 and cat_has("walk", "turn", "step", "run", "jog", "forward movement"):
            return "Basic Locomotion Neutral"
        return "Baseline"

    if cat_has("look", "head movements", "face direction") or text_has("look"):
        return "Gestures"

    if cat_has("perform", "misc. abstract action") or text_has("magic"):
        return "Magic"

    if (
        cat_has("hand movements", "arm movements", "raising body part", "lowering body part", "touching body part", "waist movements")
        or text_has("pose", "tpose", "a pose")
    ):
        return "Baseline"

    return "Other"


def _reference_category_counts(reference_manifest: Path) -> Counter[str]:
    records = read_manifest(reference_manifest)
    return Counter(record["category"] for record in records)


def _category_match_score(record: dict[str, Any]) -> float:
    text = " | ".join(t.lower() for t in record["texts"])
    act_cats = set(record["act_cats"])
    category = record["category"]

    def text_hits(*terms: str) -> int:
        return sum(term in text for term in terms)

    def cat_hits(*terms: str) -> int:
        return sum(term in act_cats for term in terms)

    score = 0.0
    if category == "Basic Locomotion Neutral":
        score += 4.0 * cat_hits("walk", "run", "jog", "forward movement", "turn", "step")
        score += 2.0 * text_hits("walk", "run", "jog")
        score -= 2.0 * cat_hits("transition")
        score -= 0.25 * len(act_cats)
    elif category == "Baseline":
        score += 2.0 * cat_hits("transition", "stand", "hand movements", "arm movements")
        score += 1.0 * text_hits("stand", "transition")
        score -= 2.0 * cat_hits("interact with/use object", "take/pick something up", "place something")
    elif category == "Basic Locomotion Styles":
        score += 3.0 * cat_hits("backwards movement", "sideways movement", "bend", "lean")
        score += 2.0 * text_hits("backward", "backwards", "side step", "limp", "sneak", "bend")
    elif category == "Dancing":
        score += 4.0 * cat_hits("dance")
        score += 2.0 * text_hits("dance", "waltz", "cha cha", "salsa", "ballet", "ballroom")
    elif category == "Gestures":
        score += 3.0 * cat_hits("look", "head movements", "face direction")
        score += 2.0 * text_hits("look", "head", "face")
    elif category == "Other":
        score += 1.0 * cat_hits("perform", "misc. abstract action")
    elif category == "Complex Actions":
        score += 1.5 * len(act_cats)
        score += 2.0 * cat_hits("transition")
    elif category == "Communication":
        score += 4.0 * cat_hits("gesture", "greet", "wave", "clap", "communicate (vocalise)")
        score += 2.0 * text_hits("wave", "clap", "salute", "gesture", "greet", "phone", "cellphone")
    elif category == "Sports":
        score += 4.0 * cat_hits("play sport", "sports move", "action with ball", "martial art", "exercise/training")
        score += 2.0 * text_hits("ball", "basketball", "boxing", "soccer", "tennis", "kickbox")
    elif category == "Advanced Locomotion":
        score += 4.0 * cat_hits("balance", "stumble", "evade", "hop", "jump", "move up/down incline", "sneak", "limp", "skip", "slide")
        score += 2.0 * text_hits("jump", "hop", "stumble", "limp", "stairs", "incline")
    elif category == "Household":
        score += 4.0 * cat_hits(
            "interact with/use object",
            "take/pick something up",
            "place something",
            "clean something",
            "telephone call",
            "drink",
            "press something",
            "knock",
            "move something",
            "lift something",
        )
        score += 2.0 * text_hits("object", "phone", "telephone", "door", "chair", "cup", "box", "drink", "clean", "knock")
        score -= 2.0 * text_hits("place right hand", "place left hand", "place hand")
    elif category == "Looking and Pointing":
        score += 5.0 * cat_hits("point")
        score += 3.0 * text_hits("point")
    elif category == "Magic":
        score += 4.0 * cat_hits("misc. abstract action", "perform")
        score += 2.0 * text_hits("magic", "motorcycle", "wear ")
    return score


def _balanced_take_within_category(records: list[dict[str, Any]], quota: int, rng: random.Random) -> list[dict[str, Any]]:
    if quota <= 0 or not records:
        return []
    if quota >= len(records):
        return list(records)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["feat_p"])].append(record)

    for feat_p, items in groups.items():
        groups[feat_p] = sorted(
            items,
            key=lambda item: (
                -_category_match_score(item),
                abs(float(item["duration_s"]) - 10.0),
                item["motion_stats"]["height_span"],
                -item["motion_stats"]["disp"],
                item["sample_id"],
            ),
        )

    parent_keys = list(groups.keys())
    rng.shuffle(parent_keys)
    selected: list[dict[str, Any]] = []
    indices = {key: 0 for key in parent_keys}
    active_keys = list(parent_keys)

    while active_keys and len(selected) < quota:
        next_active: list[str] = []
        for key in active_keys:
            idx = indices[key]
            if idx < len(groups[key]):
                selected.append(groups[key][idx])
                indices[key] += 1
                if indices[key] < len(groups[key]):
                    next_active.append(key)
                if len(selected) >= quota:
                    break
        active_keys = next_active

    return selected


def _build_motion_entry(entry: dict[str, Any], joint_axes_29: np.ndarray) -> dict[str, Any]:
    motion = entry["motion"]
    root_trans = np.asarray(motion["root_trans_offset"], dtype=np.float32)
    root_rot = np.asarray(motion["root_rot"], dtype=np.float32)
    dof = np.asarray(motion["dof"], dtype=np.float32)
    root_rotvec = Rotation.from_quat(root_rot).as_rotvec().astype(np.float32)
    local_joint_rotvec = dof[..., None] * joint_axes_29[None, ...]
    pose_aa = np.concatenate([root_rotvec[:, None, :], local_joint_rotvec], axis=1).astype(np.float32)

    return {
        "root_trans_offset": root_trans,
        "root_rot": root_rot,
        "dof": dof,
        "pose_aa": pose_aa,
        "fps": int(motion.get("fps", 50)),
        "motion_name": entry["sample_id"],
    }


def main(args: Args) -> None:
    rng = random.Random(args.seed)
    target_counts = _reference_category_counts(args.reference_manifest)
    joint_axes_29 = parse_joint_axes(args.mjcf_path)

    all_records: list[dict[str, Any]] = []
    for split in ("train", "val"):
        split_path = args.input_dir / f"{split}.pkl"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing BABEL split file: {split_path}")
        payload = joblib.load(split_path)
        for index, entry in enumerate(payload):
            category = _classify_category(entry)
            texts = _joined_texts(entry)
            stats = _motion_stats(entry)
            feat_p = str(entry.get("feat_p", ""))
            sample_id = f"babel::{split}::{index:06d}::{Path(feat_p).stem}"
            all_records.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "sample_index": index,
                    "feat_p": feat_p,
                    "babel_sid": int(entry.get("babel_sid", -1)),
                    "texts": texts,
                    "act_cats": sorted(_act_cats(entry)),
                    "duration_s": float(entry.get("duration", 0.0)),
                    "length": int(entry.get("length", 0)),
                    "fps": int(entry["motion"].get("fps", 50)),
                    "category": category,
                    "package": CATEGORY_TO_PACKAGE[category],
                    "motion_stats": stats,
                    "motion": entry["motion"],
                }
            )

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in all_records:
        by_category[record["category"]].append(record)

    selected: list[dict[str, Any]] = []
    shortages: dict[str, int] = {}
    for category in SEED_CATEGORY_ORDER:
        quota = int(target_counts.get(category, 0))
        pool = by_category.get(category, [])
        picks = _balanced_take_within_category(pool, quota, rng)
        selected.extend(picks)
        shortages[category] = int(max(0, quota - len(picks)))

    if sum(shortages.values()) != 0:
        raise RuntimeError(f"Insufficient BABEL candidates for quotas: {shortages}")
    if len(selected) != sum(target_counts.values()):
        raise RuntimeError(f"Expected {sum(target_counts.values())} selected records, got {len(selected)}")

    selected = sorted(selected, key=lambda item: item["sample_id"])
    manifest_records = [
        {
            "sample_id": item["sample_id"],
            "split": item["split"],
            "sample_index": item["sample_index"],
            "feat_p": item["feat_p"],
            "babel_sid": item["babel_sid"],
            "texts": item["texts"],
            "act_cats": item["act_cats"],
            "duration_s": item["duration_s"],
            "length": item["length"],
            "fps": item["fps"],
            "category": item["category"],
            "package": item["package"],
            "motion_stats": item["motion_stats"],
        }
        for item in selected
    ]

    motion_dict = {
        item["sample_id"]: _build_motion_entry(item, joint_axes_29)
        for item in selected
    }

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_motionlib.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)

    write_jsonl_records(manifest_records, args.output_manifest)
    joblib.dump(motion_dict, args.output_motionlib)

    report = {
        "input_dir": str(args.input_dir.resolve()),
        "reference_manifest": str(args.reference_manifest.resolve()),
        "output_manifest": str(args.output_manifest.resolve()),
        "output_motionlib": str(args.output_motionlib.resolve()),
        "num_input_records": len(all_records),
        "num_selected": len(selected),
        "target_category_counts": {cat: int(target_counts.get(cat, 0)) for cat in SEED_CATEGORY_ORDER},
        "selected_by_category": dict(Counter(item["category"] for item in selected)),
        "selected_by_package": dict(Counter(item["package"] for item in selected)),
        "selected_by_split": dict(Counter(item["split"] for item in selected)),
        "duration_mean_s": float(np.mean([item["duration_s"] for item in selected])),
        "duration_p50_s": float(np.percentile([item["duration_s"] for item in selected], 50)),
        "duration_p90_s": float(np.percentile([item["duration_s"] for item in selected], 90)),
        "motion_stats_mean": {
            key: float(np.mean([item["motion_stats"][key] for item in selected]))
            for key in ["disp", "height_span", "dof_abs_mean", "root_speed_mean"]
        },
        "examples_by_category": {
            category: [
                {
                    "sample_id": item["sample_id"],
                    "feat_p": item["feat_p"],
                    "duration_s": item["duration_s"],
                    "texts": item["texts"][:3],
                }
                for item in selected
                if item["category"] == category
            ][:5]
            for category in SEED_CATEGORY_ORDER
        },
    }
    args.output_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved manifest to {args.output_manifest}")
    print(f"Saved motionlib to {args.output_motionlib}")
    print(f"Saved report to {args.output_report}")
    print(f"Selected {len(selected)} BABEL clips at original 50Hz")


if __name__ == "__main__":
    main(tyro.cli(Args))
