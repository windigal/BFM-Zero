from __future__ import annotations

import json
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import torch
import tyro
import humanoidverse.agents.envs.humanoidverse_isaac as humanoidverse_isaac
from scipy.spatial.transform import Rotation
from torch.utils._pytree import tree_map

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.language.stage_a.teacher import build_teacher_env, quiet_extraction, tracking_inference_fast
from humanoidverse.language.stage_b.primitives import (
    _PrimitiveSplitWriter,
    build_primitive_rows_from_z_seq,
    write_primitive_dataset_card,
)
from humanoidverse.scripts.convert_pico_to_motion import parse_joint_axes
from humanoidverse.utils.helpers import get_backward_observation


BABEL_SPLITS = ("train", "val")
# TextOp G1 23dof omits the two 3-axis wrists from the BFM 29dof layout.
ACT_IDX_23_IN_29 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 22, 23, 24, 25]


@dataclass
class Args:
    input_dir: Path = Path("artifacts/textop_babel_h2_f8_50fps")
    checkpoint_dir: Path = Path("results/bfmzero-isaac/20260402_170709/checkpoint")
    output_dir: Path = Path("artifacts/stage_b/textop_babel_latent_h2_f8_raw")
    robot_mjcf: Path = Path("humanoidverse/data/robots/g1/g1_29dof.xml")
    device: str = "cuda"
    simulator: str = "mujoco"
    clips_per_batch: int = 64
    motions_per_load: int = 64
    history_len: int = 2
    future_len: int = 8
    primitive_stride: int | None = None
    min_overlap_s: float = 1e-6
    rows_per_shard: int = 5000
    shard_format: str = "parquet"
    storage_dtype: str = "float16"
    target_representation: str = "raw"
    dct_keep_coeffs: int | None = 3
    use_root_height_obs: bool = True
    robot_override: str = "g1/g1_29dof"
    quiet: bool = True
    overwrite_output: bool = False
    max_samples_per_split: int | None = None


def _resolve_checkpoint_dir(path: Path) -> Path:
    if path.name == "checkpoint":
        return path
    candidate = path / "checkpoint"
    if candidate.exists():
        return candidate
    return path


def _batched(items: list[tuple[int, dict[str, Any]]], batch_size: int) -> Iterable[list[tuple[int, dict[str, Any]]]]:
    if batch_size <= 0:
        raise ValueError("clips_per_batch must be > 0")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _load_split_entries(path: Path) -> list[dict[str, Any]]:
    payload = joblib.load(path)
    if not isinstance(payload, list):
        raise TypeError(f"Expected list payload in {path}, got {type(payload)}")
    return payload


def _sanitize_sample_id(split: str, sample_idx: int, feat_p: str, babel_sid: int) -> str:
    stem = feat_p.replace(".pkl", "").replace("/", "__")
    return f"clip::babel::{split}::{sample_idx:06d}::{babel_sid}::{stem}"


def _pick_primary_text(frame_ann: list[tuple[float, float, str, list[str]]]) -> str:
    best_text = ""
    best_duration = -1.0
    for start_t, end_t, proc_label, _ in frame_ann:
        text = str(proc_label).strip()
        if not text:
            continue
        duration = float(end_t) - float(start_t)
        if duration > best_duration:
            best_text = text
            best_duration = duration
    return best_text


def _build_events(frame_ann: list[tuple[float, float, str, list[str]]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for start_t, end_t, proc_label, act_cat in frame_ann:
        text = str(proc_label).strip()
        if not text:
            continue
        events.append(
            {
                "description": text,
                "start_time": float(start_t),
                "end_time": float(end_t),
                "act_cat": list(act_cat),
            }
        )
    return events


def _build_record(split: str, sample_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    frame_ann = list(entry.get("frame_ann") or [])
    primary_text = _pick_primary_text(frame_ann)
    duration_s = float(entry.get("duration") or 0.0)
    return {
        "sample_id": sample_id,
        "sample_type": "clip",
        "split": split,
        "primary_text": primary_text,
        "events": _build_events(frame_ann),
        "duration_s": duration_s,
        "feat_p": str(entry.get("feat_p") or ""),
        "babel_sid": int(entry.get("babel_sid") or -1),
    }


def _infer_motion_dof_dim(motion: dict[str, Any]) -> int:
    if "dof" not in motion:
        raise KeyError(f"Could not infer dof_dim because motion has no 'dof' field: keys={list(motion.keys())}")
    dof = np.asarray(motion["dof"], dtype=np.float32)
    if dof.ndim != 2:
        raise ValueError(f"Expected motion['dof'] to have shape (T, D), got {dof.shape}")
    return int(dof.shape[1])


def _convert_motion_to_29dof(
    motion: dict[str, Any],
    joint_axes_29: np.ndarray,
) -> dict[str, Any]:
    root_trans = np.asarray(motion["root_trans_offset"], dtype=np.float32)
    root_rot = np.asarray(motion["root_rot"], dtype=np.float32)
    dof = np.asarray(motion["dof"], dtype=np.float32)

    if root_trans.ndim != 2 or root_trans.shape[1] != 3:
        raise ValueError(f"root_trans_offset must have shape (T, 3), got {root_trans.shape}")
    if root_rot.ndim != 2 or root_rot.shape[1] != 4:
        raise ValueError(f"root_rot must have shape (T, 4), got {root_rot.shape}")
    if dof.ndim != 2:
        raise ValueError(f"Expected dof to have shape (T, D), got {dof.shape}")
    if not (len(root_trans) == len(root_rot) == len(dof)):
        raise ValueError("root_trans_offset, root_rot, and dof must have the same number of frames")
    if joint_axes_29.shape != (29, 3):
        raise ValueError(f"Expected 29 joint axes, got {joint_axes_29.shape}")

    source_dof_dim = int(dof.shape[1])
    if source_dof_dim == 29:
        dof_29 = dof.astype(np.float32, copy=False)
    elif source_dof_dim == 23:
        dof_29 = np.zeros((len(dof), 29), dtype=np.float32)
        dof_29[:, ACT_IDX_23_IN_29] = dof
    else:
        raise ValueError(f"Unsupported source dof_dim={source_dof_dim}; expected 23 or 29")

    root_rotvec = Rotation.from_quat(root_rot).as_rotvec().astype(np.float32)
    local_joint_rotvec = dof_29[..., None] * joint_axes_29[None, ...]
    pose_aa = np.concatenate([root_rotvec[:, None, :], local_joint_rotvec], axis=1).astype(np.float32)

    return {
        "root_trans_offset": root_trans,
        "root_rot": root_rot,
        "dof": dof_29,
        "pose_aa": pose_aa,
        "fps": int(motion.get("fps", 50)),
        "source_dof_dim": source_dof_dim,
    }


def main(args: Args) -> None:
    checkpoint_dir = _resolve_checkpoint_dir(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir does not exist: {checkpoint_dir}")
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input dir does not exist: {args.input_dir}")

    if args.output_dir.exists():
        has_existing = any(args.output_dir.iterdir())
        if has_existing and not args.overwrite_output:
            raise FileExistsError(f"Output directory {args.output_dir} is not empty. Pass --overwrite-output.")
        if has_existing and args.overwrite_output:
            for child in args.output_dir.iterdir():
                if child.is_dir():
                    import shutil

                    shutil.rmtree(child)
                else:
                    child.unlink()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    primitive_stride = int(args.primitive_stride or args.future_len)
    joint_axes_29 = parse_joint_axes(args.robot_mjcf)
    if joint_axes_29.shape != (29, 3):
        raise ValueError(f"Expected 29 actuated joints in {args.robot_mjcf}, got {joint_axes_29.shape}")

    entries_by_split: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    input_split_counts: dict[str, int] = {}
    input_dof_dims: set[int] = set()
    for split in BABEL_SPLITS:
        split_path = args.input_dir / f"{split}.pkl"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")
        split_entries = _load_split_entries(split_path)
        if args.max_samples_per_split is not None:
            split_entries = split_entries[: args.max_samples_per_split]
        for entry in split_entries:
            input_dof_dims.add(_infer_motion_dof_dim(entry["motion"]))
        entries_by_split[split] = list(enumerate(split_entries))
        input_split_counts[split] = len(split_entries)

    writers = {
        split: _PrimitiveSplitWriter(
            args.output_dir / split,
            args.rows_per_shard,
            shard_format=args.shard_format,
            storage_dtype=args.storage_dtype,
        )
        for split, entries in entries_by_split.items()
        if entries
    }

    prompt_dim: int | None = None
    total_rows = 0
    total_samples = 0
    split_row_counts: dict[str, int] = defaultdict(int)
    sample_type_counts: dict[str, int] = defaultdict(int)
    source_batches: list[dict[str, Any]] = []

    with quiet_extraction(args.quiet):
        model = load_model_from_checkpoint_dir(checkpoint_dir, device=args.device)
        model.to(args.device)
        model.eval()

        with torch.inference_mode(), tempfile.TemporaryDirectory(prefix="textop_babel_latent_", dir=str(args.output_dir)) as tmp_dir:
            tmp_root = Path(tmp_dir)
            for split in BABEL_SPLITS:
                split_entries = entries_by_split[split]
                if not split_entries:
                    continue
                writer = writers[split]
                for batch_idx, batch_entries in enumerate(_batched(split_entries, args.clips_per_batch), start=1):
                    print(f"[{split}] batch {batch_idx}: converting {len(batch_entries)} clips")
                    motion_dict: dict[str, Any] = {}
                    record_lookup: dict[str, dict[str, Any]] = {}
                    for sample_idx, entry in batch_entries:
                        sample_id = _sanitize_sample_id(
                            split=split,
                            sample_idx=sample_idx,
                            feat_p=str(entry.get("feat_p") or f"sample_{sample_idx:06d}"),
                            babel_sid=int(entry.get("babel_sid") or -1),
                        )
                        motion_dict[sample_id] = _convert_motion_to_29dof(entry["motion"], joint_axes_29)
                        record_lookup[sample_id] = _build_record(split=split, sample_id=sample_id, entry=entry)

                    motion_path = tmp_root / f"{split}_batch_{batch_idx:05d}.pkl"
                    joblib.dump(motion_dict, motion_path)

                    wrapped_env = None
                    try:
                        wrapped_env = build_teacher_env(
                            motion_file=motion_path,
                            device=args.device,
                            simulator=args.simulator,
                            use_root_height_obs=args.use_root_height_obs,
                            use_contact_in_obs_max=False,
                            robot_override=args.robot_override,
                            hydra_overrides=[],
                        )
                        env = wrapped_env._env
                        motion_keys = [str(key) for key in env._motion_lib._motion_data_keys.tolist()]

                        batch_rows = 0
                        batch_samples = 0
                        motions_per_load = max(int(args.motions_per_load), 1)
                        for batch_start in range(0, len(motion_keys), motions_per_load):
                            load_count = min(motions_per_load, len(motion_keys) - batch_start)
                            env._motion_lib.load_motions(
                                random_sample=False,
                                start_idx=batch_start,
                                num_motions_to_load=load_count,
                            )
                            current_keys = [str(key) for key in env._motion_lib.curr_motion_keys]
                            expected_keys = motion_keys[batch_start : batch_start + len(current_keys)]
                            if current_keys != expected_keys:
                                raise RuntimeError(
                                    f"Motion reload mismatch: expected {expected_keys[:3]}, got {current_keys[:3]}"
                                )

                            for local_motion_id, motion_key in enumerate(current_keys):
                                record = record_lookup[motion_key]
                                obs, _ = get_backward_observation(env, local_motion_id, use_root_height_obs=args.use_root_height_obs)
                                obs = tree_map(lambda value: value[1:], obs)
                                z_seq = tracking_inference_fast(model, obs).detach().cpu()
                                if prompt_dim is None:
                                    prompt_dim = int(z_seq.shape[-1])

                                rows = build_primitive_rows_from_z_seq(
                                    record=record,
                                    z_seq=z_seq,
                                    history_len=args.history_len,
                                    future_len=args.future_len,
                                    primitive_stride=primitive_stride,
                                    min_overlap_s=args.min_overlap_s,
                                    target_representation=args.target_representation,
                                    dct_keep_coeffs=args.dct_keep_coeffs,
                                )
                                writer.write_sample_rows(
                                    split=split,
                                    sample_id=motion_key,
                                    sample_type="clip",
                                    rows=rows,
                                )
                                total_rows += len(rows)
                                split_row_counts[split] += len(rows)
                                batch_rows += len(rows)
                                if rows:
                                    total_samples += 1
                                    batch_samples += 1
                                    sample_type_counts["clip"] += 1
                    finally:
                        if wrapped_env is not None:
                            wrapped_env.close()
                        humanoidverse_isaac._humanoidverse_env_singleton = None

                    source_batches.append(
                        {
                            "split": split,
                            "batch_index": batch_idx,
                            "num_input_clips": len(batch_entries),
                            "num_samples_written": batch_samples,
                            "num_rows_written": batch_rows,
                        }
                    )

                    motion_path.unlink(missing_ok=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    if prompt_dim is None:
        raise RuntimeError("No latent primitive rows were generated from the input BABEL dataset.")

    split_summaries = {split: writer.finalize() for split, writer in writers.items()}
    summary = {
        "input_dir": str(args.input_dir.resolve()),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "robot_mjcf": str(args.robot_mjcf.resolve()),
        "source_format": "textop_babel_packed_clips",
        "source_dof_dims": sorted(input_dof_dims),
        "target_robot_override": args.robot_override,
        "target_dof": 29,
        "device": args.device,
        "simulator": args.simulator,
        "history_len": args.history_len,
        "future_len": args.future_len,
        "primitive_stride": primitive_stride,
        "prompt_dim": prompt_dim,
        "target_representation": args.target_representation,
        "dct_keep_coeffs": args.dct_keep_coeffs,
        "rows_per_shard": args.rows_per_shard,
        "shard_format": args.shard_format,
        "storage_dtype": args.storage_dtype,
        "clips_per_batch": args.clips_per_batch,
        "motions_per_load": args.motions_per_load,
        "input_split_counts": input_split_counts,
        "sample_types": ["clip"],
        "total_samples": total_samples,
        "total_rows": total_rows,
        "split_row_counts": dict(split_row_counts),
        "sample_type_counts": dict(sample_type_counts),
        "splits": split_summaries,
        "source_batches": source_batches,
    }
    (args.output_dir / "meta.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_primitive_dataset_card(args.output_dir, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(tyro.cli(Args))