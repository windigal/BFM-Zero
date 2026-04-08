from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import tyro
from scipy.spatial.transform import Rotation

from humanoidverse.scripts.convert_pico_to_motion import parse_joint_axes


EXPECTED_G1_COLUMNS = 36
G1_FPS = 30


def load_lafan1_g1_array(
    csv_path: Path,
    start_frame: int = 0,
    end_frame: int | None = None,
    stride: int = 1,
) -> np.ndarray:
    if stride <= 0:
        raise ValueError("stride must be > 0")

    rows: list[list[float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx < start_frame:
                continue
            if end_frame is not None and idx >= end_frame:
                break
            if (idx - start_frame) % stride != 0:
                continue
            if len(row) != EXPECTED_G1_COLUMNS:
                raise ValueError(
                    f"Expected {EXPECTED_G1_COLUMNS} columns in {csv_path}, got {len(row)} at row {idx}"
                )
            rows.append([float(x) for x in row])

    if not rows:
        raise ValueError(f"No rows selected from {csv_path} with start={start_frame}, end={end_frame}, stride={stride}")
    return np.asarray(rows, dtype=np.float32)


def build_motion_entry_from_lafan1_g1(
    csv_path: Path,
    mjcf_path: Path,
    motion_name: str | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    stride: int = 1,
) -> dict[str, Any]:
    data = load_lafan1_g1_array(
        csv_path=csv_path,
        start_frame=start_frame,
        end_frame=end_frame,
        stride=stride,
    )
    joint_axes = parse_joint_axes(mjcf_path)

    root_pos = data[:, :3].astype(np.float32)
    root_rot_xyzw = data[:, 3:7].astype(np.float32)
    dof_pos = data[:, 7:].astype(np.float32)
    if dof_pos.shape[1] != joint_axes.shape[0]:
        raise ValueError(
            f"CSV DOF shape {dof_pos.shape} does not match MJCF joint axes {joint_axes.shape}"
        )

    root_rotvec = Rotation.from_quat(root_rot_xyzw).as_rotvec().astype(np.float32)
    local_joint_rotvec = dof_pos[..., None] * joint_axes[None, ...]
    pose_aa = np.concatenate([root_rotvec[:, None, :], local_joint_rotvec], axis=1).astype(np.float32)

    motion_entry = {
        "root_trans_offset": root_pos,
        "pose_aa": pose_aa,
        "dof": dof_pos,
        "root_rot": root_rot_xyzw,
        "fps": G1_FPS,
    }
    if motion_name is not None:
        motion_entry["motion_name"] = motion_name
    return motion_entry


def compare_motion_entries(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key in ("root_trans_offset", "root_rot", "dof", "pose_aa"):
        ref = np.asarray(reference[key], dtype=np.float32)
        cand = np.asarray(candidate[key], dtype=np.float32)
        if ref.shape != cand.shape:
            raise ValueError(f"Shape mismatch for {key}: {ref.shape} vs {cand.shape}")
        diff = cand - ref
        metrics[f"{key}_rmse"] = float(np.sqrt(np.mean(diff ** 2)))
        metrics[f"{key}_max_abs"] = float(np.max(np.abs(diff)))
    return metrics


@dataclass
class Args:
    csv_path: Path
    output: Path = Path("artifacts/lafan1/lafan1_motion.pkl")
    motion_name: str | None = None
    mjcf: Path = Path("humanoidverse/data/robots/g1/g1_29dof.xml")
    start_frame: int = 0
    end_frame: int | None = None
    stride: int = 1
    compare_pkl: Path | None = None
    compare_key: str | None = None
    compare_output: Path | None = None


def main(args: Args) -> None:
    motion_name = args.motion_name or args.csv_path.stem
    motion_entry = build_motion_entry_from_lafan1_g1(
        csv_path=args.csv_path,
        mjcf_path=args.mjcf,
        motion_name=motion_name,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        stride=args.stride,
    )
    motion_dict = {motion_name: motion_entry}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(motion_dict, args.output)

    print(f"Saved motion file to {args.output}")
    print(f"motion_name={motion_name}")
    print(f"num_frames={motion_entry['root_trans_offset'].shape[0]}")
    print(f"pose_aa_shape={motion_entry['pose_aa'].shape}")

    if args.compare_pkl is not None:
        compare_key = args.compare_key or motion_name
        reference = joblib.load(args.compare_pkl)[compare_key]
        if args.start_frame != 0 or args.end_frame is not None or args.stride != 1:
            end = args.end_frame
            reference = {
                k: (v[args.start_frame:end:args.stride] if hasattr(v, "__getitem__") and np.asarray(v).ndim > 0 else v)
                for k, v in reference.items()
            }
        metrics = compare_motion_entries(reference, motion_entry)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        if args.compare_output is not None:
            args.compare_output.parent.mkdir(parents=True, exist_ok=True)
            args.compare_output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main(tyro.cli(Args))
