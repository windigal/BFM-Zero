from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np


JOINT_COLUMNS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

ROOT_COLUMNS = [
    "root_x",
    "root_y",
    "root_z",
    "root_qx",
    "root_qy",
    "root_qz",
    "root_qw",
]


def _install_numpy_pickle_compat() -> None:
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def load_pickle(path: Path) -> dict:
    _install_numpy_pickle_compat()
    try:
        return joblib.load(path)
    except Exception:
        with path.open("rb") as f:
            return pickle.load(f)


def to_xyzw(quat: np.ndarray, fmt: str) -> np.ndarray:
    if fmt == "xyzw":
        return quat
    if fmt == "wxyz":
        return quat[..., [1, 2, 3, 0]]
    raise ValueError(f"Unsupported quaternion format: {fmt}")


def export_csv(input_path: Path, output_path: Path, root_rot_format: str) -> None:
    save_data = load_pickle(input_path)
    root_pos = np.asarray(save_data["root_pos"], dtype=np.float32)
    root_rot = np.asarray(save_data["root_rot"], dtype=np.float32)
    dof_pos = np.asarray(save_data["dof_pos"], dtype=np.float32)

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_pos must have shape (T, 3), got {root_pos.shape}")
    if root_rot.ndim != 2 or root_rot.shape[1] != 4:
        raise ValueError(f"root_rot must have shape (T, 4), got {root_rot.shape}")
    if dof_pos.ndim != 2 or dof_pos.shape[1] != len(JOINT_COLUMNS):
        raise ValueError(
            f"dof_pos must have shape (T, {len(JOINT_COLUMNS)}), got {dof_pos.shape}"
        )
    if not (len(root_pos) == len(root_rot) == len(dof_pos)):
        raise ValueError("root_pos, root_rot, dof_pos must have the same number of frames")

    root_rot_xyzw = to_xyzw(root_rot, root_rot_format)
    frame_data = np.concatenate([root_pos, root_rot_xyzw, dof_pos], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(ROOT_COLUMNS + JOINT_COLUMNS)
        writer.writerows(frame_data.tolist())

    print(f"Saved {output_path}")
    print(f"num_frames={frame_data.shape[0]}")
    print(f"num_columns={frame_data.shape[1]}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export pico save_data to a 36D CSV: root XYZQXQYQZQW + 29 joints."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("humanoidverse/data/pico.pkl"),
        help="Input pico/save_data pickle.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("humanoidverse/data/pico_qpos.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--root-rot-format",
        choices=("xyzw", "wxyz"),
        default="xyzw",
        help="Quaternion layout stored in save_data['root_rot'].",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    export_csv(args.input, args.output, args.root_rot_format)


if __name__ == "__main__":
    main()
