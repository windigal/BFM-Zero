from __future__ import annotations

import argparse
import pickle
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation


def _install_numpy_pickle_compat() -> None:
    # Some locally generated pickles reference numpy._core, which NumPy 1.26
    # exposes under numpy.core instead.
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
            raise ValueError(f"Joint {joint.attrib.get('name')} axis has invalid shape: {axis.shape}")
        axes.append(axis)

    if not axes:
        raise ValueError(f"No actuated joints found in {mjcf_path}")
    return np.stack(axes, axis=0)


def to_xyzw(quat: np.ndarray, fmt: str) -> np.ndarray:
    if fmt == "xyzw":
        return quat
    if fmt == "wxyz":
        return quat[..., [1, 2, 3, 0]]
    raise ValueError(f"Unsupported quaternion format: {fmt}")


def convert_save_data_to_motion(
    save_data: dict,
    joint_axes: np.ndarray,
    root_rot_format: str,
) -> dict:
    required = ("fps", "dof_pos", "root_pos", "root_rot")
    missing = [key for key in required if key not in save_data]
    if missing:
        raise KeyError(f"Missing keys in save_data: {missing}")

    dof_pos = np.asarray(save_data["dof_pos"], dtype=np.float32)
    root_pos = np.asarray(save_data["root_pos"], dtype=np.float32)
    root_rot = np.asarray(save_data["root_rot"], dtype=np.float32)

    if dof_pos.ndim != 2:
        raise ValueError(f"dof_pos must have shape (T, 29), got {dof_pos.shape}")
    if root_pos.shape != (dof_pos.shape[0], 3):
        raise ValueError(f"root_pos must have shape {(dof_pos.shape[0], 3)}, got {root_pos.shape}")
    if root_rot.shape != (dof_pos.shape[0], 4):
        raise ValueError(f"root_rot must have shape {(dof_pos.shape[0], 4)}, got {root_rot.shape}")
    if joint_axes.shape != (dof_pos.shape[1], 3):
        raise ValueError(
            f"MJCF joint axes shape {joint_axes.shape} does not match dof_pos shape {dof_pos.shape}"
        )

    root_rot_xyzw = to_xyzw(root_rot, root_rot_format)
    root_rotvec = Rotation.from_quat(root_rot_xyzw).as_rotvec().astype(np.float32)
    local_joint_rotvec = dof_pos[..., None] * joint_axes[None, ...]
    pose_aa = np.concatenate([root_rotvec[:, None, :], local_joint_rotvec], axis=1).astype(np.float32)

    motion_entry = {
        "root_trans_offset": root_pos.astype(np.float32),
        "pose_aa": pose_aa,
        "dof": dof_pos.astype(np.float32),
        "root_rot": root_rot_xyzw.astype(np.float32),
        "fps": int(save_data["fps"]),
    }
    return motion_entry


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a save_data-style pico pickle into motion-lib-compatible root_trans_offset/pose_aa."
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
        default=Path("humanoidverse/data/pico_motion.pkl"),
        help="Output motion pickle.",
    )
    parser.add_argument(
        "--motion-name",
        default="pico",
        help="Top-level key used in the exported motion dict.",
    )
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=Path("humanoidverse/data/robots/g1/g1_29dof.xml"),
        help="MJCF used to read joint axes.",
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
    save_data = load_pickle(args.input)
    if not isinstance(save_data, dict):
        raise TypeError(f"Expected dict from {args.input}, got {type(save_data)}")

    joint_axes = parse_joint_axes(args.mjcf)
    motion_entry = convert_save_data_to_motion(
        save_data=save_data,
        joint_axes=joint_axes,
        root_rot_format=args.root_rot_format,
    )

    output_dict = {args.motion_name: motion_entry}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(output_dict, args.output)

    print(f"Saved {args.output}")
    print(f"motion_name={args.motion_name}")
    print(f"num_frames={motion_entry['root_trans_offset'].shape[0]}")
    print(f"pose_aa_shape={motion_entry['pose_aa'].shape}")
    print(f"dof_shape={motion_entry['dof'].shape}")


if __name__ == "__main__":
    main()
