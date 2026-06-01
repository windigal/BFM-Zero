from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
import tyro
import yaml
from torch.utils._pytree import tree_map

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.language.stage_a.seed import parse_joint_axes
from humanoidverse.language.stage_a.teacher import build_teacher_env
from humanoidverse.utils.helpers import get_backward_observation


ARM_DOWN_JOINT_OVERRIDES: dict[str, float] = {
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": -0.4,
    "left_shoulder_yaw_joint": 0.7,
    "left_elbow_joint": 1.1,
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": 0.4,
    "right_shoulder_yaw_joint": -0.7,
    "right_elbow_joint": 1.1,
}


def _expand_default_joint_pos(default_joint_pos: dict[str, float], joint_names: list[str]) -> np.ndarray:
    values = np.zeros(len(joint_names), dtype=np.float32)
    for idx, joint_name in enumerate(joint_names):
        matched = False
        for pattern, value in default_joint_pos.items():
            if re.fullmatch(pattern, joint_name):
                values[idx] = float(value)
                matched = True
                break
        if not matched:
            values[idx] = 0.0
    return values


def _read_root_pose_from_mjcf(mjcf_path: Path) -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(mjcf_path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Could not find <worldbody> in {mjcf_path}")
    pelvis = worldbody.find("./body[@name='pelvis']")
    if pelvis is None:
        raise ValueError(f"Could not find pelvis body in {mjcf_path}")
    pos_attr = pelvis.get("pos", "0 0 0")
    root_pos = np.fromstring(pos_attr, sep=" ", dtype=np.float32)
    if root_pos.shape != (3,):
        raise ValueError(f"Expected pelvis pos with 3 values in {mjcf_path}, got {pos_attr!r}")
    quat_attr = pelvis.get("quat")
    if quat_attr is None:
        root_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    else:
        quat_wxyz = np.fromstring(quat_attr, sep=" ", dtype=np.float32)
        if quat_wxyz.shape != (4,):
            raise ValueError(f"Expected pelvis quat with 4 values in {mjcf_path}, got {quat_attr!r}")
        root_quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
    return root_pos, root_quat_xyzw


@dataclass
class Args:
    checkpoint_dir: Path = Path("/home/hanwei/code/BFM-Zero/results/bfmzero-isaac/20260402_170709/checkpoint")
    policy_config: Path = Path("/home/hanwei/code/BFM-zero-deploy/config/policy/motivo_newG1_issue17.yaml")
    mjcf_path: Path = Path("humanoidverse/data/robots/g1/g1_29dof.xml")
    motion_output: Path = Path("artifacts/deploy_export/init_stand_seed2k_latest_motion.pkl")
    latent_output: Path = Path("/home/hanwei/code/BFM-zero-deploy/model/tracking_seed_2k_s/zs_seed2k_init_stand_latest.pkl")
    meta_output: Path | None = None
    device: str = "cuda"
    simulator: str = "mujoco"
    num_frames: int = 64
    target_fps: int = 30
    use_root_height_obs: bool = True
    use_contact_in_obs_max: bool = False
    robot_override: str = "g1/g1_29dof"
    motion_name: str = "clip::deploy_init_stand_arms_down"
    use_arms_down_pose: bool = True


def main(args: Args) -> None:
    with args.policy_config.open("r", encoding="utf-8") as f:
        policy_cfg = yaml.safe_load(f)

    joint_names = list(policy_cfg["isaac_joint_names"])
    default_joint_pos = _expand_default_joint_pos(policy_cfg["default_joint_pos"], joint_names)
    joint_overrides = ARM_DOWN_JOINT_OVERRIDES if args.use_arms_down_pose else {}
    joint_name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    for joint_name, joint_value in joint_overrides.items():
        idx = joint_name_to_idx.get(joint_name)
        if idx is None:
            raise KeyError(f"Joint override {joint_name!r} not found in policy joint list.")
        default_joint_pos[idx] = float(joint_value)

    root_pos, root_quat_xyzw = _read_root_pose_from_mjcf(args.mjcf_path)
    joint_axes = parse_joint_axes(args.mjcf_path)
    if joint_axes.shape != (len(joint_names), 3):
        raise ValueError(
            f"Expected joint_axes shape {(len(joint_names), 3)}, got {joint_axes.shape}"
        )

    num_frames = max(int(args.num_frames), 2)
    dof = np.repeat(default_joint_pos[None, :], num_frames, axis=0).astype(np.float32)
    root_trans_offset = np.repeat(root_pos[None, :], num_frames, axis=0).astype(np.float32)
    root_rot = np.repeat(root_quat_xyzw[None, :], num_frames, axis=0).astype(np.float32)
    root_rotvec = np.zeros((num_frames, 1, 3), dtype=np.float32)
    local_joint_rotvec = dof[..., None] * joint_axes[None, ...]
    pose_aa = np.concatenate([root_rotvec, local_joint_rotvec], axis=1).astype(np.float32)

    motion_dict = {
        args.motion_name: {
            "root_trans_offset": root_trans_offset,
            "pose_aa": pose_aa,
            "dof": dof,
            "root_rot": root_rot,
            "fps": int(args.target_fps),
            "motion_name": args.motion_name,
        }
    }

    args.motion_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(motion_dict, args.motion_output)

    model = load_model_from_checkpoint_dir(args.checkpoint_dir, device=args.device)
    model.to(args.device)
    model.eval()

    wrapped_env = build_teacher_env(
        motion_file=args.motion_output,
        device=args.device,
        simulator=args.simulator,
        use_root_height_obs=args.use_root_height_obs,
        use_contact_in_obs_max=args.use_contact_in_obs_max,
        robot_override=args.robot_override,
        hydra_overrides=[],
    )
    env = wrapped_env._env

    obs, _ = get_backward_observation(env, 0, use_root_height_obs=args.use_root_height_obs)
    obs = tree_map(lambda x: x[1:], obs)
    z_seq = model.project_z(model.backward_map(obs)).cpu().numpy().astype(np.float32)

    args.latent_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(z_seq, args.latent_output)

    meta_output = args.meta_output or args.latent_output.with_suffix(".meta.json")
    meta = {
        "motion_name": args.motion_name,
        "motion_output": str(args.motion_output.resolve()),
        "latent_output": str(args.latent_output.resolve()),
        "seq_len": int(z_seq.shape[0]),
        "z_dim": int(z_seq.shape[1]),
        "target_fps": int(args.target_fps),
        "num_frames": int(num_frames),
        "root_pos": root_pos.tolist(),
        "root_quat_xyzw": root_quat_xyzw.tolist(),
        "default_joint_pos": default_joint_pos.tolist(),
        "use_root_height_obs": bool(args.use_root_height_obs),
        "use_contact_in_obs_max": bool(args.use_contact_in_obs_max),
        "source": "deploy_init_standing_pose",
        "use_arms_down_pose": bool(args.use_arms_down_pose),
        "arm_down_joint_overrides": joint_overrides,
        "description": "Static standing prompt sequence exported from deploy init/default pose with optional arm-only prompt overrides.",
    }
    meta_output.parent.mkdir(parents=True, exist_ok=True)
    with meta_output.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved motion file to {args.motion_output}")
    print(f"Saved tracking z sequence to {args.latent_output}")
    print(f"Saved metadata to {meta_output}")
    print(f"z_shape={tuple(z_seq.shape)}")


if __name__ == "__main__":
    main(tyro.cli(Args))