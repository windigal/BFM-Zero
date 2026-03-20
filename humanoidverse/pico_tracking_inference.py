from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import joblib
import mujoco
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.envs.legged_robot_motions.legged_robot_motions import compute_humanoid_observations_max
from humanoidverse.utils.torch_utils import quat_rotate_inverse


BFM_ZERO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_ROOT = BFM_ZERO_ROOT.parent / "BFM-Zero-deploy"

TARGET_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "head_link",
]
HEAD_OFFSET = np.array([0.0, 0.0, 0.35], dtype=np.float32)


def _install_numpy_pickle_compat() -> None:
    np_core = getattr(np, "_core", np.core)
    sys.modules.setdefault("numpy._core", np_core)
    sys.modules.setdefault("numpy._core.multiarray", np_core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np_core.numeric)


def load_pickle(path: Path) -> dict[str, Any]:
    _install_numpy_pickle_compat()
    try:
        return joblib.load(path)
    except Exception:
        with path.open("rb") as f:
            return pickle.load(f)


def to_wxyz(quat: np.ndarray, fmt: str) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    if fmt == "wxyz":
        return quat
    if fmt == "xyzw":
        return quat[..., [3, 0, 1, 2]]
    raise ValueError(f"Unsupported quaternion format: {fmt}")


def wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    return quat[..., [1, 2, 3, 0]]


def default_dof_from_policy(policy_config: dict[str, Any]) -> np.ndarray:
    isaac_joint_names = policy_config["isaac_joint_names"]
    default_joint_pos = policy_config["default_joint_pos"]
    default_dof_pos = np.zeros(len(isaac_joint_names), dtype=np.float32)
    for idx, joint_name in enumerate(isaac_joint_names):
        for pattern, value in default_joint_pos.items():
            if pattern.startswith(".*_"):
                if joint_name.endswith(pattern[3:]):
                    default_dof_pos[idx] = value
            elif pattern == joint_name:
                default_dof_pos[idx] = value
    return default_dof_pos


def finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    dt = 1.0 / float(fps)
    return np.gradient(values, dt, axis=0).astype(np.float32)


def angular_velocity_from_quat_xyzw(quat_xyzw: np.ndarray, fps: float) -> np.ndarray:
    if quat_xyzw.shape[0] == 1:
        return np.zeros(quat_xyzw.shape[:-1] + (3,), dtype=np.float32)

    flat_next = Rotation.from_quat(quat_xyzw[1:].reshape(-1, 4))
    flat_prev = Rotation.from_quat(quat_xyzw[:-1].reshape(-1, 4))
    delta = flat_next * flat_prev.inv()
    ang = (delta.as_rotvec() * fps).reshape(quat_xyzw.shape[0] - 1, *quat_xyzw.shape[1:-1], 3)

    out = np.zeros(quat_xyzw.shape[:-1] + (3,), dtype=np.float32)
    out[1:] = ang.astype(np.float32)
    out[0] = out[1]
    return out


def compute_body_kinematics(
    root_pos: np.ndarray,
    root_rot_wxyz: np.ndarray,
    dof_pos: np.ndarray,
    robot_xml_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(robot_xml_path))
    data = mujoco.MjData(model)
    body_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in TARGET_BODY_NAMES
        if name != "head_link"
    }
    torso_idx = TARGET_BODY_NAMES.index("torso_link")

    num_frames = dof_pos.shape[0]
    body_pos = np.zeros((num_frames, len(TARGET_BODY_NAMES), 3), dtype=np.float32)
    body_rot = np.zeros((num_frames, len(TARGET_BODY_NAMES), 4), dtype=np.float32)

    for i in range(num_frames):
        data.qpos[:3] = root_pos[i]
        data.qpos[3:7] = root_rot_wxyz[i]
        data.qpos[7:] = dof_pos[i]
        mujoco.mj_forward(model, data)

        for body_idx, body_name in enumerate(TARGET_BODY_NAMES[:-1]):
            mj_body_id = body_ids[body_name]
            body_pos[i, body_idx] = data.xpos[mj_body_id]
            body_rot[i, body_idx] = data.xquat[mj_body_id]

        torso_pos = body_pos[i, torso_idx]
        torso_rot = body_rot[i, torso_idx][None, :]
        head_offset_world = Rotation.from_quat(wxyz_to_xyzw(torso_rot)).apply(HEAD_OFFSET[None, :])[0]
        body_pos[i, -1] = torso_pos + head_offset_world
        body_rot[i, -1] = torso_rot[0]

    return body_pos, body_rot


def build_tracking_observation(
    save_data: dict[str, Any],
    policy_config: dict[str, Any],
    robot_xml_path: Path,
    root_rot_format: str,
    expected_state_dim: int,
    expected_privileged_dim: int,
) -> dict[str, np.ndarray]:
    dof_pos = np.asarray(save_data["dof_pos"], dtype=np.float32)
    root_pos = np.asarray(save_data["root_pos"], dtype=np.float32)
    root_rot_wxyz = to_wxyz(np.asarray(save_data["root_rot"]), root_rot_format)
    fps = int(save_data["fps"])

    default_dof_pos = default_dof_from_policy(policy_config)
    body_pos, body_rot_wxyz = compute_body_kinematics(root_pos, root_rot_wxyz, dof_pos, robot_xml_path)
    body_rot_xyzw = wxyz_to_xyzw(body_rot_wxyz)
    dof_vel = finite_difference(dof_pos, fps)
    body_vel = finite_difference(body_pos, fps)
    body_ang_vel = angular_velocity_from_quat_xyzw(body_rot_xyzw, fps)
    root_ang_vel = body_ang_vel[:, 0, :]

    body_pos_t = torch.as_tensor(body_pos, dtype=torch.float32)
    body_rot_t = torch.as_tensor(body_rot_xyzw, dtype=torch.float32)
    body_vel_t = torch.as_tensor(body_vel, dtype=torch.float32)
    body_ang_vel_t = torch.as_tensor(body_ang_vel, dtype=torch.float32)
    obs_dict = compute_humanoid_observations_max(
        body_pos_t,
        body_rot_t,
        body_vel_t,
        body_ang_vel_t,
        True,
        True,
    )
    privileged_state = torch.cat([value for value in obs_dict.values()], dim=-1).cpu().numpy().astype(np.float32)
    gravity = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32).repeat(dof_pos.shape[0], 1)
    projected_gravity = quat_rotate_inverse(
        torch.as_tensor(wxyz_to_xyzw(root_rot_wxyz), dtype=torch.float32),
        gravity,
        w_last=True,
    ).cpu().numpy().astype(np.float32)

    state = np.concatenate(
        [dof_pos - default_dof_pos, dof_vel, projected_gravity, root_ang_vel],
        axis=-1,
    ).astype(np.float32)

    if state.shape[-1] != expected_state_dim:
        raise ValueError(f"Expected state dim {expected_state_dim}, got {state.shape[-1]}")
    if privileged_state.shape[-1] != expected_privileged_dim:
        raise ValueError(
            f"Expected privileged_state dim {expected_privileged_dim}, got {privileged_state.shape[-1]}"
        )

    return {
        "state": state,
        "privileged_state": privileged_state,
    }


def tracking_inference_no_delay(model: Any, obs: dict[str, torch.Tensor]) -> torch.Tensor:
    z = model.backward_map(obs)
    return model.project_z(z)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert pico.pkl into deploy-compatible tracking z latents.")
    parser.add_argument(
        "--pico_path",
        type=Path,
        default=BFM_ZERO_ROOT / "humanoidverse" / "data" / "pico2.pkl",
        help="Input pico save_data pickle.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=DEPLOY_ROOT / "model" / "checkpoint",
        help="Checkpoint directory used to infer tracking z.",
    )
    parser.add_argument(
        "--policy_config",
        type=Path,
        default=DEPLOY_ROOT / "config" / "policy" / "motivo_newG1.yaml",
        help="Deploy policy config used for default joint positions.",
    )
    parser.add_argument(
        "--robot_xml_path",
        type=Path,
        default=BFM_ZERO_ROOT / "humanoidverse" / "data" / "robots" / "g1" / "g1_29dof.xml",
        help="MuJoCo XML used to reconstruct body kinematics from pico qpos.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEPLOY_ROOT / "model" / "tracking_inference" / "zs_pico2.pkl",
        help="Output z trajectory for deploy tracking mode.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for tracking z inference.",
    )
    parser.add_argument(
        "--root_rot_format",
        choices=("xyzw", "wxyz"),
        default="xyzw",
        help="Quaternion layout stored in pico save_data.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    save_data = load_pickle(args.pico_path)
    with args.policy_config.open("r") as f:
        policy_config = yaml.load(f, Loader=yaml.FullLoader)

    model = load_model_from_checkpoint_dir(args.checkpoint_dir, device=args.device)
    model.to(args.device)
    model.eval()

    with (args.checkpoint_dir / "model" / "init_kwargs.json").open("r") as f:
        init_kwargs = yaml.safe_load(f)
    obs_space = init_kwargs["obs_space"]["spaces"]
    expected_state_dim = int(obs_space["state"]["shape"][0])
    expected_privileged_dim = int(obs_space["privileged_state"]["shape"][0])

    obs_np = build_tracking_observation(
        save_data=save_data,
        policy_config=policy_config,
        robot_xml_path=args.robot_xml_path,
        root_rot_format=args.root_rot_format,
        expected_state_dim=expected_state_dim,
        expected_privileged_dim=expected_privileged_dim,
    )
    obs = {
        key: torch.as_tensor(value, dtype=torch.float32, device=args.device)
        for key, value in obs_np.items()
    }

    obs = {key: value[1:] for key, value in obs.items()}

    with torch.inference_mode():
        z = tracking_inference_no_delay(model, obs).cpu().numpy().astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(z, args.output)
    print(f"Saved {args.output}")
    print(f"z_shape={z.shape}")
    print(f"fps={save_data['fps']}")


if __name__ == "__main__":
    main()
