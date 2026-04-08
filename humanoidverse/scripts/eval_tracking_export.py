from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
import tyro
from scipy.spatial.transform import Rotation

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.language.stage_a.teacher import build_teacher_env
from humanoidverse.utils.helpers import get_backward_observation


def _quat_angle_deg(q1_xyzw: np.ndarray, q2_xyzw: np.ndarray) -> np.ndarray:
    delta = Rotation.from_quat(q1_xyzw.reshape(-1, 4)) * Rotation.from_quat(q2_xyzw.reshape(-1, 4)).inv()
    return np.rad2deg(np.linalg.norm(delta.as_rotvec(), axis=-1)).reshape(q1_xyzw.shape[:-1])


@dataclass
class Args:
    motion_file: Path
    latent_path: Path
    checkpoint_dir: Path = Path("checkpoint")
    output_json: Path | None = None
    device: str = "cuda"
    simulator: str = "mujoco"
    use_root_height_obs: bool = True


def main(args: Args) -> None:
    model = load_model_from_checkpoint_dir(args.checkpoint_dir, device=args.device)
    model.to(args.device)
    model.eval()

    wrapped_env = build_teacher_env(
        motion_file=args.motion_file,
        device=args.device,
        simulator=args.simulator,
        use_root_height_obs=args.use_root_height_obs,
        robot_override="g1/g1_29dof",
        hydra_overrides=[],
        include_history_actor=True,
    )
    env = wrapped_env._env
    num_envs = 1

    z_seq = joblib.load(args.latent_path)
    z_seq_t = torch.as_tensor(z_seq, dtype=torch.float32, device=args.device)

    obs, obs_dict = get_backward_observation(env, 0, use_root_height_obs=args.use_root_height_obs)

    ref_root_rot = obs_dict["ref_body_rots"][0, 0].clone()
    ref_root_init_state = torch.cat(
        [
            obs_dict["ref_body_pos"][0, 0],
            ref_root_rot,
            obs_dict["ref_body_vels"][0, 0],
            obs_dict["ref_body_angular_vels"][0, 0],
        ]
    )
    dof_init_state = torch.zeros_like(wrapped_env._env.simulator.dof_state.view(num_envs, -1, 2)[0])
    dof_init_state[..., 0] = obs_dict["dof_pos"][0]
    dof_init_state[..., 1] = obs_dict["ref_dof_vel"][0]
    target_states = {
        "dof_states": dof_init_state,
        "root_states": torch.stack([ref_root_init_state.clone() for _ in range(num_envs)]),
    }

    env_ids = torch.arange(num_envs, dtype=torch.long)
    wrapped_env.reset(to_numpy=False)
    wrapped_env._env.reset_envs_idx(env_ids, target_states=target_states)
    wrapped_env.step(torch.zeros((num_envs, wrapped_env.action_space.shape[-1]), dtype=torch.float32, device=args.device), to_numpy=False)
    observation = wrapped_env._get_g1env_observation(to_numpy=False)

    rollout_root_pos = []
    rollout_root_rot = []
    rollout_dof = []
    terminated_steps = []
    truncated_steps = []

    for i in range(len(z_seq_t)):
        action = model.act(observation, z_seq_t[i].repeat(num_envs, 1), mean=True)
        observation, reward, terminated, truncated, info = wrapped_env.step(action, to_numpy=False)
        rollout_root_pos.append(wrapped_env._env.simulator.robot_root_states[0, :3].detach().cpu().numpy().copy())
        rollout_root_rot.append(wrapped_env._env.simulator.robot_root_states[0, 3:7].detach().cpu().numpy().copy())
        rollout_dof.append(wrapped_env._env.simulator.dof_state[:, :, 0][0].detach().cpu().numpy().copy())
        terminated_steps.append(bool(terminated[0].item()))
        truncated_steps.append(bool(truncated[0].item()))

    rollout_root_pos = np.asarray(rollout_root_pos, dtype=np.float32)
    rollout_root_rot = np.asarray(rollout_root_rot, dtype=np.float32)
    rollout_dof = np.asarray(rollout_dof, dtype=np.float32)

    ref_root_pos = obs_dict["ref_body_pos"][1 : 1 + len(z_seq_t), 0].detach().cpu().numpy()
    ref_root_rot = obs_dict["ref_body_rots"][1 : 1 + len(z_seq_t), 0].detach().cpu().numpy()
    ref_dof = obs_dict["dof_pos"][1 : 1 + len(z_seq_t)].detach().cpu().numpy()

    metrics = {
        "motion_file": str(args.motion_file.resolve()),
        "latent_path": str(args.latent_path.resolve()),
        "num_steps": int(len(z_seq_t)),
        "root_pos_rmse_m": float(np.sqrt(np.mean((rollout_root_pos - ref_root_pos) ** 2))),
        "root_height_rmse_m": float(np.sqrt(np.mean((rollout_root_pos[:, 2] - ref_root_pos[:, 2]) ** 2))),
        "dof_rmse_rad": float(np.sqrt(np.mean((rollout_dof - ref_dof) ** 2))),
        "root_rot_mean_deg": float(_quat_angle_deg(rollout_root_rot, ref_root_rot).mean()),
        "final_root_height_error_m": float(rollout_root_pos[-1, 2] - ref_root_pos[-1, 2]),
        "rollout_root_height_span_m": float(rollout_root_pos[:, 2].ptp()),
        "reference_root_height_span_m": float(ref_root_pos[:, 2].ptp()),
        "terminated_any": bool(any(terminated_steps)),
        "truncated_any": bool(any(truncated_steps)),
        "terminated_count": int(sum(terminated_steps)),
        "truncated_count": int(sum(truncated_steps)),
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved metrics to {args.output_json}")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(tyro.cli(Args))
