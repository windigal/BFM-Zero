from __future__ import annotations

import contextlib
import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.utils.helpers import get_backward_observation


@dataclass(slots=True)
class TeacherExtractionConfig:
    checkpoint_dir: Path
    motion_file: Path
    output_path: Path
    device: str = "cuda"
    simulator: str = "mujoco"
    num_chunks: int = 8
    motions_per_load: int = 64
    save_z_seq: bool = False
    use_root_height_obs: bool = True
    robot_override: str = "g1/g1_29dof"
    quiet: bool = True
    hydra_overrides: list[str] = field(default_factory=list)


def _project_like_bfm(z: torch.Tensor) -> torch.Tensor:
    return (z.shape[-1] ** 0.5) * torch.nn.functional.normalize(z, dim=-1)


def pool_clip_latent(z_seq: torch.Tensor) -> torch.Tensor:
    return _project_like_bfm(z_seq.mean(dim=0, keepdim=True)).squeeze(0)


def chunk_latent_sequence(z_seq: torch.Tensor, num_chunks: int) -> torch.Tensor:
    if num_chunks <= 0:
        raise ValueError("num_chunks must be > 0")
    total_steps = z_seq.shape[0]
    if total_steps == 0:
        raise ValueError("Received empty z sequence")

    boundaries = np.linspace(0, total_steps, num_chunks + 1, dtype=np.int64)
    chunked = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        start = int(start)
        end = int(end)
        if end <= start:
            idx = min(start, total_steps - 1)
            chunk = z_seq[idx]
        else:
            chunk = z_seq[start:end].mean(dim=0)
        chunked.append(chunk)
    stacked = torch.stack(chunked, dim=0)
    return _project_like_bfm(stacked)


def tracking_inference_fast(model: Any, next_obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    z = model.backward_map(next_obs)
    seq_length = max(int(model.cfg.seq_length), 1)
    if seq_length > 1 and z.shape[0] > 1:
        cumsum = torch.cat(
            [torch.zeros((1, z.shape[-1]), device=z.device, dtype=z.dtype), torch.cumsum(z, dim=0)],
            dim=0,
        )
        starts = torch.arange(z.shape[0], device=z.device)
        ends = torch.clamp(starts + seq_length, max=z.shape[0])
        z = (cumsum[ends] - cumsum[starts]) / (ends - starts).unsqueeze(-1)
    return model.project_z(z)


@contextlib.contextmanager
def quiet_extraction(enabled: bool = True):
    if not enabled:
        yield
        return

    from loguru import logger as loguru_logger
    import humanoidverse.utils.motion_lib.motion_lib_base as motion_lib_base
    import humanoidverse.utils.motion_lib.torch_humanoid_batch as torch_humanoid_batch

    original_motion_track = motion_lib_base.track
    original_batch_track = torch_humanoid_batch.track
    motion_lib_base.track = lambda sequence, *args, **kwargs: sequence
    torch_humanoid_batch.track = lambda sequence, *args, **kwargs: sequence
    loguru_logger.disable("humanoidverse")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            yield
        finally:
            motion_lib_base.track = original_motion_track
            torch_humanoid_batch.track = original_batch_track
            loguru_logger.enable("humanoidverse")


def build_teacher_env(
    motion_file: Path,
    device: str,
    simulator: str,
    use_root_height_obs: bool,
    use_contact_in_obs_max: bool,
    robot_override: str,
    hydra_overrides: list[str],
    include_history_actor: bool = False,
):
    hydra_overrides = [
        f"simulator={simulator}",
        "env.config.headless=True",
        "env.config.max_episode_length_s=10000",
    ] + hydra_overrides
    if robot_override:
        hydra_overrides.append(f"robot={robot_override}")
    env_cfg = HumanoidVerseIsaacConfig(
        lafan_tail_path=str(motion_file.resolve()),
        device=device,
        disable_domain_randomization=True,
        disable_obs_noise=True,
        root_height_obs=use_root_height_obs,
        use_contact_in_obs_max=use_contact_in_obs_max,
        include_history_actor=include_history_actor,
        hydra_overrides=hydra_overrides,
    )
    wrapped_env, _ = env_cfg.build(1)
    return wrapped_env


def extract_teacher_latents(cfg: TeacherExtractionConfig) -> dict[str, Any]:
    with quiet_extraction(cfg.quiet):
        model = load_model_from_checkpoint_dir(cfg.checkpoint_dir, device=cfg.device)
        model.to(cfg.device)
        model.eval()

        wrapped_env = build_teacher_env(
            motion_file=cfg.motion_file,
            device=cfg.device,
            simulator=cfg.simulator,
            use_root_height_obs=cfg.use_root_height_obs,
            use_contact_in_obs_max=False,
            robot_override=cfg.robot_override,
            hydra_overrides=cfg.hydra_overrides,
        )
        env = wrapped_env._env

        motion_keys = [str(k) for k in env._motion_lib._motion_data_keys.tolist()]
        results: dict[str, Any] = {
            "__meta__": {
                **{k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
                "motion_keys": motion_keys,
                "z_dim": int(model.cfg.archi.z_dim),
            }
        }

        motions_per_load = max(int(cfg.motions_per_load), 1)
        with torch.inference_mode():
            for batch_start in range(0, len(motion_keys), motions_per_load):
                batch_size = min(motions_per_load, len(motion_keys) - batch_start)
                env._motion_lib.load_motions(random_sample=False, start_idx=batch_start, num_motions_to_load=batch_size)
                current_keys = [str(k) for k in env._motion_lib.curr_motion_keys]
                expected_keys = motion_keys[batch_start : batch_start + len(current_keys)]
                if current_keys != expected_keys:
                    raise RuntimeError(
                        f"Motion reload mismatch: expected {expected_keys[:3]}, got {current_keys[:3]}"
                    )

                for local_motion_id, motion_key in enumerate(current_keys):
                    obs, _ = get_backward_observation(env, local_motion_id, use_root_height_obs=cfg.use_root_height_obs)
                    obs = tree_map(lambda x: x[1:], obs)
                    z_seq = tracking_inference_fast(model, obs)
                    clip_z = pool_clip_latent(z_seq)
                    chunk_z = chunk_latent_sequence(z_seq, cfg.num_chunks)
                    motion_result: dict[str, Any] = {
                        "z_clip": clip_z.cpu().numpy().astype(np.float32),
                        "z_chunks": chunk_z.cpu().numpy().astype(np.float32),
                        "seq_len": int(z_seq.shape[0]),
                    }
                    if cfg.save_z_seq:
                        motion_result["z_seq"] = z_seq.cpu().numpy().astype(np.float32)
                    results[motion_key] = motion_result

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(results, cfg.output_path)
    meta_path = cfg.output_path.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(results["__meta__"], f, indent=2, default=str)
    return results
