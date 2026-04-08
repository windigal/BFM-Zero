from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import tyro
from torch.utils._pytree import tree_map

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.language.stage_a.seed import (
    load_seed_metadata,
    manifest_records_to_motion_dict,
    resolve_g1_csv_path,
)
from humanoidverse.language.stage_a.teacher import build_teacher_env
from humanoidverse.utils.helpers import get_backward_observation


@dataclass
class Args:
    filename: str = "jump_twice_R_001__A533"
    dataset_root: Path = Path("~/dataset/seed").expanduser()
    contact_dataset_root: Path = Path("~/dataset/LAFAN1/g1_seed").expanduser()
    metadata_csv: Path = Path("~/dataset/seed/metadata/seed_metadata_v003.csv").expanduser()
    checkpoint_dir: Path = Path("checkpoint")
    mjcf_path: Path = Path("humanoidverse/data/robots/g1/g1_29dof.xml")
    motion_output: Path = Path("artifacts/stage_a/debug/seed_tracking_motion.pkl")
    latent_output: Path = Path("~/code/BFM-zero-deploy/model/tracking_inference/zs_seed_clip.pkl").expanduser()
    meta_output: Path | None = None
    device: str = "cuda"
    simulator: str = "mujoco"
    target_fps: int = 30
    use_root_height_obs: bool = True
    use_contact_in_obs_max: bool = False
    include_foot_contact_binary: bool = False
    root_euler_order: str = "xyz"


def _find_row(rows: list[dict[str, str]], filename: str) -> dict[str, str]:
    for row in rows:
        if row.get("filename") == filename:
            return row
    raise ValueError(f"Could not find filename={filename} in metadata")


def main(args: Args) -> None:
    rows = load_seed_metadata(args.metadata_csv)
    row = _find_row(rows, args.filename)
    motion_csv_path = resolve_g1_csv_path(args.dataset_root, row)

    record = {
        "sample_id": f"clip::{args.filename}",
        "motion_csv_path": str(motion_csv_path),
        "start_frame": 0,
        "end_frame": int(row["move_duration_frames"]),
    }
    motion_dict = manifest_records_to_motion_dict(
        [record],
        mjcf_path=args.mjcf_path,
        target_fps=args.target_fps,
        root_euler_order=args.root_euler_order,
        include_foot_contact_binary=args.include_foot_contact_binary,
        contact_dataset_root=args.contact_dataset_root,
    )
    if len(motion_dict) != 1:
        raise RuntimeError(f"Expected one motion entry, got {len(motion_dict)}")

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
        robot_override="g1/g1_29dof",
        hydra_overrides=[],
    )
    env = wrapped_env._env
    motion_key = str(env._motion_lib._motion_data_keys.tolist()[0])
    obs, _ = get_backward_observation(env, 0, use_root_height_obs=args.use_root_height_obs)
    obs = tree_map(lambda x: x[1:], obs)
    z_seq = model.project_z(model.backward_map(obs)).cpu().numpy()

    args.latent_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(z_seq, args.latent_output)

    meta_output = args.meta_output or args.latent_output.with_suffix(".meta.json")
    meta = {
        "filename": args.filename,
        "motion_key": motion_key,
        "source_csv": str(motion_csv_path),
        "motion_output": str(args.motion_output.resolve()),
        "latent_output": str(args.latent_output.resolve()),
        "seq_len": int(z_seq.shape[0]),
        "z_dim": int(z_seq.shape[1]),
        "target_fps": int(motion_dict[motion_key]["fps"]),
        "move_duration_frames": int(row["move_duration_frames"]),
        "content_short_description": row.get("content_short_description", ""),
        "content_natural_desc_1": row.get("content_natural_desc_1", ""),
        "content_technical_description": row.get("content_technical_description", ""),
        "use_root_height_obs": args.use_root_height_obs,
        "use_contact_in_obs_max": args.use_contact_in_obs_max,
        "include_foot_contact_binary": args.include_foot_contact_binary,
        "contact_dataset_root": str(args.contact_dataset_root.resolve()),
        "root_euler_order": args.root_euler_order,
        "inference_mode": "no_delay",
    }
    meta_output.parent.mkdir(parents=True, exist_ok=True)
    with meta_output.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved motion file to {args.motion_output}")
    print(f"Saved tracking z sequence to {args.latent_output}")
    print(f"Saved metadata to {meta_output}")
    print(f"motion_key={motion_key}")
    print(f"z_shape={tuple(z_seq.shape)}")


if __name__ == "__main__":
    main(tyro.cli(Args))
