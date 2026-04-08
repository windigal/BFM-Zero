from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import tyro
from torch.utils._pytree import tree_map

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.language.stage_a.teacher import build_teacher_env
from humanoidverse.scripts.convert_lafan1_csv_to_motion import build_motion_entry_from_lafan1_g1
from humanoidverse.utils.helpers import get_backward_observation


@dataclass
class Args:
    csv_path: Path
    checkpoint_dir: Path = Path("checkpoint")
    mjcf_path: Path = Path("humanoidverse/data/robots/g1/g1_29dof.xml")
    motion_output: Path = Path("artifacts/lafan1/lafan1_motion.pkl")
    latent_output: Path = Path("~/code/BFM-zero-deploy/model/tracking_inference/zs_lafan1.pkl").expanduser()
    meta_output: Path | None = None
    motion_name: str | None = None
    start_frame: int = 0
    end_frame: int | None = None
    stride: int = 1
    device: str = "cuda"
    simulator: str = "mujoco"
    use_root_height_obs: bool = True
    use_contact_in_obs_max: bool = False


def main(args: Args) -> None:
    motion_name = args.motion_name or args.csv_path.stem
    motion_entry = build_motion_entry_from_lafan1_g1(
        csv_path=args.csv_path,
        mjcf_path=args.mjcf_path,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        stride=args.stride,
    )
    motion_dict = {motion_name: motion_entry}
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
        "csv_path": str(args.csv_path.resolve()),
        "motion_key": motion_key,
        "motion_output": str(args.motion_output.resolve()),
        "latent_output": str(args.latent_output.resolve()),
        "start_frame": int(args.start_frame),
        "end_frame": None if args.end_frame is None else int(args.end_frame),
        "stride": int(args.stride),
        "seq_len": int(z_seq.shape[0]),
        "z_dim": int(z_seq.shape[1]),
        "target_fps": int(motion_entry["fps"]),
        "use_root_height_obs": args.use_root_height_obs,
        "use_contact_in_obs_max": args.use_contact_in_obs_max,
        "inference_mode": "no_delay",
    }
    meta_output.parent.mkdir(parents=True, exist_ok=True)
    meta_output.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved motion file to {args.motion_output}")
    print(f"Saved tracking z sequence to {args.latent_output}")
    print(f"Saved metadata to {meta_output}")
    print(f"motion_key={motion_key}")
    print(f"z_shape={tuple(z_seq.shape)}")


if __name__ == "__main__":
    main(tyro.cli(Args))
