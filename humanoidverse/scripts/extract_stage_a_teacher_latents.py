from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro

from humanoidverse.language.stage_a.teacher import TeacherExtractionConfig, extract_teacher_latents


@dataclass
class Args:
    checkpoint_dir: Path = Path("checkpoint")
    motion_file: Path = Path("artifacts/stage_a/motionlib/seed_stage_a_00000.pkl")
    output_path: Path = Path("artifacts/stage_a/latents/seed_stage_a_00000.joblib")
    device: str = "cuda"
    simulator: str = "mujoco"
    num_chunks: int = 8
    motions_per_load: int = 64
    save_z_seq: bool = False
    use_root_height_obs: bool = True
    quiet: bool = True


def main(args: Args) -> None:
    extract_teacher_latents(
        TeacherExtractionConfig(
            checkpoint_dir=args.checkpoint_dir,
            motion_file=args.motion_file,
            output_path=args.output_path,
            device=args.device,
            simulator=args.simulator,
            num_chunks=args.num_chunks,
            motions_per_load=args.motions_per_load,
            save_z_seq=args.save_z_seq,
            use_root_height_obs=args.use_root_height_obs,
            quiet=args.quiet,
        )
    )
    print(f"Saved teacher latents to {args.output_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
