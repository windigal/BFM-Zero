from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import tyro

from humanoidverse.language.stage_b.primitives import BFMTextOpPrimitiveExportConfig, export_bfm_textop_primitives


@dataclass
class Args:
    checkpoint_dir: Path = Path("/home/hanwei/code/BFM-Zero/results/bfmzero-isaac/20260402_170709/checkpoint")
    manifest_path: Path = Path("artifacts/stage_a/seed_manifest.jsonl")
    motion_glob: str = "artifacts/stage_a/motionlib/*.pkl"
    output_dir: Path = Path("artifacts/stage_b/primitives_seed")
    device: str = "cuda"
    simulator: str = "mujoco"
    motions_per_load: int = 64
    history_len: int = 4
    future_len: int = 16
    primitive_stride: int = 16
    sample_types: tuple[str, ...] = ("clip",)
    min_overlap_s: float = 1e-6
    rows_per_shard: int = 5000
    shard_format: str = "parquet"
    storage_dtype: str = "float16"
    target_representation: str = "dct"
    dct_keep_coeffs: int | None = 4
    use_root_height_obs: bool = True
    robot_override: str = "g1/g1_29dof"
    quiet: bool = True
    overwrite_output: bool = False
    max_motion_files: int | None = None
    max_motion_keys_per_file: int | None = None
    hf_repo_id: str | None = None
    hf_token: str | None = None
    hf_private: bool = False
    hf_path_in_repo: str = ""
    hf_commit_message: str | None = None
    hf_create_repo: bool = True
    hf_upload: bool = False
    hf_use_upload_large_folder: bool = False


def main(args: Args) -> None:
    summary = export_bfm_textop_primitives(
        BFMTextOpPrimitiveExportConfig(
            checkpoint_dir=args.checkpoint_dir,
            manifest_path=args.manifest_path,
            motion_glob=args.motion_glob,
            output_dir=args.output_dir,
            device=args.device,
            simulator=args.simulator,
            motions_per_load=args.motions_per_load,
            history_len=args.history_len,
            future_len=args.future_len,
            primitive_stride=args.primitive_stride,
            sample_types=args.sample_types,
            min_overlap_s=args.min_overlap_s,
            rows_per_shard=args.rows_per_shard,
            shard_format=args.shard_format,
            storage_dtype=args.storage_dtype,
            target_representation=args.target_representation,
            dct_keep_coeffs=args.dct_keep_coeffs,
            use_root_height_obs=args.use_root_height_obs,
            robot_override=args.robot_override,
            quiet=args.quiet,
            overwrite_output=args.overwrite_output,
            max_motion_files=args.max_motion_files,
            max_motion_keys_per_file=args.max_motion_keys_per_file,
            hf_repo_id=args.hf_repo_id,
            hf_token=args.hf_token,
            hf_private=args.hf_private,
            hf_path_in_repo=args.hf_path_in_repo,
            hf_commit_message=args.hf_commit_message,
            hf_create_repo=args.hf_create_repo,
            hf_upload=args.hf_upload,
            hf_use_upload_large_folder=args.hf_use_upload_large_folder,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(tyro.cli(Args))
