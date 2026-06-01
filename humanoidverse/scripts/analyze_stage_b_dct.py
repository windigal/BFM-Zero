from __future__ import annotations

import glob
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import tyro
from torch.utils._pytree import tree_map

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.language.stage_a.dataset import load_latent_store, load_manifest_records
from humanoidverse.language.stage_a.teacher import build_teacher_env, quiet_extraction, tracking_inference_fast
from humanoidverse.language.stage_b.frequency import DCTFutureCodec
from humanoidverse.utils.helpers import get_backward_observation


@dataclass
class Args:
    manifest_path: Path = Path("artifacts/stage_a/seed_manifest.jsonl")
    latent_paths: list[Path] | None = None
    latent_glob: str = "artifacts/stage_a/latents/*.joblib"
    checkpoint_dir: Path | None = None
    motion_glob: str | None = None
    device: str = "cuda"
    simulator: str = "mujoco"
    motions_per_load: int = 64
    use_root_height_obs: bool = True
    robot_override: str = "g1/g1_29dof"
    quiet: bool = True
    split: str = "train"
    sample_types: tuple[str, ...] = ("clip",)
    future_len: int = 16
    window_stride: int = 16
    max_sequences: int = 2000
    max_windows: int = 50000
    max_motion_files: int | None = None
    max_motion_keys_per_file: int | None = None
    seed: int = 42
    output_json: Path | None = None


def _resolve_latent_paths(args: Args) -> list[Path]:
    if args.latent_paths:
        return args.latent_paths
    return sorted(Path(path) for path in glob.glob(args.latent_glob))


def _filtered_manifest_records(args: Args) -> list[dict]:
    return [
        record
        for record in load_manifest_records(args.manifest_path)
        if record.get("split") == args.split and (not args.sample_types or record.get("sample_type") in args.sample_types)
    ]


def _iter_z_sequences_from_latents(args: Args, records: list[dict], rng: random.Random) -> Iterator[torch.Tensor]:
    latent_store = load_latent_store(_resolve_latent_paths(args))
    shuffled_records = list(records)
    rng.shuffle(shuffled_records)
    for record in shuffled_records:
        latent = latent_store.get(str(record["sample_id"]))
        if latent is None or "z_seq" not in latent:
            continue
        z_seq = torch.as_tensor(latent["z_seq"], dtype=torch.float32)
        if z_seq.ndim == 2:
            yield z_seq


def _iter_z_sequences_from_motionlib(args: Args, records: list[dict]) -> Iterator[torch.Tensor]:
    if args.checkpoint_dir is None or not args.motion_glob:
        return

    manifest_lookup = {str(record["sample_id"]): record for record in records}
    motion_files = sorted(Path(path) for path in glob.glob(args.motion_glob))
    if args.max_motion_files is not None:
        motion_files = motion_files[: args.max_motion_files]
    if len(motion_files) > 1:
        raise ValueError(
            "Motionlib-backed DCT analysis currently supports one motion file per run because "
            "HumanoidVerse cannot be reinitialized with multiple motionlib configs inside one Python process. "
            "Use --max-motion-files 1, or build a temporary raw primitive dataset subset and analyze that."
        )

    with quiet_extraction(args.quiet):
        model = load_model_from_checkpoint_dir(args.checkpoint_dir, device=args.device)
        model.to(args.device)
        model.eval()
        with torch.inference_mode():
            for motion_file in motion_files:
                wrapped_env = build_teacher_env(
                    motion_file=motion_file,
                    device=args.device,
                    simulator=args.simulator,
                    use_root_height_obs=args.use_root_height_obs,
                    use_contact_in_obs_max=False,
                    robot_override=args.robot_override,
                    hydra_overrides=[],
                )
                env = wrapped_env._env
                motion_keys = [str(key) for key in env._motion_lib._motion_data_keys.tolist()]
                if args.max_motion_keys_per_file is not None:
                    motion_keys = motion_keys[: args.max_motion_keys_per_file]

                motions_per_load = max(int(args.motions_per_load), 1)
                for batch_start in range(0, len(motion_keys), motions_per_load):
                    batch_size = min(motions_per_load, len(motion_keys) - batch_start)
                    env._motion_lib.load_motions(random_sample=False, start_idx=batch_start, num_motions_to_load=batch_size)
                    current_keys = [str(key) for key in env._motion_lib.curr_motion_keys]

                    for local_motion_id, motion_key in enumerate(current_keys):
                        if motion_key not in manifest_lookup:
                            continue
                        obs, _ = get_backward_observation(
                            env,
                            local_motion_id,
                            use_root_height_obs=args.use_root_height_obs,
                        )
                        obs = tree_map(lambda value: value[1:], obs)
                        z_seq = tracking_inference_fast(model, obs).detach().cpu()
                        if z_seq.ndim == 2:
                            yield z_seq


def _iter_window_starts(seq_len: int, future_len: int, window_stride: int, rng: random.Random) -> list[int]:
    starts = list(range(0, seq_len - future_len + 1, max(window_stride, 1)))
    rng.shuffle(starts)
    return starts


def main(args: Args) -> None:
    rng = random.Random(args.seed)
    records = _filtered_manifest_records(args)
    codec_full = DCTFutureCodec(future_len=args.future_len, keep_coeffs=args.future_len)
    codecs = [DCTFutureCodec(future_len=args.future_len, keep_coeffs=k) for k in range(1, args.future_len + 1)]

    coeff_energy = torch.zeros(args.future_len, dtype=torch.float64)
    mae_sum = torch.zeros(args.future_len, dtype=torch.float64)
    rmse_sum = torch.zeros(args.future_len, dtype=torch.float64)
    used_sequences = 0
    used_windows = 0

    if args.checkpoint_dir is not None and args.motion_glob:
        sequence_iter = _iter_z_sequences_from_motionlib(args, records)
        source = "motionlib"
    else:
        sequence_iter = _iter_z_sequences_from_latents(args, records, rng)
        source = "latents"

    max_windows_per_sequence = max(1, args.max_windows // max(args.max_sequences, 1))
    for z_seq in sequence_iter:
        if used_sequences >= args.max_sequences or used_windows >= args.max_windows:
            break
        if z_seq.shape[0] < args.future_len:
            continue

        starts = _iter_window_starts(int(z_seq.shape[0]), args.future_len, args.window_stride, rng)
        for start in starts[:max_windows_per_sequence]:
            future = z_seq[start : start + args.future_len]
            coeffs = codec_full.encode(future)
            coeff_energy += coeffs.square().mean(dim=-1).to(torch.float64)
            for k, codec in enumerate(codecs, start=1):
                recon = codec.decode(coeffs[:k])
                diff = recon - future
                mae_sum[k - 1] += diff.abs().mean().to(torch.float64)
                rmse_sum[k - 1] += diff.square().mean().sqrt().to(torch.float64)
            used_windows += 1
            if used_windows >= args.max_windows:
                break
        used_sequences += 1

    if used_windows == 0:
        raise RuntimeError(
            "No future windows were found. Use z_seq-enabled latents or pass checkpoint_dir + motion_glob "
            "to analyze DCT statistics directly from motionlib."
        )

    coeff_energy /= used_windows
    energy_ratio = coeff_energy / coeff_energy.sum().clamp(min=1e-12)
    cumulative_energy = energy_ratio.cumsum(dim=0)
    mae = mae_sum / used_windows
    rmse = rmse_sum / used_windows

    def _first_k(threshold: float) -> int:
        for idx, value in enumerate(cumulative_energy.tolist(), start=1):
            if value >= threshold:
                return idx
        return args.future_len

    report = {
        "source": source,
        "split": args.split,
        "future_len": args.future_len,
        "window_stride": args.window_stride,
        "used_sequences": used_sequences,
        "used_windows": used_windows,
        "energy_ratio": [float(x) for x in energy_ratio],
        "cumulative_energy": [float(x) for x in cumulative_energy],
        "mae_by_keep_coeffs": [float(x) for x in mae],
        "rmse_by_keep_coeffs": [float(x) for x in rmse],
        "recommended_keep_coeffs": {
            "energy_90": _first_k(0.90),
            "energy_95": _first_k(0.95),
            "energy_99": _first_k(0.99),
        },
        "compression_ratio_by_keep_coeffs": {
            str(k): float(args.future_len / k) for k in range(1, args.future_len + 1)
        },
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(tyro.cli(Args))
