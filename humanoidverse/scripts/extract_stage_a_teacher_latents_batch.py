from __future__ import annotations

import glob
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro


@dataclass
class Args:
    checkpoint_dir: Path = Path("checkpoint")
    motion_glob: str = "artifacts/stage_a/motionlib/*.pkl"
    output_dir: Path = Path("artifacts/stage_a/latents")
    output_report: Path = Path("artifacts/stage_a/latents_report.json")
    device: str = "cuda"
    simulator: str = "mujoco"
    num_chunks: int = 8
    motions_per_load: int = 64
    save_z_seq: bool = False
    use_root_height_obs: bool = True
    robot_override: str = "g1/g1_29dof"
    skip_existing: bool = True
    stop_on_error: bool = True
    quiet: bool = True


def _run_single_shard(
    *,
    checkpoint_dir: Path,
    motion_file: Path,
    output_path: Path,
    device: str,
    simulator: str,
    num_chunks: int,
    motions_per_load: int,
    save_z_seq: bool,
    use_root_height_obs: bool,
    quiet: bool,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "humanoidverse.scripts.extract_stage_a_teacher_latents",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--motion-file",
        str(motion_file),
        "--output-path",
        str(output_path),
        "--device",
        device,
        "--simulator",
        simulator,
        "--num-chunks",
        str(num_chunks),
        "--motions-per-load",
        str(motions_per_load),
    ]
    if not save_z_seq:
        cmd.append("--no-save-z-seq")
    if not use_root_height_obs:
        cmd.append("--no-use-root-height-obs")
    if not quiet:
        cmd.append("--no-quiet")
    return subprocess.run(cmd, capture_output=True, text=True)


def main(args: Args) -> None:
    motion_files = sorted(Path(path) for path in glob.glob(args.motion_glob))
    if not motion_files:
        raise FileNotFoundError(f"No motion files matched motion_glob={args.motion_glob!r}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_rows: list[dict[str, object]] = []
    total_outputs = 0

    for motion_file in motion_files:
        output_path = args.output_dir / f"{motion_file.stem}.joblib"
        row = {
            "motion_file": str(motion_file.resolve()),
            "output_path": str(output_path.resolve()),
            "status": "pending",
        }
        if args.skip_existing and output_path.exists():
            row["status"] = "skipped_existing"
            report_rows.append(row)
            print(f"Skipping existing latent shard: {output_path}")
            continue

        proc = _run_single_shard(
            checkpoint_dir=args.checkpoint_dir,
            motion_file=motion_file,
            output_path=output_path,
            device=args.device,
            simulator=args.simulator,
            num_chunks=args.num_chunks,
            motions_per_load=args.motions_per_load,
            save_z_seq=args.save_z_seq,
            use_root_height_obs=args.use_root_height_obs,
            quiet=args.quiet,
        )
        row["returncode"] = proc.returncode
        if proc.stdout:
            row["stdout_tail"] = proc.stdout.strip().splitlines()[-10:]
            print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
        if proc.stderr:
            row["stderr_tail"] = proc.stderr.strip().splitlines()[-10:]
            print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n", file=sys.stderr)

        if proc.returncode != 0:
            row["status"] = "failed"
            report_rows.append(row)
            if args.stop_on_error:
                args.output_report.parent.mkdir(parents=True, exist_ok=True)
                args.output_report.write_text(
                    json.dumps(
                        {
                            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
                            "motion_glob": args.motion_glob,
                            "output_dir": str(args.output_dir.resolve()),
                            "num_motion_files": len(motion_files),
                            "num_outputs_written": total_outputs,
                            "rows": report_rows,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                raise RuntimeError(f"Latent extraction failed for shard {motion_file}")
            continue

        row["status"] = "ok"
        meta_path = output_path.with_suffix(".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            row["num_motion_keys"] = len(meta.get("motion_keys", []))
        report_rows.append(row)
        total_outputs += 1
        print(f"Saved latent shard to {output_path}")

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(
            {
                "checkpoint_dir": str(args.checkpoint_dir.resolve()),
                "motion_glob": args.motion_glob,
                "output_dir": str(args.output_dir.resolve()),
                "num_motion_files": len(motion_files),
                "num_outputs_written": total_outputs,
                "rows": report_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved batch extraction report to {args.output_report}")


if __name__ == "__main__":
    main(tyro.cli(Args))
