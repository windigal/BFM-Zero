from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro


@dataclass
class Args:
    gmr_python: Path = Path("/home/hanwei/miniforge3/envs/gmr/bin/python")
    textop_root: Path = Path("~/code/TextOp").expanduser()
    babel_dir: Path = Path("~/dataset/babel_v1-0_release/babel_v1.0_release").expanduser()
    amass_hf_dir: Path = Path("~/dataset/AMASS").expanduser()
    amass_babel_smplx_dir: Path = Path("~/dataset/AMASS_babel_smplx").expanduser()
    amass_robot_dir: Path = Path("~/dataset/AMASS_robot_g1_29dof_30fps").expanduser()
    amass_robot_50fps_dir: Path = Path("~/dataset/AMASS_robot_g1_29dof_50fps").expanduser()
    textop_pack_dir: Path = Path("~/code/TextOp/dataset/BABEL-AMASS-ROBOT-29dof-50fps-TEACH").expanduser()
    output_dir: Path = Path("artifacts/textop_babel_h2_f8_50fps").expanduser()
    robot_name: str = "unitree_g1"
    robot_config: Path = Path("~/code/TextOp/TextOpRobotMDAR/robotmdar/config/skeleton/g1_29dof.yaml").expanduser()
    overwrite_output: bool = False


def _run(cmd: list[str], *, cwd: Path) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd))
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def _check_smplx_models(textop_root: Path) -> None:
    body_root = textop_root / "assets" / "body_models" / "smplx"
    required = [
        body_root / "SMPLX_NEUTRAL.pkl",
        body_root / "SMPLX_MALE.pkl",
        body_root / "SMPLX_FEMALE.pkl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing SMPL-X body models required by TextOp/GMR retargeting. "
            f"Expected files under {body_root}.\nMissing:\n" + "\n".join(missing)
        )


def main(args: Args) -> None:
    if not args.gmr_python.exists():
        raise FileNotFoundError(f"Missing python interpreter: {args.gmr_python}")
    if not args.textop_root.exists():
        raise FileNotFoundError(f"Missing TextOp repo: {args.textop_root}")
    _check_smplx_models(args.textop_root)

    if args.overwrite_output and args.output_dir.exists():
        for child in args.output_dir.iterdir():
            if child.is_dir():
                for nested in child.rglob("*"):
                    if nested.is_file():
                        nested.unlink()
                for nested in sorted(child.rglob("*"), reverse=True):
                    if nested.is_dir():
                        nested.rmdir()
                child.rmdir()
            else:
                child.unlink()

    _run(
        [
            str(args.gmr_python),
            "-m",
            "humanoidverse.scripts.extract_babel_required_amass_raw",
            "--babel-dir",
            str(args.babel_dir),
            "--amass-hf-dir",
            str(args.amass_hf_dir),
            "--output-dir",
            str(args.amass_babel_smplx_dir),
        ],
        cwd=Path("/home/hanwei/code/BFM-Zero"),
    )

    _run(
        [
            str(args.gmr_python),
            "dataset/smplx_to_robot_dataset.py",
            "--src_folder",
            str(args.amass_babel_smplx_dir),
            "--tgt_folder",
            str(args.amass_robot_dir),
            "--robot",
            args.robot_name,
        ],
        cwd=args.textop_root,
    )

    _run(
        [
            str(args.gmr_python),
            "dataset/process_retarget_data.py",
            "--input_dir",
            str(args.amass_robot_dir),
            "--output_dir",
            str(args.amass_robot_50fps_dir),
            "--robot_config",
            str(args.robot_config),
            "--dof_layout",
            "full",
        ],
        cwd=args.textop_root,
    )

    _run(
        [
            str(args.gmr_python),
            "dataset/pack_dataset.py",
            "--amass_robot",
            str(args.amass_robot_50fps_dir),
            "--babel",
            str(args.babel_dir),
            "--output_dir",
            str(args.textop_pack_dir),
        ],
        cwd=args.textop_root,
    )

    _run(
        [
            str(args.gmr_python),
            "-m",
            "humanoidverse.scripts.build_textop_babel_dataset",
            "--amass-robot-dir",
            str(args.amass_robot_50fps_dir),
            "--babel-dir",
            str(args.babel_dir),
            "--output-dir",
            str(args.output_dir),
            "--overwrite-output",
        ],
        cwd=Path("/home/hanwei/code/BFM-Zero"),
    )

    summary = {
        "textop_root": str(args.textop_root.resolve()),
        "babel_dir": str(args.babel_dir.resolve()),
        "amass_hf_dir": str(args.amass_hf_dir.resolve()),
        "amass_babel_smplx_dir": str(args.amass_babel_smplx_dir.resolve()),
        "amass_robot_dir": str(args.amass_robot_dir.resolve()),
        "amass_robot_50fps_dir": str(args.amass_robot_50fps_dir.resolve()),
        "textop_pack_dir": str(args.textop_pack_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(tyro.cli(Args))
