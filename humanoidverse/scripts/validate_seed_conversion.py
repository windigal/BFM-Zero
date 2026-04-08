from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import tyro
from easydict import EasyDict
from scipy.spatial.transform import Rotation

from humanoidverse.language.stage_a.seed import (
    VALID_ROOT_EULER_ORDERS,
    load_seed_g1_rows,
    load_seed_metadata,
    parse_joint_axes,
    resolve_g1_csv_path,
    seed_rows_to_motion_entry,
)
from humanoidverse.pico_tracking_inference import compute_body_kinematics, wxyz_to_xyzw
from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch


@dataclass
class Args:
    filename: str = "Neutral_walk_forward_002__A057"
    dataset_root: Path = Path("~/dataset/seed").expanduser()
    metadata_csv: Path = Path("~/dataset/seed/metadata/seed_metadata_v003.csv").expanduser()
    mjcf_path: Path = Path("humanoidverse/data/robots/g1/g1_29dof.xml")
    output_dir: Path = Path("artifacts/stage_a/validation")
    target_fps: int = 30
    simulator: str = "mujoco"
    device: str = "cuda"
    root_euler_orders: tuple[str, ...] = VALID_ROOT_EULER_ORDERS
    export_motion_pkls: bool = True


def _find_row(rows: list[dict[str, str]], filename: str) -> dict[str, str]:
    for row in rows:
        if row.get("filename") == filename:
            return row
    raise ValueError(f"Could not find filename={filename} in metadata")


def _xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)
    return quat_xyzw[..., [3, 0, 1, 2]]


def _quat_angle_deg(q1_xyzw: np.ndarray, q2_xyzw: np.ndarray) -> np.ndarray:
    delta = Rotation.from_quat(q1_xyzw.reshape(-1, 4)) * Rotation.from_quat(q2_xyzw.reshape(-1, 4)).inv()
    return np.rad2deg(np.linalg.norm(delta.as_rotvec(), axis=-1)).reshape(q1_xyzw.shape[:-1])


def _heading_velocity_error_deg(root_pos: np.ndarray, root_quat_xyzw: np.ndarray) -> float:
    horiz_vel = np.gradient(root_pos[:, :2], axis=0)
    speed = np.linalg.norm(horiz_vel, axis=-1)
    valid = speed > 1e-4
    if not np.any(valid):
        return float("nan")
    heading_vec = Rotation.from_quat(root_quat_xyzw).apply(np.tile(np.array([1.0, 0.0, 0.0]), (len(root_quat_xyzw), 1)))
    heading_xy = heading_vec[:, :2]
    heading_xy /= np.linalg.norm(heading_xy, axis=-1, keepdims=True) + 1e-8
    vel_xy = horiz_vel / (speed[:, None] + 1e-8)
    cos = np.clip(np.sum(heading_xy * vel_xy, axis=-1), -1.0, 1.0)
    angles = np.rad2deg(np.arccos(cos))
    return float(angles[valid].mean())


def _upright_tilt_deg(root_quat_xyzw: np.ndarray) -> float:
    up = Rotation.from_quat(root_quat_xyzw).apply(np.tile(np.array([0.0, 0.0, 1.0]), (len(root_quat_xyzw), 1)))
    cos = np.clip(up[:, 2], -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos)).mean())


def _build_motion_dict(
    csv_path: Path,
    mjcf_path: Path,
    start_frame: int,
    end_frame: int,
    target_fps: int,
    root_euler_order: str,
) -> dict[str, dict[str, Any]]:
    stride = max(1, round(120 / target_fps))
    target_fps = int(120 / stride)
    joint_axes = parse_joint_axes(mjcf_path)
    rows = load_seed_g1_rows(csv_path=csv_path, start_frame=start_frame, end_frame=end_frame, stride=stride)
    motion_entry = seed_rows_to_motion_entry(
        rows=rows,
        joint_axes=joint_axes,
        target_fps=target_fps,
        root_euler_order=root_euler_order,
    )
    return {f"clip::{csv_path.stem}": motion_entry}


def _build_humanoid_batch(mjcf_path: Path) -> Humanoid_Batch:
    cfg = EasyDict(
        asset=EasyDict(
            assetRoot=str(mjcf_path.resolve().parent),
            assetFileName=mjcf_path.name,
        ),
        extend_config=[
            EasyDict(
                joint_name="head_link",
                parent_name="torso_link",
                pos=[0.0, 0.0, 0.35],
                rot=[1.0, 0.0, 0.0, 0.0],
            )
        ],
    )
    return Humanoid_Batch(cfg)


def _collect_motionlib_metrics(
    fk_model: Humanoid_Batch,
    motion_entry: dict[str, Any],
    mjcf_path: Path,
) -> dict[str, float]:
    pose_aa = np.asarray(motion_entry["pose_aa"], dtype=np.float32)
    trans = np.asarray(motion_entry["root_trans_offset"], dtype=np.float32)
    motion = fk_model.fk_batch(
        pose=torch.from_numpy(pose_aa[None, ...]),
        trans=torch.from_numpy(trans[None, ...]),
        return_full=True,
        dt=1.0 / float(motion_entry["fps"]),
    )

    hv_body_pos = motion.global_translation.squeeze(0).detach().cpu().numpy()
    hv_body_rot = motion.global_rotation.squeeze(0).detach().cpu().numpy()
    hv_dof = motion.dof_pos.squeeze(0).detach().cpu().numpy()

    mujoco_body_pos, mujoco_body_rot_wxyz = compute_body_kinematics(
        root_pos=np.asarray(motion_entry["root_trans_offset"], dtype=np.float32),
        root_rot_wxyz=_xyzw_to_wxyz(np.asarray(motion_entry["root_rot"], dtype=np.float32)),
        dof_pos=np.asarray(motion_entry["dof"], dtype=np.float32),
        robot_xml_path=mjcf_path,
    )
    mujoco_body_rot = wxyz_to_xyzw(mujoco_body_rot_wxyz)

    body_count = min(hv_body_pos.shape[1], mujoco_body_pos.shape[1])
    hv_local = hv_body_pos[:, :body_count] - hv_body_pos[:, :1]
    mj_local = mujoco_body_pos[:, :body_count] - mujoco_body_pos[:, :1]

    pos_rmse = float(np.sqrt(np.mean((hv_local - mj_local) ** 2)))
    rot_err = float(_quat_angle_deg(hv_body_rot[:, :body_count], mujoco_body_rot[:, :body_count]).mean())
    dof_rmse = float(np.sqrt(np.mean((hv_dof - np.asarray(motion_entry["dof"], dtype=np.float32)) ** 2)))
    root_height_offset = float(np.mean(hv_body_pos[:, 0, 2] - mujoco_body_pos[:, 0, 2]))

    return {
        "motionlib_local_body_pos_rmse_m": pos_rmse,
        "motionlib_body_rot_mean_deg": rot_err,
        "motionlib_dof_rmse_rad": dof_rmse,
        "motionlib_root_height_offset_m": root_height_offset,
    }


def _semantic_metrics(motion_entry: dict[str, Any]) -> dict[str, float]:
    root_pos = np.asarray(motion_entry["root_trans_offset"], dtype=np.float32)
    root_quat = np.asarray(motion_entry["root_rot"], dtype=np.float32)
    horiz_disp = float(np.linalg.norm(root_pos[-1, :2] - root_pos[0, :2]))
    return {
        "heading_velocity_error_deg": _heading_velocity_error_deg(root_pos, root_quat),
        "upright_tilt_deg": _upright_tilt_deg(root_quat),
        "horizontal_displacement_m": horiz_disp,
        "root_height_mean_m": float(root_pos[:, 2].mean()),
    }


def _semantic_score(metrics: dict[str, float]) -> float:
    return float(metrics["heading_velocity_error_deg"] + 0.5 * metrics["upright_tilt_deg"])


def main(args: Args) -> None:
    rows = load_seed_metadata(args.metadata_csv)
    row = _find_row(rows, args.filename)
    csv_path = resolve_g1_csv_path(args.dataset_root, row)

    sample_output_dir = args.output_dir / args.filename
    motion_dir = sample_output_dir / "motions"
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    if args.export_motion_pkls:
        motion_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    fk_model = _build_humanoid_batch(args.mjcf_path)
    for order in args.root_euler_orders:
        motion_dict = _build_motion_dict(
            csv_path=csv_path,
            mjcf_path=args.mjcf_path,
            start_frame=0,
            end_frame=int(row["move_duration_frames"]),
            target_fps=args.target_fps,
            root_euler_order=order,
        )
        motion_key = next(iter(motion_dict))
        motion_entry = motion_dict[motion_key]

        motion_file = motion_dir / f"{args.filename}__{order}.pkl"
        if args.export_motion_pkls:
            joblib.dump(motion_dict, motion_file)
        else:
            motion_file = sample_output_dir / f"__tmp_{args.filename}__{order}.pkl"
            joblib.dump(motion_dict, motion_file)

        metrics = {
            **_collect_motionlib_metrics(
                fk_model=fk_model,
                motion_entry=motion_entry,
                mjcf_path=args.mjcf_path,
            ),
            **_semantic_metrics(motion_entry),
        }
        metrics["semantic_score"] = _semantic_score(metrics)
        results.append(
            {
                "root_euler_order": order,
                "motion_file": str(motion_file.resolve()),
                **metrics,
            }
        )

        if not args.export_motion_pkls:
            motion_file.unlink(missing_ok=True)

    results.sort(key=lambda item: item["semantic_score"])
    report = {
        "filename": args.filename,
        "source_csv": str(csv_path),
        "target_fps": args.target_fps,
        "results": results,
    }

    json_path = sample_output_dir / "root_euler_validation.json"
    md_path = sample_output_dir / "root_euler_validation.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# Root Euler Validation for `{args.filename}`",
        "",
        f"- Source CSV: `{csv_path}`",
        f"- Target FPS: `{args.target_fps}`",
        "",
        "| order | local body pos RMSE (m) | body rot err (deg) | dof RMSE (rad) | heading-vel err (deg) | upright tilt (deg) | semantic score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        lines.append(
            "| {order} | {pos:.5f} | {rot:.3f} | {dof:.5f} | {head:.3f} | {tilt:.3f} | {score:.3f} |".format(
                order=item["root_euler_order"],
                pos=item["motionlib_local_body_pos_rmse_m"],
                rot=item["motionlib_body_rot_mean_deg"],
                dof=item["motionlib_dof_rmse_rad"],
                head=item["heading_velocity_error_deg"],
                tilt=item["upright_tilt_deg"],
                score=item["semantic_score"],
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved JSON report to {json_path}")
    print(f"Saved Markdown report to {md_path}")
    print("Top candidates:")
    for item in results[:3]:
        print(
            f"  order={item['root_euler_order']} "
            f"score={item['semantic_score']:.3f} "
            f"heading_vel={item['heading_velocity_error_deg']:.3f} "
            f"tilt={item['upright_tilt_deg']:.3f}"
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
