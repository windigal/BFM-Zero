from __future__ import annotations

import glob
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro

from humanoidverse.language.stage_b.primitives import upload_primitive_dataset, write_primitive_dataset_card


@dataclass
class Args:
    checkpoint_dir: Path = Path("/home/hanwei/code/BFM-Zero/results/bfmzero-isaac/20260402_170709/checkpoint")
    manifest_path: Path = Path("artifacts/stage_a/seed_manifest.jsonl")
    motion_glob: str = "artifacts/stage_a/motionlib/*.pkl"
    output_dir: Path = Path("artifacts/stage_b/primitives_seed_full")
    device: str = "cuda"
    simulator: str = "mujoco"
    motions_per_load: int = 64
    history_len: int = 4
    future_len: int = 16
    primitive_stride: int = 16
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
    hf_repo_id: str | None = None
    hf_token: str | None = None
    hf_private: bool = False
    hf_path_in_repo: str = ""
    hf_commit_message: str | None = None
    hf_create_repo: bool = True
    hf_upload: bool = False
    hf_use_upload_large_folder: bool = False


def _run_single_motion_export(args: Args, motion_file: Path, tmp_output: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "humanoidverse.scripts.build_stage_b_primitive_dataset",
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--manifest-path",
        str(args.manifest_path),
        "--motion-glob",
        str(motion_file),
        "--output-dir",
        str(tmp_output),
        "--device",
        args.device,
        "--simulator",
        args.simulator,
        "--motions-per-load",
        str(args.motions_per_load),
        "--history-len",
        str(args.history_len),
        "--future-len",
        str(args.future_len),
        "--primitive-stride",
        str(args.primitive_stride),
        "--rows-per-shard",
        str(args.rows_per_shard),
        "--shard-format",
        args.shard_format,
        "--storage-dtype",
        args.storage_dtype,
        "--target-representation",
        args.target_representation,
        "--robot-override",
        args.robot_override,
    ]
    if args.dct_keep_coeffs is not None:
        cmd.extend(["--dct-keep-coeffs", str(args.dct_keep_coeffs)])
    if not args.use_root_height_obs:
        cmd.append("--no-use-root-height-obs")
    if not args.quiet:
        cmd.append("--no-quiet")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
        if proc.stderr:
            print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n", file=sys.stderr)
        raise RuntimeError(f"Primitive export failed for motion file: {motion_file}")


def _merge_single_output(
    *,
    tmp_output: Path,
    final_output: Path,
    combined_index_handles: dict[str, Any],
) -> dict[str, Any]:
    submeta = json.loads((tmp_output / "meta.json").read_text(encoding="utf-8"))
    source_tag = Path(submeta["source_files"][0]["motion_file"]).stem

    for split in ("train", "val"):
        split_dir = tmp_output / split
        if not split_dir.exists():
            continue
        final_split_dir = final_output / split
        final_split_dir.mkdir(parents=True, exist_ok=True)

        for shard_path in sorted(split_dir.glob("part-*")):
            renamed = f"{source_tag}-{shard_path.name}"
            shutil.move(str(shard_path), str(final_split_dir / renamed))

        index_path = split_dir / "_index.jsonl"
        if index_path.exists():
            handle = combined_index_handles.setdefault(
                split,
                (final_split_dir / "_index.jsonl").open("a", encoding="utf-8"),
            )
            with index_path.open("r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    entry["shard"] = f"{source_tag}-{entry['shard']}"
                    handle.write(json.dumps(entry, ensure_ascii=False))
                    handle.write("\n")
    return submeta


def _close_index_handles(index_handles: dict[str, Any]) -> None:
    for handle in index_handles.values():
        handle.close()


def _build_final_summary(args: Args, submeta_list: list[dict[str, Any]]) -> dict[str, Any]:
    split_row_counts: dict[str, int] = defaultdict(int)
    sample_type_counts: dict[str, int] = defaultdict(int)
    source_files: list[dict[str, Any]] = []
    total_rows = 0
    total_samples = 0
    prompt_dim = None

    split_shards: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val"):
        split_dir = args.output_dir / split
        shards = []
        if split_dir.exists():
            for shard in sorted(split_dir.iterdir()):
                if not shard.is_file() or shard.name == "_index.jsonl":
                    continue
                shards.append({"path": shard.name, "num_rows": None})
        if shards:
            split_shards[split] = shards

    for submeta in submeta_list:
        total_rows += int(submeta["total_rows"])
        total_samples += int(submeta["total_samples"])
        source_files.extend(submeta.get("source_files", []))
        if prompt_dim is None:
            prompt_dim = int(submeta["prompt_dim"])
        for split, count in (submeta.get("split_row_counts") or {}).items():
            split_row_counts[split] += int(count)
        for sample_type, count in (submeta.get("sample_type_counts") or {}).items():
            sample_type_counts[sample_type] += int(count)

    if prompt_dim is None:
        raise RuntimeError("No submeta entries were produced during batch export.")

    split_summaries = {}
    for split, shards in split_shards.items():
        index_path = args.output_dir / split / "_index.jsonl"
        num_rows = 0
        if index_path.exists():
            with index_path.open("r", encoding="utf-8") as f:
                for line in f:
                    num_rows += int(json.loads(line)["num_rows"])
        split_summaries[split] = {
            "total_rows": num_rows,
            "num_shards": len(shards),
            "shards": shards,
            "index_path": "_index.jsonl",
        }

    summary = {
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "manifest_path": str(args.manifest_path.resolve()),
        "motion_glob": args.motion_glob,
        "output_dir": str(args.output_dir.resolve()),
        "device": args.device,
        "simulator": args.simulator,
        "history_len": args.history_len,
        "future_len": args.future_len,
        "primitive_stride": args.primitive_stride,
        "prompt_dim": prompt_dim,
        "target_representation": args.target_representation,
        "dct_keep_coeffs": args.dct_keep_coeffs,
        "rows_per_shard": args.rows_per_shard,
        "shard_format": args.shard_format,
        "storage_dtype": args.storage_dtype,
        "sample_types": ["clip"],
        "total_samples": total_samples,
        "total_rows": total_rows,
        "split_row_counts": dict(split_row_counts),
        "sample_type_counts": dict(sample_type_counts),
        "skipped_missing_manifest": 0,
        "skipped_filtered_type": 0,
        "splits": split_summaries,
        "source_files": source_files,
        "hf_repo_id": args.hf_repo_id,
    }
    return summary


def main(args: Args) -> None:
    motion_files = sorted(Path(path) for path in glob.glob(args.motion_glob))
    if args.max_motion_files is not None:
        motion_files = motion_files[: args.max_motion_files]
    if not motion_files:
        raise FileNotFoundError(f"No motion files matched motion_glob={args.motion_glob!r}")

    if args.output_dir.exists():
        has_existing = any(args.output_dir.iterdir())
        if has_existing and not args.overwrite_output:
            raise FileExistsError(f"Output directory {args.output_dir} is not empty. Pass --overwrite-output.")
        if has_existing and args.overwrite_output:
            shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = args.output_dir / "_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    submeta_list: list[dict[str, Any]] = []
    combined_index_handles: dict[str, Any] = {}
    try:
        for idx, motion_file in enumerate(motion_files, start=1):
            tmp_output = tmp_root / motion_file.stem
            print(f"[{idx}/{len(motion_files)}] exporting {motion_file.name}")
            _run_single_motion_export(args, motion_file, tmp_output)
            submeta = _merge_single_output(
                tmp_output=tmp_output,
                final_output=args.output_dir,
                combined_index_handles=combined_index_handles,
            )
            submeta_list.append(submeta)
            shutil.rmtree(tmp_output, ignore_errors=True)
    finally:
        _close_index_handles(combined_index_handles)

    shutil.rmtree(tmp_root, ignore_errors=True)
    summary = _build_final_summary(args, submeta_list)
    (args.output_dir / "meta.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_primitive_dataset_card(args.output_dir, summary)

    if args.hf_upload:
        if not args.hf_repo_id:
            raise ValueError("hf_repo_id must be provided when hf_upload=True")
        summary["hf_url"] = upload_primitive_dataset(
            output_dir=args.output_dir,
            repo_id=args.hf_repo_id,
            token=args.hf_token,
            private=args.hf_private,
            path_in_repo=args.hf_path_in_repo,
            commit_message=args.hf_commit_message,
            create_repo=args.hf_create_repo,
            use_upload_large_folder=args.hf_use_upload_large_folder,
        )
        (args.output_dir / "meta.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(tyro.cli(Args))
