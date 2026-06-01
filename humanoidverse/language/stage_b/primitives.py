from __future__ import annotations

import glob
import gzip
import json
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.language.stage_a.dataset import load_manifest_records
from humanoidverse.language.stage_a.teacher import build_teacher_env, quiet_extraction, tracking_inference_fast
from humanoidverse.utils.helpers import get_backward_observation

from .dataset import select_primitive_text
from .frequency import DCTFutureCodec


@dataclass(slots=True)
class BFMTextOpPrimitiveExportConfig:
    checkpoint_dir: Path
    manifest_path: Path
    motion_glob: str
    output_dir: Path
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
    hydra_overrides: list[str] = field(default_factory=list)


def _normalize_shard_format(shard_format: str) -> str:
    normalized = shard_format.lower().strip()
    if normalized not in {"jsonl", "parquet"}:
        raise ValueError(f"Unsupported shard_format={shard_format!r}. Expected 'jsonl' or 'parquet'.")
    return normalized


def _normalize_storage_dtype(storage_dtype: str) -> str:
    normalized = storage_dtype.lower().strip()
    if normalized not in {"float16", "float32"}:
        raise ValueError(f"Unsupported storage_dtype={storage_dtype!r}. Expected 'float16' or 'float32'.")
    return normalized


def build_primitive_rows_from_z_seq(
    record: dict[str, Any],
    z_seq: torch.Tensor,
    history_len: int,
    future_len: int,
    primitive_stride: int,
    min_overlap_s: float,
    *,
    target_representation: str,
    dct_keep_coeffs: int | None,
) -> list[dict[str, Any]]:
    if z_seq.ndim != 2:
        raise ValueError(f"Expected z_seq with rank 2, got shape={tuple(z_seq.shape)}")
    if history_len <= 0 or future_len <= 0 or primitive_stride <= 0:
        raise ValueError("history_len, future_len, and primitive_stride must all be > 0")

    seq_len, prompt_dim = z_seq.shape
    context_len = history_len + future_len
    if seq_len < context_len:
        return []
    normalized_target_representation = target_representation.lower().strip()
    if normalized_target_representation not in {"raw", "dct"}:
        raise ValueError(
            f"Unsupported target_representation={target_representation!r}. Expected 'raw' or 'dct'."
        )
    dct_codec = (
        DCTFutureCodec(future_len=future_len, keep_coeffs=int(dct_keep_coeffs))
        if normalized_target_representation == "dct"
        else None
    )

    rows: list[dict[str, Any]] = []
    sample_id = str(record["sample_id"])
    clip_text = str(record.get("primary_text") or "").strip()
    sample_type = str(record.get("sample_type") or "")
    split = str(record.get("split") or "")

    primitive_idx = 0
    for primitive_start in range(0, seq_len - context_len + 1, primitive_stride):
        hist = z_seq[primitive_start : primitive_start + history_len]
        fut = z_seq[primitive_start + history_len : primitive_start + context_len]
        future_start_idx = primitive_start + history_len
        future_end_idx = future_start_idx + future_len
        rows.append(
            {
                "clip_id": sample_id,
                "sample_type": sample_type,
                "split": split,
                "chunk_id": primitive_idx,
                "text_chunk": select_primitive_text(
                    record=record,
                    future_start_idx=future_start_idx,
                    future_end_idx=future_end_idx,
                    seq_len=seq_len,
                    min_overlap_s=min_overlap_s,
                ),
                "clip_text": clip_text,
                "z_hist_raw": hist.reshape(-1).cpu().tolist(),
                "z_fut_raw": fut.reshape(-1).cpu().tolist(),
                "z_fut_dct": (
                    dct_codec.encode(fut).reshape(-1).cpu().tolist()
                    if dct_codec is not None
                    else None
                ),
                "history_len": history_len,
                "future_len": future_len,
                "prompt_dim": prompt_dim,
                "target_representation": normalized_target_representation,
                "dct_keep_coeffs": dct_codec.keep_coeffs if dct_codec is not None else 0,
                "window_start": primitive_start,
                "t_start": future_start_idx,
                "t_end": future_end_idx,
            }
        )
        primitive_idx += 1
    return rows


class _PrimitiveSplitWriter:
    def __init__(
        self,
        split_dir: Path,
        rows_per_shard: int,
        *,
        shard_format: str,
        storage_dtype: str,
    ) -> None:
        self.split_dir = split_dir
        self.rows_per_shard = rows_per_shard
        self.shard_format = _normalize_shard_format(shard_format)
        self.storage_dtype = _normalize_storage_dtype(storage_dtype)
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self.current_handle = None
        self.current_path: Path | None = None
        self.current_rows = 0
        self.current_buffer: list[dict[str, Any]] = []
        self.shard_idx = 0
        self.total_rows = 0
        self.shards: list[dict[str, Any]] = []
        self.sample_index_entries: list[dict[str, Any]] = []

    def _finalize_current_shard(self) -> None:
        if self.current_path is None:
            return
        if self.shard_format == "parquet":
            self._write_parquet_shard()
        elif self.current_handle is not None:
            self.current_handle.close()
        self.shards.append(
            {
                "path": str(self.current_path.name),
                "num_rows": self.current_rows,
            }
        )
        self.current_handle = None
        self.current_path = None
        self.current_rows = 0
        self.current_buffer = []

    def _write_parquet_shard(self) -> None:
        if not self.current_buffer or self.current_path is None:
            return

        import pyarrow as pa
        import pyarrow.parquet as pq

        value_type = pa.float16() if self.storage_dtype == "float16" else pa.float32()
        hist_size = len(self.current_buffer[0]["z_hist_raw"])
        fut_size = len(self.current_buffer[0]["z_fut_raw"])
        has_dct = self.current_buffer[0].get("z_fut_dct") is not None
        dct_size = len(self.current_buffer[0]["z_fut_dct"]) if has_dct else 0
        arrays = [
            pa.array([str(row["clip_id"]) for row in self.current_buffer], type=pa.string()),
            pa.array([str(row["sample_type"]) for row in self.current_buffer], type=pa.string()),
            pa.array([str(row["split"]) for row in self.current_buffer], type=pa.string()),
            pa.array([int(row["chunk_id"]) for row in self.current_buffer], type=pa.int32()),
            pa.array([str(row["text_chunk"]) for row in self.current_buffer], type=pa.string()),
            pa.array([str(row["clip_text"]) for row in self.current_buffer], type=pa.string()),
            pa.array([row["z_hist_raw"] for row in self.current_buffer], type=pa.list_(value_type, hist_size)),
            pa.array([row["z_fut_raw"] for row in self.current_buffer], type=pa.list_(value_type, fut_size)),
            (
                pa.array([row["z_fut_dct"] for row in self.current_buffer], type=pa.list_(value_type, dct_size))
                if has_dct
                else pa.nulls(len(self.current_buffer))
            ),
            pa.array([int(row["history_len"]) for row in self.current_buffer], type=pa.int16()),
            pa.array([int(row["future_len"]) for row in self.current_buffer], type=pa.int16()),
            pa.array([int(row["prompt_dim"]) for row in self.current_buffer], type=pa.int16()),
            pa.array([str(row["target_representation"]) for row in self.current_buffer], type=pa.string()),
            pa.array([int(row["dct_keep_coeffs"]) for row in self.current_buffer], type=pa.int16()),
            pa.array([int(row["window_start"]) for row in self.current_buffer], type=pa.int32()),
            pa.array([int(row["t_start"]) for row in self.current_buffer], type=pa.int32()),
            pa.array([int(row["t_end"]) for row in self.current_buffer], type=pa.int32()),
        ]
        field_names = [
            "clip_id",
            "sample_type",
            "split",
            "chunk_id",
            "text_chunk",
            "clip_text",
            "z_hist_raw",
            "z_fut_raw",
            "z_fut_dct",
            "history_len",
            "future_len",
            "prompt_dim",
            "target_representation",
            "dct_keep_coeffs",
            "window_start",
            "t_start",
            "t_end",
        ]
        table = pa.Table.from_arrays(arrays, names=field_names)
        pq.write_table(table, self.current_path, compression="zstd")

    def _open_next_shard(self) -> None:
        if self.current_path is not None:
            self._finalize_current_shard()
        suffix = ".parquet" if self.shard_format == "parquet" else ".jsonl.gz"
        self.current_path = self.split_dir / f"part-{self.shard_idx:05d}{suffix}"
        if self.shard_format == "jsonl":
            self.current_handle = gzip.open(self.current_path, "wt", encoding="utf-8")
        else:
            self.current_handle = None
        self.current_rows = 0
        self.current_buffer = []
        self.shard_idx += 1

    def write_sample_rows(
        self,
        *,
        split: str,
        sample_id: str,
        sample_type: str,
        rows: list[dict[str, Any]],
    ) -> None:
        if not rows:
            return
        if self.current_path is None or (self.shard_format == "jsonl" and self.current_handle is None):
            self._open_next_shard()
        elif self.current_rows > 0 and self.current_rows + len(rows) > self.rows_per_shard:
            self._open_next_shard()

        if self.current_path is None or (self.shard_format == "jsonl" and self.current_handle is None):
            raise RuntimeError("Primitive shard writer failed to open output file.")

        if self.shard_format == "parquet":
            self.current_buffer.extend(rows)
        else:
            for row in rows:
                self.current_handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                self.current_handle.write("\n")
        self.current_rows += len(rows)
        self.total_rows += len(rows)
        self.sample_index_entries.append(
            {
                "split": split,
                "sample_id": sample_id,
                "sample_type": sample_type,
                "num_rows": len(rows),
                "shard": self.current_path.name,
            }
        )

    def finalize(self) -> dict[str, Any]:
        if self.current_path is not None:
            self._finalize_current_shard()

        index_path = self.split_dir / "_index.jsonl"
        with index_path.open("w", encoding="utf-8") as f:
            for entry in self.sample_index_entries:
                f.write(json.dumps(entry, ensure_ascii=False))
                f.write("\n")

        return {
            "total_rows": self.total_rows,
            "num_shards": len(self.shards),
            "shards": self.shards,
            "index_path": str(index_path.name),
        }


def write_primitive_dataset_card(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# BFM+TextOp Primitive Dataset",
        "",
        "This dataset stores one BFM primitive per line for Stage B training.",
        "",
        "## Fields",
        "",
        "- `clip_id`: source motion/sample identifier",
        "- `chunk_id`: primitive index within the source clip",
        "- `text_chunk`: text aligned to the primitive future window",
        "- `clip_text`: clip-level fallback text",
        "- `z_hist_raw`: flattened history prompt `[H * Dz]`",
        "- `z_fut_raw`: flattened future prompt `[F * Dz]`",
        "- `z_fut_dct`: flattened low-frequency DCT target `[K * Dz]` when DCT export is enabled",
        "- `window_start`: primitive start index in the latent sequence",
        "- `t_start`, `t_end`: future window bounds in latent-step indices (`t_end` is exclusive)",
        "",
        "## Summary",
        "",
        f"- Total rows: {summary['total_rows']}",
        f"- History length: {summary['history_len']}",
        f"- Future length: {summary['future_len']}",
        f"- Primitive stride: {summary['primitive_stride']}",
        f"- Prompt dim: {summary['prompt_dim']}",
        f"- Target representation: {summary['target_representation']}",
        f"- DCT coeffs kept: {summary['dct_keep_coeffs']}",
        f"- Shard format: {summary['shard_format']}",
        f"- Storage dtype: {summary['storage_dtype']}",
        "",
        "Rows are stored under `train/` and `val/` as one-row-per-primitive shards.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def upload_primitive_dataset(
    *,
    output_dir: Path,
    repo_id: str,
    token: str | None,
    private: bool,
    path_in_repo: str,
    commit_message: str | None,
    create_repo: bool,
    use_upload_large_folder: bool,
) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    who = api.whoami()
    token_role = str(((who.get("auth") or {}).get("accessToken") or {}).get("role") or "")
    if token_role.lower() == "read":
        raise PermissionError(
            "The local Hugging Face token is read-only. Configure a write token before using --hf-upload."
        )

    if create_repo:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    commit_message = commit_message or "Upload BFM+TextOp primitive dataset shards"
    if use_upload_large_folder:
        api.upload_large_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(output_dir),
        )
    else:
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(output_dir),
            path_in_repo=path_in_repo or None,
            commit_message=commit_message,
        )
    return f"https://huggingface.co/datasets/{repo_id}"


def export_bfm_textop_primitives(cfg: BFMTextOpPrimitiveExportConfig) -> dict[str, Any]:
    motion_files = sorted(Path(path) for path in glob.glob(cfg.motion_glob))
    if cfg.max_motion_files is not None:
        motion_files = motion_files[: cfg.max_motion_files]
    if not motion_files:
        raise FileNotFoundError(f"No motion files matched motion_glob={cfg.motion_glob!r}")

    manifest_records = load_manifest_records(cfg.manifest_path)
    manifest_lookup = {str(record["sample_id"]): record for record in manifest_records}

    if cfg.output_dir.exists():
        has_existing = any(cfg.output_dir.iterdir())
        if has_existing and not cfg.overwrite_output:
            raise FileExistsError(
                f"Output directory {cfg.output_dir} is not empty. Pass overwrite_output=True to replace it."
            )
        if has_existing and cfg.overwrite_output:
            shutil.rmtree(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    split_writers: dict[str, _PrimitiveSplitWriter] = {}
    total_rows = 0
    total_samples = 0
    prompt_dim: int | None = None
    skipped_missing_manifest = 0
    skipped_filtered_type = 0
    sample_type_counts: dict[str, int] = defaultdict(int)
    split_row_counts: dict[str, int] = defaultdict(int)
    source_files_summary: list[dict[str, Any]] = []

    with quiet_extraction(cfg.quiet):
        model = load_model_from_checkpoint_dir(cfg.checkpoint_dir, device=cfg.device)
        model.to(cfg.device)
        model.eval()

        with torch.inference_mode():
            for motion_file in motion_files:
                wrapped_env = build_teacher_env(
                    motion_file=motion_file,
                    device=cfg.device,
                    simulator=cfg.simulator,
                    use_root_height_obs=cfg.use_root_height_obs,
                    use_contact_in_obs_max=False,
                    robot_override=cfg.robot_override,
                    hydra_overrides=cfg.hydra_overrides,
                )
                env = wrapped_env._env
                motion_keys = [str(key) for key in env._motion_lib._motion_data_keys.tolist()]
                if cfg.max_motion_keys_per_file is not None:
                    motion_keys = motion_keys[: cfg.max_motion_keys_per_file]

                file_rows = 0
                file_samples = 0
                motions_per_load = max(int(cfg.motions_per_load), 1)
                for batch_start in range(0, len(motion_keys), motions_per_load):
                    batch_size = min(motions_per_load, len(motion_keys) - batch_start)
                    env._motion_lib.load_motions(random_sample=False, start_idx=batch_start, num_motions_to_load=batch_size)
                    current_keys = [str(key) for key in env._motion_lib.curr_motion_keys]
                    expected_keys = motion_keys[batch_start : batch_start + len(current_keys)]
                    if current_keys != expected_keys:
                        raise RuntimeError(
                            f"Motion reload mismatch: expected {expected_keys[:3]}, got {current_keys[:3]}"
                        )

                    for local_motion_id, motion_key in enumerate(current_keys):
                        record = manifest_lookup.get(motion_key)
                        if record is None:
                            skipped_missing_manifest += 1
                            continue
                        if cfg.sample_types and record.get("sample_type") not in cfg.sample_types:
                            skipped_filtered_type += 1
                            continue

                        obs, _ = get_backward_observation(env, local_motion_id, use_root_height_obs=cfg.use_root_height_obs)
                        obs = tree_map(lambda value: value[1:], obs)
                        z_seq = tracking_inference_fast(model, obs).detach().cpu()
                        if prompt_dim is None:
                            prompt_dim = int(z_seq.shape[-1])
                        rows = build_primitive_rows_from_z_seq(
                            record=record,
                            z_seq=z_seq,
                            history_len=cfg.history_len,
                            future_len=cfg.future_len,
                            primitive_stride=cfg.primitive_stride,
                            min_overlap_s=cfg.min_overlap_s,
                            target_representation=cfg.target_representation,
                            dct_keep_coeffs=cfg.dct_keep_coeffs,
                        )
                        split = str(record["split"])
                        writer = split_writers.setdefault(
                            split,
                            _PrimitiveSplitWriter(
                                cfg.output_dir / split,
                                cfg.rows_per_shard,
                                shard_format=cfg.shard_format,
                                storage_dtype=cfg.storage_dtype,
                            ),
                        )
                        writer.write_sample_rows(
                            split=split,
                            sample_id=motion_key,
                            sample_type=str(record.get("sample_type") or ""),
                            rows=rows,
                        )
                        total_rows += len(rows)
                        total_samples += 1
                        file_rows += len(rows)
                        file_samples += 1
                        sample_type_counts[str(record.get("sample_type") or "")] += 1
                        split_row_counts[split] += len(rows)

                source_files_summary.append(
                    {
                        "motion_file": str(motion_file.resolve()),
                        "num_motion_keys": len(motion_keys),
                        "num_samples_written": file_samples,
                        "num_rows_written": file_rows,
                    }
                )

    split_summaries = {split: writer.finalize() for split, writer in split_writers.items()}
    if prompt_dim is None:
        raise RuntimeError("No primitive rows were exported.")

    summary = {
        "checkpoint_dir": str(cfg.checkpoint_dir.resolve()),
        "manifest_path": str(cfg.manifest_path.resolve()),
        "motion_glob": cfg.motion_glob,
        "output_dir": str(cfg.output_dir.resolve()),
        "device": cfg.device,
        "simulator": cfg.simulator,
        "history_len": cfg.history_len,
        "future_len": cfg.future_len,
        "primitive_stride": cfg.primitive_stride,
        "prompt_dim": prompt_dim,
        "target_representation": cfg.target_representation,
        "dct_keep_coeffs": cfg.dct_keep_coeffs,
        "rows_per_shard": cfg.rows_per_shard,
        "shard_format": _normalize_shard_format(cfg.shard_format),
        "storage_dtype": _normalize_storage_dtype(cfg.storage_dtype),
        "sample_types": list(cfg.sample_types),
        "total_samples": total_samples,
        "total_rows": total_rows,
        "split_row_counts": dict(split_row_counts),
        "sample_type_counts": dict(sample_type_counts),
        "skipped_missing_manifest": skipped_missing_manifest,
        "skipped_filtered_type": skipped_filtered_type,
        "splits": split_summaries,
        "source_files": source_files_summary,
        "hf_repo_id": cfg.hf_repo_id,
    }
    (cfg.output_dir / "meta.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_primitive_dataset_card(cfg.output_dir, summary)

    if cfg.hf_upload:
        if not cfg.hf_repo_id:
            raise ValueError("hf_repo_id must be provided when hf_upload=True")
        summary["hf_url"] = upload_primitive_dataset(
            output_dir=cfg.output_dir,
            repo_id=cfg.hf_repo_id,
            token=cfg.hf_token,
            private=cfg.hf_private,
            path_in_repo=cfg.hf_path_in_repo,
            commit_message=cfg.hf_commit_message,
            create_repo=cfg.hf_create_repo,
            use_upload_large_folder=cfg.hf_use_upload_large_folder,
        )
        (cfg.output_dir / "meta.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return summary
