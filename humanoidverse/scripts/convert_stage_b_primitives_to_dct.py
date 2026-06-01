from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import tyro

from humanoidverse.language.stage_b.frequency import DCTFutureCodec


@dataclass
class Args:
    input_dir: Path = Path("artifacts/stage_b/primitives_seed_full_parquet")
    output_dir: Path = Path("artifacts/stage_b/primitives_seed_full_dct_f8_k3")
    future_len: int = 8
    dct_keep_coeffs: int = 3
    rows_per_log: int = 20
    overwrite_output: bool = False


def _build_output_table(table: pa.Table, codec: DCTFutureCodec) -> pa.Table:
    z_hist_raw = table.column("z_hist_raw")
    z_fut_raw = table.column("z_fut_raw")
    hist_size = z_hist_raw.type.list_size
    fut_size = z_fut_raw.type.list_size
    prompt_dim = fut_size // codec.future_len
    if fut_size != codec.future_len * prompt_dim:
        raise ValueError(f"Invalid future list size {fut_size} for future_len={codec.future_len}")

    z_fut_np = z_fut_raw.combine_chunks().values.to_numpy(zero_copy_only=False).reshape(
        len(table), codec.future_len, prompt_dim
    )
    z_fut_t = torch.from_numpy(z_fut_np.astype("float32"))
    z_fut_dct = codec.encode(z_fut_t).reshape(len(table), codec.keep_coeffs * prompt_dim).numpy()

    value_type = pa.float16()
    arrays = [
        table.column("clip_id"),
        table.column("sample_type"),
        table.column("split"),
        table.column("chunk_id"),
        table.column("text_chunk"),
        table.column("clip_text"),
        z_hist_raw,
        z_fut_raw,
        pa.array(z_fut_dct.tolist(), type=pa.list_(value_type, codec.keep_coeffs * prompt_dim)),
        table.column("history_len"),
        table.column("future_len"),
        table.column("prompt_dim"),
        pa.array(["dct"] * len(table), type=pa.string()),
        pa.array([codec.keep_coeffs] * len(table), type=pa.int16()),
        table.column("window_start"),
        table.column("t_start"),
        table.column("t_end"),
    ]
    names = [
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
    return pa.Table.from_arrays(arrays, names=names)


def _copy_split_index(input_split_dir: Path, output_split_dir: Path) -> int:
    output_split_dir.mkdir(parents=True, exist_ok=True)
    index_path = input_split_dir / "_index.jsonl"
    if not index_path.exists():
        return 0
    shutil.copy2(index_path, output_split_dir / "_index.jsonl")
    with index_path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main(args: Args) -> None:
    if args.output_dir.exists():
        has_existing = any(args.output_dir.iterdir())
        if has_existing and not args.overwrite_output:
            raise FileExistsError(f"{args.output_dir} is not empty. Pass --overwrite-output.")
        if has_existing and args.overwrite_output:
            shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((args.input_dir / "meta.json").read_text(encoding="utf-8"))
    if int(meta["future_len"]) != args.future_len:
        raise ValueError(
            f"Input dataset future_len={meta['future_len']} does not match requested future_len={args.future_len}"
        )
    codec = DCTFutureCodec(future_len=args.future_len, keep_coeffs=args.dct_keep_coeffs)

    total_shards = 0
    converted_rows = 0
    for split in ("train", "val"):
        input_split_dir = args.input_dir / split
        if not input_split_dir.exists():
            continue
        output_split_dir = args.output_dir / split
        _copy_split_index(input_split_dir, output_split_dir)
        shard_paths = sorted(input_split_dir.glob("*.parquet"))
        for shard_idx, shard_path in enumerate(shard_paths, start=1):
            table = pq.read_table(shard_path)
            out_table = _build_output_table(table, codec)
            pq.write_table(out_table, output_split_dir / shard_path.name, compression="zstd")
            converted_rows += out_table.num_rows
            total_shards += 1
            if shard_idx % max(args.rows_per_log, 1) == 0 or shard_idx == len(shard_paths):
                print(
                    f"[{split}] {shard_idx}/{len(shard_paths)} shards converted | "
                    f"rows={converted_rows} total_shards={total_shards}",
                    flush=True,
                )

    summary = dict(meta)
    summary["output_dir"] = str(args.output_dir.resolve())
    summary["target_representation"] = "dct"
    summary["dct_keep_coeffs"] = int(args.dct_keep_coeffs)
    summary["rows_per_shard"] = int(meta.get("rows_per_shard", 5000))
    (args.output_dir / "meta.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    readme_lines = [
        "# Stage B Primitive Dataset",
        "",
        f"- source dataset: `{args.input_dir.resolve()}`",
        f"- history_len: `{meta['history_len']}`",
        f"- future_len: `{meta['future_len']}`",
        f"- target_representation: `dct`",
        f"- dct_keep_coeffs: `{args.dct_keep_coeffs}`",
        f"- total_rows: `{meta['total_rows']}`",
        "",
        "This dataset was produced by converting existing raw primitive shards and appending `z_fut_dct`.",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "input_dir": str(args.input_dir.resolve()),
                "output_dir": str(args.output_dir.resolve()),
                "future_len": args.future_len,
                "dct_keep_coeffs": args.dct_keep_coeffs,
                "total_shards": total_shards,
                "converted_rows": converted_rows,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
