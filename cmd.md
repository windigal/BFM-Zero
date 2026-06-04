# Commands

默认已经在仓库根目录，且终端已激活 `bfm-zero` 环境。  
本文档只保留：

- 要执行的脚本命令
- 对应可选命令行参数

---

## 常用命令

### 1. 用 `SEED 2k` 训练 BFM

当前 [train.py](/home/hanwei/code/BFM-Zero/humanoidverse/train.py) 已把训练数据写成：

- `humanoidverse/data/seed_train_10s_2000.pkl`

直接运行：

```bash
python -m humanoidverse.train
```

注意：
- 这个入口当前**没有**通用命令行参数
- 如果要切换 `SEED 5k / LAFAN / 其他 motionlib`，需要改 [train.py](/home/hanwei/code/BFM-Zero/humanoidverse/train.py) 里的 `lafan_tail_path`

---

## Stage B / TextOp 数据构建

### 2. 构建 SEED 短标签 Stage B 数据集 v1

```bash
python -m humanoidverse.scripts.build_stage_b_seed_clip_shortlabel_dataset
```

输出：`artifacts/stage_b/primitives_seed_clip_shortlabel_h2_f8_k3`（旧版，已被 v2 取代，仅留作对比）。

### 3. 构建 SEED 短标签 Stage B 数据集 v2（transition-aware，当前 demo 路径）

```bash
python -m humanoidverse.scripts.build_stage_b_seed_clip_shortlabel_v2_dataset
```

输出：`artifacts/stage_b/primitives_seed_clip_shortlabel_h2_f8_k3_v2`

相对 v1 的三项改动：每个 chunk 按概率注入 history（standing / cross-label / same-label-different-clip）、
稀有 label 按 clip 复制、保证每个 label 至少有一个 val clip。重建已存在目录需加 `--overwrite-output`。

可选参数（默认值见脚本）：

- `--input-dir`（默认 `artifacts/stage_b/primitives_seed_full_parquet`）
- `--manifest-path`
- `--output-dir`
- `--init-history-path`
- `--history-len`（默认 2）
- `--future-len`（默认 8）
- `--prompt-dim`（默认 256）
- `--dct-keep-coeffs`（默认 3）
- `--rows-per-shard`
- `--static-step-threshold` / `--moving-history-threshold` / `--min-chunks-per-clip`
- `--p-inject`（默认 0.35）
- `--injection-mix-standing` / `--injection-mix-cross-label` / `--injection-mix-same-label`（须和为 1.0，默认 0.25 / 0.45 / 0.30）
- `--target-clips-per-label`（默认 400）
- `--max-clip-replication`（默认 16）
- `--global-pool-size`（默认 4096）
- `--overwrite-output`
- `--seed`

### 4. 构建 TextOp-BABEL 29dof latent primitive 数据集（对比/扩展路径）

```bash
python -m humanoidverse.scripts.build_textop_babel_latent_primitives
```

输出：`artifacts/stage_b/textop_babel_latent_h2_f8_raw`。非当前 demo checkpoint。

---

## Stage B / TextOp 训练

### 5. 训练 v2 raw diffusion Stage B（当前 demo 模型）

```bash
python -m humanoidverse.scripts.train_stage_b \
  --primitive-dataset-dir artifacts/stage_b/primitives_seed_clip_shortlabel_h2_f8_k3_v2 \
  --output-dir results/stage_b_seed_clip_shortlabel_v2_raw_d30 \
  --history-len 2 \
  --future-len 8 \
  --target-representation raw \
  --objective-type diffusion \
  --diffusion-steps 30 \
  --batch-size 64 \
  --num-epochs 10
```

产出：`results/stage_b_seed_clip_shortlabel_v2_raw_d30/<时间戳>/`（含 `stage_b_best.pt`、`config.json`、`summary.json`、`train.log`、`tb/`）。
输出目录自动加时间戳，重训会生成新的子目录。本轮已训练 run：`20260417_162319`，best_val_loss=0.0813（epoch 9）。

恢复/续训：

```bash
python -m humanoidverse.scripts.train_stage_b \
  --primitive-dataset-dir artifacts/stage_b/primitives_seed_clip_shortlabel_h2_f8_k3_v2 \
  --output-dir results/stage_b_seed_clip_shortlabel_v2_raw_d30 \
  --resume-from results/stage_b_seed_clip_shortlabel_v2_raw_d30/20260417_162319/stage_b_last.pt \
  --history-len 2 --future-len 8 --target-representation raw \
  --objective-type diffusion --diffusion-steps 30 --num-epochs 5
```

关键参数说明：

- `--target-representation`：`raw` 或 `dct`（demo 用 raw）
- `--objective-type`：`diffusion` 或 `flow`
- `--diffusion-steps`：模型 config 的去噪 schedule 长度（**不是** deploy 端的 `--sampling-steps`）
- `--primitive-dataset-dir`：数据集根目录，需含 `train/` 和 `val/`
- 其余：`--batch-size` `--num-workers` `--num-epochs` `--lr` `--hidden-dim` `--num-layers` `--num-heads` `--dropout` `--cond-mask-prob` `--freeze-text-encoder` `--resume-from` `--seed`

### 6. 评测 Stage B

```bash
python -m humanoidverse.scripts.eval_stage_b \
  --checkpoint-path results/stage_b_seed_clip_shortlabel_v2_raw_d30/20260417_162319/stage_b_best.pt
```

---

## Deploy（在 sibling 仓 `../BFM-zero-deploy` 内运行）

### 7. 运行 TextOp deploy demo

```bash
cd /home/hanwei/code/BFM-zero-deploy
bash rl_policy/textop.sh
```

配置文件：`config/exp/textop/dct_f8_k3_demo.yaml`。当前指向 v2 checkpoint
`results/stage_b_seed_clip_shortlabel_v2_raw_d30/20260417_162319/stage_b_best.pt`，
`guidance_scale=2.0`、`sampling_seed=1234`、`sampling_method=ddim`、`sampling_steps=30`、`async_generation=true`。
换模型只需改该 yaml 的 `stage_b_checkpoint` 一行。键位：`p` 复现/重置，`n` 切下一个 prompt。

---

## BFM 训练数据

### 8. 构建 `SEED` 训练 manifest

```bash
python -m humanoidverse.scripts.build_seed_train_manifest
```

可选参数：

- `--dataset-root`
- `--metadata-csv`
- `--output-manifest`
- `--output-report`
- `--include-mirrors`
- `--min-clip-duration-s`
- `--max-duration-s`

### 9. 构建训练 tier 子集

```bash
python -m humanoidverse.scripts.build_seed_train_tiers
```

可选参数：

- `--manifest-path`
- `--output-dir`
- `--target-fps`
- `--clip-length-s`
- `--clip-step-s`
- `--tiers`
- `--seed`

### 10. 生成 `SEED 2k` motionlib

```bash
python -m humanoidverse.scripts.build_seed_train_motionlib \
  --manifest-path artifacts/seed_train/tiers/seed_train_10s_2000.jsonl \
  --manifest-is-clipped true \
  --output-motionlib humanoidverse/data/seed_train_10s_2000.pkl \
  --output-clipped-manifest artifacts/seed_train/tiers/seed_train_10s_2000.jsonl \
  --output-report artifacts/seed_train/tiers/seed_train_10s_2000.motionlib_report.json \
  --target-fps 30 \
  --clip-length-s 10.0 \
  --root-euler-order xyz
```

### 11. 生成 `SEED 5k` motionlib

```bash
python -m humanoidverse.scripts.build_seed_train_motionlib \
  --manifest-path artifacts/seed_train/tiers/seed_train_10s_5000.jsonl \
  --manifest-is-clipped true \
  --output-motionlib humanoidverse/data/seed_train_10s_5000.pkl \
  --output-clipped-manifest artifacts/seed_train/tiers/seed_train_10s_5000.jsonl \
  --output-report artifacts/seed_train/tiers/seed_train_10s_5000.motionlib_report.json \
  --target-fps 30 \
  --clip-length-s 10.0 \
  --root-euler-order xyz
```

`build_seed_train_motionlib` 可选参数：

- `--manifest-path`
- `--output-motionlib`
- `--output-clipped-manifest`
- `--output-report`
- `--mjcf-path`
- `--target-fps`
- `--clip-length-s`
- `--clip-step-s`
- `--root-euler-order`
- `--manifest-is-clipped`

---

## Tracking 导出与验证

### 12. 导出 `SEED` tracking context

```bash
python -m humanoidverse.scripts.export_seed_tracking_inference \
  --filename Neutral_walk_forward_002__A057 \
  --motion-output artifacts/stage_a/debug/neutral_walk_seed_motion_xyz.pkl \
  --latent-output /home/hanwei/code/BFM-zero-deploy/model/tracking_inference/zs_seed_neutral_walk_xyz.pkl \
  --device cuda \
  --simulator mujoco \
  --root-euler-order xyz
```

可选参数：

- `--filename`
- `--dataset-root`
- `--metadata-csv`
- `--checkpoint-dir`
- `--mjcf-path`
- `--motion-output`
- `--latent-output`
- `--meta-output`
- `--device`
- `--simulator`
- `--target-fps`
- `--use-root-height-obs`
- `--root-euler-order`

### 13. 验证 `SEED` 转换

```bash
python -m humanoidverse.scripts.validate_seed_conversion \
  --filename Neutral_walk_forward_002__A057 \
  --root-euler-orders xyz yxz zyx
```

可选参数：

- `--filename`
- `--dataset-root`
- `--metadata-csv`
- `--mjcf-path`
- `--output-dir`
- `--target-fps`
- `--simulator`
- `--device`
- `--root-euler-orders`
- `--export-motion-pkls`

### 14. `LAFAN1` CSV 转 motion

```bash
python -m humanoidverse.scripts.convert_lafan1_csv_to_motion \
  --csv-path ~/dataset/LAFAN1/g1/walk1_subject1.csv \
  --output artifacts/lafan1/walk1_subject1_300_motion.pkl \
  --motion-name walk1_subject1 \
  --end-frame 300
```

可选参数：

- `--csv-path`
- `--output`
- `--motion-name`
- `--mjcf`
- `--start-frame`
- `--end-frame`
- `--stride`
- `--compare-pkl`
- `--compare-key`
- `--compare-output`

### 15. 导出 `LAFAN1` tracking context

```bash
python -m humanoidverse.scripts.export_lafan1_tracking_inference \
  --csv-path ~/dataset/LAFAN1/g1/walk1_subject1.csv \
  --motion-output artifacts/lafan1/walk1_subject1_300_motion.pkl \
  --latent-output /home/hanwei/code/BFM-zero-deploy/model/tracking_inference/zs_lafan1_walk1_subject1_300.pkl \
  --motion-name walk1_subject1 \
  --end-frame 300 \
  --device cuda \
  --simulator mujoco
```

可选参数：

- `--csv-path`
- `--checkpoint-dir`
- `--mjcf-path`
- `--motion-output`
- `--latent-output`
- `--meta-output`
- `--motion-name`
- `--start-frame`
- `--end-frame`
- `--stride`
- `--device`
- `--simulator`
- `--use-root-height-obs`

### 16. tracking 导出闭环评测

```bash
python -m humanoidverse.scripts.eval_tracking_export \
  --motion-file artifacts/lafan1/walk1_subject1_300_motion.pkl \
  --latent-path /home/hanwei/code/BFM-zero-deploy/model/tracking_inference/zs_lafan1_walk1_subject1_300.pkl \
  --checkpoint-dir checkpoint \
  --output-json artifacts/lafan1/walk1_subject1_300_rollout.json
```

可选参数：

- `--motion-file`
- `--latent-path`
- `--checkpoint-dir`
- `--output-json`
- `--device`
- `--simulator`
- `--use-root-height-obs`

---

## PICO 辅助脚本

### 17. 导出 PICO qpos CSV

```bash
python -m humanoidverse.scripts.export_pico_qpos_csv \
  --input humanoidverse/data/pico.pkl \
  --output humanoidverse/data/pico_qpos.csv
```

可选参数：

- `--input`
- `--output`
- `--root-rot-format`

### 18. PICO 转 motion

```bash
python -m humanoidverse.scripts.convert_pico_to_motion \
  --input humanoidverse/data/pico.pkl \
  --output humanoidverse/data/pico_motion.pkl \
  --motion-name pico
```

可选参数：

- `--input`
- `--output`
- `--motion-name`
- `--mjcf`
- `--root-rot-format`
