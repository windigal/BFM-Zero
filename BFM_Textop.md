# BFM + TextOp Pipeline — Working Handoff

Last updated: 2026-06-04

> Read this if you are a fresh agent picking up the BFM-Zero text-to-motion (TextOp) work.
> **All paths below are absolute under `/home/hanwei/code/`. On a new machine, adjust the repo
> root and the sibling deploy repo path first.** The two repos are:
> - main: `/home/hanwei/code/BFM-Zero` (branch `dev`, dirty working tree — do not revert unrelated changes)
> - deploy: `/home/hanwei/code/BFM-zero-deploy` (sibling, also dirty)

## 1. What this system is

It does **not** generate robot joint trajectories directly. It generates **BFM prompt sequences**
(the 256-dim `z` latents that the BFM actor consumes):

```text
text + recent BFM prompts (z_hist) -> future BFM prompts (z_fut) -> BFM ONNX actor -> robot action
```

The language generator (Stage B) sits *above* the BFM motor interface. The existing BFM actor
handles physical control. `prompt_dim = 256`.

### Stage A / teacher latents
Motion clips become BFM prompt sequences through the existing BFM tracking path:
`retargeted motion -> observations -> BFM backward map / tracking inference -> z_seq`.

### Stage B / TextOp generator
Main files:
- `humanoidverse/language/stage_b/model.py` — diffusion/flow transformer; CLIP text + z_hist + timestep -> z_fut
- `humanoidverse/language/stage_b/dataset.py` — parquet/jsonl loader (see contract below)
- `humanoidverse/language/stage_b/controller.py`
- `humanoidverse/scripts/train_stage_b.py` — training entrypoint
- `humanoidverse/scripts/eval_stage_b.py`

**Loader contract (important):** in `raw` mode the loader only reads `z_hist_raw` and `z_fut_raw`.
`z_fut_dct` is only read when `target_representation=dct`. All other columns
(`history_source`, `replicate_idx`, `source_*`, `trim_*`, `window_start`, ...) are **pure metadata**
— safe to add/change without touching training code. New text label strings are also safe: CLIP
text encoder is frozen and tokenizes any string.

Current demo model config: `history_len=2`, `future_len=8`, `prompt_dim=256`,
`objective_type=diffusion`, `target_representation=raw`, `diffusion_steps=30`, frozen CLIP.
Note: `--diffusion-steps` controls the denoise schedule length baked into the model config.
Do **not** confuse it with `--sampling-steps` (deploy-time DDIM steps).

## 2. The problem this round of work addressed

Demo symptoms reported by the user:
1. some actions (e.g. `walk forward`) looked unstandard / sloppy,
2. the robot stops after finishing a clip if you don't switch text,
3. switching text mid-action responds slowly.

Root cause found by inspecting the **v1** dataset
`artifacts/stage_b/primitives_seed_clip_shortlabel_h2_f8_k3/meta.json`:

- **history distribution was ~98% within-clip continuation.** Of 209,658 rows, `clip_previous_frames`
  = 97.7%, `standing` = 1.6%, `similar_moving` = 0.7%. Only each clip's *first* chunk ever got a
  non-continuation history. So the model almost never saw "currently doing A, text says B, produce B"
  — explaining slow switching and no natural transition.
- **label distribution was extremely skewed.** dance 2089 / jump backward 1026 / jog forward 960 /
  jump high 401 = ~94% of clips. `walk forward` had only **8 clips**, `run forward` 20.
  "walk forward is sloppy" is structural data starvation, not undertraining.

## 3. Dataset v2 (transition-aware) — what was built

Builder: `humanoidverse/scripts/build_stage_b_seed_clip_shortlabel_v2_dataset.py`
(copied from the v1 builder `build_stage_b_seed_clip_shortlabel_dataset.py`, reusing
`ReservoirPool`, `_trim_static_edges`, `_reconstruct_clip_sequence`, `_filter_clip_record`,
`_iter_clip_rows`, `ClipSplitWriter`, `DCTFutureCodec`, `_load_standing_history`).

Output: `artifacts/stage_b/primitives_seed_clip_shortlabel_h2_f8_k3_v2`

Three substantive changes over v1:

**(A) Per-chunk history injection** (v1 only injected on chunk 0).
Each chunk, with probability `p_inject=0.35`, replaces `z_hist` with an injected sample;
otherwise it falls back to v1 behavior (`clip_previous_frames`). Conditional injection mix:
- `standing` 25% — start from rest
- `cross_label_moving` 45% — **the key fix**: history is a moving snippet from a *different* label,
  teaching "doing X, asked for Y -> produce Y". Drawn from a global labeled reservoir
  (`LabeledReservoirPool`, size 4096); rejects same-label up to 3 times, then degrades to same-label.
- `same_label_different_clip` 30% — teaches clip-to-clip continuation (keep dancing after a dance clip).

Each replication uses an independent deterministic key
`history_key=f"{clip_id}:{chunk_idx}:{replicate_idx}"`, so replicas are **not** byte-identical —
the same future is seen with varied history/label pairings. Injection source is recorded in the
`history_source` column (metadata only).

**(B) Rare-label clip replication.**
`replication_factor = min(max_clip_replication=16, ceil(target_clips_per_label=400 / label_count))`.
High-freq labels get 1×; rare labels are amplified. Replication affects **train only** (val is never
replicated — replicating val would artificially lower val loss).

**(C) Guaranteed val coverage.**
If a label has no val clip upstream, the lowest-hash train clip is promoted to val, so every label
has val signal (v1 val was only 52 clips with almost no rare-label coverage).

### v2 dataset realized stats (from meta.json)
- total emitted clips: 6109 (train 6045 / val 64); total rows: 263,203 (train 261,629 / val 1,574)
- history_source distribution (post-injection):
  - `clip_previous_frames` 167,357 (63.6%)
  - `same_label_different_clip` 39,614 (15.1%)
  - `cross_label_moving` 32,616 (12.4%)  ← up from 0
  - `standing` 23,616 (9.0%)
  - (cross is a touch below the nominal 15.7% because some cross draws degrade to same-label)
- per-label replication factors: walk forward / run forward / walk right / jog right / run around /
  macarena dance / moonwalk dance / jump high forward = 16×; jog backward 10×; walk backward 3×;
  dance / jog forward / jump backward / jump high = 1×.
- val_label_histogram: every one of the 14 labels has ≥1 val clip.

### Rebuild command
```bash
python -m humanoidverse.scripts.build_stage_b_seed_clip_shortlabel_v2_dataset
# all v2 knobs are CLI flags with the defaults above; pass --overwrite-output to rebuild in place.
# tunable: --p-inject, --injection-mix-{standing,cross-label,same-label} (must sum to 1.0),
#          --target-clips-per-label, --max-clip-replication
```

### Verifying a rebuilt v2 dataset
Check `meta.json` after building:
- `history_source_counts_post_injection`: `cross_label_moving` should be ~12–16% (was 0 in v1),
  `clip_previous_frames` ~64%, `standing` ~9%.
- `per_label_replication_factor`: rare labels (walk forward, run forward) should be 16.
- `val_label_histogram`: every label ≥ 1.
- Sanity: pull a few `cross_label_moving` rows and confirm the decoded history's source label differs
  from `text_chunk`.

## 4. Training the v2 model

No training-code changes were needed — `train_stage_b.py` already has every flag. The model that the
deploy config currently points to was produced with:

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

Output run: `results/stage_b_seed_clip_shortlabel_v2_raw_d30/20260417_162319/`
(contains `stage_b_best.pt`, `stage_b_last.pt`, `config.json`, `summary.json`, `train.log`, `tb/`).
Output dir is auto-timestamped (`use_timestamp_subdir=True`); on rebuild you get a new `YYYYMMDD_HHMMSS`
subdir, so update the deploy config to whichever timestamp you want to ship.

Per-epoch val loss (from `train.log`) — still descending at the end:

| epoch | val loss |
|------:|---------:|
| 1 | 0.1407 |
| 2 | 0.1160 |
| 3 | 0.1070 |
| 4 | 0.0999 |
| 5 | 0.0915 |
| 6 | 0.0896 |
| 7 | 0.0851 |
| 8 | 0.0845 |
| **9 (best)** | **0.0813** |
| 10 | 0.0824 |

`best_val_loss = 0.0813` (epoch 9). Training took ~47 min on this machine (10 epochs, 3427 iters/epoch,
train_sequences=219314 / val_sequences=1126 after windowing).

**Do not compare this 0.0813 against the v1 number 0.0514 directly.** The v2 val set is deliberately
harder: ~36% of its rows are transition cases (cross_label + standing), whereas v1 val was ~98%
within-clip continuation. A higher number on a harder val set is expected. Loss was still falling at
epoch 10 — if more capacity is wanted, resume:

```bash
python -m humanoidverse.scripts.train_stage_b \
  --primitive-dataset-dir artifacts/stage_b/primitives_seed_clip_shortlabel_h2_f8_k3_v2 \
  --output-dir results/stage_b_seed_clip_shortlabel_v2_raw_d30 \
  --resume-from results/stage_b_seed_clip_shortlabel_v2_raw_d30/20260417_162319/stage_b_last.pt \
  --history-len 2 --future-len 8 --target-representation raw \
  --objective-type diffusion --diffusion-steps 30 --num-epochs 5
```

## 5. Deploy

Deploy lives in the sibling repo `/home/hanwei/code/BFM-zero-deploy`. Key files:
- `rl_policy/textop_runtime.py` — diffusion sampler / runtime; supports deterministic initial noise
- `rl_policy/bfm_zero.py` — keyboard handlers, rollout reset, prompt-switch logic, Stage B generator wiring
- `rl_policy/textop.sh` — launcher (points at `FBcprAuxModel_seed2k_latest.onnx` + the demo config)
- `config/exp/textop/dct_f8_k3_demo.yaml` — active config

### The only deploy change this round
One line in `config/exp/textop/dct_f8_k3_demo.yaml` — repoint the checkpoint from the v1 run to the v2 run:

```yaml
stage_b_checkpoint: /home/hanwei/code/BFM-Zero/results/stage_b_seed_clip_shortlabel_v2_raw_d30/20260417_162319/stage_b_best.pt
```

Everything else in that config was kept as-is (per prior deploy debugging): `guidance_scale=2.0`,
`sampling_seed=1234`, `sampling_method=ddim`, `sampling_steps=30`, `ddim_eta=0.0`,
`async_generation=true`, `generator_rate_hz=6.25`, `init_history_mode=first`, `loop_texts=true`.
The config file name still says `dct_f8_k3` but it points at a **raw** checkpoint — name is historical.

`sampling_seed=1234` makes repeated same-text rollouts bit-exact (set during earlier debugging — same
text + same seed previously gave same_text_l2 ≈ 0). Leave it unless you intentionally want stochastic
rollouts again; removing it restores fresh Gaussian noise per generation.

### Run it
```bash
cd /home/hanwei/code/BFM-zero-deploy
bash rl_policy/textop.sh
```

### Runtime behavior
1. load Stage B checkpoint; 2. load initial BFM prompt history (`zs_seed2k_init_stand_latest.pkl`);
3. generate current future chunk; 4. async-prefetch next chunk; 5. feed one prompt per low-level
control step to the BFM ONNX actor; 6. on text switch, invalidate stale chunks and prefetch a new one
(`p` = reproduce/reset, `n` = next prompt).

### What to verify on hardware (not yet done — needs the real robot)
- `walk forward` should look more like standard walking than the v1 model.
- After `dance` with no new command, the robot should keep moving rather than visibly stopping.
- Switching text mid-action (e.g. dance → walk backward) should respond within ~1 future window (8 frames).
- Same seed + same text still reproduces exactly.
- Async underrun was previously diagnosed but not redesigned — confirm prompt-generation latency on the
  target machine doesn't force fallback reuse of the last prompt.

## 6. Other existing assets (for reference, not the active demo)

**SEED Short-Label v1** — `artifacts/stage_b/primitives_seed_clip_shortlabel_h2_f8_k3`
(4747 samples / 209,658 rows). Checkpoint
`results/stage_b_seed_clip_shortlabel_raw_d30/20260416_174449/stage_b_best.pt`, best_val 0.0514.
Superseded by v2 for the demo; kept for comparison and reproducibility.

**TextOp-BABEL 29dof** — `artifacts/stage_b/textop_babel_latent_h2_f8_raw`
(8535 samples / 659,862 rows, source_dof=29, target_dof=29). Checkpoint
`results/stage_b_textop_babel_h2_f8_raw/20260414_144928/stage_b_best.pt`, best_val 0.0160.
Builder `humanoidverse/scripts/build_textop_babel_latent_primitives.py`. A separate 29dof track,
useful for data scaling; not the active demo checkpoint.

## 7. Known issues / caveats
- `walk forward` / `run forward` are still the weakest actions — source labels are sparse even after
  16× replication (8 and 20 unique clips). Replication helps but cannot manufacture new motion diversity;
  more real clips for these labels is the real fix.
- jumping quality is limited by the low-level BFM actor itself, not Stage B.
- v2 val loss (0.0813) is not comparable to v1 (0.0514) — harder val set (see §4).
- deterministic sampling (`sampling_seed=1234`) is intentional for debugging/reproducibility.
- both repos have dirty working trees; some modified files under `humanoidverse/agents/` and
  `humanoidverse/train.py` may be orthogonal to this TextOp thread — verify before touching.

## 8. Next engineering step (paper direction, not yet started)
Make Stage B use BFM signals more directly, then chase a research contribution:
1. add an offline BFM-critic / discriminator-guided sampler (test-time guidance from the BFM value fn);
2. compare guided vs unguided prompt quality on held-out transitions;
3. only move guidance into the deploy runtime after offline metrics look sane;
4. use the result as a baseline for later DDPO/DPPO-style Stage B fine-tuning.
Also candidate: state-aware conditioning (feed current BFM z/value as extra condition) and warm-start
sampling (start diffusion from the previous window + small noise instead of pure Gaussian) to cut
switch latency.
