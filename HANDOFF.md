# BFM-Zero Text-to-Motion Handoff

Last updated: 2026-06-04

## Current Goal

Continue the BFM-Zero text-to-motion / TextOp-style work without rerunning completed data builds or training jobs.

The active project direction is:

- train a high-level text-conditioned generator that maps `text + prompt history -> future BFM prompts`
- execute the generated prompt sequence with the existing BFM low-level actor
- use BFM's own strengths, including smooth latent interpolation, tracking/goal/reward inference, critic/discriminator signals, and sim-to-real robustness, as the basis for the next paper-level innovation

## Source of Truth

Use the Claude conversation at:

`/home/hanwei/.claude/projects/-home-hanwei-code-BFM-Zero/db3906fc-173e-4736-b562-6206ade27e20.jsonl`

The older chat todo list and parts of previous repo docs are stale. In particular, "run Stage B training" is no longer generally pending.

## Completed Work

### Stage B Core

Implemented in:

- `humanoidverse/language/stage_b/model.py`
- `humanoidverse/language/stage_b/dataset.py`
- `humanoidverse/language/stage_b/controller.py`
- `humanoidverse/scripts/train_stage_b.py`
- `humanoidverse/scripts/eval_stage_b.py`

Current model:

- input: CLIP text embedding + BFM prompt history
- output: future BFM prompt sequence
- supports `objective_type=diffusion` and `objective_type=flow`
- supports `target_representation=raw` and `target_representation=dct`
- current demo path uses `raw`, `history_len=2`, `future_len=8`, `prompt_dim=256`, `diffusion_steps=30`

### SEED Short-Label v1

Dataset:

`artifacts/stage_b/primitives_seed_clip_shortlabel_h2_f8_k3`

Summary:

- total_samples: 4747
- total_rows: 209658
- train clips: 4695
- val clips: 52
- dominant labels: dance, jump backward, jog forward, jump high, walk backward
- history was mostly within-clip continuation, so it did not sufficiently train text switching

Training:

`results/stage_b_seed_clip_shortlabel_raw_d30/20260416_174449`

Summary:

- target: raw
- objective: diffusion
- diffusion_steps: 30
- best_val_loss: 0.051412111547376425
- best checkpoint: `stage_b_best.pt`

### SEED Short-Label v2

Dataset:

`artifacts/stage_b/primitives_seed_clip_shortlabel_h2_f8_k3_v2`

Builder:

`humanoidverse/scripts/build_stage_b_seed_clip_shortlabel_v2_dataset.py`

Key changes over v1:

- every chunk may receive injected history, not only the first clip chunk
- injection probability: `p_inject=0.35`
- injection mix: standing 25%, cross-label moving 45%, same-label different-clip 30%
- rare labels are replicated, capped by `max_clip_replication=16`
- every label is guaranteed at least one validation clip

Dataset summary:

- total_samples: 6109
- total_rows: 263203
- train clips: 6045
- val clips: 64
- train rows: 261629
- val rows: 1574
- history_source_counts_post_injection:
  - clip_previous_frames: 167357
  - cross_label_moving: 32616
  - same_label_different_clip: 39614
  - standing: 23616

Training:

`results/stage_b_seed_clip_shortlabel_v2_raw_d30/20260417_162319`

Summary:

- target: raw
- objective: diffusion
- diffusion_steps: 30
- best_val_loss: 0.0812746880369054
- best epoch: 9
- best checkpoint: `stage_b_best.pt`

Important interpretation:

- v2 val loss is higher than v1 because the val set includes much harder transition cases.
- v2 produced a better demo-level behavior in user testing: response speed and action switching were acceptable.
- remaining poor labels include forward walk/run, mainly because those labels are still sparse/noisy in the source data.
- jump quality is currently treated as a BFM low-level limitation, not a Stage B priority.

### TextOp-BABEL 29dof Path

Dataset:

`artifacts/stage_b/textop_babel_latent_h2_f8_raw`

Builder:

`humanoidverse/scripts/build_textop_babel_latent_primitives.py`

Summary:

- source format: TextOp packed BABEL clips
- source_dof_dims: `[29]`
- target_dof: 29
- total_samples: 8535
- total_rows: 659862
- train rows: 480495
- val rows: 179367
- target_representation: raw

Training:

`results/stage_b_textop_babel_h2_f8_raw/20260414_144928`

Summary:

- best_val_loss: 0.015998343413660195
- best checkpoint: `stage_b_best.pt`

This path is complete enough to use as a data/training comparison, but the current demo path is the SEED short-label v2 checkpoint.

## Deploy State

Deploy work is in the sibling repo:

`../BFM-zero-deploy`

Active launcher:

`../BFM-zero-deploy/rl_policy/textop.sh`

Active config:

`../BFM-zero-deploy/config/exp/textop/dct_f8_k3_demo.yaml`

Despite the file name, the active config now points to the v2 raw checkpoint:

`/home/hanwei/code/BFM-Zero/results/stage_b_seed_clip_shortlabel_v2_raw_d30/20260417_162319/stage_b_best.pt`

Current deploy settings:

- `guidance_scale: 2.0`
- `sampling_seed: 1234`
- `sampling_method: ddim`
- `sampling_steps: 30`
- `ddim_eta: 0.0`
- `async_generation: true`
- `generator_rate_hz: 6.25`
- init history: `../BFM-zero-deploy/model/tracking_seed_2k_s/zs_seed2k_init_stand_latest.pkl`

Deploy fixes already made:

- deterministic initial diffusion noise via `sampling_seed`
- DDIM sampling support
- text embedding cache
- async prefetch
- prompt switch invalidates stale rollout requests
- repeated same-text same-seed behavior is reproducible
- different texts separate better with `guidance_scale=2.0`

Do not redo the p-reset randomness debug unless a new issue appears.

## Asset Backup and Environment Reproducibility (2026-06-04)

This section covers the machine-migration prep done on 2026-06-04: large gitignored
assets were backed up to private Hugging Face model repos, and both repos were given
reproducible `uv` lockfiles. The goal is that a fresh machine can fully restore code +
assets + environments.

### Hugging Face asset repos (private)

Two private HF model repos under the `FloatRIslet` namespace hold the gitignored data
that GitHub cannot carry. Both were uploaded with the proxy + xet disabled (see
"Upload gotchas" below).

`FloatRIslet/BFM-Zero-assets` (709 files) — mirrors BFM-Zero relative paths:

- `artifacts/stage_b/primitives_seed_full_parquet/` (541 files, ~10G) — **unrebuildable
  latent root data**; the original SEED motion source is gone, this is the only copy.
- `artifacts/stage_b/primitives_seed_clip_shortlabel_h2_f8_k3_v2/` (58 files) — v2 dataset.
- `humanoidverse/data/seed_train_*.pkl` (7 files) — SEED RL motion inputs (babel/lafan-only
  pkls intentionally excluded).
- `results/stage_b_seed_clip_shortlabel_v2_raw_d30/20260417_162319/` (6 files) — v2 Stage B
  run: `stage_b_best.pt` + `stage_b_last.pt` + config/summary/log/tensorboard. Per-epoch
  `checkpoints/` excluded.
- `results/bfmzero-isaac/<7 SEED runs>/` (95 files) — SEED RL runs. Each keeps
  `checkpoint/model/` + `optimizers.pth` + config + tensorboard + exported ONNX.
  **Excluded: the 25G `checkpoint/buffers/` replay buffer per run, and the babel run
  `20260414_195813` entirely.**

`FloatRIslet/BFM-zero-deploy-model` (97 files) — mirrors `BFM-zero-deploy/model/`:
checkpoint (~3.2G) + exported ONNX (~225M) + tracking/goal/reward latent dirs.
`model/.cache/` excluded.

### Fresh-machine restore

Clone the two repos as **siblings** (the deploy config uses a sibling-relative path to
the BFM-Zero checkpoint — see Deploy State), then pull each repo's HF assets:

```bash
cd ~/code
git clone -b dev https://github.com/windigal/BFM-Zero.git
git clone https://github.com/windigal/BFM-zero-deploy.git

cd BFM-Zero          && hf download FloatRIslet/BFM-Zero-assets        --repo-type model --local-dir .
cd ../BFM-zero-deploy && hf download FloatRIslet/BFM-zero-deploy-model --repo-type model --local-dir ./model
```

A write-enabled HF token is needed to re-upload; a read token suffices to download.

### Environment lockfiles (uv)

Both repos now have reproducible `uv` lockfiles pinned to the **currently working
conda envs** — the envs were NOT modified; the lockfiles were generated/regenerated to
match reality. On a fresh machine, `uv sync` in each repo reproduces the env.

`BFM-Zero` — conda env `bfm-zero` (Python 3.11.15). `uv.lock` regenerated (313 packages,
committed as `fe4ee7b`). Three `pyproject.toml` fixes were required to make the lock
resolvable against isaacsim/isaaclab's messy metadata — keep them:

- `[[tool.uv.dependency-metadata]]` for `flatdict==4.0.1` (its sdist build needs
  setuptools but doesn't declare it; it has no runtime deps, so metadata is declared
  directly to skip the build).
- `environments = ["sys_platform == 'linux' and platform_machine == 'x86_64'"]` — lock
  only for the platform actually used; avoids resolving unused aarch64 / non-CPython forks.
- `override-dependencies = ["typing-extensions>=4.13.0"]` — `isaacsim-kernel==5.1.0.0`
  hard-pins `typing-extensions==4.12.2`, conflicting with `tyro>=0.9.18` (needs >=4.13.0).
  The env runs 4.15.0 fine; the override matches reality.

`BFM-zero-deploy` — conda env `bfmdeploy` (Python 3.10.20). Previously had only an
unversioned `requirements.txt` and no lockfile. Created `pyproject.toml` + `uv.lock`
(102 packages, committed as `df26f0a`), pinned to the env's actual versions. Notes:

- `requirements.txt` listed `zmq` (a broken placeholder, version 0.0.0); the real
  package is `pyzmq==27.1.0` and that is what `pyproject.toml` declares.
- `torch==2.10.0+cu128` comes from the PyTorch cu128 index (declared as a `tool.uv.index`).
- `requirements.txt` is now superseded by `pyproject.toml`/`uv.lock`.

### Upload gotchas (if re-uploading large assets to HF)

This network needs the local proxy, and HF's default xet backend crashes here:

- Export the proxy: `HTTPS_PROXY`/`HTTP_PROXY`/`ALL_PROXY = http://127.0.0.1:7897`
  (Clash mixed port — adjust to the actual port on the new machine).
- Set `HF_HUB_DISABLE_XET=1`. With xet enabled, uploads on a flaky SSL link die with
  `TokenRefreshFailure: Cannot send a request, as the client has been closed` plus a
  Python segfault, losing all progress each crash.
- Use `HfApi().upload_large_folder(...)` (NOT `hf upload`): it commits file-by-file with
  on-disk resume state, so a drop only costs the current file instead of the whole batch.


## BFM Components Relevant to Next Work

BFM-Zero exposes the following useful modules:

- Backward map `B(obs) -> z`
- Forward map `F(obs, z, action) -> next_z`
- Actor `(obs, z) -> action`
- Critic `Q(obs, z, action) -> scalar`
- Discriminator `D(obs, z) -> expert/generated score`

In public inference wrappers these methods often run under `@torch.no_grad()`. For critic-guided sampling or RL fine-tuning, use the underlying modules carefully and preserve normalization behavior.

Current BFM inference modes:

- tracking inference: reference motion to prompt sequence
- goal inference: target pose/state to prompt
- reward inference: reward function to prompt
- prompt interpolation and prompt-level adaptation are central paper strengths

## Do Not Repeat Unless Requirements Change

- Do not rebuild `artifacts/stage_b/primitives_seed_clip_shortlabel_h2_f8_k3`.
- Do not rebuild `artifacts/stage_b/primitives_seed_clip_shortlabel_h2_f8_k3_v2`.
- Do not retrain `results/stage_b_seed_clip_shortlabel_raw_d30/20260416_174449` just to recover a working checkpoint.
- Do not retrain `results/stage_b_seed_clip_shortlabel_v2_raw_d30/20260417_162319` unless explicitly continuing or changing the dataset/config.
- Do not rebuild `artifacts/stage_b/textop_babel_latent_h2_f8_raw`.
- Do not rerun TextOp-BABEL training unless doing a controlled comparison.
- Do not revert dirty working tree changes unless the user explicitly asks.

## Recommended Next Technical Step

The recommended next research/engineering step is:

1. Implement a lightweight `Critic/Discriminator-Guided Sampling` prototype for Stage B.
2. Use it as a test-time baseline before implementing full DDPO/DPPO-style RL fine-tuning.
3. Evaluate whether BFM critic/discriminator guidance improves transition quality, forward locomotion consistency, and physical plausibility without breaking text separation.

This directly connects the generator to BFM instead of using BFM only as a low-level executor.

## Open Questions

- Should `sampling_seed: 1234` remain in demo configs, or should stochastic generation be restored after debugging?
- Are the modified tracked files under `humanoidverse/agents/` and `humanoidverse/train.py` intended to be part of this delivery or separate BFM training changes?
- Should forward walk/run be fixed by another data curation pass, by guidance/RL, or both?
- Should TextOp-BABEL become the main training path later, or remain a comparison dataset?
- Should deploy-side async underrun be measured on the real target machine before more generator work?

## Current Repo State

As of 2026-06-04 both repos are committed and the working trees are clean.

- `BFM-Zero` branch `dev`, HEAD `fe4ee7b update uv.lock`. The Stage B / TextOp
  implementation and the modified agent/training files described in earlier handoff
  versions are now committed (HEAD `2d96a65 BFM+Textop` and predecessors). All Stage B
  code lives under `humanoidverse/language/stage_b/` and `humanoidverse/scripts/`.
- `BFM-zero-deploy` branch `main`, HEAD `df26f0a update uv.lock` (preceded by
  `a0bd012 textop: use relative path for stage_b_checkpoint`).
- Neither HEAD has been pushed yet — confirm with the user before pushing to a remote.

Key implementation files (now tracked) for orientation:

- `humanoidverse/language/stage_b/` — Stage B model/dataset/controller
- `humanoidverse/scripts/train_stage_b.py`, `eval_stage_b.py`
- `humanoidverse/scripts/build_stage_b_seed_clip_shortlabel_v2_dataset.py`
- `humanoidverse/scripts/build_textop_babel_latent_primitives.py`
- deploy: `../BFM-zero-deploy/rl_policy/{textop_runtime.py,bfm_zero.py,textop.sh}`,
  `../BFM-zero-deploy/config/exp/textop/`


## Cleanup Candidates

Do not clean these without explicit approval. Candidates are listed here only to make future cleanup easier.

Likely safe to remove after confirming no active comparison depends on them:

- `.codex`
- `artifacts/stage_b/batch_smoke/`
- `artifacts/stage_b/batch_smoke_parquet/`
- `artifacts/stage_b/primitives_seed_clip_shortlabel_smoke/`
- `artifacts/stage_b/primitives_seed_full_parquet_smoke/`
- `artifacts/stage_b/smoke_primitives_seed/`
- `artifacts/stage_b/smoke_primitives_seed_dct_4x16/`
- `artifacts/stage_b/smoke_primitives_seed_jsonl64/`
- `artifacts/stage_b/smoke_primitives_seed_parquet/`
- `artifacts/stage_b/textop_babel_smoke_h2_f8_raw/`
- `artifacts/stage_b/dct_analysis_smoke_f16.json`
- `results/stage_b_smoke_dct_4x16/`
- `results/stage_b_textop_babel_smoke_raw/`
- failed or partial Stage B logs under `results/stage_b/20260414_100242`, `20260414_104528`, `20260414_104632`, `20260414_104643`

Keep for now:

- `artifacts/stage_b/primitives_seed_clip_shortlabel_h2_f8_k3/`
- `artifacts/stage_b/primitives_seed_clip_shortlabel_h2_f8_k3_v2/`
- `artifacts/stage_b/textop_babel_latent_h2_f8_raw/`
- `results/stage_b_seed_clip_shortlabel_raw_d30/20260416_174449/`
- `results/stage_b_seed_clip_shortlabel_v2_raw_d30/20260417_162319/`
- `results/stage_b_textop_babel_h2_f8_raw/20260414_144928/`
- all deploy TextOp runtime/config files until a PR-quality cleanup pass happens
