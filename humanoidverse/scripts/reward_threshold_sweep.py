"""Validate whether a reward THRESHOLD on buffer samples improves inferred z.

Baseline reward_wr_inference weights every state by softmax(10*r): low-reward (often
noisy / behavior-irrelevant) states still leak into z. This script tries hard thresholds:
only states with reward > thr * max_reward contribute to the reward-weighted backward
embedding. For each task and threshold it infers z, rolls it out in BFM-Zero's own env,
and relabels realized reward -- so the comparison is sim2sim-free and uses the same
mujoco reward model as inference.

Usage:
  python -m humanoidverse.scripts.reward_threshold_sweep \
    --model-folder results/bfmzero-isaac/20260606_020139 \
    --buffer clean --thresholds 0.0 0.5 0.8 0.9 --episode-length 500
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import mujoco

import humanoidverse
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim
from humanoidverse.agents.wrappers.humenvbench import get_next
from humanoidverse.envs.g1_env_helper.robot import make_from_name
from humanoidverse.envs.g1_env_helper.bench.reward_eval_hv import relabel
from torch.utils._pytree import tree_map

if getattr(humanoidverse, "__file__", None) is not None:
    HUMANOIDVERSE_DIR = Path(humanoidverse.__file__).parent
else:
    HUMANOIDVERSE_DIR = Path(__file__).parent.parent.parent

# A representative subset spanning stand / locomotion / arms / sit, so the sweep is
# quick but covers the regimes where thresholding should matter most.
TASKS = [
    "move-ego-0-0",
    "move-ego-low0.5-0-0",
    "move-ego-0-0.7",
    "move-ego-0-0.3",
    "move-ego-90-0.3",
    "raisearms-m-m",
    "move-arms-0-0.7-m-m",
    "crouch-0",
    "crouch-0.25",
]

REWARD_XML = str(
    HUMANOIDVERSE_DIR / "data" / "robots" / "g1" / "scene_29dof_freebase_noadditional_actuators.xml"
)


def _z_from_reward_threshold(model, next_obs, reward, thr_frac):
    """z = project( sum_{r > thr} softmax(10*r_kept) * B(next_obs_kept) ).

    thr_frac is a fraction of the max reward in the batch; thr_frac=0 reproduces the
    baseline reward_wr_inference (no states dropped). All masking is done on CPU to match
    the buffer tensors; model.reward_inference moves the kept obs to its device internally.
    """
    reward = reward.detach().cpu().reshape(-1, 1)
    if thr_frac <= 0.0:
        mask = torch.ones_like(reward, dtype=torch.bool)
    else:
        thr = thr_frac * float(reward.max())
        mask = reward > thr
    n_kept = int(mask.sum())
    if n_kept == 0:
        return None, 0
    idx = mask.reshape(-1).nonzero(as_tuple=True)[0]
    kept_obs = tree_map(lambda x: x[idx], next_obs)
    kept_r = reward[idx]
    weight = F.softmax(10.0 * kept_r, dim=0)
    z = model.reward_inference(kept_obs, kept_r, weight).reshape(1, -1)
    return z, n_kept


def main(
    model_folder: Path,
    buffer: str = "clean",
    thresholds: tuple[float, ...] = (0.0, 0.5, 0.8, 0.9),
    num_samples: int = 150_000,
    episode_length: int = 500,
    device: str = "cuda",
    simulator: str = "mujoco",
    max_workers: int = 24,
):
    model_folder = Path(model_folder)
    model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=device)
    model.to(device)
    model.eval()
    with open(model_folder / "config.json") as f:
        config = json.load(f)

    # Load buffer (train = noisy obs+qpos/qvel; train_clean = clean obs only).
    print("Loading buffers...", flush=True)
    t0 = time.time()
    dataset = TrajectoryDictBufferMultiDim.load(model_folder / "checkpoint/buffers/train", device="cpu")
    for k in ("qpos", "qvel"):
        if k not in dataset.output_key_tp1:
            dataset.output_key_tp1.append(k)
    clean_dataset = None
    if buffer == "clean":
        clean_dataset = TrajectoryDictBufferMultiDim.load(
            model_folder / "checkpoint/buffers/train_clean", device="cpu"
        )
        if "observation" not in clean_dataset.output_key_tp1:
            clean_dataset.output_key_tp1.append("observation")
    print(f"loaded in {time.time()-t0:.1f}s; sampling {num_samples}...", flush=True)

    # Sample ONCE; reuse the same transitions across all tasks/thresholds.
    use_full = num_samples >= dataset.size() and hasattr(dataset, "get_full_buffer")
    data = dataset.get_full_buffer() if use_full else dataset.sample(num_samples)
    qpos = np.asarray(get_next("qpos", data))
    qvel = np.asarray(get_next("qvel", data))
    action = np.asarray(data["action"])
    if buffer == "clean" and clean_dataset is not None and dataset.size() == clean_dataset.size():
        next_idxs = getattr(dataset, "last_sample_next_idxs", None)
        if use_full:
            next_obs = get_next("observation", clean_dataset.get_full_buffer())
        elif next_idxs is not None:
            next_obs = tree_map(lambda x: x[next_idxs], clean_dataset.storage["observation"])
        else:
            next_obs = get_next("observation", data)
    else:
        next_obs = get_next("observation", data)

    rmodel = mujoco.MjModel.from_xml_path(REWARD_XML)
    env_cfg = dict(config["env"])
    env_cfg["hydra_overrides"] = list(env_cfg.get("hydra_overrides", [])) + [
        "env.config.max_episode_length_s=10000",
        "env.config.headless=True",
        f"simulator={simulator}",
    ]
    env_cfg["disable_domain_randomization"] = True
    env_cfg["disable_obs_noise"] = True
    env, _ = HumanoidVerseIsaacConfig(**env_cfg).build(1)

    def rollout(z):
        z_t = torch.as_tensor(np.asarray(z, dtype=np.float32), device=device).reshape(1, -1)
        obs, info = env.reset(to_numpy=False, reset_to_default_pose=True)
        fn = make_from_name  # task fn bound per outer loop below
        return obs, info, z_t

    results = {}
    for task in TASKS:
        print(f"\n=== task {task} ===", flush=True)
        reward_np = relabel(rmodel, qpos, qvel, action, make_from_name(task),
                            max_workers=max_workers, process_executor=True)
        reward = torch.as_tensor(reward_np, dtype=torch.float32, device=device).reshape(-1, 1)
        rmax = float(reward.max())
        reward_fn = make_from_name(task)
        results[task] = {}
        for thr in thresholds:
            z, n_kept = _z_from_reward_threshold(model, next_obs, reward, thr)
            if z is None:
                print(f"  thr={thr}: 0 states kept, skipped")
                results[task][str(thr)] = {"mean_reward": None, "n_kept": 0}
                continue
            z_t = torch.as_tensor(z.detach().float().cpu().numpy(), device=device).reshape(1, -1)
            obs, info = env.reset(to_numpy=False, reset_to_default_pose=True)
            rewards = []
            for i in range(episode_length):
                a = model.act(obs, z_t, mean=True)
                obs, _, term, trunc, info = env.step(a, to_numpy=False)
                qp = info["qpos"]; qv = info["qvel"]
                if isinstance(qp, torch.Tensor):
                    qp = qp.cpu().numpy(); qv = qv.cpu().numpy()
                r = relabel(rmodel, qp[:1], qv[:1], np.zeros((1, 29), dtype=np.float32),
                            reward_fn, max_workers=1, process_executor=False)
                rewards.append(float(r.reshape(-1)[0]))
            mr = float(np.mean(rewards))
            results[task][str(thr)] = {"mean_reward": mr, "n_kept": n_kept, "reward_max": rmax}
            print(f"  thr={thr:<4} kept={n_kept:>7} (max_r={rmax:.3f})  realized_mean={mr:.4f}")

    # Summary table
    print("\n=== realized reward by threshold (buffer=%s) ===" % buffer)
    header = "task".ljust(22) + "".join(f"thr={t}".rjust(11) for t in thresholds)
    print(header)
    print("-" * len(header))
    sums = {str(t): [] for t in thresholds}
    for task in TASKS:
        row = task.ljust(22)
        for t in thresholds:
            v = results[task][str(t)]["mean_reward"]
            row += ("--".rjust(11) if v is None else f"{v:.4f}".rjust(11))
            if v is not None:
                sums[str(t)].append(v)
        print(row)
    print("-" * len(header))
    mrow = "MEAN".ljust(22)
    best_t = None; best_v = -1
    for t in thresholds:
        m = float(np.mean(sums[str(t)])) if sums[str(t)] else float("nan")
        mrow += f"{m:.4f}".rjust(11)
        if m > best_v:
            best_v = m; best_t = t
    print(mrow)
    print(f"\n==> best threshold = {best_t} (mean realized reward {best_v:.4f}); thr=0.0 is the baseline.")

    out = model_folder / "reward_inference" / f"threshold_sweep_{buffer}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"buffer": buffer, "thresholds": list(thresholds), "results": results}, f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
