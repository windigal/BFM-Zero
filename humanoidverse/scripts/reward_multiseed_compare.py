"""Multi-seed averaging of realized reward for the 4 z configs.

For each seed we re-sample the buffer (the ONLY source of randomness: rollouts are
deterministic given z -- mean action, fixed default-pose reset, DR/obs-noise disabled),
then from that single sample compute 4 z per task:
  noisy / clean  x  thr0.0 (baseline) / thr0.5 (reward-thresholded, min_count floor)
roll each out, relabel realized reward, and average mean-reward across seeds (report std).

Usage:
  python -m humanoidverse.scripts.reward_multiseed_compare \
    --model-folder results/bfmzero-isaac/20260606_020139 \
    --seeds 0 1 2 --threshold-frac 0.5 --min-count 500 --episode-length 500
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
from humanoidverse.scripts.reward_inference_compare import TASKS as ALL_TASKS
from humanoidverse.scripts.reward_threshold_z_gen import _z_thresholded, _load_next_obs, REWARD_XML
from torch.utils._pytree import tree_map


CONFIGS = ["noisy", "clean", "noisy+thr", "clean+thr"]

# High-variance / sparse-high-reward tasks where a single rollout was unreliable
# (few states pass the threshold, so z depends heavily on which samples are drawn).
HIGH_VARIANCE_TASKS = [
    "move-ego-0-0",
    "move-ego-low0.5-0-0",
    "move-ego-90-0.3",
    "rotate-z-5-0.5",
    "rotate-z--5-0.5",
    "raisearms-m-l",
    "raisearms-m-m",
    "move-arms-180-0.4-m-m",
    "move-arms-90-0.7-m-l",
    "crouch-0",
    "crouch-0.25",
    "spin-arms--5-m-l",
]


def main(
    model_folder: Path,
    seeds: tuple[int, ...] = (0, 1, 2),
    threshold_frac: float = 0.5,
    min_count: int = 500,
    num_samples: int = 150_000,
    episode_length: int = 500,
    device: str = "cuda",
    simulator: str = "mujoco",
    max_workers: int = 24,
    use_all_tasks: bool = False,
):
    TASKS = ALL_TASKS if use_all_tasks else HIGH_VARIANCE_TASKS
    model_folder = Path(model_folder)
    model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=device)
    model.to(device)
    model.eval()
    with open(model_folder / "config.json") as f:
        config = json.load(f)

    print("Loading buffers...", flush=True)
    t0 = time.time()
    dataset = TrajectoryDictBufferMultiDim.load(model_folder / "checkpoint/buffers/train", device="cpu")
    for k in ("qpos", "qvel"):
        if k not in dataset.output_key_tp1:
            dataset.output_key_tp1.append(k)
    clean_dataset = TrajectoryDictBufferMultiDim.load(
        model_folder / "checkpoint/buffers/train_clean", device="cpu"
    )
    if "observation" not in clean_dataset.output_key_tp1:
        clean_dataset.output_key_tp1.append("observation")
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

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

    def rollout_reward(z, reward_fn):
        z_t = torch.as_tensor(np.asarray(z, dtype=np.float32), device=device).reshape(1, -1)
        obs, info = env.reset(to_numpy=False, reset_to_default_pose=True)
        rs = []
        for _ in range(episode_length):
            a = model.act(obs, z_t, mean=True)
            obs, _, term, trunc, info = env.step(a, to_numpy=False)
            qp = info["qpos"]; qv = info["qvel"]
            if isinstance(qp, torch.Tensor):
                qp = qp.cpu().numpy(); qv = qv.cpu().numpy()
            rr = relabel(rmodel, qp[:1], qv[:1], np.zeros((1, 29), dtype=np.float32),
                         reward_fn, max_workers=1, process_executor=False)
            rs.append(float(rr.reshape(-1)[0]))
        return float(np.mean(rs))

    # per_seed[config][task] = realized mean reward
    per_seed = {c: {t: [] for t in TASKS} for c in CONFIGS}
    for si, seed in enumerate(seeds):
        print(f"\n########## seed {seed} ({si+1}/{len(seeds)}) ##########", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        data = dataset.sample(num_samples)
        qpos = np.asarray(get_next("qpos", data))
        qvel = np.asarray(get_next("qvel", data))
        action = np.asarray(data["action"])
        next_obs = {
            "noisy": _load_next_obs("noisy", dataset, clean_dataset, data, False),
            "clean": _load_next_obs("clean", dataset, clean_dataset, data, False),
        }
        for task in TASKS:
            reward_np = relabel(rmodel, qpos, qvel, action, make_from_name(task),
                                max_workers=max_workers, process_executor=True)
            reward = torch.as_tensor(reward_np, dtype=torch.float32).reshape(-1, 1)
            reward_fn = make_from_name(task)
            for buf in ("noisy", "clean"):
                for thr, cfg in ((0.0, buf), (threshold_frac, buf + "+thr")):
                    z, _, _ = _z_thresholded(model, next_obs[buf], reward, thr, min_count)
                    mr = rollout_reward(z.detach().float().cpu().numpy(), reward_fn)
                    per_seed[cfg][task].append(mr)
            row = " ".join(f"{c}={per_seed[c][task][-1]:.3f}" for c in CONFIGS)
            print(f"[seed {seed}] {task:<22} {row}", flush=True)

    # Aggregate: mean +/- std across seeds
    print(f"\n=== mean realized reward over {len(seeds)} seeds (±std) ===")
    hdr = f"{'task':<22}" + "".join(f"{c:>18}" for c in CONFIGS)
    print(hdr); print("-" * len(hdr))
    agg = {c: {} for c in CONFIGS}
    for task in TASKS:
        row = f"{task:<22}"
        for c in CONFIGS:
            arr = np.asarray(per_seed[c][task])
            agg[c][task] = {"mean": float(arr.mean()), "std": float(arr.std())}
            row += f"{arr.mean():>10.4f}±{arr.std():.3f}"
        print(row)
    print("-" * len(hdr))
    means = {c: float(np.mean([agg[c][t]["mean"] for t in TASKS])) for c in CONFIGS}
    mrow = f"{'OVERALL MEAN':<22}"
    for c in CONFIGS:
        mrow += f"{means[c]:>18.4f}"
    print(mrow)
    best = max(means, key=means.get)
    print(f"\n==> best config overall = {best} ({means[best]:.4f})")
    # per-config #1 counts (by seed-averaged mean)
    cnt = {c: 0 for c in CONFIGS}
    for task in TASKS:
        vals = {c: agg[c][task]["mean"] for c in CONFIGS}
        b = max(vals, key=vals.get)
        if vals[b] - sorted(vals.values())[-2] > 1e-4:
            cnt[b] += 1
    print("per-task #1 counts:", cnt)

    out = model_folder / "reward_inference" / f"multiseed_compare_thr{threshold_frac}.json"
    with open(out, "w") as f:
        json.dump({"seeds": list(seeds), "threshold_frac": threshold_frac,
                   "overall_mean": means, "per_task": agg}, f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
