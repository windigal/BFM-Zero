"""Generate thresholded reward-inferred z for ALL preset tasks, on BOTH buffers.

For each task and each buffer (noisy = train obs, clean = train_clean obs), build
  z = project( sum_{r > 0.5*max_r} softmax(10*r) * B(next_obs) )
with a min-count floor (if too few states pass the threshold, fall back to the top-K
highest-reward states) so z never collapses onto a handful of samples. Records the kept
ratio and the rollout realized reward per task, and saves deploy-ready z pkls.

Usage:
  python -m humanoidverse.scripts.reward_threshold_z_gen \
    --model-folder results/bfmzero-isaac/20260606_020139 \
    --threshold-frac 0.5 --min-count 500 --episode-length 500
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import pickle
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
from humanoidverse.scripts.reward_inference_compare import TASKS
from torch.utils._pytree import tree_map

if getattr(humanoidverse, "__file__", None) is not None:
    HUMANOIDVERSE_DIR = Path(humanoidverse.__file__).parent
else:
    HUMANOIDVERSE_DIR = Path(__file__).parent.parent.parent

REWARD_XML = str(
    HUMANOIDVERSE_DIR / "data" / "robots" / "g1" / "scene_29dof_freebase_noadditional_actuators.xml"
)


def _z_thresholded(model, next_obs, reward, thr_frac, min_count):
    """z from states with reward > thr_frac*max, with a top-K fallback floor.

    Returns (z[1,d], n_kept, total). All masking on CPU (buffer tensors are CPU;
    model.reward_inference moves kept obs to device internally).
    """
    reward = reward.detach().cpu().reshape(-1, 1)
    total = reward.shape[0]
    rmax = float(reward.max())
    if thr_frac <= 0.0 or rmax <= 0.0:
        idx = torch.arange(total)
    else:
        mask = (reward > thr_frac * rmax).reshape(-1)
        n = int(mask.sum())
        if n >= min_count:
            idx = mask.nonzero(as_tuple=True)[0]
        else:
            # too few above threshold -> take the top-min_count highest-reward states
            k = min(min_count, total)
            idx = torch.topk(reward.reshape(-1), k).indices
    kept_obs = tree_map(lambda x: x[idx], next_obs)
    kept_r = reward[idx]
    weight = F.softmax(10.0 * kept_r, dim=0)
    z = model.reward_inference(kept_obs, kept_r, weight).reshape(1, -1)
    return z, int(idx.numel()), total


def _load_next_obs(buffer_tag, dataset, clean_dataset, data, use_full):
    if buffer_tag == "noisy":
        return get_next("observation", data)
    if dataset.size() != clean_dataset.size():
        return get_next("observation", data)
    if use_full:
        return get_next("observation", clean_dataset.get_full_buffer())
    next_idxs = getattr(dataset, "last_sample_next_idxs", None)
    if next_idxs is not None:
        return tree_map(lambda x: x[next_idxs], clean_dataset.storage["observation"])
    return get_next("observation", data)


def main(
    model_folder: Path,
    threshold_frac: float = 0.5,
    min_count: int = 500,
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
    print(f"loaded in {time.time()-t0:.1f}s; sampling {num_samples}...", flush=True)

    use_full = num_samples >= dataset.size() and hasattr(dataset, "get_full_buffer")
    data = dataset.get_full_buffer() if use_full else dataset.sample(num_samples)
    qpos = np.asarray(get_next("qpos", data))
    qvel = np.asarray(get_next("qvel", data))
    action = np.asarray(data["action"])
    next_obs_by_buffer = {
        "noisy": _load_next_obs("noisy", dataset, clean_dataset, data, use_full),
        "clean": _load_next_obs("clean", dataset, clean_dataset, data, use_full),
    }

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
    out_dir = model_folder / "reward_inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Relabel each task's reward ONCE (shared across buffers).
    print("Relabeling task rewards...", flush=True)
    reward_by_task = {}
    for task in TASKS:
        r = relabel(rmodel, qpos, qvel, action, make_from_name(task),
                    max_workers=max_workers, process_executor=True)
        reward_by_task[task] = torch.as_tensor(r, dtype=torch.float32).reshape(-1, 1)

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

    summary = {"threshold_frac": threshold_frac, "min_count": min_count, "tasks": {}}
    z_dicts = {"noisy": {}, "clean": {}}
    for buffer_tag in ("noisy", "clean"):
        next_obs = next_obs_by_buffer[buffer_tag]
        print(f"\n########## buffer={buffer_tag} ##########", flush=True)
        for task in TASKS:
            reward = reward_by_task[task]
            z, n_kept, total = _z_thresholded(model, next_obs, reward, threshold_frac, min_count)
            z_np = z.detach().float().cpu().numpy()
            z_dicts[buffer_tag][task] = [z_np]
            mr = rollout_reward(z_np, make_from_name(task))
            kept_frac = n_kept / max(total, 1)
            summary["tasks"].setdefault(task, {})[buffer_tag] = {
                "kept": n_kept, "total": total, "kept_frac": kept_frac,
                "reward_max": float(reward.max()), "realized_mean": mr,
            }
            print(f"[{buffer_tag}] {task:<22} kept={n_kept:>7}/{total} ({kept_frac*100:5.2f}%)  realized={mr:.4f}", flush=True)
        # save deploy-ready z pkl per buffer
        out_pkl = out_dir / f"reward_locomotion_{buffer_tag}_thr{threshold_frac}.pkl"
        with open(out_pkl, "wb") as f:
            pickle.dump(z_dicts[buffer_tag], f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[{buffer_tag}] saved {out_pkl}")

    # Summary table: realized reward noisy vs clean (both thresholded)
    print(f"\n=== thresholded (thr={threshold_frac}, min_count={min_count}) realized reward ===")
    print(f"{'task':<22}{'kept%(n)':>10}{'kept%(c)':>10}{'noisy':>9}{'clean':>9}{'Δ(c-n)':>9}")
    print("-" * 69)
    sn = sc = 0.0; n = 0; cwin = nwin = 0
    for task in TASKS:
        tn = summary["tasks"][task]["noisy"]; tc = summary["tasks"][task]["clean"]
        d = tc["realized_mean"] - tn["realized_mean"]
        if d > 1e-4: cwin += 1
        elif d < -1e-4: nwin += 1
        sn += tn["realized_mean"]; sc += tc["realized_mean"]; n += 1
        print(f"{task:<22}{tn['kept_frac']*100:>9.2f}{tc['kept_frac']*100:>10.2f}"
              f"{tn['realized_mean']:>9.4f}{tc['realized_mean']:>9.4f}{d:>9.4f}")
    print("-" * 69)
    print(f"{'MEAN':<22}{'':>10}{'':>10}{sn/n:>9.4f}{sc/n:>9.4f}{(sc-sn)/n:>9.4f}")
    print(f"wins: clean={cwin} noisy={nwin}")

    summary["mean_realized"] = {"noisy": sn / n, "clean": sc / n}
    out_json = out_dir / f"threshold_z_gen_thr{threshold_frac}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {out_json}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
