import os

os.environ["MUJOCO_GL"] = "egl"  # Use EGL for rendering
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import json
import pickle
import numpy as np
import torch
import joblib
import rich
import time

import humanoidverse
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim
from humanoidverse.agents.buffers.transition import DictBuffer
from humanoidverse.envs.g1_env_helper.bench import RewardWrapperHV
from humanoidverse.utils.helpers import export_meta_policy_as_onnx

# Resolve humanoidverse root directory
if getattr(humanoidverse, "__file__", None) is not None:
    HUMANOIDVERSE_DIR = Path(humanoidverse.__file__).parent
else:
    HUMANOIDVERSE_DIR = Path(__file__).parent.parent.parent


# Same preset task list as reward_inference.py (kept in sync intentionally).
TASKS = [
    # stand
    "move-ego-0-0",
    "move-ego-low0.5-0-0",
    # locomotion medium
    "move-ego-0-0.7",
    # locomotion slow
    "move-ego-0-0.3",
    "move-ego-90-0.3",
    "move-ego-180-0.3",
    "move-ego--90-0.3",
    # spin
    "rotate-z-5-0.5",
    "rotate-z--5-0.5",
    # raise arms
    "raisearms-l-l",
    "raisearms-l-m",
    "raisearms-m-l",
    "raisearms-m-m",
    # move + arms
    "move-arms-0-0.7-m-m",
    "move-arms-90-0.7-m-m",
    "move-arms-180-0.4-m-m",
    "move-arms--90-0.7-m-m",
    "move-arms-0-0.7-l-m",
    "move-arms-90-0.7-l-m",
    "move-arms-180-0.4-l-m",
    "move-arms--90-0.7-l-m",
    "move-arms-0-0.7-m-l",
    "move-arms-90-0.7-m-l",
    "move-arms-180-0.4-m-l",
    "move-arms--90-0.7-m-l",
    "move-arms-0-0.7-l-l",
    "move-arms-90-0.7-l-l",
    "move-arms-180-0.4-l-l",
    "move-arms--90-0.7-l-l",
    # spin + arms
    "spin-arms-5-l-l",
    "spin-arms--5-l-l",
    "spin-arms-5-l-m",
    "spin-arms--5-l-m",
    "spin-arms-5-m-l",
    "spin-arms--5-m-l",
    # sit
    "crouch-0",
    "crouch-0.25",
    "sitonground",
]


def _build_wrapper(model, dataset, clean_dataset, num_samples, max_workers, process_executor):
    return RewardWrapperHV(
        model=model,
        inference_dataset=dataset,
        clean_inference_dataset=clean_dataset,
        num_samples_per_inference=num_samples,
        inference_function="reward_wr_inference",
        max_workers=max_workers,
        process_executor=process_executor,
        env_model=str(
            HUMANOIDVERSE_DIR / "data" / "robots" / "g1" / "scene_29dof_freebase_noadditional_actuators.xml"
        ),
    )


def _infer_all_tasks(wrapper, tasks, n_inferences, out_path, tag):
    z_dict = {}
    for r in range(n_inferences):
        for task in tasks:
            print(f"[{tag}] inference for {task}...", end=" ", flush=True)
            start_t = time.time()
            z = wrapper.reward_inference(task=task)
            # Save as float32 numpy: z may be bfloat16 on GPU, which deploy's
            # np.concatenate cannot handle. Convert here so the pkl is deploy-ready.
            z_np = z.detach().float().cpu().numpy()
            z_dict[task] = z_dict.get(task, []) + [z_np]
            print(f"done in {time.time()-start_t:.1f}s")
            # Deploy reads reward z via pickle.load (not joblib), so save plain pickle.
            with open(out_path, "wb") as f:
                pickle.dump(z_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[{tag}] saved {out_path}")
    return z_dict


def _compare_z(clean_dict, noisy_dict):
    print("\n=== clean vs noisy z comparison ===")
    print(f"{'task':<28} {'cosine':>8} {'L2':>10}")
    for task in clean_dict:
        if task not in noisy_dict:
            continue
        zc = torch.as_tensor(clean_dict[task][0]).reshape(-1).float()
        zn = torch.as_tensor(noisy_dict[task][0]).reshape(-1).float()
        cos = torch.nn.functional.cosine_similarity(zc, zn, dim=0).item()
        l2 = torch.linalg.norm(zc - zn).item()
        flag = "" if cos > 0.9 else "  <-- LOW COSINE"
        print(f"{task:<28} {cos:>8.4f} {l2:>10.4f}{flag}")


def _rollout_realized_reward(model, env, z_dict, tasks, env_model_path, episode_length, device, tag):
    """Roll out each task's z in BFM-Zero's own env and accumulate realized task reward.

    Reward is relabeled per step from the env's ground-truth qpos/qvel using the same
    task reward function used at inference time -- so clean vs noisy is compared with an
    identical, sim2sim-free reward signal.
    """
    import mujoco
    from humanoidverse.envs.g1_env_helper.robot import make_from_name
    from humanoidverse.envs.g1_env_helper.bench.reward_eval_hv import relabel

    rmodel = mujoco.MjModel.from_xml_path(env_model_path)
    results = {}
    for task in tasks:
        z = z_dict[task][0]
        if isinstance(z, torch.Tensor):
            z = z.float().cpu().numpy()
        z_t = torch.as_tensor(np.asarray(z, dtype=np.float32), device=device).reshape(1, -1)
        reward_fn = make_from_name(task)
        obs, info = env.reset(to_numpy=False, reset_to_default_pose=True)
        rewards = []
        fell_at = -1
        for i in range(episode_length):
            action = model.act(obs, z_t, mean=True)
            obs, _, terminated, truncated, info = env.step(action, to_numpy=False)
            qpos = info["qpos"]
            qvel = info["qvel"]
            if isinstance(qpos, torch.Tensor):
                qpos = qpos.cpu().numpy()
                qvel = qvel.cpu().numpy()
            r = relabel(rmodel, qpos[:1], qvel[:1], np.zeros((1, 29), dtype=np.float32),
                        reward_fn, max_workers=1, process_executor=False)
            rewards.append(float(r.reshape(-1)[0]))
            term = bool(terminated.any()) if hasattr(terminated, "any") else bool(terminated)
            if term and fell_at < 0:
                fell_at = i
        rewards = np.asarray(rewards)
        results[task] = {
            "mean_reward": float(rewards.mean()),
            "sum_reward": float(rewards.sum()),
            "fell_at": fell_at,
        }
        print(f"[{tag}] {task:<22} mean={rewards.mean():.4f} sum={rewards.sum():.2f} fell_at={fell_at}")
    return results


def _print_rollout_compare(clean_res, noisy_res, out_json):
    print("\n=== clean vs noisy realized reward (BFM-Zero env rollout) ===")
    print(f"{'task':<24} {'clean':>9} {'noisy':>9} {'Δ(c-n)':>9}  winner")
    print("-" * 64)
    clean_wins = noisy_wins = 0
    sc = sn = 0.0
    n = 0
    for task in clean_res:
        c = clean_res[task]["mean_reward"]
        no = noisy_res[task]["mean_reward"]
        delta = c - no
        winner = "clean" if delta > 1e-4 else ("noisy" if delta < -1e-4 else "tie")
        if winner == "clean":
            clean_wins += 1
        elif winner == "noisy":
            noisy_wins += 1
        sc += c
        sn += no
        n += 1
        print(f"{task:<24} {c:>9.4f} {no:>9.4f} {delta:>9.4f}  {winner}")
    print("-" * 64)
    if n:
        print(f"{'MEAN':<24} {sc/n:>9.4f} {sn/n:>9.4f} {(sc-sn)/n:>9.4f}")
    print(f"\nwins: clean={clean_wins} noisy={noisy_wins}")
    if n:
        verdict = "CLEAN" if sc > sn else ("NOISY" if sn > sc else "TIE")
        print(f"==> {verdict} buffer z achieves higher mean realized reward overall.")
    out = {"clean": clean_res, "noisy": noisy_res}
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {out_json}")


def main(
    model_folder: Path,
    data_path: Path | None = None,
    device: str = "cuda",
    simulator: str = "isaacsim",
    num_samples: int = 150_000,
    n_inferences: int = 1,
    max_workers: int = 24,
    process_executor: bool = True,
    skip_rollouts: bool = True,
    export_onnx: bool = True,
    rollout_compare: bool = False,
    episode_length: int = 500,
    reuse_z: bool = False,
):
    """Produce TWO z-dicts (clean + noisy) from the same run's buffers for comparison.

    Unlike reward_inference.py (single z-dict), this builds two RewardWrapperHV:
    - noisy: B network fed with noisy next-obs (clean_inference_dataset=None)
    - clean: B network fed with clean next-obs (clean_inference_dataset=train_clean)
    Both relabel reward from the SAME noisy buffer qpos/qvel, so only the B input differs.
    """
    model_folder = Path(model_folder)
    model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=device)
    model.to(device)
    model.eval()
    model_name = model.__class__.__name__
    with open(model_folder / "config.json", "r") as f:
        config = json.load(f)

    out_dir = model_folder / "reward_inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    if reuse_z:
        # Skip buffer load + inference; load the already-saved z pkls and (optionally) roll out.
        print("reuse_z=True: loading existing z pkls, skipping buffer load + inference")
        with open(out_dir / "reward_locomotion_noisy.pkl", "rb") as f:
            noisy_dict = pickle.load(f)
        with open(out_dir / "reward_locomotion_clean.pkl", "rb") as f:
            clean_dict = pickle.load(f)
    else:
        # Load the main (noisy) buffer + the clean buffer.
        print("Loading buffers...", flush=True)
        start_t = time.time()
        buffer_path = model_folder / "checkpoint/buffers/train"
        dataset = TrajectoryDictBufferMultiDim.load(buffer_path, device="cpu")
        clean_buffer_path = model_folder / "checkpoint/buffers/train_clean"
        if not clean_buffer_path.is_dir():
            raise FileNotFoundError(
                f"clean buffer not found at {clean_buffer_path}; this script requires both train and train_clean"
            )
        clean_dataset = TrajectoryDictBufferMultiDim.load(clean_buffer_path, device="cpu")
        print(f"loaded train + train_clean in {time.time()-start_t:.1f}s")
        if dataset.size() != clean_dataset.size():
            print(
                f"WARNING: buffer sizes differ (train={dataset.size()} clean={clean_dataset.size()}). "
                "Clean next-obs alignment falls back to noisy; comparison may be invalid."
            )

        if export_onnx:
            output_dir = model_folder / "exported"
            output_dir.mkdir(parents=True, exist_ok=True)
            export_meta_policy_as_onnx(
                model,
                output_dir,
                f"{model_name}.onnx",
                {"actor_obs": torch.randn(1, model._actor.input_filter.output_space.shape[0] + model.cfg.archi.z_dim)},
                z_dim=model.cfg.archi.z_dim,
                history=("history_actor" in model.cfg.archi.actor.input_filter.key),
                use_29dof=True,
            )
            print(f"Exported ONNX to {output_dir}/{model_name}.onnx")

        # noisy z
        noisy_wrapper = _build_wrapper(model, dataset, None, num_samples, max_workers, process_executor)
        noisy_dict = _infer_all_tasks(
            noisy_wrapper, TASKS, n_inferences, out_dir / "reward_locomotion_noisy.pkl", "noisy"
        )

        # clean z
        clean_wrapper = _build_wrapper(model, dataset, clean_dataset, num_samples, max_workers, process_executor)
        clean_dict = _infer_all_tasks(
            clean_wrapper, TASKS, n_inferences, out_dir / "reward_locomotion_clean.pkl", "clean"
        )

    _compare_z(clean_dict, noisy_dict)

    if rollout_compare:
        print("\nBuilding env for in-repo rollout comparison...", flush=True)
        env_cfg = dict(config["env"])
        env_cfg["hydra_overrides"] = list(env_cfg.get("hydra_overrides", [])) + [
            "env.config.max_episode_length_s=10000",
            "env.config.headless=True",
            f"simulator={simulator}",
        ]
        env_cfg["disable_domain_randomization"] = True
        env_cfg["disable_obs_noise"] = True
        wrapped_env, _ = HumanoidVerseIsaacConfig(**env_cfg).build(1)
        env_model_path = str(
            HUMANOIDVERSE_DIR / "data" / "robots" / "g1" / "scene_29dof_freebase_noadditional_actuators.xml"
        )
        noisy_res = _rollout_realized_reward(
            model, wrapped_env, noisy_dict, TASKS, env_model_path, episode_length, device, "noisy"
        )
        clean_res = _rollout_realized_reward(
            model, wrapped_env, clean_dict, TASKS, env_model_path, episode_length, device, "clean"
        )
        _print_rollout_compare(clean_res, noisy_res, out_dir / "realized_reward_compare.json")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
