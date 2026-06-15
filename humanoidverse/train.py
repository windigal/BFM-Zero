# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys


def _normalize_gpu_id(gpu_id: str | int | None) -> str | None:
    if gpu_id is None:
        return None
    gpu_id_text = str(gpu_id).strip()
    if not gpu_id_text:
        raise ValueError("--gpu-id cannot be empty")
    if gpu_id_text.startswith("cuda:"):
        gpu_id_text = gpu_id_text.split(":", 1)[1]

    gpu_ids = [part.strip() for part in gpu_id_text.split(",")]
    if any(not part.isdigit() for part in gpu_ids):
        raise ValueError(f"--gpu-id must be a CUDA device index like '0' or '1'; got {gpu_id!r}")
    return ",".join(str(int(part)) for part in gpu_ids)


def _configure_cuda_visible_devices(gpu_id: str | int | None) -> str | None:
    normalized_gpu_id = _normalize_gpu_id(gpu_id)
    if normalized_gpu_id is None:
        return None

    current_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if current_visible_devices == normalized_gpu_id:
        return normalized_gpu_id

    torch_module = sys.modules.get("torch")
    if torch_module is not None and torch_module.cuda.is_initialized():
        raise RuntimeError(
            "Cannot apply --gpu-id after CUDA has already been initialized. "
            "Launch a fresh process with --gpu-id, or set CUDA_VISIBLE_DEVICES before Python starts."
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = normalized_gpu_id
    return normalized_gpu_id


def _preparse_gpu_id(argv: list[str]) -> str | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu-id", default=None)
    args, _ = parser.parse_known_args(argv[1:])
    return args.gpu_id


_PRESELECTED_GPU_ID = _configure_cuda_visible_devices(_preparse_gpu_id(sys.argv))

from humanoidverse.agents.evaluations.humanoidverse_isaac import (
    HumanoidVerseIsaacTrackingEvaluation,
    HumanoidVerseIsaacTrackingEvaluationConfig,
)
from humanoidverse.agents.envs.humanoidverse_isaac import load_expert_trajectories_from_motion_lib, HumanoidVerseIsaacConfig

os.environ["OMP_NUM_THREADS"] = "1"

import torch

torch.set_float32_matmul_precision("high")

import json
import time
import typing as tp
import warnings
from decimal import Decimal
from pathlib import Path
from typing import Dict, List
from collections import OrderedDict
from torch.utils._pytree import tree_map

# import exca as xk
import gymnasium
import numpy as np
import pydantic
import torch  # better to use scoped import if we use processes
import tyro
import wandb
from packaging.version import Version
from torch.utils.tensorboard import SummaryWriter
from torch.utils._pytree import tree_map
from tqdm import tqdm


from humanoidverse.agents.base import BaseConfig
from humanoidverse.agents.buffers.load_data import load_expert_trajectories
from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim
from humanoidverse.agents.buffers.transition import DictBuffer, dtype_numpytotorch_lower_precision
from humanoidverse.agents.fb_cpr.agent import FBcprAgentConfig
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
from humanoidverse.agents.misc.loggers import CSVLogger
from humanoidverse.agents.utils import EveryNStepsChecker, get_local_workdir, set_seed_everywhere

TRAIN_LOG_FILENAME = "train_log.txt"
REWARD_EVAL_LOG_FILENAME = "reward_eval_log.csv"
TRACKING_EVAL_LOG_FILENAME = "tracking_eval_log.csv"

CHECKPOINT_DIR_NAME = "checkpoint"
WANDB_RUN_INFO_FILENAME = "wandb_run.json"

TRAIN_METRIC_GROUPS = OrderedDict(
    [
        ("FB", ["B", "B_norm", "z_norm", "F1", "M1", "target_M", "fb_diag", "fb_offdiag", "orth_loss_diag", "orth_loss_offdiag", "orth_loss", "q_loss", "fb_loss"]),
        ("Discriminator", ["disc_expert_loss", "disc_train_loss", "disc_wgan_gp_loss", "disc_loss", "mean_disc_reward"]),
        ("Critic", ["target_Q", "Q1", "mean_next_Q", "unc_Q", "critic_loss"]),
        ("Aux Critic", ["target_auxQ", "auxQ1", "mean_next_auxQ", "unc_auxQ", "aux_critic_loss", "mean_aux_reward"]),
        ("Actor", ["Q_discriminator", "Q_aux", "Q_fb", "actor_loss"]),
        (
            "Aux Rewards",
            [
                "aux_rew/penalty_torques",
                "aux_rew/penalty_action_rate",
                "aux_rew/limits_dof_pos",
                "aux_rew/limits_torque",
                "aux_rew/penalty_undesired_contact",
                "aux_rew/penalty_feet_ori",
                "aux_rew/penalty_ankle_roll",
                "aux_rew/penalty_slippage",
            ],
        ),
    ]
)


def _format_metric_value(value: float) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    text = format(Decimal(str(float(value))), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in {"", "-0"}:
        return "0"
    return text


def _format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _make_timestamped_result_dir(base_dir: str) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return str(Path(base_dir) / timestamp)


def _read_checkpoint_time(run_dir: Path) -> int | None:
    status_path = run_dir / CHECKPOINT_DIR_NAME / "train_status.json"
    if not status_path.exists():
        return None
    try:
        with status_path.open("r") as f:
            train_status = json.load(f)
        return int(train_status["time"])
    except Exception:
        return None


def _find_latest_resumable_run(base_dir: Path, *, max_time: int | None = None) -> Path | None:
    if not base_dir.exists():
        return None
    candidates: list[Path] = []
    for run_dir in sorted(base_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        checkpoint_time = _read_checkpoint_time(run_dir)
        if checkpoint_time is None:
            continue
        if max_time is not None and checkpoint_time >= max_time:
            continue
        candidates.append(run_dir)
    return candidates[0] if candidates else None


def _build_grouped_metrics(metrics: dict[str, float]) -> OrderedDict[str, list[tuple[str, float]]]:
    grouped = OrderedDict()
    used = set()
    for group_name, keys in TRAIN_METRIC_GROUPS.items():
        items = [(key, metrics[key]) for key in keys if key in metrics]
        if items:
            grouped[group_name] = items
            used.update(key for key, _ in items)
    leftover = [(key, metrics[key]) for key in sorted(metrics.keys()) if key not in used]
    if leftover:
        grouped["Other"] = leftover
    return grouped


def _iter_tracked_group_metrics(
    grouped_metrics: OrderedDict[str, list[tuple[str, float]]],
) -> tp.Iterator[tuple[str, str, float]]:
    for group_name in TRAIN_METRIC_GROUPS.keys():
        for key, value in grouped_metrics.get(group_name, []):
            yield group_name, key, value


def _render_grouped_train_log(
    *,
    grouped_metrics: OrderedDict[str, list[tuple[str, float]]],
    timestep: int,
    max_timesteps: int,
    iteration: int,
    max_iterations: int,
    steps_per_second: float,
    collection_time: float,
    learning_time: float,
    elapsed_time: float,
    eta_seconds: float,
) -> str:
    width = 88
    summary_labels = ["Computation", "Total timesteps", "Iteration time", "Time elapsed", "ETA"]
    metric_labels = [
        key if "/" in key else f"{group_name}/{key}"
        for group_name, items in grouped_metrics.items()
        for key, _ in items
    ]
    label_width = max([len(label) for label in metric_labels + summary_labels], default=0)

    lines = [
        "#" * width,
        f"{'Learning iteration ' + str(iteration) + '/' + str(max_iterations):^{width}}",
        "",
        (
            f"{'Computation':>{label_width}}: "
            f"{_format_metric_value(steps_per_second)} steps/s "
            f"(collection: {collection_time:.3f}s, learning: {learning_time:.3f}s)"
        ),
    ]

    for group_name, items in grouped_metrics.items():
        for key, value in items:
            metric_label = key if "/" in key else f"{group_name}/{key}"
            lines.append(f"{metric_label:>{label_width}}: {_format_metric_value(value)}")

    lines.extend(
        [
            "-" * width,
            f"{'Total timesteps':>{label_width}}: {timestep}",
            f"{'Iteration time':>{label_width}}: {collection_time + learning_time:.2f}s",
            f"{'Time elapsed':>{label_width}}: {_format_duration(elapsed_time)}",
            f"{'ETA':>{label_width}}: {_format_duration(eta_seconds)}",
        ]
    )
    return "\n".join(lines)


def _print_grouped_train_log(
    *,
    grouped_metrics: OrderedDict[str, list[tuple[str, float]]],
    timestep: int,
    max_timesteps: int,
    iteration: int,
    max_iterations: int,
    steps_per_second: float,
    collection_time: float,
    learning_time: float,
    elapsed_time: float,
    eta_seconds: float,
) -> str:
    log_text = _render_grouped_train_log(
        grouped_metrics=grouped_metrics,
        timestep=timestep,
        max_timesteps=max_timesteps,
        iteration=iteration,
        max_iterations=max_iterations,
        steps_per_second=steps_per_second,
        collection_time=collection_time,
        learning_time=learning_time,
        elapsed_time=elapsed_time,
        eta_seconds=eta_seconds,
    )
    print(log_text, flush=True)
    return log_text


def _log_grouped_tensorboard(
    writer: SummaryWriter | None,
    grouped_metrics: OrderedDict[str, list[tuple[str, float]]],
    timestep: int,
    *,
    collection_time: float,
    learning_time: float,
    elapsed_time: float,
    eta_seconds: float,
    iteration: int,
    formatted_log: str | None = None,
    text_log_every_iterations: int = 100,
) -> None:
    if writer is None:
        return
    for group_name, key, value in _iter_tracked_group_metrics(grouped_metrics):
        group_slug = group_name.lower().replace(" ", "_")
        tag_key = key.replace("aux_rew/", "")
        writer.add_scalar(f"train/{group_slug}/{tag_key}", value, timestep)
    writer.flush()

_ENC_CONFIG_TO_EXPERT_DATA_OBS_MAPPER = {
    HumanoidVerseIsaacConfig: None,
}



Evaluation = tp.Annotated[
    tp.Union[
        HumanoidVerseIsaacTrackingEvaluationConfig,
    ],
    pydantic.Field(discriminator="name"),
]

Agent = FBcprAgentConfig | FBcprAuxAgentConfig


class TrainConfig(BaseConfig):
    # The "pydantic.Field" field is used to explicitely tell which field is the discriminative
    # feature
    agent: Agent = pydantic.Field(discriminator="name")
    motions: str | None = None
    motions_root: str | None = None

    env: HumanoidVerseIsaacConfig = pydantic.Field(discriminator="name")

    work_dir: str = pydantic.Field(default_factory=lambda: get_local_workdir("g1mujoco_train"))

    seed: int = 0
    online_parallel_envs: int = 50
    # Note: this is in env steps (multiples of online_parallel_envs)
    log_every_updates: int = 100_000
    num_env_steps: int = 30_000_000
    # Note: this is in env steps (multiples of online_parallel_envs)
    update_agent_every: int = 500
    # Note: this is in env steps (multiples of online_parallel_envs)
    num_seed_steps: int = 50_000
    num_agent_updates: int = 50
    # Note: this is in env steps (multiples of online_parallel_envs)
    checkpoint_every_steps: int = 5_000_000
    checkpoint_every_iterations: int | None = None
    checkpoint_buffer: bool = True
    checkpoint_buffer_every_steps: int = 100_000
    checkpoint_buffer_every_iterations: int | None = None
    resume_latest_run: bool = False
    prioritization: bool = False
    prioritization_min_val: float = 0.5
    prioritization_max_val: float = 5
    prioritization_scale: float = 2
    prioritization_mode: str = "bin"  # ["bin", "exp", "lin"]
    padding_beginning: int = 0
    padding_end: int = 0

    # Buffer
    use_trajectory_buffer: bool = False
    buffer_size: int = 5_000_000
    clean_reward_buffer: bool = False
    clean_reward_buffer_name: str = "train_clean"

    # Post-training from an existing checkpoint.
    post_train_freeze_fb: bool = False
    post_train_require_checkpoint_buffer: bool = False
    post_train_add_env_steps: int | None = None
    post_train_rollout_expert_trajectories_percentage: float | None = None

    # WANDB
    use_wandb: bool = True
    wandb_ename: str | None = "windigal"
    wandb_gname: str | None = "first-test"
    wandb_pname: str | None = "BFM-Zero"

    # misc
    gpu_id: str | None = None
    load_isaac_expert_data: bool = True
    buffer_device: str = "cpu"
    # Default to True; otherwise you will spam the console with tqdm
    disable_tqdm: bool = True
    tensorboard_text_every_iterations: int = 256

    # If you want to add more available evaluations, Update "Evaluations" type above
    evaluations: Dict[str, Evaluation] | List[Evaluation] = pydantic.Field(default_factory=lambda: [])
    # Note: this is in env steps (multiples of online_parallel_envs)
    eval_every_steps: int = 1_000_000

    tags: dict = pydantic.Field(default_factory=lambda: {})

    # exca
    # infra: xk.TaskInfra = xk.TaskInfra(version="1")

    def model_post_init(self, context):
        if self.checkpoint_every_iterations is not None:
            object.__setattr__(
                self,
                "checkpoint_every_steps",
                int(self.checkpoint_every_iterations * self.online_parallel_envs),
            )
        if self.checkpoint_buffer_every_iterations is not None:
            object.__setattr__(
                self,
                "checkpoint_buffer_every_steps",
                int(self.checkpoint_buffer_every_iterations * self.online_parallel_envs),
            )
        if self.post_train_add_env_steps is not None and self.post_train_add_env_steps <= 0:
            raise ValueError("post_train_add_env_steps must be positive when set")
        if self.post_train_rollout_expert_trajectories_percentage is not None:
            ratio = self.post_train_rollout_expert_trajectories_percentage
            if ratio < 0 or ratio > 1:
                raise ValueError("post_train_rollout_expert_trajectories_percentage must be in [0, 1]")
        # TODO prioritization needs tracking eval to work, but this is bit hacky to check for it
        if self.load_isaac_expert_data and not isinstance(self.env, HumanoidVerseIsaacConfig):
            raise ValueError("Loading expert isaac data is only supported for HumanoidVerseIsaacConfig")

        if self.prioritization:
            has_prioritization_eval = False
            for eval_type in self.evaluations:
                if isinstance(eval_type, (HumanoidVerseIsaacTrackingEvaluationConfig)):
                    has_prioritization_eval = True
                    break
            if not has_prioritization_eval:
                raise ValueError("Prioritization requires tracking evaluation to be enabled")


        if self.motions is None or self.motions_root is None:
            if self.prioritization:
                raise ValueError("Prioritization requires expert data to be provided (motions and motions_root)")
            elif self.agent == FBcprAgentConfig:
                # TODO how to do checks like these in pydantic or more systematically?
                raise ValueError("FBcprAgent requires expert data to be provided (motions and motions_root)")

        # Ensure all evaluations have unique log names
        if isinstance(self.evaluations, list):
            log_names = set()
            for eval_cfg in self.evaluations:
                if eval_cfg.name_in_logs in log_names:
                    raise ValueError(
                        f"Duplicate evaluation name_in_logs found: {eval_cfg.name}. These should be unique so we do not overwrite any logs"
                    )
                log_names.add(eval_cfg.name_in_logs)

    def build(self):
        """In case of cluster run, use exca and process instead of explivit build"""
        return Workspace(self)


def _checkpoint_agent_config_update(cfg: TrainConfig) -> dict[str, tp.Any] | None:
    update: dict[str, tp.Any] = {}
    train_update: dict[str, tp.Any] = {}
    if cfg.post_train_rollout_expert_trajectories_percentage is not None:
        train_update["rollout_expert_trajectories"] = True
        train_update["rollout_expert_trajectories_percentage"] = cfg.post_train_rollout_expert_trajectories_percentage
    if train_update:
        update["train"] = train_update
    if cfg.post_train_freeze_fb:
        update["compile"] = False
        update["cudagraphs"] = False
    return update or None


def create_agent_or_load_checkpoint(work_dir: Path, cfg: TrainConfig, agent_build_kwargs: dict[str, tp.Any]):
    checkpoint_dir = work_dir / CHECKPOINT_DIR_NAME
    checkpoint_time = 0
    train_status = {"time": 0, "buffer_time": None}
    if checkpoint_dir.exists():
        with (checkpoint_dir / "train_status.json").open("r") as f:
            train_status = json.load(f)
        checkpoint_time = int(train_status["time"])

        print(f"Loading the agent at time {checkpoint_time}")
        agent = cfg.agent.object_class.load(
            checkpoint_dir,
            device=cfg.agent.model.device,
            config_update=_checkpoint_agent_config_update(cfg),
        )
        cfg = cfg.model_copy(update={"agent": agent.cfg})
    else:
        if cfg.post_train_freeze_fb:
            raise ValueError("post_train_freeze_fb=True requires work_dir to contain a checkpoint")
        agent = cfg.agent.build(**agent_build_kwargs)
    return agent, cfg, checkpoint_time, train_status


def _read_wandb_run_info(work_dir: Path) -> dict[str, tp.Any] | None:
    info_path = work_dir / WANDB_RUN_INFO_FILENAME
    if not info_path.exists():
        return None
    try:
        with info_path.open("r") as f:
            info = json.load(f)
        if not isinstance(info, dict):
            return None
        return info
    except Exception:
        return None


def _write_wandb_run_info(work_dir: Path, run: tp.Any) -> None:
    info_path = work_dir / WANDB_RUN_INFO_FILENAME
    payload = {
        "id": run.id,
        "name": run.name,
        "project": run.project,
        "entity": run.entity,
        "group": run.group,
        "url": run.url,
    }
    with info_path.open("w") as f:
        json.dump(payload, f, indent=4)


def init_wandb(cfg: TrainConfig, work_dir: Path):
    exp_name = "BFM-Zero"
    wandb_name = exp_name
    wandb_config = cfg.model_dump()
    existing_run_info = _read_wandb_run_info(work_dir)
    init_kwargs = dict(
        entity=cfg.wandb_ename,
        project=cfg.wandb_pname,
        group=cfg.wandb_gname,
        name=(existing_run_info or {}).get("name", wandb_name),
        config=wandb_config,
        dir="./_wandb",
    )
    if existing_run_info is not None and existing_run_info.get("id"):
        init_kwargs["id"] = existing_run_info["id"]
        init_kwargs["resume"] = "must"
    run = wandb.init(**init_kwargs)
    _write_wandb_run_info(work_dir, run)


class Workspace:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        requested_work_dir = Path(self.cfg.work_dir)
        if self.cfg.resume_latest_run and not (requested_work_dir / CHECKPOINT_DIR_NAME).exists():
            latest_run = _find_latest_resumable_run(requested_work_dir.parent, max_time=self.cfg.num_env_steps)
            if latest_run is not None:
                print(f"Resuming latest checkpointed run: {latest_run}")
                self.cfg = self.cfg.model_copy(update={"work_dir": str(latest_run)})

        # HACK with Isaac, we can not recreate environments with current code, so we need to
        #      create the environment with desired number of envs here
        if isinstance(cfg.env, HumanoidVerseIsaacConfig):
            from omegaconf import OmegaConf

            self.train_env, self.train_env_info = cfg.env.build(num_envs=cfg.online_parallel_envs)
            self.obs_space = self.train_env.single_observation_space
            self.action_space = self.train_env.single_action_space
        else:
            sample_env, _ = cfg.env.build(num_envs=1)
            self.obs_space = sample_env.observation_space
            self.action_space = sample_env.action_space

        assert "time" in self.obs_space.keys(), "Observation space must contain 'obs' and 'time' (TimeAwareObservation wrapper)"
        assert len(self.action_space.shape) == 1, "Only 1D action space is supported (first dim should be vector env)"
        # TODO for backwards consistency, we do not pass "time" to the agent, so we remove it from the obs_space we pass to the agent/model
        #      but would we need it at some point?
        del self.obs_space.spaces["time"]

        self.action_dim = self.action_space.shape[0]

        print(f"Workdir: {self.cfg.work_dir}")
        self.work_dir = Path(self.cfg.work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)

        if isinstance(cfg.env, HumanoidVerseIsaacConfig):
            with open(self.work_dir / "config.yaml", "w") as file:
                OmegaConf.save(self.train_env_info["unresolved_conf"], file)

        self.train_logger = CSVLogger(filename=self.work_dir / TRAIN_LOG_FILENAME)
        self.tb_writer = SummaryWriter(log_dir=str(self.work_dir / "tensorboard"))

        set_seed_everywhere(self.cfg.seed)

        self.agent, self.cfg, self._checkpoint_time, self._train_status = create_agent_or_load_checkpoint(
            self.work_dir, self.cfg, agent_build_kwargs=dict(obs_space=self.obs_space, action_dim=self.action_dim)
        )
        if self.cfg.post_train_add_env_steps is not None:
            if self._checkpoint_time <= 0:
                raise ValueError("post_train_add_env_steps requires work_dir to contain a checkpoint")
            self.cfg = self.cfg.model_copy(
                update={"num_env_steps": self._checkpoint_time + self.cfg.post_train_add_env_steps}
            )
        self.agent._model.train()
        if self.cfg.post_train_freeze_fb:
            self.agent.freeze_fb_maps(sync_targets=True)
            print(
                "Post-training mode: froze F/B maps and synchronized target F/B maps. "
                f"rollout_expert_trajectories={self.agent.cfg.train.rollout_expert_trajectories}, "
                f"rollout_expert_trajectories_percentage={self.agent.cfg.train.rollout_expert_trajectories_percentage}, "
                f"num_env_steps={self.cfg.num_env_steps}"
            )

        if isinstance(self.cfg.evaluations, list):
            self.evaluations = {eval_cfg.name_in_logs: eval_cfg.build() for eval_cfg in self.cfg.evaluations}
        else:
            self.evaluations = {eval_cfg: eval_cfg.build() for name, eval_cfg in self.cfg.evaluations.items()}
        self.evaluate = len(self.evaluations) > 0

        self.eval_loggers = {name: CSVLogger(filename=self.work_dir / f"{name}.csv") for name in self.evaluations.keys()}

        if self.cfg.use_wandb:
            init_wandb(self.cfg, self.work_dir)

        with (self.work_dir / "config.json").open("w") as f:
            f.write(self.cfg.model_dump_json(indent=4))

        self.priorization_eval_name = None
        if self.cfg.prioritization:
            for name, evaluation in self.evaluations.items():
                if isinstance(evaluation.cfg, HumanoidVerseIsaacTrackingEvaluationConfig):
                    self.priorization_eval_name = name
                    break
            if self.priorization_eval_name is None:
                raise ValueError("Prioritization requires tracking evaluation to be enabled")

        self.training_with_expert_data = True

        self.manager = None
        self._latest_train_timestep = self._checkpoint_time
        self._latest_replay_buffer = None
        self._latest_buffer_checkpoint_time = int(self._train_status.get("buffer_time") or 0)

    def train(self):
        self.start_time = time.time()
        try:
            self.train_online()
        except KeyboardInterrupt:
            if self._latest_replay_buffer is not None:
                print(f"Interrupted. Saving latest checkpoint at time {self._latest_train_timestep}")
                self.save(self._latest_train_timestep, self._latest_replay_buffer, save_buffer=False)
            raise
        finally:
            self.tb_writer.close()
            if self.cfg.use_wandb and wandb.run is not None:
                wandb.finish()

    def train_online(self) -> None:
        if self.training_with_expert_data:
            if self.cfg.load_isaac_expert_data:
                expert_buffer = load_expert_trajectories_from_motion_lib(self.train_env._env, self.cfg.agent, device=self.cfg.buffer_device)
            else:
                print("Loading expert trajectories")
                expert_buffer = load_expert_trajectories(
                    self.cfg.motions,
                    self.cfg.motions_root,
                    seq_length=self.agent.cfg.model.seq_length,
                    device=self.cfg.buffer_device,
                    # TODO data stored in disk does not have dictionary obs, so we need to manually
                    #      define what obs key the data on disk corresponds to
                    obs_dict_mapper=_ENC_CONFIG_TO_EXPERT_DATA_OBS_MAPPER[self.cfg.env.__class__],
                )
        print("Creating the training environment")

        if isinstance(self.cfg.env, HumanoidVerseIsaacConfig):
            train_env = self.train_env
            train_env_info = self.train_env_info
        else:
            train_env, train_env_info = self.cfg.env.build(num_envs=self.cfg.online_parallel_envs)

        print("Allocating buffers")
        replay_buffer = {}
        checkpoint_dir = self.work_dir / CHECKPOINT_DIR_NAME
        train_buffer_loaded = (checkpoint_dir / "buffers/train").exists()
        if self.cfg.post_train_require_checkpoint_buffer and not train_buffer_loaded:
            raise ValueError(
                "post_train_require_checkpoint_buffer=True but checkpoint/buffers/train is missing. "
                "Use a checkpoint with saved buffers or disable the requirement."
            )
        if train_buffer_loaded:
            print("Loading checkpointed buffer")
            if self.cfg.use_trajectory_buffer:
                replay_buffer["train"] = TrajectoryDictBufferMultiDim.load(checkpoint_dir / "buffers/train", device=self.cfg.buffer_device)
            else:
                replay_buffer["train"] = DictBuffer.load(checkpoint_dir / "buffers/train", device=self.cfg.buffer_device)
            print(f"Loaded buffer of size {len(replay_buffer['train'])}")
        else:
            if self.cfg.use_trajectory_buffer:
                output_key_t = ["observation", "action", "z", "terminated", "truncated", "step_count", "reward"]
                # TODO this interface should be more elegant (how to inform buffer what keys are coming in / need to be sampled?)
                if isinstance(self.cfg.agent, (FBcprAuxAgentConfig)):
                    output_key_t.append("aux_rewards")

                replay_buffer["train"] = TrajectoryDictBufferMultiDim(
                    capacity=self.cfg.buffer_size // self.cfg.online_parallel_envs,  # make sure to divide by num_envs
                    device=self.cfg.buffer_device,
                    n_dim=2,
                    end_key="truncated",
                    output_key_t=output_key_t,  # TODO(team): fix this. in principle we could avoid to sample qpos, qvel for training but we need them for reward evaluation
                    output_key_tp1=["observation", "terminated"],
                )
            else:
                replay_buffer["train"] = DictBuffer(capacity=self.cfg.buffer_size, device=self.cfg.buffer_device)
        use_clean_reward_buffer = self.cfg.clean_reward_buffer and hasattr(train_env, "get_clean_observation")
        if self.cfg.clean_reward_buffer and not use_clean_reward_buffer:
            warnings.warn("clean_reward_buffer=True requires an environment with get_clean_observation(); skipping clean buffer.")
        if use_clean_reward_buffer:
            clean_buffer_path = checkpoint_dir / "buffers" / self.cfg.clean_reward_buffer_name
            if clean_buffer_path.exists():
                print("Loading checkpointed clean reward buffer")
                if self.cfg.use_trajectory_buffer:
                    replay_buffer[self.cfg.clean_reward_buffer_name] = TrajectoryDictBufferMultiDim.load(
                        clean_buffer_path,
                        device=self.cfg.buffer_device,
                    )
                else:
                    replay_buffer[self.cfg.clean_reward_buffer_name] = DictBuffer.load(clean_buffer_path, device=self.cfg.buffer_device)
                print(f"Loaded clean buffer of size {len(replay_buffer[self.cfg.clean_reward_buffer_name])}")
            elif train_buffer_loaded:
                warnings.warn(
                    f"{self.cfg.clean_reward_buffer_name} is missing for this checkpoint; "
                    "skipping clean reward buffer to avoid misaligned replay samples."
                )
                use_clean_reward_buffer = False
            elif self.cfg.use_trajectory_buffer:
                replay_buffer[self.cfg.clean_reward_buffer_name] = TrajectoryDictBufferMultiDim(
                    capacity=self.cfg.buffer_size // self.cfg.online_parallel_envs,
                    device=self.cfg.buffer_device,
                    n_dim=2,
                    end_key="truncated",
                    output_key_t=["observation"],
                    output_key_tp1=["observation"],
                )
            else:
                replay_buffer[self.cfg.clean_reward_buffer_name] = DictBuffer(capacity=self.cfg.buffer_size, device=self.cfg.buffer_device)
        if self.training_with_expert_data:
            replay_buffer["expert_slicer"] = expert_buffer
        self._latest_replay_buffer = replay_buffer

        print("Starting training")
        progb = tqdm(total=self.cfg.num_env_steps, disable=self.cfg.disable_tqdm)
        td, info = train_env.reset()
        clean_td = train_env.get_clean_observation(to_numpy=True) if use_clean_reward_buffer else None
        # see https://farama.org/Vector-Autoreset-Mode
        terminated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
        truncated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
        done = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
        total_metrics, context = None, None
        start_time = time.time()
        collection_time_acc = 0.0
        learning_time_acc = 0.0
        steps_since_last_log = 0
        train_time_total = 0.0
        checkpoint_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.checkpoint_every_steps)
        buffer_checkpoint_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.checkpoint_buffer_every_steps)
        eval_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.eval_every_steps)
        update_agent_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.update_agent_every)
        log_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.log_every_updates)

        eval_instances = []
        for evaluation_name in self.evaluations.keys():
            evaluation = self.evaluations[evaluation_name]
            eval_instances.append(isinstance(evaluation, HumanoidVerseIsaacTrackingEvaluation))
        uses_humanoidverse_eval = True if any(eval_instances) else False

        for t in range(self._checkpoint_time, self.cfg.num_env_steps + self.cfg.online_parallel_envs, self.cfg.online_parallel_envs):
            should_save_checkpoint = (t != self._checkpoint_time) and checkpoint_time_checker.check(t)
            should_save_buffer = (
                self.cfg.checkpoint_buffer
                and (t != self._checkpoint_time)
                and buffer_checkpoint_time_checker.check(t)
            )
            if should_save_checkpoint or should_save_buffer:
                if should_save_checkpoint:
                    checkpoint_time_checker.update_last_step(t)
                if should_save_buffer:
                    buffer_checkpoint_time_checker.update_last_step(t)
                self.save(t, replay_buffer, save_buffer=should_save_buffer)

            if (self.evaluate and eval_time_checker.check(t)) or (self.evaluate and t == self._checkpoint_time):
                eval_metrics = self.eval(t, replay_buffer=replay_buffer)
                eval_time_checker.update_last_step(t)
                if uses_humanoidverse_eval:
                    # reset if there is a humanoidverse evaluation
                    td, info = train_env.reset()
                    clean_td = train_env.get_clean_observation(to_numpy=True) if use_clean_reward_buffer else None
                    terminated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
                    truncated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
                    done = np.zeros(self.cfg.online_parallel_envs, dtype=bool)

                if self.cfg.prioritization:
                    assert len(eval_metrics[self.priorization_eval_name]) == len(replay_buffer["expert_slicer"].motion_ids), (
                        "Mismatch in number of motions returned by the eval"
                    )
                    # priorities
                    index_in_buffer, name_in_buffer = {}, {}
                    for i, motion_id in enumerate(replay_buffer["expert_slicer"].motion_ids):
                        index_in_buffer[motion_id] = i
                        if hasattr(replay_buffer["expert_slicer"], "file_names"):
                            name_in_buffer[motion_id] = replay_buffer["expert_slicer"].file_names[i]
                    motions_id, priorities, idxs = [], [], []
                    for _, metr in eval_metrics[self.priorization_eval_name].items():
                        motions_id.append(metr["motion_id"])
                        priorities.append(metr["emd"])
                        idxs.append(index_in_buffer[metr["motion_id"]])
                    priorities = (
                        torch.clamp(
                            torch.tensor(priorities, dtype=torch.float32, device=self.agent.device),
                            min=self.cfg.prioritization_min_val,
                            max=self.cfg.prioritization_max_val,
                        )
                        * self.cfg.prioritization_scale
                    )

                    if self.cfg.prioritization_mode == "lin":
                        pass
                    elif self.cfg.prioritization_mode == "exp":
                        priorities = 2**priorities
                    elif self.cfg.prioritization_mode == "bin":
                        bins = torch.floor(priorities)
                        for i in range(int(bins.min().item()), int(bins.max().item()) + 1):
                            mask = bins == i
                            n = mask.sum().item()
                            if n > 0:
                                priorities[mask] = 1 / n
                    else:
                        raise ValueError(f"Unsupported prioritization mode {self.cfg.prioritization_mode}")

                    train_env._env._motion_lib.update_sampling_weight_by_id(
                        priorities=list(priorities), motions_id=idxs, file_name=name_in_buffer
                    )

                    replay_buffer["expert_slicer"].update_priorities(
                        priorities=priorities.to(self.cfg.buffer_device), idxs=torch.tensor(np.array(idxs), device=self.cfg.buffer_device)
                    )

            rollout_start = time.time()
            with torch.no_grad():
                obs = tree_map(lambda x: torch.tensor(x, dtype=dtype_numpytotorch_lower_precision(x.dtype), device=self.agent.device), td)
                # TODO consistency with obs_space: remove time assigned by TimeAwareObservationWrapper
                step_count = obs.pop("time")

                history_context = None
                if "history" in obs:
                    # this works in inference mode
                    if len(obs["history"]["action"]) == 0:
                        history_context = self.agent._model._context_encoder.get_initial_context(self.cfg.online_parallel_envs)
                    else:
                        history_context = self.agent.history_inference(obs=obs["history"]["observation"], action=obs["history"]["action"])[
                            :, -1
                        ].clone()

                context = self.agent.maybe_update_rollout_context(z=context, step_count=step_count, replay_buffer=replay_buffer)
                if t < self.cfg.num_seed_steps:
                    action = train_env.action_space.sample().astype(np.float32)
                else:
                    # this works in inference mode
                    if history_context is not None:
                        action = self.agent.act(obs=obs, z=context, context=history_context, mean=False)
                    else:
                        action = self.agent.act(obs=obs, z=context, mean=False)
                    # TODO a bit hard-coded -- just to avoid moving stuff from cpu to cuda
                    if not isinstance(self.cfg.env, HumanoidVerseIsaacConfig):
                        action = action.cpu().detach().numpy()
            new_td, new_reward, new_terminated, new_truncated, new_info = train_env.step(action)
            new_clean_td = train_env.get_clean_observation(to_numpy=True) if use_clean_reward_buffer else None
            collection_time_acc += time.time() - rollout_start
            steps_since_last_log += self.cfg.online_parallel_envs

            # we check if at the next iteration we will evaluate
            next_t = t + self.cfg.online_parallel_envs
            if (self.evaluate and eval_time_checker.check(next_t)) or (self.evaluate and next_t == self._checkpoint_time):
                if isinstance(self.cfg.env, HumanoidVerseIsaacConfig) and uses_humanoidverse_eval:
                    # make sure we set truncated since at the next iteration we are forced to reset the environment
                    # after the evaluation. This is because we share the environment with the evaluation
                    new_truncated = np.ones_like(new_truncated, dtype=bool)
                    truncated = np.ones_like(new_truncated, dtype=bool)

            if Version(gymnasium.__version__) >= Version("1.0"):
                if self.cfg.use_trajectory_buffer:
                    data = {
                        "observation": tree_map(lambda x: x[None, ...], obs),
                        "action": action[None, ...],
                        "terminated": terminated[None, ..., None],
                        "truncated": truncated[None, ..., None],
                        "step_count": step_count[None, ..., None],
                        "reward": new_reward[None, ..., None],
                    }
                    data["observation"].pop("history", None)
                    if context is not None:
                        data["z"] = context[None, ...]
                    if history_context is not None:
                        data["history_context"] = history_context[None, ...]
                    if "qpos" in info:
                        data["qpos"] = info["qpos"][None, ...]
                    if "qvel" in info:
                        data["qvel"] = info["qvel"][None, ...]
                    if "aux_rewards" in new_info:
                        data["aux_rewards"] = {k: v[None, ..., None] for k, v in new_info["aux_rewards"].items() if not k.startswith("_")}
                else:
                    # We add only transitions corresponding to environments that have not reset in the previous step.
                    # For environments that have reset in the previous step, the new observation corresponds to the state after reset.
                    indexes = ~done

                    real_next_obs = tree_map(lambda x: x.astype(np.float32 if x.dtype == np.float64 else x.dtype)[indexes], new_td)
                    # TODO again, we need to remove "time" from the observation (to stay consistent with obs_space)
                    _ = real_next_obs.pop("time")
                    _ = real_next_obs.pop("history", None)

                    data = {
                        "observation": tree_map(lambda x: x[indexes], obs),
                        "action": action[indexes],
                        "step_count": step_count[indexes],
                        "reward": new_reward[indexes].reshape(-1, 1),
                        "next": {
                            "observation": real_next_obs,
                            "terminated": new_terminated[indexes].reshape(-1, 1),
                            "truncated": new_truncated[indexes].reshape(-1, 1),
                        },
                    }
                    data["observation"].pop("history", None)
                    if context is not None:
                        data["z"] = context[indexes]
                    if history_context is not None:
                        data["history_context"] = history_context[indexes]
                    if "qpos" in info:
                        data["qpos"] = info["qpos"][indexes]
                        data["next"]["qpos"] = new_info["qpos"][indexes]
                    if "qvel" in info:
                        data["qvel"] = info["qvel"][indexes]
                        data["next"]["qvel"] = new_info["qvel"][indexes]
                    if "aux_rewards" in new_info:
                        data["aux_rewards"] = {
                            k: v[indexes].reshape(-1, 1) for k, v in new_info["aux_rewards"].items() if not k.startswith("_")
                        }
            else:
                raise NotImplementedError("still some work to do for gymnasium < 1.0")
            replay_buffer["train"].extend(data)
            if use_clean_reward_buffer:
                if self.cfg.use_trajectory_buffer:
                    clean_data = {
                        "observation": tree_map(lambda x: x[None, ...], clean_td),
                        "truncated": truncated[None, ..., None],
                    }
                else:
                    clean_data = {
                        "observation": tree_map(
                            lambda x: x.astype(np.float32 if x.dtype == np.float64 else x.dtype)[indexes],
                            clean_td,
                        ),
                        "truncated": new_truncated[indexes].reshape(-1, 1),
                        "next": {
                            "observation": tree_map(
                                lambda x: x.astype(np.float32 if x.dtype == np.float64 else x.dtype)[indexes],
                                new_clean_td,
                            ),
                        },
                    }
                replay_buffer[self.cfg.clean_reward_buffer_name].extend(clean_data)

            if len(replay_buffer["train"]) > 0 and t > self.cfg.num_seed_steps and update_agent_time_checker.check(t):
                update_agent_time_checker.update_last_step(t)
                learning_start = time.time()
                for _ in range(self.cfg.num_agent_updates):
                    metrics = self.agent.update(replay_buffer, t)
                    if total_metrics is None:
                        num_metrics_updates = 1
                        total_metrics = {k: metrics[k].float().clone() for k in metrics.keys()}
                    else:
                        num_metrics_updates += 1
                        total_metrics = {k: total_metrics[k] + metrics[k].float() for k in metrics.keys()}
                learning_time_acc += time.time() - learning_start

            if log_time_checker.check(t) and total_metrics is not None:
                log_time_checker.update_last_step(t)
                current_timestep = min(t + self.cfg.online_parallel_envs, self.cfg.num_env_steps)
                m_dict = {}
                for k in sorted(list(total_metrics.keys())):
                    tmp = total_metrics[k] / num_metrics_updates
                    m_dict[k] = np.round(tmp.mean().item(), 6)
                interval_train_time = collection_time_acc + learning_time_acc
                train_time_total += interval_train_time
                interval_fps = steps_since_last_log / max(interval_train_time, 1e-9)
                interval_iterations = max(steps_since_last_log // self.cfg.online_parallel_envs, 1)
                avg_collection_time = collection_time_acc / interval_iterations
                avg_learning_time = learning_time_acc / interval_iterations
                total_train_steps = max(current_timestep - self._checkpoint_time, 0)
                avg_train_fps = total_train_steps / max(train_time_total, 1e-9) if total_train_steps > 0 else interval_fps
                grouped_metrics = _build_grouped_metrics(m_dict)
                if self.cfg.use_wandb:
                    wandb.log(
                        {f"train/{key}": value for _, key, value in _iter_tracked_group_metrics(grouped_metrics)},
                        step=current_timestep,
                    )
                current_iteration = current_timestep // self.cfg.online_parallel_envs
                max_iterations = self.cfg.num_env_steps // self.cfg.online_parallel_envs
                elapsed_time = time.time() - start_time
                remaining_steps = max(self.cfg.num_env_steps - current_timestep, 0)
                eta_seconds = remaining_steps / max(avg_train_fps, 1e-9) if remaining_steps > 0 else 0.0
                log_text = _print_grouped_train_log(
                    grouped_metrics=grouped_metrics,
                    timestep=current_timestep,
                    max_timesteps=self.cfg.num_env_steps,
                    iteration=current_iteration,
                    max_iterations=max_iterations,
                    steps_per_second=interval_fps,
                    collection_time=avg_collection_time,
                    learning_time=avg_learning_time,
                    elapsed_time=elapsed_time,
                    eta_seconds=eta_seconds,
                )
                _log_grouped_tensorboard(
                    self.tb_writer,
                    grouped_metrics,
                    current_timestep,
                    collection_time=avg_collection_time,
                    learning_time=avg_learning_time,
                    elapsed_time=elapsed_time,
                    eta_seconds=eta_seconds,
                    iteration=current_iteration,
                    formatted_log=log_text,
                    text_log_every_iterations=self.cfg.tensorboard_text_every_iterations,
                )
                total_metrics = None
                collection_time_acc = 0.0
                learning_time_acc = 0.0
                steps_since_last_log = 0
                m_dict["timestep"] = current_timestep
                m_dict["iteration"] = current_iteration
                self.train_logger.log(m_dict)

            progb.update(self.cfg.online_parallel_envs)
            self._latest_train_timestep = min(t + self.cfg.online_parallel_envs, self.cfg.num_env_steps)
            td = new_td
            clean_td = new_clean_td if use_clean_reward_buffer else None
            terminated = new_terminated
            truncated = new_truncated
            done = np.logical_or(new_terminated.ravel(), new_truncated.ravel())
            info = new_info
        train_env.close()

    def eval(self, t, replay_buffer):
        print(f"Starting evaluation at time {t}")
        evaluation_results = {}

        # This will contain the results, mapping evaluation.cfg.name --> dict of metrics
        evaluation_results = {}
        for evaluation_name in self.evaluations.keys():
            logger = self.eval_loggers[evaluation_name]
            evaluation = self.evaluations[evaluation_name]

            # NOTE we have this inside the loop so that the agent is not moved to cpu if we don't evaluate
            if not isinstance(self.cfg.env, HumanoidVerseIsaacConfig):
                self.agent._model.to("cpu")
            self.agent._model.train(False)

            if isinstance(self.cfg.env, HumanoidVerseIsaacConfig):
                # Pass train env
                evaluation_metrics, wandb_dict = evaluation.run(
                    timestep=t, agent_or_model=self.agent, replay_buffer=replay_buffer, logger=logger, env=self.train_env
                )
            else:
                evaluation_metrics, wandb_dict = evaluation.run(
                    timestep=t,
                    agent_or_model=self.agent,
                    replay_buffer=replay_buffer,
                    logger=logger,
                )
            # For wandb dict, put it on wandb
            if self.cfg.use_wandb and wandb_dict is not None:
                wandb.log(
                    {f"eval/{evaluation_name}/{k}": v for k, v in wandb_dict.items()},
                    step=t,
                )
            if wandb_dict is not None:
                for k, v in wandb_dict.items():
                    self.tb_writer.add_scalar(f"eval/{evaluation_name}/{k}", v, t)

            evaluation_results[evaluation_name] = evaluation_metrics

        # ---------------------------------------------------------------
        # this is important, move back the agent to cuda and
        # restart the training
        if not isinstance(self.cfg.env, HumanoidVerseIsaacConfig):
            self.agent._model.to(self.cfg.agent.model.device)
        self.agent._model.train()
        if self.cfg.post_train_freeze_fb:
            self.agent.freeze_fb_maps(sync_targets=False)

        return evaluation_results

    def save(self, time: int, replay_buffer: Dict[str, tp.Any], *, save_buffer: bool | None = None) -> None:
        save_buffer = self.cfg.checkpoint_buffer if save_buffer is None else save_buffer
        print(f"Checkpointing at time {time} (save_buffer={save_buffer})")
        self.agent.save(str(self.work_dir / CHECKPOINT_DIR_NAME))
        if save_buffer:
            replay_buffer["train"].save(self.work_dir / CHECKPOINT_DIR_NAME / "buffers" / "train")
            if self.cfg.clean_reward_buffer_name in replay_buffer:
                replay_buffer[self.cfg.clean_reward_buffer_name].save(
                    self.work_dir / CHECKPOINT_DIR_NAME / "buffers" / self.cfg.clean_reward_buffer_name
                )
            self._latest_buffer_checkpoint_time = time
        with (self.work_dir / CHECKPOINT_DIR_NAME / "train_status.json").open("w+") as f:
            json.dump({"time": time, "buffer_time": self._latest_buffer_checkpoint_time}, f, indent=4)


def train_bfm_zero(
    gpu_id: str | None = None,
    work_dir: str | None = None,
    num_env_steps: int | None = None,
    clean_reward_buffer: bool = True,
    post_train_add_env_steps: int | None = None,
    post_train_freeze_fb: bool = False,
    post_train_require_checkpoint_buffer: bool = False,
    post_train_rollout_expert_trajectories_percentage: float | None = None,
):
    """Launch BFM-Zero training.

    Args:
        gpu_id: Physical CUDA device index to expose to this process, e.g. "0" or "1".
            Internally the selected device is still addressed as cuda/cuda:0.
        work_dir: Result directory. If it contains checkpoint/, training resumes from it.
        num_env_steps: Absolute final env-step target. Leave unset for the default recipe.
        clean_reward_buffer: Store/load the aligned clean reward buffer.
        post_train_add_env_steps: Additional env steps to run after the loaded checkpoint time.
        post_train_freeze_fb: Freeze F/B maps after loading a checkpoint.
        post_train_require_checkpoint_buffer: Require checkpoint/buffers/train to exist.
        post_train_rollout_expert_trajectories_percentage: Override expert-z rollout env ratio after loading.
    """
    selected_gpu_id = _configure_cuda_visible_devices(gpu_id if gpu_id is not None else _PRESELECTED_GPU_ID)
    if selected_gpu_id is not None:
        print(f"Using physical CUDA device(s): {selected_gpu_id} via CUDA_VISIBLE_DEVICES={selected_gpu_id}")

    from humanoidverse.agents.fb_cpr_aux.model import FBcprAuxModelArchiConfig, FBcprAuxModelConfig
    from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentTrainConfig
    from humanoidverse.agents.nn_models import ForwardArchiConfig, BackwardArchiConfig, ActorArchiConfig, ActorArchiConfig, DiscriminatorArchiConfig, RewardNormalizerConfig, MoEBackwardArchiConfig
    from humanoidverse.agents.normalizers import ObsNormalizerConfig, BatchNormNormalizerConfig
    from humanoidverse.agents.nn_filters import DictInputFilterConfig

    cfg = TrainConfig(
        name='TrainConfig',
        agent=FBcprAuxAgentConfig(
            name='FBcprAuxAgent',
            model=FBcprAuxModelConfig(
                name='FBcprAuxModel',
                device='cuda',
                archi=FBcprAuxModelArchiConfig(
                    name='FBcprAuxModelArchiConfig',
                    z_dim=256,
                    norm_z=True,
                    f=ForwardArchiConfig(name='ForwardArchi', hidden_dim=1024, model='residual', hidden_layers=4, embedding_layers=2, num_parallel=2, ensemble_mode='batch', input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'privileged_state', 'last_action', 'history_actor'])),
                    b=BackwardArchiConfig(name='BackwardArchi', hidden_dim=256, hidden_layers=1, norm=True, input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'privileged_state'])),
                    # b=MoEBackwardArchiConfig(name='MoEBackwardArchi', expert_num=4, hidden_dim=256, hidden_layers=1, norm=True, input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'privileged_state'])),
                    actor=ActorArchiConfig(name='actor', model='residual', hidden_dim=1024, hidden_layers=4, embedding_layers=2, input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'last_action', 'history_actor'])),
                    critic=ForwardArchiConfig(name='ForwardArchi', hidden_dim=1024, model='residual', hidden_layers=4, embedding_layers=2, num_parallel=2, ensemble_mode='batch', input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'privileged_state', 'last_action', 'history_actor'])),
                    discriminator=DiscriminatorArchiConfig(name='DiscriminatorArchi', hidden_dim=512, hidden_layers=3, num_obs_steps=1, input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'privileged_state'])),
                    aux_critic=ForwardArchiConfig(name='ForwardArchi', hidden_dim=1024, model='residual', hidden_layers=4, embedding_layers=2, num_parallel=2, ensemble_mode='batch', input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'privileged_state', 'last_action', 'history_actor']))
                ),
                # archi=FBcprAuxModelArchiConfig(
                #     name='FBcprAuxModelArchiConfig',
                #     z_dim=256,
                #     norm_z=True,
                #     f=ForwardArchiConfig(name='ForwardArchi', hidden_dim=2048, model='residual', hidden_layers=6, embedding_layers=2, num_parallel=2, ensemble_mode='batch', input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'privileged_state', 'last_action', 'history_actor'])),
                #     b=BackwardArchiConfig(name='BackwardArchi', hidden_dim=256, hidden_layers=1, norm=True, input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'privileged_state'])),
                #     actor=ActorArchiConfig(name='actor', model='residual', hidden_dim=2048, hidden_layers=6, embedding_layers=2, input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'last_action', 'history_actor'])),
                #     critic=ForwardArchiConfig(name='ForwardArchi', hidden_dim=2048, model='residual', hidden_layers=6, embedding_layers=2, num_parallel=2, ensemble_mode='batch', input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'privileged_state', 'last_action', 'history_actor'])),
                #     discriminator=DiscriminatorArchiConfig(name='DiscriminatorArchi', hidden_dim=1024, hidden_layers=3, input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'privileged_state'])),
                #     aux_critic=ForwardArchiConfig(name='ForwardArchi', hidden_dim=2048, model='residual', hidden_layers=6, embedding_layers=2, num_parallel=2, ensemble_mode='batch', input_filter=DictInputFilterConfig(name='DictInputFilterConfig', key=['state', 'privileged_state', 'last_action', 'history_actor']))
                # ),
                obs_normalizer=ObsNormalizerConfig(
                    name='ObsNormalizerConfig',
                    normalizers={
                        'state': BatchNormNormalizerConfig(name='BatchNormNormalizerConfig', momentum=0.01),
                        'privileged_state': BatchNormNormalizerConfig(name='BatchNormNormalizerConfig', momentum=0.01),
                        'last_action': BatchNormNormalizerConfig(name='BatchNormNormalizerConfig', momentum=0.01),
                        'history_actor': BatchNormNormalizerConfig(name='BatchNormNormalizerConfig', momentum=0.01)
                    },
                    allow_mismatching_keys=True
                ),
                inference_batch_size=500000,
                seq_length=8,
                actor_std=0.05,
                amp=True,
                norm_aux_reward=RewardNormalizerConfig(name='RewardNormalizer', translate=False, scale=True)
            ),
            train=FBcprAuxAgentTrainConfig(
                name='FBcprAuxAgentTrainConfig',
                lr_f=0.0003,
                lr_b=1e-05,
                lr_actor=0.0003,
                weight_decay=0.0,
                clip_grad_norm=0.0,
                fb_target_tau=0.01,
                ortho_coef=100.0,
                train_goal_ratio=0.2,
                fb_pessimism_penalty=0.0,
                actor_pessimism_penalty=0.5,
                stddev_clip=0.3,
                q_loss_coef=0.0,
                batch_size=1024,
                discount=0.98,
                use_mix_rollout=True,
                update_z_every_step=100,
                z_buffer_size=8192,
                rollout_expert_trajectories=True,
                rollout_expert_trajectories_length=250,
                rollout_expert_trajectories_percentage=0.5,
                lr_discriminator=1e-05,
                lr_critic=0.0003,
                critic_target_tau=0.005,
                critic_pessimism_penalty=0.5,
                reg_coeff=0.05,
                scale_reg=True,
                expert_asm_ratio=0.6,
                relabel_ratio=0.8,
                grad_penalty_discriminator=10.0,
                weight_decay_discriminator=0.0,
                lr_aux_critic=0.0003,
                reg_coeff_aux=0.02,
                aux_critic_pessimism_penalty=0.5
            ),
            aux_rewards=['penalty_torques', 'penalty_action_rate', 'limits_dof_pos', 'limits_torque', 'penalty_undesired_contact', 'penalty_feet_ori', 'penalty_ankle_roll', 'penalty_slippage'],
            aux_rewards_scaling={'penalty_action_rate': -0.1, 'penalty_feet_ori': -0.4, 'penalty_ankle_roll': -4.0, 'limits_dof_pos': -10.0, 'penalty_slippage': -2.0, 'penalty_undesired_contact': -1.0, 'penalty_torques': 0.0, 'limits_torque': 0.0},
            cudagraphs=False,
            compile=True
        ),
        motions='',
        motions_root='',
        env=HumanoidVerseIsaacConfig(
            name='humanoidverse_isaac',
            device='cuda:0',
            # TODO this needs to be updated to point to a path with lafan dataset chunked into 10s clips
            # lafan_tail_path='humanoidverse/data/lafan_29dof_10s-clipped.pkl',
            lafan_tail_path='humanoidverse/data/seed_train_10s_2000.pkl',
            # lafan_tail_path='humanoidverse/data/seed_train_10s_2000_jump08_pair_with_contact.pkl',
            # lafan_tail_path='humanoidverse/data/babel_train_seed2k_windowed_2000_10s_50fps.pkl',
            enable_cameras=False,
            camera_render_save_dir='isaac_videos',
            max_episode_length_s=None,
            disable_obs_noise=False,
            disable_domain_randomization=False,
            relative_config_path='exp/bfm_zero/bfm_zero',
            include_last_action=True,
            hydra_overrides=['robot=g1/g1_29dof_hard_waist', 'robot.control.action_scale=0.25', 'robot.control.action_clip_value=5.0', 'robot.control.normalize_action_to=5.0', 'env.config.lie_down_init=True', 'env.config.lie_down_init_prob=0.3'],
            context_length=None,
            include_dr_info=False,
            included_dr_obs_names=None,
            include_history_actor=True,
            include_history_noaction=False,
            make_config_g1env_compatible=False,
            root_height_obs=True,
            use_contact_in_obs_max=False,
        ),
        work_dir=work_dir or _make_timestamped_result_dir('results/bfmzero-isaac'),
        seed=42,
        online_parallel_envs=1024,
        log_every_updates=51200, # 50 iter * 1024 env steps/iter
        num_env_steps=num_env_steps if num_env_steps is not None else 384000000,
        update_agent_every=1024,
        num_seed_steps=10240,
        num_agent_updates=16,
        checkpoint_every_steps=9600000,
        checkpoint_every_iterations=5000,
        checkpoint_buffer=True,
        checkpoint_buffer_every_iterations=100000,
        resume_latest_run=False,
        prioritization=True,
        prioritization_min_val=0.5,
        prioritization_max_val=2.0,
        prioritization_scale=2.0,
        prioritization_mode='exp',
        use_trajectory_buffer=True,
        buffer_size=5120000,
        clean_reward_buffer=clean_reward_buffer,
        post_train_freeze_fb=post_train_freeze_fb,
        post_train_require_checkpoint_buffer=post_train_require_checkpoint_buffer,
        post_train_add_env_steps=post_train_add_env_steps,
        post_train_rollout_expert_trajectories_percentage=post_train_rollout_expert_trajectories_percentage,
        use_wandb=True,
        wandb_ename='windigal',  # your wandb entity (username/team), empty = default from wandb login
        wandb_gname='first-test',  # run group
        wandb_pname='BFM-Zero',  # your wandb project name
        gpu_id=selected_gpu_id,
        load_isaac_expert_data=True,
        buffer_device='cpu',
        disable_tqdm=True,
        evaluations=[HumanoidVerseIsaacTrackingEvaluationConfig(name='HumanoidVerseIsaacTrackingEvaluationConfig', generate_videos=False, videos_dir='videos', video_name_prefix='unknown_agent', name_in_logs='humanoidverse_tracking_eval', env=None, num_envs=1024, n_episodes_per_motion=1)],
        eval_every_steps=9600000,
        tags={},
    )
    workspace = cfg.build()
    workspace.train()


if __name__ == "__main__":
    # This is the bare minimum CLI interface to launch experiments, but ideally you should
    # launch your experiments from Python code (e.g., see under "scripts")
    tyro.cli(train_bfm_zero)

# uv run --no-cache -m humanoidverse.meta_online_entry_point
