from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRETRAIN_EPOCHS = 1500
DEFAULT_PRETRAIN_BATCH_SIZE = 2048
DEFAULT_PRETRAIN_LR = 2e-4
DEFAULT_PRETRAIN_MIN_LR = 2e-5
DEFAULT_PRETRAIN_TRAIN_SPLIT = 0.95
DEFAULT_PRETRAIN_VAL_FREQ = 10
DEFAULT_PRETRAIN_SAVE_FREQ = 50
DEFAULT_FINETUNE_ITERS = 201
DEFAULT_FINETUNE_FT_DENOISING_STEPS = 10
DEFAULT_FINETUNE_SAVE_FREQ = 100
DEFAULT_EVAL_EPISODES = 20
DEFAULT_CHECKPOINT_EVAL_N_ENVS = 256
DEFAULT_WANDB_PROJECT_PREFIX = "dppo_mimicgen"
_VARIANT_SUFFIX_RE = re.compile(r"_[doDO]\d+$")


def _task_stem(dataset_id: str) -> str:
    """Strip the trailing variant suffix (``_d<N>`` / ``_o<N>``) from a dataset id.

    Keeps the project stable across variants so ``stack_d0`` and ``stack_d1``
    share a project while still logging the variant as the wandb group.
    """
    return _VARIANT_SUFFIX_RE.sub("", dataset_id)


def _render_wandb_block(stage: str, dataset_id: str, entity: str | None, group: str | None) -> str:
    entity_yaml = "null" if entity is None else entity
    group_yaml = "null" if group is None else group
    return f"""wandb:
  entity: {entity_yaml}
  project: {DEFAULT_WANDB_PROJECT_PREFIX}_{_task_stem(dataset_id)}-{stage}
  group: {group_yaml}
  run: ${{now:%H-%M-%S}}_${{name}}
  log_model: True"""


def _ensure_local_repo_paths() -> None:
    import sys

    os.environ.setdefault("MUJOCO_GL", "glx")
    for repo_name in ("mimicgen", "robomimic", "robosuite", "robosuite-task-zoo", "dppo"):
        repo_path = REPO_ROOT / "resources" / repo_name
        if repo_path.exists():
            repo_path_str = str(repo_path)
            if repo_path_str not in sys.path:
                sys.path.insert(0, repo_path_str)


@dataclass(frozen=True, slots=True)
class MimicGenLowDimSpec:
    dataset_id: str
    source_hdf5: Path
    env_name: str
    low_dim_keys: tuple[str, ...]
    obs_shapes: dict[str, tuple[int, ...]]
    obs_dim: int
    action_dim: int
    horizon: int
    num_demos: int
    num_transitions: int


def _sorted_demo_keys(data_group: h5py.Group) -> list[str]:
    return sorted(data_group.keys(), key=lambda key: int(key.split("_")[1]))


def _default_low_dim_keys(obs_group: h5py.Group) -> tuple[str, ...]:
    return tuple(sorted(key for key in obs_group.keys() if "image" not in key.lower()))


def _safe_range(min_value: np.ndarray, max_value: np.ndarray) -> np.ndarray:
    value_range = max_value - min_value
    value_range[value_range < 1e-6] = 1.0
    return value_range.astype(np.float32, copy=False)


def _normalize_min_max(values: np.ndarray, min_value: np.ndarray, max_value: np.ndarray) -> np.ndarray:
    value_range = _safe_range(min_value, max_value)
    return (2.0 * (values - min_value) / value_range) - 1.0


def _resolve_horizon(dataset_id: str, env_args: dict[str, Any], traj_lengths: list[int]) -> int:
    env_horizon = env_args.get("env_kwargs", {}).get("horizon")
    if env_horizon is not None:
        return int(env_horizon)

    _ensure_local_repo_paths()
    try:
        import mimicgen

        for task_table in mimicgen.DATASET_REGISTRY.values():
            if dataset_id in task_table:
                return int(task_table[dataset_id]["horizon"])
    except Exception:
        pass

    return int(max(traj_lengths))


def inspect_mimicgen_lowdim_dataset(
    source_hdf5: str | Path,
    *,
    dataset_id: str | None = None,
    obs_keys: list[str] | tuple[str, ...] | None = None,
    max_demos: int | None = None,
) -> MimicGenLowDimSpec:
    source_path = Path(source_hdf5).expanduser().resolve()
    resolved_dataset_id = dataset_id or source_path.stem

    with h5py.File(source_path, "r") as file_obj:
        data_group = file_obj["data"]
        demo_keys = _sorted_demo_keys(data_group)
        if max_demos is not None:
            demo_keys = demo_keys[:max_demos]
        if not demo_keys:
            raise ValueError(f"No demos found in {source_path}")

        sample_demo = data_group[demo_keys[0]]
        sample_obs_group = sample_demo["obs"]
        selected_keys = tuple(obs_keys) if obs_keys else _default_low_dim_keys(sample_obs_group)
        obs_shapes = {key: tuple(np.asarray(sample_obs_group[key]).shape[1:]) for key in selected_keys}
        action_dim = int(np.asarray(sample_demo["actions"]).shape[1])
        traj_lengths = [int(np.asarray(data_group[demo_key]["actions"]).shape[0]) for demo_key in demo_keys]
        env_args = json.loads(data_group.attrs["env_args"])

    obs_dim = int(sum(int(np.prod(obs_shapes[key])) for key in selected_keys))
    return MimicGenLowDimSpec(
        dataset_id=resolved_dataset_id,
        source_hdf5=source_path,
        env_name=str(env_args["env_name"]),
        low_dim_keys=selected_keys,
        obs_shapes=obs_shapes,
        obs_dim=obs_dim,
        action_dim=action_dim,
        horizon=_resolve_horizon(resolved_dataset_id, env_args, traj_lengths),
        num_demos=len(traj_lengths),
        num_transitions=int(sum(traj_lengths)),
    )


def _stack_low_dim_obs(obs_group: h5py.Group, low_dim_keys: tuple[str, ...]) -> np.ndarray:
    return np.hstack([np.asarray(obs_group[key], dtype=np.float32) for key in low_dim_keys])


def _empty_like_columns(num_cols: int, dtype: np.dtype[np.floating]) -> np.ndarray:
    return np.empty((0, num_cols), dtype=dtype)


def _build_official_split_payload(
    *,
    demo_ids: list[int],
    states_by_demo: list[np.ndarray],
    actions_by_demo: list[np.ndarray],
    rewards_by_demo: list[np.ndarray],
    obs_min: np.ndarray,
    obs_max: np.ndarray,
    action_min: np.ndarray,
    action_max: np.ndarray,
) -> dict[str, np.ndarray]:
    if not demo_ids:
        obs_dim = int(obs_min.shape[0])
        action_dim = int(action_min.shape[0])
        return {
            "states": _empty_like_columns(obs_dim, np.float32),
            "actions": _empty_like_columns(action_dim, np.float32),
            "rewards": np.empty((0,), dtype=np.float32),
            "terminals": np.empty((0,), dtype=bool),
            "traj_lengths": np.empty((0,), dtype=np.int32),
        }

    states = np.concatenate([states_by_demo[index] for index in demo_ids], axis=0)
    actions = np.concatenate([actions_by_demo[index] for index in demo_ids], axis=0)
    rewards = np.concatenate([rewards_by_demo[index] for index in demo_ids], axis=0)
    traj_lengths = np.asarray([actions_by_demo[index].shape[0] for index in demo_ids], dtype=np.int32)
    terminals = np.zeros((states.shape[0],), dtype=bool)
    return {
        "states": _normalize_min_max(states, obs_min, obs_max).astype(np.float32, copy=False),
        "actions": _normalize_min_max(actions, action_min, action_max).astype(np.float32, copy=False),
        "rewards": rewards.astype(np.float32, copy=False),
        "terminals": terminals,
        "traj_lengths": traj_lengths,
    }


def _yaml_list(values: tuple[str, ...]) -> str:
    return json.dumps(list(values))


def _render_lowdim_env_block(
    spec: MimicGenLowDimSpec,
    *,
    normalization_ref: str,
    include_warp_defaults: bool = False,
    save_video: bool = False,
) -> str:
    warp_block = (
        """
  env_type: warp
  warp:
    graph_capture: False
    njmax_per_env: null
    naconmax_per_env: null
    physics_timestep: null"""
        if include_warp_defaults
        else ""
    )
    save_video_str = "true" if save_video else "false"
    return f"""env:
  n_envs: 50
  name: ${{env_name}}
  success_info_key: success
  best_reward_threshold_for_success: 1
  max_episode_steps: {spec.horizon}
  save_video: {save_video_str}{warp_block}
  wrappers:
    robomimic_lowdim:
      normalization_path: {normalization_ref}
      low_dim_keys: {_yaml_list(spec.low_dim_keys)}
    multi_step:
      n_obs_steps: ${{cond_steps}}
      n_action_steps: ${{act_steps}}
      max_episode_steps: ${{env.max_episode_steps}}
      reset_within_step: True
"""


def _render_diffusion_mlp_network_block() -> str:
    return """network:
    _target_: model.diffusion.mlp_diffusion.DiffusionMLP
    time_dim: 32
    mlp_dims: [1024, 1024, 1024]
    cond_mlp_dims: [512, 64]
    residual_style: True
    cond_dim: ${eval:'${obs_dim} * ${cond_steps}'}
    horizon_steps: ${horizon_steps}
    action_dim: ${action_dim}"""


def _render_ppo_model_block(*, base_policy_ref: str) -> str:
    return f"""model:
  _target_: model.diffusion.diffusion_ppo.PPODiffusion
  gamma_denoising: 0.99
  clip_ploss_coef: 0.01
  clip_ploss_coef_base: 0.001
  clip_ploss_coef_rate: 3
  randn_clip_value: 3
  min_sampling_denoising_std: 0.1
  min_logprob_denoising_std: 0.1
  network_path: {base_policy_ref}
  actor:
    _target_: model.diffusion.mlp_diffusion.DiffusionMLP
    time_dim: 32
    mlp_dims: [1024, 1024, 1024]
    cond_mlp_dims: [512, 64]
    residual_style: True
    cond_dim: ${{eval:'${{obs_dim}} * ${{cond_steps}}'}}
    horizon_steps: ${{horizon_steps}}
    action_dim: ${{action_dim}}
  critic:
    _target_: model.common.critic.CriticObs
    mlp_dims: [256, 256, 256]
    activation_type: Mish
    residual_style: True
    cond_dim: ${{eval:'${{obs_dim}} * ${{cond_steps}}'}}
  ft_denoising_steps: ${{ft_denoising_steps}}
  horizon_steps: ${{horizon_steps}}
  obs_dim: ${{obs_dim}}
  action_dim: ${{action_dim}}
  denoising_steps: ${{denoising_steps}}
  device: ${{device}}
"""


def _render_pretrain_config(
    spec: MimicGenLowDimSpec,
    artifact_dir: Path,
    config_dir: Path,
    log_root: Path,
    wandb_entity: str | None,
    wandb_group: str | None,
) -> str:
    train_dataset_path = (artifact_dir / "train.npz").as_posix()
    logdir = (log_root / "pretrain" / spec.dataset_id / "${name}" / "${now:%Y-%m-%d}_${now:%H-%M-%S}_${seed}").as_posix()
    wandb_block = _render_wandb_block("pretrain", spec.dataset_id, wandb_entity, wandb_group)
    return f"""defaults:
  - _self_
hydra:
  run:
    dir: ${{logdir}}
_target_: agent.pretrain.train_diffusion_agent.TrainDiffusionAgent

name: {spec.dataset_id}_pre_diffusion_mlp_ta${{horizon_steps}}_td${{denoising_steps}}
logdir: {logdir}
train_dataset_path: {train_dataset_path}

seed: 42
device: cuda:0
env: {spec.dataset_id}
obs_dim: {spec.obs_dim}
action_dim: {spec.action_dim}
denoising_steps: 20
horizon_steps: 4
cond_steps: 1

{wandb_block}

train:
  n_epochs: {DEFAULT_PRETRAIN_EPOCHS}
  batch_size: {DEFAULT_PRETRAIN_BATCH_SIZE}
  num_workers: 0
  learning_rate: {DEFAULT_PRETRAIN_LR}
  weight_decay: 1e-6
  train_split: {DEFAULT_PRETRAIN_TRAIN_SPLIT}
  val_freq: {DEFAULT_PRETRAIN_VAL_FREQ}
  use_bf16: True
  use_compile: True
  lr_scheduler:
    first_cycle_steps: {DEFAULT_PRETRAIN_EPOCHS}
    warmup_steps: 100
    min_lr: {DEFAULT_PRETRAIN_MIN_LR}
  save_model_freq: {DEFAULT_PRETRAIN_SAVE_FREQ}
  checkpoint_eval:
    enabled: True
    async_enabled: True
    async_queue_size: 2
    script_path: {(REPO_ROOT / "resources" / "dppo" / "script" / "eval_checkpoint_sweep.py").as_posix()}
    config_dir: {config_dir.as_posix()}
    config_name: eval_diffusion_mlp
    output_dir: {logdir}/checkpoint_eval
    device: ${{device}}
    n_envs: {DEFAULT_CHECKPOINT_EVAL_N_ENVS}
    n_steps: {spec.horizon}
    max_episode_steps: {spec.horizon}
    every_n: 1
    video_checkpoints: best
    render_num: 1
    skip_existing: True
    copy_best_to: {logdir}/best_checkpoint.pt

model:
  _target_: model.diffusion.diffusion.DiffusionModel
  predict_epsilon: True
  denoised_clip_value: 1.0
  network:
    _target_: model.diffusion.mlp_diffusion.DiffusionMLP
    time_dim: 32
    mlp_dims: [1024, 1024, 1024]
    cond_mlp_dims: [512, 64]
    residual_style: True
    cond_dim: ${{eval:'${{obs_dim}} * ${{cond_steps}}'}}
    horizon_steps: ${{horizon_steps}}
    action_dim: ${{action_dim}}
  horizon_steps: ${{horizon_steps}}
  obs_dim: ${{obs_dim}}
  action_dim: ${{action_dim}}
  denoising_steps: ${{denoising_steps}}
  device: ${{device}}

ema:
  decay: 0.995

train_dataset:
  _target_: agent.dataset.sequence.StitchedSequenceDataset
  dataset_path: ${{train_dataset_path}}
  horizon_steps: ${{horizon_steps}}
  cond_steps: ${{cond_steps}}
  max_n_episodes: {spec.num_demos}
  device: ${{device}}
"""


def _render_finetune_config(spec: MimicGenLowDimSpec, artifact_dir: Path, log_root: Path, wandb_entity: str | None, wandb_group: str | None) -> str:
    normalization_path = (artifact_dir / "normalization.npz").as_posix()
    env_meta_path = (artifact_dir / "env_meta.json").as_posix()
    base_policy_path = (artifact_dir / "override_base_policy_path.pt").as_posix()
    logdir = (log_root / "finetune" / spec.dataset_id / "${name}" / "${now:%Y-%m-%d}_${now:%H-%M-%S}_${seed}").as_posix()
    wandb_block = _render_wandb_block("finetune", spec.dataset_id, wandb_entity, wandb_group)
    return f"""defaults:
  - _self_
hydra:
  run:
    dir: ${{logdir}}
_target_: agent.finetune.train_ppo_diffusion_agent.TrainPPODiffusionAgent

name: {spec.dataset_id}_ft_diffusion_mlp_ta${{horizon_steps}}_td${{denoising_steps}}_tdf${{ft_denoising_steps}}
logdir: {logdir}
base_policy_path: {base_policy_path}
robomimic_env_cfg_path: {env_meta_path}
normalization_path: {normalization_path}

seed: 42
device: cuda:0
env_name: {spec.dataset_id}
obs_dim: {spec.obs_dim}
action_dim: {spec.action_dim}
denoising_steps: 20
ft_denoising_steps: {DEFAULT_FINETUNE_FT_DENOISING_STEPS}
cond_steps: 1
horizon_steps: 4
act_steps: 4

{_render_lowdim_env_block(spec, normalization_ref='${normalization_path}', include_warp_defaults=True, save_video=True)}

{wandb_block}

train:
  n_train_itr: {DEFAULT_FINETUNE_ITERS}
  n_critic_warmup_itr: 2
  n_steps: {spec.horizon}
  gamma: 0.999
  actor_lr: 1e-4
  actor_weight_decay: 0
  actor_lr_scheduler:
    first_cycle_steps: ${{train.n_train_itr}}
    warmup_steps: 10
    min_lr: 1e-4
  critic_lr: 1e-3
  critic_weight_decay: 0
  critic_lr_scheduler:
    first_cycle_steps: ${{train.n_train_itr}}
    warmup_steps: 10
    min_lr: 1e-3
  save_model_freq: {DEFAULT_FINETUNE_SAVE_FREQ}
  val_freq: 10
  render:
    freq: 25
    num: 1
  reward_scale_running: True
  reward_scale_const: 1.0
  gae_lambda: 0.95
  batch_size: 10000
  update_epochs: 5
  vf_coef: 0.5
  target_kl: 1

{_render_ppo_model_block(base_policy_ref='${base_policy_path}')}
"""


def _render_eval_bc_config(spec: MimicGenLowDimSpec, artifact_dir: Path, log_root: Path) -> str:
    normalization_path = (artifact_dir / "normalization.npz").as_posix()
    env_meta_path = (artifact_dir / "env_meta.json").as_posix()
    base_policy_path = (artifact_dir / "override_base_policy_path.pt").as_posix()
    logdir = (log_root / "eval" / spec.dataset_id / "${name}" / "${now:%Y-%m-%d}_${now:%H-%M-%S}_${seed}").as_posix()
    return f"""defaults:
  - _self_
hydra:
  run:
    dir: ${{logdir}}
_target_: agent.eval.eval_diffusion_agent.EvalDiffusionAgent

name: {spec.dataset_id}_eval_bc_diffusion_mlp_ta${{horizon_steps}}_td${{denoising_steps}}
logdir: {logdir}
base_policy_path: {base_policy_path}
robomimic_env_cfg_path: {env_meta_path}
normalization_path: {normalization_path}

seed: 42
device: cuda:0
env_name: {spec.dataset_id}
obs_dim: {spec.obs_dim}
action_dim: {spec.action_dim}
denoising_steps: 20
cond_steps: 1
horizon_steps: 4
act_steps: 4
ft_denoising_steps: 0

n_episodes: {DEFAULT_EVAL_EPISODES}
n_steps: {spec.horizon}
render_num: 0

{_render_lowdim_env_block(spec, normalization_ref='${normalization_path}', include_warp_defaults=True)}

model:
  _target_: model.diffusion.diffusion_eval.DiffusionEval
  ft_denoising_steps: ${{ft_denoising_steps}}
  predict_epsilon: True
  denoised_clip_value: 1.0
  randn_clip_value: 3
  network_path: ${{base_policy_path}}
  {_render_diffusion_mlp_network_block()}
  horizon_steps: ${{horizon_steps}}
  obs_dim: ${{obs_dim}}
  action_dim: ${{action_dim}}
  denoising_steps: ${{denoising_steps}}
  device: ${{device}}
"""


def _render_eval_rl_init_config(spec: MimicGenLowDimSpec, artifact_dir: Path, log_root: Path) -> str:
    normalization_path = (artifact_dir / "normalization.npz").as_posix()
    env_meta_path = (artifact_dir / "env_meta.json").as_posix()
    base_policy_path = (artifact_dir / "override_base_policy_path.pt").as_posix()
    logdir = (log_root / "eval" / spec.dataset_id / "${name}" / "${now:%Y-%m-%d}_${now:%H-%M-%S}_${seed}").as_posix()
    return f"""defaults:
  - _self_
hydra:
  run:
    dir: ${{logdir}}
_target_: agent.eval.eval_diffusion_agent.EvalDiffusionAgent

name: {spec.dataset_id}_eval_rl_init_diffusion_mlp_ta${{horizon_steps}}_td${{denoising_steps}}_tdf${{ft_denoising_steps}}
logdir: {logdir}
base_policy_path: {base_policy_path}
robomimic_env_cfg_path: {env_meta_path}
normalization_path: {normalization_path}

seed: 42
device: cuda:0
env_name: {spec.dataset_id}
obs_dim: {spec.obs_dim}
action_dim: {spec.action_dim}
denoising_steps: 20
ft_denoising_steps: {DEFAULT_FINETUNE_FT_DENOISING_STEPS}
cond_steps: 1
horizon_steps: 4
act_steps: 4

n_episodes: {DEFAULT_EVAL_EPISODES}
n_steps: {spec.horizon}
render_num: 0

{_render_lowdim_env_block(spec, normalization_ref='${normalization_path}', include_warp_defaults=True)}

{_render_ppo_model_block(base_policy_ref='${base_policy_path}')}
"""


def write_official_dppo_lowdim_configs(
    spec: MimicGenLowDimSpec,
    *,
    artifact_dir: str | Path,
    config_dir: str | Path,
    log_root: str | Path,
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
) -> dict[str, Path]:
    artifact_path = Path(artifact_dir).expanduser().resolve()
    config_path = Path(config_dir).expanduser().resolve()
    log_root_path = Path(log_root).expanduser().resolve()
    config_path.mkdir(parents=True, exist_ok=True)

    rendered = {
        "pre_diffusion_mlp.yaml": _render_pretrain_config(
            spec,
            artifact_path,
            config_path,
            log_root_path,
            wandb_entity,
            wandb_group,
        ),
        "ft_ppo_diffusion_mlp.yaml": _render_finetune_config(spec, artifact_path, log_root_path, wandb_entity, wandb_group),
        "eval_diffusion_mlp.yaml": _render_eval_bc_config(spec, artifact_path, log_root_path),
        "eval_bc_diffusion_mlp.yaml": _render_eval_bc_config(spec, artifact_path, log_root_path),
        "eval_rl_init_diffusion_mlp.yaml": _render_eval_rl_init_config(spec, artifact_path, log_root_path),
    }
    written: dict[str, Path] = {}
    for filename, payload in rendered.items():
        target = config_path / filename
        target.write_text(payload, encoding="utf-8")
        written[filename] = target
    return written


def write_task_manifest(
    spec: MimicGenLowDimSpec,
    *,
    artifact_dir: str | Path,
    config_dir: str | Path,
    log_root: str | Path,
    config_files: dict[str, Path],
) -> Path:
    manifest = {
        "dataset_id": spec.dataset_id,
        "source_hdf5": spec.source_hdf5.as_posix(),
        "artifact_dir": Path(artifact_dir).expanduser().resolve().as_posix(),
        "config_dir": Path(config_dir).expanduser().resolve().as_posix(),
        "log_root": Path(log_root).expanduser().resolve().as_posix(),
        "env_name": spec.env_name,
        "obs_dim": spec.obs_dim,
        "action_dim": spec.action_dim,
        "horizon": spec.horizon,
        "low_dim_keys": list(spec.low_dim_keys),
        "configs": {name: path.as_posix() for name, path in config_files.items()},
    }
    target = Path(config_dir).expanduser().resolve() / "task_manifest.json"
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target


def materialize_mimicgen_lowdim_port(
    source_hdf5: str | Path,
    *,
    output_root: str | Path,
    dataset_id: str | None = None,
    obs_keys: list[str] | tuple[str, ...] | None = None,
    max_demos: int | None = None,
    val_split: float = 0.0,
    split_seed: int = 0,
    log_root: str | Path | None = None,
    config_root: str | Path | None = None,
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
) -> dict[str, Any]:
    spec = inspect_mimicgen_lowdim_dataset(
        source_hdf5,
        dataset_id=dataset_id,
        obs_keys=obs_keys,
        max_demos=max_demos,
    )
    output_root_path = Path(output_root).expanduser().resolve()
    artifact_dir = output_root_path / "data" / spec.dataset_id
    generated_cfg_root = (
        Path(config_root).expanduser().resolve()
        if config_root is not None
        else REPO_ROOT / "resources" / "dppo" / "cfg" / "mimicgen" / "generated"
    )
    config_dir = generated_cfg_root / spec.dataset_id
    log_root_path = Path(log_root).expanduser().resolve() if log_root is not None else REPO_ROOT / "logs" / "official_dppo" / "mimicgen"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(spec.source_hdf5, "r") as file_obj:
        data_group = file_obj["data"]
        demo_keys = _sorted_demo_keys(data_group)
        if max_demos is not None:
            demo_keys = demo_keys[:max_demos]
        env_args = json.loads(data_group.attrs["env_args"])

        states_by_demo: list[np.ndarray] = []
        actions_by_demo: list[np.ndarray] = []
        rewards_by_demo: list[np.ndarray] = []
        obs_min = np.full((spec.obs_dim,), np.inf, dtype=np.float32)
        obs_max = np.full((spec.obs_dim,), -np.inf, dtype=np.float32)
        action_min = np.full((spec.action_dim,), np.inf, dtype=np.float32)
        action_max = np.full((spec.action_dim,), -np.inf, dtype=np.float32)

        for demo_key in demo_keys:
            demo_group = data_group[demo_key]
            states = _stack_low_dim_obs(demo_group["obs"], spec.low_dim_keys).astype(np.float32, copy=False)
            actions = np.asarray(demo_group["actions"], dtype=np.float32)
            rewards = np.asarray(demo_group["rewards"], dtype=np.float32) if "rewards" in demo_group else np.zeros((actions.shape[0],), dtype=np.float32)
            states_by_demo.append(states)
            actions_by_demo.append(actions)
            rewards_by_demo.append(rewards)
            obs_min = np.minimum(obs_min, states.min(axis=0))
            obs_max = np.maximum(obs_max, states.max(axis=0))
            action_min = np.minimum(action_min, actions.min(axis=0))
            action_max = np.maximum(action_max, actions.max(axis=0))

    demo_indices = list(range(len(demo_keys)))
    train_demo_ids = demo_indices
    val_demo_ids: list[int] = []
    if val_split > 0.0 and len(demo_indices) > 1:
        rng = random.Random(split_seed)
        shuffled = demo_indices[:]
        rng.shuffle(shuffled)
        num_val = max(1, int(round(len(shuffled) * val_split)))
        val_demo_ids = sorted(shuffled[:num_val])
        train_demo_ids = sorted(shuffled[num_val:])

    train_payload = _build_official_split_payload(
        demo_ids=train_demo_ids,
        states_by_demo=states_by_demo,
        actions_by_demo=actions_by_demo,
        rewards_by_demo=rewards_by_demo,
        obs_min=obs_min,
        obs_max=obs_max,
        action_min=action_min,
        action_max=action_max,
    )
    val_payload = _build_official_split_payload(
        demo_ids=val_demo_ids,
        states_by_demo=states_by_demo,
        actions_by_demo=actions_by_demo,
        rewards_by_demo=rewards_by_demo,
        obs_min=obs_min,
        obs_max=obs_max,
        action_min=action_min,
        action_max=action_max,
    )

    np.savez_compressed(artifact_dir / "train.npz", **train_payload)
    np.savez_compressed(artifact_dir / "val.npz", **val_payload)
    np.savez_compressed(
        artifact_dir / "normalization.npz",
        obs_min=obs_min,
        obs_max=obs_max,
        action_min=action_min,
        action_max=action_max,
        obs_mean=np.concatenate(states_by_demo, axis=0).mean(axis=0).astype(np.float32),
        obs_std=np.concatenate(states_by_demo, axis=0).std(axis=0).astype(np.float32),
        action_mean=np.concatenate(actions_by_demo, axis=0).mean(axis=0).astype(np.float32),
        action_std=np.concatenate(actions_by_demo, axis=0).std(axis=0).astype(np.float32),
    )
    (artifact_dir / "env_meta.json").write_text(json.dumps(env_args, indent=2), encoding="utf-8")
    metadata = {
        "dataset_id": spec.dataset_id,
        "source_hdf5": spec.source_hdf5.as_posix(),
        "env_name": spec.env_name,
        "low_dim_keys": list(spec.low_dim_keys),
        "obs_shapes": {key: list(value) for key, value in spec.obs_shapes.items()},
        "obs_dim": spec.obs_dim,
        "action_dim": spec.action_dim,
        "horizon": spec.horizon,
        "num_demos": spec.num_demos,
        "num_transitions": spec.num_transitions,
        "train_demo_ids": train_demo_ids,
        "val_demo_ids": val_demo_ids,
    }
    (artifact_dir / "dataset_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    written_configs = write_official_dppo_lowdim_configs(
        spec,
        artifact_dir=artifact_dir,
        config_dir=config_dir,
        log_root=log_root_path,
        wandb_entity=wandb_entity,
        wandb_group=wandb_group,
    )
    manifest_path = write_task_manifest(
        spec,
        artifact_dir=artifact_dir,
        config_dir=config_dir,
        log_root=log_root_path,
        config_files=written_configs,
    )
    return {
        "spec": spec,
        "artifact_dir": artifact_dir,
        "config_dir": config_dir,
        "log_root": log_root_path,
        "configs": written_configs,
        "task_manifest": manifest_path,
    }


def verify_mimicgen_lowdim_env_reset(spec: MimicGenLowDimSpec, *, env_meta_path: str | Path | None = None) -> dict[str, Any]:
    _ensure_local_repo_paths()
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils

    if env_meta_path is None:
        with h5py.File(spec.source_hdf5, "r") as file_obj:
            env_meta = json.loads(file_obj["data"].attrs["env_args"])
    else:
        env_meta = json.loads(Path(env_meta_path).read_text(encoding="utf-8"))
    ObsUtils.initialize_obs_utils_with_obs_specs(
        [
            {
                "obs": {
                    "low_dim": list(spec.low_dim_keys),
                    "rgb": [],
                    "depth": [],
                    "scan": [],
                },
                "goal": {
                    "low_dim": [],
                    "rgb": [],
                    "depth": [],
                    "scan": [],
                },
            }
        ]
    )
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=spec.env_name,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
        use_depth_obs=False,
    )
    try:
        obs = env.reset()
        missing_keys = [key for key in spec.low_dim_keys if key not in obs]
        if missing_keys:
            raise RuntimeError(f"Reset observation is missing low-dim keys: {missing_keys}")
        return {
            "env_name": spec.env_name,
            "num_obs_keys": len(spec.low_dim_keys),
            "missing_keys": missing_keys,
            "action_dim": int(env.action_dimension),
        }
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
