from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from rl_mimicgen.dppo.envs.task_registry import DPPOTaskSpec, get_task_spec

WORKSPACE_DIR = Path(__file__).resolve().parents[3]


def _ensure_local_repo_paths() -> None:
    import sys

    os.environ.setdefault("MUJOCO_GL", "glx")

    for repo_name in ("mimicgen", "robomimic", "robosuite", "robosuite-task-zoo"):
        repo_path = WORKSPACE_DIR / repo_name
        if repo_path.exists():
            repo_path_str = str(repo_path)
            if repo_path_str not in sys.path:
                sys.path.insert(0, repo_path_str)


class MimicGenLowDimEnv:
    def __init__(self, env: Any, obs_keys: tuple[str, ...], horizon: int | None = None) -> None:
        self._env = env
        self.obs_keys = obs_keys
        base_env = getattr(env, "env", env)
        self.horizon = int(horizon if horizon is not None else getattr(base_env, "horizon", 0))
        low, high = base_env.action_spec
        self.action_low = np.asarray(low, dtype=np.float32)
        self.action_high = np.asarray(high, dtype=np.float32)
        self.action_dim = int(np.prod(self.action_low.shape))

    def _filter_obs(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {key: np.asarray(obs[key], dtype=np.float32) for key in self.obs_keys}

    def reset(self) -> dict[str, np.ndarray]:
        return self._filter_obs(self._env.reset())

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        obs, reward, done, info = self._env.step(action)
        success = bool(self._env.is_success().get("task", False))
        info = dict(info)
        info["success"] = success
        return self._filter_obs(obs), float(reward), bool(done or success), info

    def close(self) -> None:
        close_fn = getattr(self._env, "close", None)
        if callable(close_fn):
            close_fn()


def make_mimicgen_lowdim_env(
    task: str,
    variant: str,
    dataset_path: str | None = None,
    render: bool = False,
    render_offscreen: bool = False,
    use_image_obs: bool = False,
    reward_shaping: bool | None = None,
) -> MimicGenLowDimEnv:
    _ensure_local_repo_paths()

    import mimicgen  # noqa: F401
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils
    from robomimic.utils.file_utils import get_env_metadata_from_dataset

    spec = get_task_spec(task=task, variant=variant)
    ObsUtils.initialize_obs_utils_with_obs_specs(
        [
            {
                "obs": {
                    "low_dim": list(spec.obs_keys),
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
    env_meta = get_env_metadata_from_dataset(dataset_path or spec.dataset_path)
    env_meta = deepcopy(env_meta)
    if reward_shaping is not None:
        env_meta["env_kwargs"]["reward_shaping"] = reward_shaping
    env_meta["env_kwargs"]["ignore_done"] = False
    env_meta["env_kwargs"]["horizon"] = int(spec.horizon)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=spec.env_name,
        render=render,
        render_offscreen=render_offscreen,
        use_image_obs=use_image_obs,
        use_depth_obs=False,
    )
    return MimicGenLowDimEnv(env=env, obs_keys=spec.obs_keys, horizon=spec.horizon)
