from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np
import torch
from tensordict import TensorDict


@dataclass(slots=True)
class ActionSpaceSpec:
    shape: tuple[int, ...]
    low: np.ndarray
    high: np.ndarray


WORKSPACE_DIR = Path(__file__).resolve().parents[2]


def _ensure_local_repo_paths() -> None:
    import sys

    os.environ.setdefault("MUJOCO_GL", "glx")

    for repo_name in ("mimicgen", "robomimic", "robosuite", "robosuite-task-zoo"):
        repo_path = WORKSPACE_DIR / repo_name
        if repo_path.exists():
            repo_path_str = str(repo_path)
            if repo_path_str not in sys.path:
                sys.path.insert(0, repo_path_str)


@dataclass(slots=True)
class RobosuiteCheckpointFactory:
    checkpoint_path: str
    env_name: str
    render: bool = False
    render_offscreen: bool = False
    reward_shaping: bool | None = None
    horizon: int | None = None

    def __call__(self) -> Any:
        _ensure_local_repo_paths()

        import mimicgen  # noqa: F401
        import robosuite
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.obs_utils as ObsUtils
        from robomimic.utils.file_utils import config_from_checkpoint, load_dict_from_checkpoint

        if not hasattr(robosuite, "__version__"):
            try:
                from importlib.metadata import version

                robosuite.__version__ = version("robosuite")
            except Exception:
                robosuite.__version__ = "1.4.1"

        ckpt_dict = load_dict_from_checkpoint(self.checkpoint_path)
        config, _ = config_from_checkpoint(ckpt_dict=ckpt_dict, verbose=False)
        ObsUtils.initialize_obs_utils_with_config(config)

        env_meta = deepcopy(ckpt_dict["env_metadata"])
        env_meta["env_kwargs"]["ignore_done"] = False
        if self.reward_shaping is not None:
            env_meta["env_kwargs"]["reward_shaping"] = self.reward_shaping
        if self.horizon is not None:
            env_meta["env_kwargs"]["horizon"] = int(self.horizon)
        shape_meta = ckpt_dict["shape_metadata"]

        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            env_name=self.env_name,
            render=self.render,
            render_offscreen=self.render_offscreen,
            use_image_obs=shape_meta.get("use_images", False),
            use_depth_obs=shape_meta.get("use_depths", False),
        )
        return EnvUtils.wrap_env_from_config(env, config=config)


def _stack_obs(obs_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.stack([obs[key] for obs in obs_list], axis=0) for key in obs_list[0]}


def _get_env_goal(env: Any) -> dict[str, np.ndarray] | None:
    get_goal = getattr(env, "get_goal", None)
    if not callable(get_goal):
        return None
    try:
        goal = get_goal()
    except (AttributeError, NotImplementedError):
        return None
    if goal is None:
        return None
    return {key: np.asarray(value) for key, value in goal.items()}


class SerialRobomimicVectorEnv:
    def __init__(self, env_fns: list[Callable[[], Any]]) -> None:
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.device = torch.device("cpu")
        self.cfg = {}
        low, high = self.envs[0].base_env.action_spec
        self.num_actions = int(np.prod(low.shape))
        self.single_action_space = ActionSpaceSpec(
            shape=tuple(low.shape),
            low=low.astype(np.float32),
            high=high.astype(np.float32),
        )
        sample_obs = self.envs[0].reset()
        self.single_observation_space = {key: value.shape for key, value in sample_obs.items()}
        self._timesteps = np.zeros(self.num_envs, dtype=np.int32)
        self._horizons = np.array([int(env.base_env.horizon) for env in self.envs], dtype=np.int32)
        self.max_episode_length = int(self._horizons.max())
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        obs_list = []
        for idx, env in enumerate(self.envs):
            if seed is not None:
                np.random.seed(seed + idx)
            obs_list.append(env.reset())
        self._timesteps[:] = 0
        self.episode_length_buf.zero_()
        return _stack_obs(obs_list), {}

    def step(self, actions: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        obs_list = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        final_info: list[dict[str, Any] | None] = [None] * self.num_envs

        for idx, (env, action) in enumerate(zip(self.envs, actions, strict=True)):
            obs, reward, done, info = env.step(action)
            self._timesteps[idx] += 1
            self.episode_length_buf[idx] += 1
            success = bool(env.is_success().get("task", False))
            timeout = bool(getattr(env.base_env, "done", False) or (self._timesteps[idx] >= self._horizons[idx]))

            rewards[idx] = float(reward)
            terminated[idx] = success
            truncated[idx] = timeout and not success

            if success or timeout or done:
                final_info[idx] = dict(info)
                final_info[idx]["success"] = success
                final_info[idx]["reward"] = float(reward)
                obs = env.reset()
                self._timesteps[idx] = 0
                self.episode_length_buf[idx] = 0

            obs_list.append(obs)

        return _stack_obs(obs_list), rewards, terminated, truncated, {"final_info": final_info}

    def close(self) -> None:
        for env in self.envs:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()

    def get_observations(self) -> TensorDict:
        obs, _ = self.reset()
        tensor_obs = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in obs.items()}
        return TensorDict(tensor_obs, batch_size=[self.num_envs])

    def get_goal(self) -> dict[str, np.ndarray] | None:
        goals = [_get_env_goal(env) for env in self.envs]
        if any(goal is None for goal in goals):
            return None
        return _stack_obs(goals)


def _worker(remote: Any, env_fn_bytes: bytes) -> None:
    _ensure_local_repo_paths()
    env_fn = cloudpickle.loads(env_fn_bytes)
    env = env_fn()
    timestep = 0
    horizon = int(env.base_env.horizon)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "metadata":
                low, high = env.base_env.action_spec
                remote.send(
                    {
                        "action_low": low.astype(np.float32),
                        "action_high": high.astype(np.float32),
                        "sample_obs": env.reset(),
                        "horizon": horizon,
                    }
                )
                timestep = 0
            elif cmd == "reset":
                if data is not None:
                    np.random.seed(data)
                timestep = 0
                remote.send(env.reset())
            elif cmd == "step":
                obs, reward, done, info = env.step(data)
                timestep += 1
                success = bool(env.is_success().get("task", False))
                timeout = bool(getattr(env.base_env, "done", False) or (timestep >= horizon))
                terminated = success
                truncated = timeout and not success
                final_info = None
                if success or timeout or done:
                    final_info = dict(info)
                    final_info["success"] = success
                    final_info["reward"] = float(reward)
                    obs = env.reset()
                    timestep = 0
                remote.send((obs, float(reward), terminated, truncated, final_info))
            elif cmd == "get_goal":
                remote.send(_get_env_goal(env))
            elif cmd == "close":
                close_fn = getattr(env, "close", None)
                if callable(close_fn):
                    close_fn()
                remote.close()
                break
            else:
                raise ValueError(f"Unsupported worker command: {cmd}")
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


class ParallelRobomimicVectorEnv:
    def __init__(self, env_fns: list[Callable[[], Any]], start_method: str = "spawn") -> None:
        self.num_envs = len(env_fns)
        self.device = torch.device("cpu")
        self.cfg = {}
        ctx = mp.get_context(start_method)
        self.remotes = []
        self.processes = []
        for env_fn in env_fns:
            parent_remote, child_remote = ctx.Pipe()
            process = ctx.Process(target=_worker, args=(child_remote, cloudpickle.dumps(env_fn)))
            process.daemon = True
            process.start()
            child_remote.close()
            self.remotes.append(parent_remote)
            self.processes.append(process)

        self.remotes[0].send(("metadata", None))
        metadata = self.remotes[0].recv()
        self.single_action_space = ActionSpaceSpec(
            shape=tuple(metadata["action_low"].shape),
            low=metadata["action_low"],
            high=metadata["action_high"],
        )
        self.num_actions = int(np.prod(self.single_action_space.shape))
        self.single_observation_space = {key: value.shape for key, value in metadata["sample_obs"].items()}
        self.max_episode_length = int(metadata["horizon"])
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)

        for remote in self.remotes[1:]:
            remote.send(("reset", None))
            remote.recv()

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        for idx, remote in enumerate(self.remotes):
            worker_seed = None if seed is None else seed + idx
            remote.send(("reset", worker_seed))
        obs_list = [remote.recv() for remote in self.remotes]
        self.episode_length_buf.zero_()
        return _stack_obs(obs_list), {}

    def step(self, actions: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        for remote, action in zip(self.remotes, actions, strict=True):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]
        obs_list = [result[0] for result in results]
        rewards = np.asarray([result[1] for result in results], dtype=np.float32)
        terminated = np.asarray([result[2] for result in results], dtype=bool)
        truncated = np.asarray([result[3] for result in results], dtype=bool)
        final_info = [result[4] for result in results]
        self.episode_length_buf += 1
        for idx, info in enumerate(final_info):
            if info is not None:
                self.episode_length_buf[idx] = 0
        return _stack_obs(obs_list), rewards, terminated, truncated, {"final_info": final_info}

    def close(self) -> None:
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for process in self.processes:
            process.join(timeout=1.0)

    def get_observations(self) -> TensorDict:
        obs, _ = self.reset()
        tensor_obs = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in obs.items()}
        return TensorDict(tensor_obs, batch_size=[self.num_envs])

    def get_goal(self) -> dict[str, np.ndarray] | None:
        for remote in self.remotes:
            remote.send(("get_goal", None))
        goals = [remote.recv() for remote in self.remotes]
        if any(goal is None for goal in goals):
            return None
        return _stack_obs(goals)


def make_robosuite_env_from_checkpoint(
    checkpoint_path: str,
    task_override: str | None = None,
    *,
    render: bool = False,
    render_offscreen: bool = False,
    reward_shaping_override: bool | None = None,
    horizon_override: int | None = None,
) -> tuple[Callable[[], Any], Any, dict]:
    _ensure_local_repo_paths()
    import mimicgen  # noqa: F401
    from robomimic.utils.file_utils import config_from_checkpoint, load_dict_from_checkpoint

    ckpt_dict = load_dict_from_checkpoint(checkpoint_path)
    config, _ = config_from_checkpoint(ckpt_dict=ckpt_dict, verbose=False)

    env_meta = deepcopy(ckpt_dict["env_metadata"])
    env_meta["env_kwargs"]["ignore_done"] = False
    env_name = task_override or env_meta["env_name"]
    reward_shaping = env_meta["env_kwargs"].get("reward_shaping")
    if reward_shaping_override is not None:
        reward_shaping = reward_shaping_override
    horizon = env_meta["env_kwargs"].get("horizon")
    if horizon_override is not None:
        horizon = int(horizon_override)
    factory = RobosuiteCheckpointFactory(
        checkpoint_path=checkpoint_path,
        env_name=env_name,
        render=render,
        render_offscreen=render_offscreen,
        reward_shaping=reward_shaping,
        horizon=horizon,
    )

    return factory, config, ckpt_dict
