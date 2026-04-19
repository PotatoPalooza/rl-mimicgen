"""Batched robosuite env wrapper for DPPO with warp + non-warp backends.

Exposes a common dict-of-tensors step/reset interface so the DPPO fine-tune
loop in ``rl_mimicgen/dppo/finetune/train_dppo_agent.py`` is agnostic to
whether rollouts run on a single CPU robosuite env (``use_warp=False``,
``num_envs=1`` — the default, matches the prior single-env loop) or on a
GPU-parallel MuJoCo Warp batch (``use_warp=True``, ``num_envs>=N`` — shares
the fidelity machinery from ``rl_mimicgen.rsl_rl.wrappers.robomimic_vec_env``:
masked per-env reset, NaN scrub, early-termination buckets, video capture).

Both backends must stay feature-complete so warp-vs-CPU is A/B comparable.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rl_mimicgen.dppo.envs.task_registry import DPPOTaskSpec, get_task_spec


WORKSPACE_DIR = Path(__file__).resolve().parents[3]


def _ensure_local_repo_paths() -> None:
    import sys

    os.environ.setdefault("MUJOCO_GL", "glx")
    for repo_name in ("mimicgen", "robomimic", "robosuite", "robosuite-task-zoo"):
        repo_path = WORKSPACE_DIR / "resources" / repo_name
        if repo_path.exists():
            repo_path_str = str(repo_path)
            if repo_path_str not in sys.path:
                sys.path.insert(0, repo_path_str)


def _load_env_meta(spec: DPPOTaskSpec, dataset_path: str | None, reward_shaping: bool | None) -> dict:
    from robomimic.utils.file_utils import get_env_metadata_from_dataset

    env_meta = deepcopy(get_env_metadata_from_dataset(dataset_path or spec.dataset_path))
    if reward_shaping is not None:
        env_meta["env_kwargs"]["reward_shaping"] = reward_shaping
    env_meta["env_kwargs"]["ignore_done"] = False
    env_meta["env_kwargs"]["horizon"] = int(spec.horizon)
    return env_meta


def _initialize_obs_utils(spec: DPPOTaskSpec) -> None:
    import robomimic.utils.obs_utils as ObsUtils

    ObsUtils.initialize_obs_utils_with_obs_specs(
        [
            {
                "obs": {
                    "low_dim": list(spec.obs_keys),
                    "rgb": [],
                    "depth": [],
                    "scan": [],
                },
                "goal": {"low_dim": [], "rgb": [], "depth": [], "scan": []},
            }
        ]
    )


class DPPOMimicGenVecEnv:
    """Common dict-obs vec-env interface for DPPO rollouts.

    Construction goes through :func:`build_dppo_vec_env`; direct ``__init__``
    is an internal. API mirrors what the DPPO fine-tune loop needs:

    - ``reset() -> dict[str, torch.Tensor]`` — per-key ``(num_envs, *)``
      tensors on ``device``.
    - ``step(actions: torch.Tensor) -> tuple[dict, Tensor, Tensor, dict]`` —
      ``(obs_dict, rewards: (N,), dones: (N,) bool, extras)``. Terminated
      envs auto-reset in-place; ``dones`` reflects the closing step.
    - ``close()`` — releases underlying robosuite resources.

    Both backends compute ``done`` as ``timeout | success | diverged |
    early_term``. Success is ``env.is_success()["task"]`` (matches the
    previous single-env loop's semantics via ``done or success``). Reward
    from terminated envs is zeroed on the closing step to match the RSL-RL
    wrapper and avoid leaking divergence noise into returns.
    """

    def __init__(
        self,
        *,
        spec: DPPOTaskSpec,
        use_warp: bool,
        num_envs: int,
        device: str | torch.device,
        horizon: int,
        clip_actions: float | None,
        env: Any,
        inner_vec_env: Any | None = None,
    ) -> None:
        self.spec = spec
        self.use_warp = use_warp
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.horizon = int(horizon)
        self.clip_actions = clip_actions
        self._env = env
        self._inner = inner_vec_env  # RobomimicVecEnv when use_warp=True, else None

        base = getattr(env, "env", env)
        low, high = base.action_spec
        self.action_low = np.asarray(low, dtype=np.float32)
        self.action_high = np.asarray(high, dtype=np.float32)
        self.action_dim = int(np.prod(self.action_low.shape))
        self.obs_keys = tuple(spec.obs_keys)

        # Serial-backend internal state.
        self._serial_step: int = 0
        self._serial_last_obs: dict[str, np.ndarray] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> dict[str, torch.Tensor]:
        if self.use_warp:
            obs_td, _ = self._inner.reset()
            return self._tensordict_to_dict(obs_td)
        raw = self._env.reset()
        self._serial_last_obs = self._filter_obs(raw)
        self._serial_step = 0
        return self._serial_obs_to_dict(self._serial_last_obs)

    def step(
        self, actions: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
        if self.use_warp:
            return self._warp_step(actions)
        return self._serial_step_impl(actions)

    def close(self) -> None:
        close_fn = getattr(self._env, "close", None)
        if callable(close_fn):
            close_fn()

    # ------------------------------------------------------------------
    # Warp backend
    # ------------------------------------------------------------------

    def _warp_step(
        self, actions: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
        if actions.device != self.device:
            actions = actions.to(self.device)
        if actions.dtype != torch.float32:
            actions = actions.float()
        obs_td, reward, dones, extras = self._inner.step(actions)
        return self._tensordict_to_dict(obs_td), reward, dones.bool(), extras

    def _tensordict_to_dict(self, obs_td: Any) -> dict[str, torch.Tensor]:
        # RobomimicVecEnv(flatten_obs=False) returns a TensorDict keyed by obs name.
        return {k: obs_td[k] for k in self.obs_keys}

    # ------------------------------------------------------------------
    # Serial (non-warp) backend — num_envs=1
    # ------------------------------------------------------------------

    def _filter_obs(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {key: np.asarray(obs[key], dtype=np.float32) for key in self.obs_keys}

    def _serial_obs_to_dict(self, obs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for k in self.obs_keys:
            v = torch.as_tensor(obs[k], dtype=torch.float32, device=self.device)
            out[k] = v.unsqueeze(0)  # prepend num_envs=1 batch dim
        return out

    def _serial_step_impl(
        self, actions: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
        assert actions.shape[0] == 1, (
            f"Non-warp backend expects num_envs=1 but got actions shape {tuple(actions.shape)}"
        )
        action_np = actions[0].detach().cpu().numpy().astype(np.float32, copy=False)
        if self.clip_actions is not None:
            action_np = np.clip(action_np, -self.clip_actions, self.clip_actions)
        obs_raw, reward, done, info = self._env.step(action_np)
        success = bool(self._env.is_success().get("task", False))
        self._serial_step += 1
        timeout = self._serial_step >= self.horizon
        closing = bool(done) or success or timeout

        if closing:
            # Reward on closing step is zeroed to mirror the warp wrapper's
            # behavior on divergence/early-term; success is treated as a
            # neutral termination so returns reflect the trajectory to that
            # point. This preserves parity with the prior single-env loop,
            # which also consumed ``reward`` on the closing step then reset.
            reward_on_close = float(reward)
            obs_raw = self._env.reset()
            self._serial_step = 0
            reward = reward_on_close
        self._serial_last_obs = self._filter_obs(obs_raw)
        obs_dict = self._serial_obs_to_dict(self._serial_last_obs)
        reward_t = torch.tensor([float(reward)], dtype=torch.float32, device=self.device)
        done_t = torch.tensor([closing], dtype=torch.bool, device=self.device)
        extras = {"time_outs": torch.tensor([timeout and not (bool(done) or success)],
                                            dtype=torch.bool, device=self.device),
                  "info": info,
                  "success": success}
        return obs_dict, reward_t, done_t, extras

    # ------------------------------------------------------------------
    # Rendering (used by eval/video)
    # ------------------------------------------------------------------

    def render_rgb_array(
        self,
        camera_name: str = "agentview",
        height: int = 512,
        width: int = 512,
    ) -> np.ndarray:
        return np.asarray(
            self._env.render(
                mode="rgb_array",
                camera_name=camera_name,
                height=height,
                width=width,
            )
        )


def build_dppo_vec_env(
    *,
    task: str,
    variant: str,
    use_warp: bool = False,
    num_envs: int = 1,
    device: str | torch.device = "cuda:0",
    dataset_path: str | None = None,
    reward_shaping: bool | None = None,
    clip_actions: float | None = 1.0,
    render_offscreen: bool = False,
    warp_njmax_per_env: int | None = None,
    warp_naconmax_per_env: int | None = None,
    warp_graph_capture: bool = False,
    physics_timestep: float | None = None,
) -> DPPOMimicGenVecEnv:
    """Construct a DPPO-ready vec env for the given task/variant.

    ``use_warp=False`` forces ``num_envs=1`` (warp is the only batched
    backend). Warp-specific knobs (``warp_njmax_per_env``,
    ``warp_naconmax_per_env``, ``warp_graph_capture``, ``physics_timestep``)
    are ignored when ``use_warp=False``.
    """
    _ensure_local_repo_paths()

    import mimicgen  # noqa: F401
    import robomimic.utils.env_utils as EnvUtils

    spec = get_task_spec(task=task, variant=variant)
    _initialize_obs_utils(spec)

    env_meta = _load_env_meta(spec, dataset_path, reward_shaping)

    if use_warp:
        from rl_mimicgen.rsl_rl.wrappers.robomimic_vec_env import RobomimicVecEnv
        from rl_mimicgen.rsl_rl.warp_buffer_sizes import resolve_warp_buffer_sizes

        warp_caps: dict[str, int] = {}
        resolved = resolve_warp_buffer_sizes(spec.env_name) or {}
        warp_caps.update({k: int(v) for k, v in resolved.items()})
        if warp_njmax_per_env is not None:
            warp_caps["njmax_per_env"] = int(warp_njmax_per_env)
        if warp_naconmax_per_env is not None:
            warp_caps["naconmax_per_env"] = int(warp_naconmax_per_env)
        if warp_caps:
            env_meta.setdefault("env_kwargs", {}).update(warp_caps)

        if physics_timestep is not None:
            import robosuite

            robosuite.macros.SIMULATION_TIMESTEP = float(physics_timestep)

        os.environ["ROBOSUITE_WARP_GRAPH"] = "1" if warp_graph_capture else "0"

        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            env_name=spec.env_name,
            render=False,
            render_offscreen=render_offscreen,
            use_image_obs=False,
            use_depth_obs=False,
            use_warp=True,
            num_envs=int(num_envs),
        )
        inner = RobomimicVecEnv(
            env=env,
            horizon=spec.horizon,
            device=str(device),
            clip_actions=clip_actions,
            obs_keys=list(spec.obs_keys),
            flatten_obs=False,
        )
        return DPPOMimicGenVecEnv(
            spec=spec,
            use_warp=True,
            num_envs=int(num_envs),
            device=device,
            horizon=spec.horizon,
            clip_actions=clip_actions,
            env=env,
            inner_vec_env=inner,
        )

    if num_envs != 1:
        raise ValueError(
            f"use_warp=False requires num_envs=1 (got {num_envs}); "
            "the CPU robosuite backend is not batched."
        )

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=spec.env_name,
        render=False,
        render_offscreen=render_offscreen,
        use_image_obs=False,
        use_depth_obs=False,
    )
    return DPPOMimicGenVecEnv(
        spec=spec,
        use_warp=False,
        num_envs=1,
        device=device,
        horizon=spec.horizon,
        clip_actions=clip_actions,
        env=env,
    )
