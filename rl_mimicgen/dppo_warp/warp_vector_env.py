"""Warp-batched gym-vector-API adapter for upstream DPPO fine-tuning.

Upstream DPPO (``resources/dppo/agent/finetune/train_ppo_diffusion_agent.py``)
expects a vec env matching the ``AsyncVectorEnv(RobomimicLowdimWrapper +
MultiStep)`` contract:

- ``venv.step(action_venv)`` where ``action_venv`` is
  ``(n_envs, act_steps, action_dim)`` numpy in normalized ``[-1, 1]``.
  Returns ``(obs_dict, reward, terminated, truncated, info_list)`` where
  ``obs_dict = {"state": (n_envs, n_obs_steps, obs_dim)}`` (min-max
  normalized), ``reward`` is a sum over the act_steps, ``terminated``
  and ``truncated`` are ``(n_envs,)`` bool. Terminated envs auto-reset
  inside ``step`` to match ``reset_within_step=True`` MultiStep
  semantics.
- ``venv.reset_arg(options_list)`` — per-env options dicts; supports
  ``options[i]["video_path"]`` to start an mp4 writer for env i.
- ``venv.reset_one_arg(env_ind, options)``.
- ``venv.seed([...])``.
- ``venv.n_envs``, ``venv.single_action_space``.
- ``venv.close()``.

This class wraps a single MuJoCo-Warp batched robosuite env
(``use_warp=True, num_envs=N``), delegating masked per-env reset, NaN
scrub, early-termination hooks, and robot qpos caching to
``rl_mimicgen.rsl_rl.wrappers.robomimic_vec_env.RobomimicVecEnv`` with
``flatten_obs=False``. On top of that we add:

- min-max obs normalization and action unnormalization (reading
  ``normalization.npz`` once at init),
- an ``(n_envs, n_obs_steps, obs_dim)`` rolling obs history,
- per-env ``MultiStep.cnt`` tracking and ``max_episode_steps`` truncation,
- auto-reset of envs that finish within an ``act_steps`` sub-loop,
- per-env video writers that flush on next reset, using a CPU-MjData
  render fast path per env to avoid warp's full-SoA GPU->CPU sync cost.

Success signal is propagated as ``info[i]["success"]`` from the robosuite
env's ``is_success()["task"]`` mask.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _ensure_local_repo_paths() -> None:
    import sys

    workspace = Path(__file__).resolve().parents[2]
    os.environ.setdefault("MUJOCO_GL", "egl")
    for repo in ("mimicgen", "robomimic", "robosuite", "robosuite-task-zoo"):
        p = workspace / "resources" / repo
        if p.exists():
            p_str = str(p)
            if p_str not in sys.path:
                sys.path.insert(0, p_str)


class WarpRobomimicVectorEnv:
    """Drop-in warp-backed replacement for ``AsyncVectorEnv + RobomimicLowdimWrapper + MultiStep``."""

    def __init__(
        self,
        *,
        robomimic_env_cfg_path: str,
        normalization_path: str,
        low_dim_keys: list[str],
        num_envs: int,
        max_episode_steps: int,
        n_obs_steps: int,
        n_action_steps: int,
        device: str = "cuda:0",
        reward_shaping: bool = False,
        save_video: bool = False,
        render_hw: tuple[int, int] = (256, 256),
        render_camera_name: str = "agentview",
        warp_graph_capture: bool = False,
        njmax_per_env: int | None = None,
        naconmax_per_env: int | None = None,
        physics_timestep: float | None = None,
    ) -> None:
        _ensure_local_repo_paths()

        import gym
        from gym import spaces
        import mimicgen  # noqa: F401
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.obs_utils as ObsUtils
        from rl_mimicgen.rsl_rl.warp_buffer_sizes import resolve_warp_buffer_sizes
        from rl_mimicgen.rsl_rl.wrappers.robomimic_vec_env import RobomimicVecEnv

        self.n_envs = int(num_envs)
        self.num_envs = self.n_envs  # alias; AsyncVectorEnv exposes both
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.max_episode_steps = int(max_episode_steps)
        self.save_video = bool(save_video)
        self.render_hw = tuple(render_hw)
        self.render_camera_name = render_camera_name
        self.device = torch.device(device)
        self.low_dim_keys = list(low_dim_keys)

        # Normalization stats — upstream RobomimicLowdimWrapper uses min-max
        # bounds (not mean/std) for obs, and uses min/max for action unnormalize.
        norm = np.load(normalization_path)
        self._obs_min = np.asarray(norm["obs_min"], dtype=np.float32)
        self._obs_max = np.asarray(norm["obs_max"], dtype=np.float32)
        self._action_min = np.asarray(norm["action_min"], dtype=np.float32)
        self._action_max = np.asarray(norm["action_max"], dtype=np.float32)
        self._action_range = np.maximum(self._action_max - self._action_min, 1e-8).astype(np.float32)
        self._obs_range = np.maximum(self._obs_max - self._obs_min, 1e-6).astype(np.float32)

        # GPU-side normalize/unnormalize tensors — avoid per-substep host round-trips.
        self._action_min_t = torch.as_tensor(self._action_min, device=self.device)
        self._action_range_t = torch.as_tensor(self._action_range, device=self.device)
        self._obs_min_t = torch.as_tensor(self._obs_min, device=self.device)
        self._obs_max_t = torch.as_tensor(self._obs_max, device=self.device)

        # Build env_meta for warp — merge per-task warp buffer caps + CLI overrides.
        with open(robomimic_env_cfg_path, "r", encoding="utf-8") as f:
            env_meta = json.load(f)
        env_name = env_meta.get("env_name")
        env_meta.setdefault("env_kwargs", {})
        env_meta["env_kwargs"]["reward_shaping"] = bool(reward_shaping)
        env_meta["env_kwargs"]["ignore_done"] = False
        env_meta["env_kwargs"]["horizon"] = int(max_episode_steps)

        warp_caps: dict[str, int] = {}
        resolved = resolve_warp_buffer_sizes(env_name) or {}
        warp_caps.update({k: int(v) for k, v in resolved.items()})
        if njmax_per_env is not None:
            warp_caps["njmax_per_env"] = int(njmax_per_env)
        if naconmax_per_env is not None:
            warp_caps["naconmax_per_env"] = int(naconmax_per_env)
        if warp_caps:
            env_meta["env_kwargs"].update(warp_caps)

        if physics_timestep is not None:
            import robosuite

            robosuite.macros.SIMULATION_TIMESTEP = float(physics_timestep)

        os.environ["ROBOSUITE_WARP_GRAPH"] = "1" if warp_graph_capture else "0"

        # Obs modality init — upstream make_async does this per process; we do
        # it once globally since warp uses a single sim instance.
        ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": list(self.low_dim_keys)})

        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=self.save_video,
            use_image_obs=False,
            use_depth_obs=False,
            use_warp=True,
            num_envs=self.n_envs,
        )
        base = env.env  # robosuite env
        if hasattr(base, "hard_reset"):
            base.hard_reset = False

        self._env = env
        self._inner = RobomimicVecEnv(
            env=env,
            horizon=self.max_episode_steps,
            device=str(self.device),
            clip_actions=None,
            obs_keys=list(self.low_dim_keys),
            terminate_on_success=False,  # upstream never terminates early
            flatten_obs=False,
        )

        # Action / observation space — gym-vector exposes single_* on the base
        # env; agent only reads single_action_space in the finetune path.
        action_dim = int(env.action_dimension)
        self.action_dim = action_dim
        self.obs_dim = int(self._obs_min.shape[0])
        self.single_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_action_steps, action_dim), dtype=np.float32
        )
        self.single_observation_space = spaces.Dict(
            {
                "state": spaces.Box(
                    low=-1.0, high=1.0, shape=(n_obs_steps, self.obs_dim), dtype=np.float32
                )
            }
        )
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space

        # Per-env episode-step counters for MultiStep-style truncation. GPU
        # tensor so we can mask it without a CPU round-trip per substep.
        self._cnt_t = torch.zeros(self.n_envs, dtype=torch.int64, device=self.device)

        # Obs history ring buffer — shape (n_envs, n_obs_steps, obs_dim), stores
        # the most recent n_obs_steps normalized flat obs per env (with left-
        # padding after a reset so the earliest slot is the reset obs). Lives
        # on GPU to avoid a per-substep device->host copy; synced to CPU only
        # when returning from step()/reset().
        self._obs_history_t = torch.zeros(
            (self.n_envs, self.n_obs_steps, self.obs_dim),
            dtype=torch.float32,
            device=self.device,
        )

        # Per-env video writers: list[imageio.Writer | None]. Only env indexes
        # with options["video_path"] on reset get a writer.
        self._video_writers: list[Any] = [None] * self.n_envs
        self._render_mj_data = None
        self._render_camera_id: int | None = None

        # Do an initial reset so downstream probes for observation_space /
        # action_space have real data to work with; caller is free to invoke
        # reset_arg again to start recording clips.
        self._initial_reset_done = False
        self._seed_deferred: list[int] | None = None

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def _unnormalize_action_torch(self, actions_norm: torch.Tensor) -> torch.Tensor:
        # Upstream: (action + 1) / 2 * (max - min) + min. Action comes in as
        # [-1, 1]; we feed unnormalized to the sim so it matches what the
        # demos used.
        return (actions_norm + 1.0) * 0.5 * self._action_range_t + self._action_min_t

    def _flatten_raw_obs_gpu(self, obs_td) -> torch.Tensor:
        """Concatenate per-key (N, *) tensors from RobomimicVecEnv into a
        flat ``(N, obs_dim)`` GPU tensor in the declared key order. No
        host copy — concat stays on device.
        """
        chunks: list[torch.Tensor] = []
        for key in self.low_dim_keys:
            v = obs_td[key]
            if not isinstance(v, torch.Tensor):
                v = torch.as_tensor(np.asarray(v, dtype=np.float32), device=self.device)
            else:
                v = v.float()
                if v.device != self.device:
                    v = v.to(self.device)
            if v.dim() == 1:
                v = v.unsqueeze(-1)
            elif v.dim() > 2:
                v = v.reshape(v.shape[0], -1)
            chunks.append(v)
        return torch.cat(chunks, dim=1)

    def _normalize_obs_gpu(self, flat_t: torch.Tensor) -> torch.Tensor:
        # Match RobomimicLowdimWrapper.normalize_obs: 2 * ((x - min) / (max - min + 1e-6) - 0.5) -> [-1, 1].
        return 2.0 * ((flat_t - self._obs_min_t) / (self._obs_max_t - self._obs_min_t + 1e-6) - 0.5)

    def _stacked_obs_dict(self) -> dict[str, np.ndarray]:
        # Single CPU sync per outer step / reset.
        return {"state": self._obs_history_t.detach().cpu().numpy()}

    def _push_obs_gpu(self, obs_normalized_t: torch.Tensor) -> None:
        # Roll history: drop oldest slot, push current to last slot.
        if self.n_obs_steps > 1:
            self._obs_history_t[:, :-1] = self._obs_history_t[:, 1:].clone()
        self._obs_history_t[:, -1] = obs_normalized_t

    def _fill_obs_on_reset_gpu(
        self, obs_normalized_t: torch.Tensor, reset_mask_t: torch.Tensor | None
    ) -> None:
        """Broadcast a fresh obs to every history slot for reset envs."""
        if reset_mask_t is None:
            self._obs_history_t[:] = obs_normalized_t.unsqueeze(1)
            return
        if not reset_mask_t.any():
            return
        # Broadcast across the time dim for masked rows only.
        self._obs_history_t[reset_mask_t] = obs_normalized_t[reset_mask_t].unsqueeze(1)

    # ------------------------------------------------------------------
    # gym.vector-style API
    # ------------------------------------------------------------------

    def seed(self, seeds):
        if isinstance(seeds, (int, np.integer)):
            torch.manual_seed(int(seeds))
            np.random.seed(int(seeds))
        else:
            seeds = list(seeds)
            if seeds:
                torch.manual_seed(int(seeds[0]))
                np.random.seed(int(seeds[0]))

    def reset(self) -> dict[str, np.ndarray]:
        return self.reset_arg(options_list=[{} for _ in range(self.n_envs)])

    def reset_arg(self, options_list: list[dict]) -> dict[str, np.ndarray]:
        # Close any writers from previous iter and open new ones for envs that
        # requested a video_path in this reset.
        self._close_all_video_writers()
        if self.save_video:
            for i, options in enumerate(options_list):
                vp = options.get("video_path")
                if vp:
                    Path(vp).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
                    self._video_writers[i] = self._open_video_writer(vp)

        # Full-batch reset.
        obs_td, _ = self._inner.reset()
        self._cnt_t.zero_()
        flat_t = self._flatten_raw_obs_gpu(obs_td)
        normalized_t = self._normalize_obs_gpu(flat_t)
        self._fill_obs_on_reset_gpu(normalized_t, reset_mask_t=None)
        self._initial_reset_done = True
        return self._stacked_obs_dict()

    def reset_one_arg(self, env_ind: int, options: dict | None = None) -> dict[str, np.ndarray]:
        options = options or {}
        if self.save_video and options.get("video_path"):
            self._close_video_writer(env_ind)
            Path(options["video_path"]).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            self._video_writers[env_ind] = self._open_video_writer(options["video_path"])

        mask_t = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
        mask_t[env_ind] = True
        self._inner._reset_envs(mask_t)
        self._cnt_t[env_ind] = 0
        obs_td = self._inner.get_observations()
        flat_t = self._flatten_raw_obs_gpu(obs_td)
        normalized_t = self._normalize_obs_gpu(flat_t)
        self._obs_history_t[env_ind] = normalized_t[env_ind].unsqueeze(0).expand(self.n_obs_steps, -1)
        return self._stacked_obs_dict()

    def step(
        self, action_venv
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Apply a chunk of ``act_steps`` normalized actions per env.

        ``action_venv``: ``(n_envs, act_steps, action_dim)`` in ``[-1, 1]``.
        Accepts numpy *or* a torch tensor already on ``self.device`` — the
        finetune agent can hand actions over without a D2H+H2D round trip.

        All per-substep accumulators live on GPU; one CPU sync happens at
        the end for the outer return values.
        """
        if action_venv.ndim == 2:  # (n_envs, action_dim) — wrap single-step
            action_venv = action_venv[:, None]
        assert action_venv.shape[0] == self.n_envs
        act_steps = action_venv.shape[1]
        N = self.n_envs
        device = self.device

        # GPU-side accumulators.
        reward_sum_t = torch.zeros(N, dtype=torch.float32, device=device)
        terminated_t = torch.zeros(N, dtype=torch.bool, device=device)
        truncated_t = torch.zeros(N, dtype=torch.bool, device=device)
        success_any_t = torch.zeros(N, dtype=torch.bool, device=device)
        done_mask_t = torch.zeros(N, dtype=torch.bool, device=device)

        # Full action chunk -> GPU + unnormalize. Fast path when caller
        # already handed us a matching GPU tensor.
        if isinstance(action_venv, torch.Tensor) and action_venv.device == device:
            actions_chunk_t = action_venv.to(dtype=torch.float32)
        else:
            actions_chunk_t = torch.as_tensor(
                action_venv, dtype=torch.float32, device=device
            )
        actions_chunk_t = self._unnormalize_action_torch(actions_chunk_t)

        max_ep_t = torch.tensor(self.max_episode_steps, dtype=self._cnt_t.dtype, device=device)

        for sub in range(act_steps):
            actions_t = actions_chunk_t[:, sub]  # (N, action_dim)
            obs_td, reward_t, dones_t, _extras = self._inner.step(actions_t)
            reward_t = reward_t.to(device=device, dtype=torch.float32)
            dones_t = dones_t.to(device=device, dtype=torch.bool)

            # Success per env from the underlying robosuite env (warp returns a (N,) tensor).
            success_raw = self._inner.env.is_success().get("task", None)
            if isinstance(success_raw, torch.Tensor):
                success_t = success_raw.to(device=device, dtype=torch.bool).view(N)
            elif success_raw is None:
                success_t = torch.zeros(N, dtype=torch.bool, device=device)
            else:
                success_t = torch.as_tensor(np.asarray(success_raw).astype(bool), device=device).view(N)
            success_any_t |= success_t

            # Credit reward + count only for envs still active at this substep.
            active_t = ~done_mask_t
            reward_sum_t = reward_sum_t + torch.where(active_t, reward_t, torch.zeros_like(reward_t))
            self._cnt_t = self._cnt_t + active_t.to(self._cnt_t.dtype)

            # terminated vs truncated: inner dones are physics early-term /
            # divergence; our own counter drives timeout truncation. Separating
            # them keeps GAE bootstrap correct for the agent.
            fresh_done_t = dones_t & active_t
            fresh_timeout_t = (self._cnt_t >= max_ep_t) & active_t
            truncated_t |= fresh_timeout_t
            terminated_t |= fresh_done_t & ~fresh_timeout_t
            close_mask_t = fresh_done_t | fresh_timeout_t
            done_mask_t |= close_mask_t

            # Render + video IO (per-env CPU renders, only for active writers).
            if self.save_video:
                self._append_video_frames()

            # Post-step obs -> normalized, pushed onto GPU history.
            flat_t = self._flatten_raw_obs_gpu(obs_td)
            normalized_t = self._normalize_obs_gpu(flat_t)
            if close_mask_t.any():
                self._push_obs_gpu(normalized_t)
                self._fill_obs_on_reset_gpu(normalized_t, reset_mask_t=close_mask_t)
                self._cnt_t[close_mask_t] = 0
            else:
                self._push_obs_gpu(normalized_t)

            if bool(done_mask_t.all()):
                break  # all envs finished; skip remaining sub-actions

        # Force keyframe reset for envs that truncated but didn't trigger the
        # inner wrapper's done branch (so the inner sim isn't held past
        # horizon for those envs).
        need_reset_t = truncated_t & ~terminated_t
        if bool(need_reset_t.any()):
            self._inner._reset_envs(need_reset_t)
            obs_td = self._inner.get_observations()
            flat_t = self._flatten_raw_obs_gpu(obs_td)
            normalized_t = self._normalize_obs_gpu(flat_t)
            self._fill_obs_on_reset_gpu(normalized_t, reset_mask_t=need_reset_t)
            self._cnt_t[need_reset_t] = 0

        # Single CPU sync for the outer return contract.
        reward_sum = reward_sum_t.detach().cpu().numpy()
        terminated = terminated_t.detach().cpu().numpy()
        truncated = truncated_t.detach().cpu().numpy()
        success_any = success_any_t.detach().cpu().numpy()
        info_list = [{"success": bool(success_any[i])} for i in range(N)]
        return self._stacked_obs_dict(), reward_sum, terminated, truncated, info_list

    def close(self) -> None:
        self._close_all_video_writers()
        close_fn = getattr(self._env, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass

    # Gym-vector convenience methods the agent may call opportunistically.
    def call(self, name: str, *args, **kwargs):
        raise NotImplementedError("WarpRobomimicVectorEnv.call is not supported")

    def get_attr(self, name: str):
        raise NotImplementedError("WarpRobomimicVectorEnv.get_attr is not supported")

    # ------------------------------------------------------------------
    # Video helpers
    # ------------------------------------------------------------------

    def _open_video_writer(self, path: str):
        """Open an mp4 writer with HTML5/wandb-browser-compatible settings.

        Defaults in imageio-ffmpeg don't guarantee H.264 + yuv420p — on some
        builds you get yuv444p, which Chrome/Firefox/wandb's player refuse to
        play ("This video has no playable media."). Pin the codec, pixel
        format, macro-block size, and faststart so the moov atom lives at the
        front of the file for streaming playback.
        """
        import imageio

        return imageio.get_writer(
            path,
            fps=30,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=16,
            ffmpeg_params=["-movflags", "+faststart"],
        )

    def close_videos(self) -> None:
        """Finalize all open video writers without clearing their slot names.

        Must be called before handing mp4 paths to ``wandb.Video(path)``:
        wandb hashes + copies the file immediately on construction
        (``wandb/sdk/data_types/base_types/media.py:_set_file``), but imageio
        only writes the mp4 moov atom on ``writer.close()``. Without this the
        copy wandb uploads is a truncated/unfinalized mp4 and the browser
        player reports "no playable media" even though VLC may recover.
        """
        self._close_all_video_writers()

    def _close_video_writer(self, env_ind: int) -> None:
        w = self._video_writers[env_ind]
        if w is not None:
            try:
                w.close()
            except Exception:
                pass
            self._video_writers[env_ind] = None

    def _close_all_video_writers(self) -> None:
        for i in range(self.n_envs):
            self._close_video_writer(i)

    def _append_video_frames(self) -> None:
        """Render + append a frame for each env with an active writer.

        Uses a CPU ``MjData`` + ``mj_forward`` fast path so rendering a single
        env doesn't force a full-SoA GPU->CPU warp sync (~220 ms at 2k envs).
        """
        active_idxs = [i for i, w in enumerate(self._video_writers) if w is not None]
        if not active_idxs:
            return
        import mujoco
        import warp as wp

        sim = self._env.env.sim
        ctx = getattr(sim, "_render_context_offscreen", None)
        if ctx is None:
            return

        if self._render_mj_data is None:
            self._render_mj_data = mujoco.MjData(sim.model._model)
            self._render_camera_id = (
                sim.model.camera_name2id(self.render_camera_name)
                if self.render_camera_name is not None
                else None
            )

        h, w = self.render_hw
        qpos_all = wp.to_torch(sim._warp_data.qpos)
        qvel_all = wp.to_torch(sim._warp_data.qvel)
        d = self._render_mj_data
        saved_data_ptr = ctx.data._data
        try:
            ctx.data._data = d
            for env_ind in active_idxs:
                d.qpos[:] = qpos_all[env_ind].detach().cpu().numpy()
                d.qvel[:] = qvel_all[env_ind].detach().cpu().numpy()
                mujoco.mj_forward(sim.model._model, d)
                ctx.render(width=w, height=h, camera_id=self._render_camera_id, segmentation=False)
                frame = ctx.read_pixels(w, h, depth=False, segmentation=False)
                frame = np.ascontiguousarray(frame[::-1])
                self._video_writers[env_ind].append_data(frame)
        finally:
            ctx.data._data = saved_data_ptr
