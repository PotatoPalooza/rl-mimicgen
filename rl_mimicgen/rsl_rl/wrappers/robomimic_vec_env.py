"""VecEnv wrapper adapting a robomimic warp environment for RSL-RL's OnPolicyRunner.

Every termination (timeout at horizon, early success, or physics divergence)
is handled per-env via ``_reset_envs(env_indices)``, which resets just the
specified envs to a freshly-sampled random initial state (keyframe qpos/qvel +
re-randomised object placements via each mimicgen task's ``_reset_internal``
hooks). Other envs' trajectories continue undisturbed. This plays well with
RSL-RL's ``init_at_random_ep_len=True``, since per-env timeouts keep episode
phases desynchronised across the batch.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import torch
import numpy as np
from tensordict import TensorDict

from rsl_rl.env import VecEnv


class _EnvCfg:
    """Small wrapper exposing a ``to_dict()`` method for RSL-RL's wandb writer.

    ``WandbSummaryWriter.store_config`` first tries ``env_cfg.to_dict()`` and
    falls back to ``dataclasses.asdict(env_cfg)`` -- a plain dict fails both.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class RobomimicVecEnv(VecEnv):
    """Wraps a robomimic warp environment for use with RSL-RL."""

    def __init__(
        self,
        env: Any,
        horizon: int,
        device: str = "cuda:0",
        clip_actions: float | None = None,
        obs_keys: list[str] | None = None,
        terminate_on_success: bool = True,
        video_trigger: Callable[[int], bool] | None = None,
        video_dir: str | None = None,
        video_width: int = 256,
        video_height: int = 256,
        video_fps: int = 20,
        video_camera: str | None = None,
        flatten_obs: bool = True,
    ) -> None:
        """
        Args:
            env: A robomimic ``EnvRobosuite`` instance created with ``use_warp=True``.
            horizon: Maximum episode length (steps).
            device: Torch device string.
            clip_actions: If set, clamp actions to [-clip_actions, clip_actions].
            obs_keys: Explicit ordering of observation keys to concatenate into the
                ``"policy"`` group. When provided, this must match the ordering used
                by the upstream model (e.g. a BC-RNN warm-start checkpoint). If
                ``None``, keys from the first reset are sorted alphabetically.
            terminate_on_success: If True (default), envs whose ``is_success()["task"]``
                returns True are marked done immediately and their sim is reset to
                keyframe in-place (no re-randomisation of object placements; that
                waits until the next horizon full-reset).
            video_trigger: Optional ``Callable[[int], bool]`` invoked each step with
                the cumulative environment step count. When it returns True, the
                wrapper *arms* recording; capture then begins on the next step at
                which env 0 is at the start of a fresh episode (i.e. just after its
                previous episode ended, or the very first step after construction).
                Frames are flushed to ``video_dir`` as an mp4 when env 0 ends that
                episode. This guarantees each recorded clip starts at an episode
                boundary rather than mid-rollout. Requires the underlying robomimic
                env to have been created with ``render_offscreen=True``.
            video_dir: Directory to write mp4 files into. Defaults to the current
                working directory if ``video_trigger`` is provided without one.
            video_width, video_height, video_fps, video_camera: mp4 framing options.
                ``video_camera=None`` uses the env's first configured camera.
            flatten_obs: If True (default, RSL-RL mode), observations are
                concatenated into a single ``(num_envs, obs_dim)`` tensor under
                the ``"policy"`` key. If False (DPPO mode), observations are
                exposed per-key as a TensorDict of ``(num_envs, *)`` tensors so
                downstream code can apply per-key normalization before
                flattening itself.
        """
        self.env = env
        self.clip_actions = clip_actions
        self.terminate_on_success = terminate_on_success
        self.flatten_obs = flatten_obs

        # --- VecEnv required attributes ---
        self.num_envs: int = env.env.num_envs
        self.num_actions: int = env.action_dimension
        self.max_episode_length: int = horizon
        self.device: torch.device = torch.device(device)
        self.episode_length_buf: torch.Tensor = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # RSL-RL's wandb writer tries env_cfg.to_dict() then dataclasses.asdict();
        # a plain dict fails both -- wrap so the writer gets something usable.
        self.cfg = _EnvCfg(
            env_name=env.env.__class__.__name__,
            num_envs=self.num_envs,
            horizon=self.max_episode_length,
            clip_actions=clip_actions,
            terminate_on_success=terminate_on_success,
            obs_keys=list(obs_keys) if obs_keys is not None else None,
            flatten_obs=flatten_obs,
        )

        # --- Internal state ---
        self._obs_keys: list[str] | None = list(obs_keys) if obs_keys is not None else None
        self._cached_obs_td: TensorDict | None = None
        self._nan_total: int = 0
        self._nan_reported: bool = False
        # Episode-outcome counts accumulated since the last horizon log
        self._success_count: int = 0          # episodes in which success was achieved
        self._term_success_count: int = 0     # ended early due to success (and not diverged)
        self._term_nan_count: int = 0         # ended early due to divergence
        self._term_early_count: int = 0       # ended early via _check_early_termination (any cause)
        self._term_timeout_count: int = 0     # ended via horizon timeout
        # Counts can overlap across causes when multiple causes fire on one env.
        self._term_cause_counts: dict[str, int] = {}
        # Sticky set: cause keys emit 0 when not firing so dashboards don't drop series.
        self._early_term_causes_seen: set[str] = set()

        # None => no curriculum hook active; skip logging a scalar.
        self._current_difficulty: float | None = None

        # Populated lazily from _get_partial_task_metrics if the env exposes it.
        self._subtask_keys: list[str] | None = None
        self._subtask_ever: dict[str, torch.Tensor] = {}
        self._subtask_buf: dict[str, list[float]] = {}

        # Video recording state (gym-style step trigger; env 0 only).
        self._video_trigger = video_trigger
        self._video_dir = video_dir or os.getcwd()
        self._video_width = video_width
        self._video_height = video_height
        self._video_fps = video_fps
        self._video_camera = video_camera
        self._video_step_count: int = 0
        self._video_active: bool = False
        self._video_armed: bool = False
        self._render_mj_data = None
        self._render_camera_id: int | None = None
        # True when the next step will be env 0's first step of a fresh episode.
        # Initial reset in __init__ satisfies this for the very first step().
        self._video_at_episode_start: bool = True
        self._video_frames: list[np.ndarray] = []
        self._video_start_step: int = 0

        # Initial reset - also discovers observation keys
        self._full_reset()

    def get_observations(self) -> TensorDict:
        assert self._cached_obs_td is not None
        return self._cached_obs_td

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        # Arm on trigger, begin capture only at env-0 episode boundary.
        if (
            self._video_trigger is not None
            and not self._video_active
            and not self._video_armed
            and self._video_trigger(self._video_step_count)
        ):
            self._video_armed = True
        if (
            self._video_armed
            and not self._video_active
            and self._video_at_episode_start
        ):
            self._video_active = True
            self._video_armed = False
            self._video_start_step = self._video_step_count
            self._video_frames = []
            print(f"[INFO] Video: started recording env 0 at step={self._video_step_count} "
                  f"(episode start; will flush when env 0's episode ends)")
        if self._video_active:
            self._video_frames.append(self._render_env0())

        obs_dict, reward, _, info = self.env.step(actions)
        self.episode_length_buf += 1
        self._video_step_count += 1

        # f32 warp physics can diverge at scale under aggressive actions.
        diverged = self._handle_divergence(obs_dict, actions)

        succeeded = self._success_mask()
        early_causes = self._early_term_dict()
        early_term = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for mask in early_causes.values():
            early_term |= mask
        early_term &= ~diverged & ~succeeded
        if early_term.any():
            obs_dict = self._reset_envs(early_term, obs_dict)

        # Keyframe-only reset; object-placement randomisation only on horizon env.reset().
        if self.terminate_on_success:
            to_reset = succeeded & ~diverged  # diverged envs were already reset
            if to_reset.any():
                obs_dict = self._reset_envs(to_reset, obs_dict)

        # Reward -> (num_envs,) float tensor on device
        if not isinstance(reward, torch.Tensor):
            reward = torch.full((self.num_envs,), float(reward), dtype=torch.float32, device=self.device)
        reward = reward.float().to(self.device)
        if diverged.any() or early_term.any():
            reward = reward.clone()
            reward[diverged | early_term] = 0.0

        # Accumulate per-subtask "ever achieved" flags.
        self._update_subtask_ever()

        # Timeout check - all envs share the same horizon so they timeout together
        timeouts = self.episode_length_buf >= self.max_episode_length
        if self.terminate_on_success:
            early = succeeded | diverged | early_term
        else:
            early = diverged | early_term
        dones = (timeouts | early).long()

        # Finalize an in-progress video recording if env 0 just ended its episode.
        env0_done = bool(dones[0].item())
        if self._video_active and env0_done:
            self._finalize_video()
        self._video_at_episode_start = env0_done

        # Per-env episode accounting
        if early.any():
            # Each early-terminated env closed out an episode
            self.episode_length_buf[early] = 0
            succ_early = int((succeeded & ~diverged & ~early_term).sum().item())
            self._success_count += succ_early
            self._term_success_count += succ_early
            self._term_nan_count += int(diverged.sum().item())
            self._term_early_count += int(early_term.sum().item())
            # Only count envs where the cause actually drove termination (not pre-empted).
            for cause, mask in early_causes.items():
                effective = mask & ~diverged & ~succeeded
                if effective.any():
                    self._term_cause_counts[cause] = (
                        self._term_cause_counts.get(cause, 0) + int(effective.sum().item())
                    )
            self._snapshot_completed(early)

        extras: dict = {"time_outs": timeouts & ~early}

        # Reset only timed-out envs -- resetting all would defeat init_at_random_ep_len.
        timed_out = timeouts & ~early
        if timed_out.any():
            final_succ = self.env.is_success()["task"]
            if isinstance(final_succ, torch.Tensor):
                self._success_count += int((final_succ.to(self.device).bool() & timed_out).sum().item())
            else:
                self._success_count += int(bool(final_succ)) * int(timed_out.sum().item())
            n_timeout = int(timed_out.sum().item())
            self._term_timeout_count += n_timeout
            self._snapshot_completed(timed_out)
            obs_dict = self._reset_envs(timed_out, obs_dict)
            self.episode_length_buf[timed_out] = 0

        if early.any() or timed_out.any():
            n_done = max(
                self._term_success_count
                + self._term_nan_count
                + self._term_early_count
                + self._term_timeout_count,
                1,
            )
            extras["log"] = {
                "Metrics/success_rate": self._success_count / n_done,
            }
            for k, buf in self._subtask_buf.items():
                if buf:
                    extras["log"][f"Metrics/{k}_rate"] = float(np.mean(buf))
                    buf.clear()

            extras["log"]["Episode_Termination/success"] = self._term_success_count / n_done
            extras["log"]["Episode_Termination/time_out"] = self._term_timeout_count / n_done
            extras["log"]["Episode_Termination/nan_term"] = self._term_nan_count / n_done
            for cause in sorted(self._early_term_causes_seen):
                n = self._term_cause_counts.get(cause, 0)
                extras["log"][f"Episode_Termination/{cause}"] = n / n_done

            if self._current_difficulty is not None:
                extras["log"]["Curriculum/difficulty"] = self._current_difficulty

            self._success_count = 0
            self._term_success_count = 0
            self._term_nan_count = 0
            self._term_early_count = 0
            self._term_timeout_count = 0
            self._term_cause_counts = {}

        self._cached_obs_td = self._obs_dict_to_tensordict(obs_dict)
        # Guard OnPolicyRunner.check_nan against residual NaN from warp's global forward().
        for key, tensor in list(self._cached_obs_td.items()):
            if torch.is_floating_point(tensor) and torch.isnan(tensor).any():
                self._cached_obs_td[key] = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.isnan(reward).any():
            reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        return self._cached_obs_td, reward, dones, extras

    def reset(self) -> tuple[TensorDict, dict]:
        self._full_reset()
        self._video_at_episode_start = True
        return self._cached_obs_td, {}

    def seed(self, seed: int = -1) -> int:
        if seed >= 0:
            torch.manual_seed(seed)
            np.random.seed(seed)
        return seed

    def set_difficulty(self, difficulty: float) -> float:
        """Push a new curriculum difficulty into the underlying env.

        Returns the clipped value actually applied (so callers can log it).
        No-op on envs that don't implement ``set_difficulty``.
        """
        fn = getattr(self.env.env, "set_difficulty", None)
        d = float(np.clip(difficulty, 0.0, 1.0))
        if callable(fn):
            fn(d)
        self._current_difficulty = d
        return d

    def close(self) -> None:
        pass

    def _render_env0(self) -> np.ndarray:
        """Render env 0 via a CPU mj_forward pass on a reusable MjData.

        The straightforward ``sim.render(env_idx=0)`` path calls
        ``mjwarp.get_data_into(world_id=0)``, which syncs the entire
        ``(num_envs, ...)`` warp SoA to CPU even though only env 0 is
        needed -- ~220ms at 2048 envs. We avoid that by copying just env
        0's qpos/qvel to a pre-allocated CPU ``MjData``, running
        ``mj_forward`` to refresh xpos/xmat/site_xpos (everything the
        renderer reads), then pointing the offscreen context's data
        pointer at it for the actual ``mjr_render`` call.

        Returns a contiguous ``uint8`` RGB array (H, W, 3).
        """
        import mujoco
        import warp as wp
        sim = self.env.env.sim
        ctx = getattr(sim, "_render_context_offscreen", None)
        if ctx is None:
            raise RuntimeError(
                "MjSimWarp has no offscreen render context; was the env created "
                "with render_offscreen=True? (rl_mimicgen.rsl_rl.train_rl enables this only when "
                "--video is passed.)"
            )

        if self._render_mj_data is None:
            self._render_mj_data = mujoco.MjData(sim.model._model)
            self._render_camera_id = (
                sim.model.camera_name2id(self._video_camera)
                if self._video_camera is not None else None
            )

        # Copy env 0's state from GPU -> CPU (two small tensor slices).
        qpos0 = wp.to_torch(sim._warp_data.qpos)[0].detach().cpu().numpy()
        qvel0 = wp.to_torch(sim._warp_data.qvel)[0].detach().cpu().numpy()
        d = self._render_mj_data
        d.qpos[:] = qpos0
        d.qvel[:] = qvel0
        mujoco.mj_forward(sim.model._model, d)

        # Redirect the render context at our CPU MjData just for this frame.
        saved_data_ptr = ctx.data._data
        ctx.data._data = d
        try:
            ctx.render(
                width=self._video_width,
                height=self._video_height,
                camera_id=self._render_camera_id,
                segmentation=False,
            )
            frame = ctx.read_pixels(
                self._video_width, self._video_height, depth=False, segmentation=False
            )
        finally:
            ctx.data._data = saved_data_ptr

        # MuJoCo's offscreen read_pixels returns the image bottom-up; flip to
        # top-down (matches robomimic.envs.env_robosuite.EnvRobosuite.render).
        return np.ascontiguousarray(frame[::-1])

    def _finalize_video(self) -> None:
        """Write the currently-buffered frames to ``video_dir`` as an mp4."""
        frames = self._video_frames
        start_step = self._video_start_step
        self._video_active = False
        self._video_frames = []
        if not frames:
            return
        os.makedirs(self._video_dir, exist_ok=True)
        path = os.path.join(self._video_dir, f"rollout_step{start_step:09d}.mp4")
        try:
            import imageio.v2 as imageio
            with imageio.get_writer(path, fps=self._video_fps, codec="libx264", quality=6) as w:
                for f in frames:
                    w.append_data(f)
            print(f"[INFO] Video: wrote {len(frames)} frames -> {path}")
        except Exception as e:
            print(f"[WARN] video write failed ({path}): {e}")

    def _update_subtask_ever(self) -> None:
        """Pull partial-task metrics from the underlying env (if any) and OR them
        into the per-env "ever achieved during this episode" flags.

        Mimicgen's Coffee/ThreePieceAssembly expose _get_partial_task_metrics()
        returning a dict like {"task", "grasp", "insertion", "rim", ...}. Other
        tasks don't override it, so we skip silently.
        """
        fn = getattr(self.env.env, "_get_partial_task_metrics", None)
        if fn is None:
            return
        try:
            metrics = fn()
        except Exception:
            return
        if not isinstance(metrics, dict):
            return

        if self._subtask_keys is None:
            # First call: discover which keys produce per-env bool tensors.
            # Drop "task" -- already covered by /episode/success_rate.
            self._subtask_keys = []
            for k, v in metrics.items():
                if k == "task":
                    continue
                if isinstance(v, torch.Tensor) and v.shape[0] == self.num_envs:
                    self._subtask_keys.append(k)
                    self._subtask_ever[k] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                    self._subtask_buf[k] = []

        for k in self._subtask_keys:
            v = metrics.get(k)
            if isinstance(v, torch.Tensor) and v.shape[0] == self.num_envs:
                self._subtask_ever[k] |= v.to(self.device).bool()

    def _snapshot_completed(self, mask: torch.Tensor) -> None:
        """Push per-env episode stats into the rolling buffers, then reset those
        per-env accumulators. Call once per completion event (early or timeout).
        """
        if not mask.any():
            return
        for k, ever in self._subtask_ever.items():
            self._subtask_buf[k].extend(ever[mask].float().cpu().tolist())
            ever[mask] = False

    def _full_reset(self) -> None:
        obs_dict = self.env.reset()
        if self._obs_keys is None:
            self._obs_keys = sorted(obs_dict.keys())
        else:
            missing = [k for k in self._obs_keys if k not in obs_dict]
            if missing:
                raise KeyError(
                    f"obs_keys {missing} not present in env observations "
                    f"(available: {sorted(obs_dict.keys())})"
                )
        self.episode_length_buf[:] = 0
        for ever in self._subtask_ever.values():
            ever[:] = False
        self._cached_obs_td = self._obs_dict_to_tensordict(obs_dict)

    def _find_diverged_envs(self, obs_dict: dict) -> torch.Tensor:
        """Return a (num_envs,) bool mask of envs with NaN state.

        For warp sims, qpos is the single source of divergence -- if physics
        NaNs out it's always in qpos first, and every observation keyed off
        qpos inherits the NaN on the next kinematics pass. So one
        ``isnan(qpos).any(dim=1)`` replaces ~15 per-key isnan kernels.

        Falls back to the per-obs-key scan when we can't reach ``qpos``
        (non-warp sims or before the inner env exposes it).
        """
        try:
            import warp as wp
            qpos_t = wp.to_torch(self.env.env.sim._warp_data.qpos)
            if qpos_t.shape[0] == self.num_envs:
                return torch.isnan(qpos_t).any(dim=1)
        except Exception:
            pass
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for k in self._obs_keys:
            v = obs_dict.get(k)
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                t = v
            else:
                try:
                    t = torch.as_tensor(np.asarray(v), device=self.device)
                except Exception:
                    continue
            if not torch.is_floating_point(t) or t.shape[0] != self.num_envs:
                continue
            m = torch.isnan(t)
            if t.dim() > 1:
                m = m.any(dim=tuple(range(1, t.dim())))
            mask |= m
        return mask

    def _handle_divergence(self, obs_dict: dict, actions: torch.Tensor) -> torch.Tensor:
        """Detect per-env physics divergence, reset offending envs in-place, and
        patch ``obs_dict`` with fresh observations. Returns the diverged bool mask.

        On the first occurrence, prints a detailed diagnostic (env idx, offending
        action, qpos/qvel state). Subsequent occurrences only increment counters.
        """
        diverged = self._find_diverged_envs(obs_dict)
        if not diverged.any():
            return diverged

        n_div = int(diverged.sum().item())
        diverged_idxs = diverged.nonzero().flatten().cpu().tolist()
        self._nan_total += n_div

        if not self._nan_reported:
            self._nan_reported = True
            first_env = diverged_idxs[0]
            step_at = int(self.episode_length_buf[first_env].item())
            print("\n" + "=" * 72)
            print(f"[NaN RECOVERY] first divergence at episode step={step_at}, "
                  f"env_idx={first_env}  ({n_div}/{self.num_envs} envs this step)")
            try:
                sim = self.env.env.sim
                qpos = sim._warp_data.qpos.numpy()
                qpos_nan = np.isnan(qpos).any(axis=1)
                print(f"  qpos NaN in {int(qpos_nan.sum())}/{self.num_envs} envs "
                      f"-> physics divergence (not an obs-extraction bug)")
            except Exception as e:
                print(f"  [could not read sim state: {e}]")
            a = actions[first_env].detach().cpu().tolist() if isinstance(actions, torch.Tensor) else list(actions[first_env])
            print(f"  last action[env={first_env}] = {[f'{x:+.3f}' for x in a]}")
            print(f"  -> resetting these envs in-place, zero reward, marking done.")
            print(f"  (subsequent recoveries silent; cumulative count logged per "
                  f"episode-boundary under /episode/nan_resets)")
            print("=" * 72 + "\n", flush=True)

        self._reset_envs(diverged, obs_dict)
        return diverged

    def _success_mask(self) -> torch.Tensor:
        """Return a (num_envs,) bool tensor marking envs whose ``is_success()["task"]``
        is True. Tolerates tensor/scalar/bool return types."""
        succ = self.env.is_success().get("task", None)
        if succ is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if isinstance(succ, torch.Tensor):
            return succ.to(self.device).bool().view(self.num_envs)
        arr = np.asarray(succ)
        if arr.shape == ():
            return torch.full((self.num_envs,), bool(arr), dtype=torch.bool, device=self.device)
        return torch.as_tensor(arr.astype(bool), device=self.device)

    def _early_term_dict(self) -> dict[str, torch.Tensor]:
        """Collect per-cause early-termination masks from the underlying env.

        The env's ``_check_early_termination()`` returns a ``dict`` keyed by
        cause name (e.g. ``"fell_off_coffee_pod"``); each value is a per-env
        bool signal (scalar for single-env sims, ``(num_envs,)`` tensor for
        warp). Scalars are broadcast to the full batch. Envs that don't
        implement the hook yield ``{}``.
        """
        fn = getattr(self.env.env, "_check_early_termination", None)
        if fn is None:
            return {}
        result = fn()
        if not isinstance(result, dict):
            return {}
        out: dict[str, torch.Tensor] = {}
        for cause, val in result.items():
            if isinstance(val, torch.Tensor):
                out[cause] = val.to(self.device).bool().view(self.num_envs)
                continue
            arr = np.asarray(val)
            if arr.shape == ():
                out[cause] = torch.full(
                    (self.num_envs,), bool(arr), dtype=torch.bool, device=self.device
                )
            else:
                out[cause] = torch.as_tensor(arr.astype(bool), device=self.device)
        self._early_term_causes_seen.update(out.keys())
        return out

    def _reset_envs(self, env_indices: torch.Tensor | list[int], obs_dict: dict | None = None) -> dict:
        """Per-env reset to a fresh random initial state (keyframe qpos/qvel +
        re-sampled object placements + task-specific reset hooks such as
        Coffee's hinge qpos).

        Slim masked path: only the envs in ``env_indices`` are touched. Kept
        envs' sim state flows through unchanged (no snapshot/restore needed)
        because every write is scoped to masked rows:

        - ``sim.reset(env_indices=idxs)`` -> ``mjwarp.reset_data(reset=mask)``
          keyframes only masked envs' qpos/qvel/time.
        - Robot arm + gripper init qpos are written into masked rows of
          ``qpos`` via an indexed torch assign.
        - ``_reset_internal()`` is called with ``_reset_env_mask`` set and
          ``_skip_robot_reset=True`` so placement sampling + hinge qpos only
          touch masked rows and the redundant full-batch ``robot.reset()``
          (which would otherwise wipe kept envs' arm qpos + rebuild the
          controller) is skipped.
        - ``_get_observations(force_update=True)`` is called once. The full
          ``self.env.reset()`` path would call it *twice* (once inside
          ``MujocoEnv.reset`` and once after our restore).

        Non-Coffee tasks whose ``_reset_internal`` doesn't consult
        ``_reset_env_mask`` still get masked robot qpos and masked keyframe
        reset, but their object placements are resampled full-batch -- which
        is a correctness-neutral waste, not a bug (unmasked rows end up
        identical to their keyframe positions).

        Args:
            env_indices: Bool ``(num_envs,)`` mask or iterable of env idxs
                to reset.
            obs_dict: If provided, overwrite the reset envs' slices in this
                dict with fresh observations and return it. If None, return
                a fresh full obs dict from the env.

        Returns:
            The obs dict (modified in place if one was passed).
        """
        rsuite_inner = self.env.env  # robosuite env (e.g. Coffee_D0)
        sim = rsuite_inner.sim

        # Normalise env_indices to a bool mask + list[int].
        if isinstance(env_indices, torch.Tensor) and env_indices.dtype == torch.bool:
            reset_mask = env_indices.to(self.device)
            idxs = reset_mask.nonzero().flatten().cpu().tolist()
        else:
            idxs = list(env_indices) if not isinstance(env_indices, torch.Tensor) else env_indices.cpu().tolist()
            reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            if idxs:
                reset_mask[torch.tensor(idxs, device=self.device)] = True
        if not idxs:
            return obs_dict if obs_dict is not None else self.env.get_observation()

        sim.reset(env_indices=idxs)

        self._apply_robot_init_qpos_masked(rsuite_inner, sim, idxs)

        # _skip_robot_reset bypasses the full-batch robot.reset() loop (arm qpos
        # written above, controller state persists -- stateful controllers would
        # leak state across episodes, so drop this if that changes).
        rsuite_inner._reset_env_mask = reset_mask
        rsuite_inner._skip_robot_reset = True
        try:
            rsuite_inner._reset_internal()
        finally:
            rsuite_inner._reset_env_mask = None
            rsuite_inner._skip_robot_reset = False

        # Kept envs keep prior obs (sim state didn't move); wrapper applies
        # object-state rename + robot-key promotion + CUDA coercion.
        raw = rsuite_inner._get_observations(force_update=True)
        fresh = self.env.get_observation(di=raw)
        if obs_dict is None:
            return fresh
        for k in list(obs_dict.keys()):
            if k in fresh:
                obs_dict[k] = fresh[k]
        return obs_dict

    def _apply_robot_init_qpos_masked(self, rsuite_inner: Any, sim: Any, idxs: list[int]) -> None:
        """Write arm + gripper init qpos into masked rows of ``sim.data.qpos``.

        Caches the init vectors + joint indexes on first call so subsequent
        calls skip the per-robot Python setup. Noise is applied per-call (not
        per-env) to match ``robot.reset()``'s broadcast behaviour.
        """
        import warp as wp
        qpos_t = wp.to_torch(sim._warp_data.qpos)  # (num_envs, nq)
        cache = getattr(self, "_robot_qpos_cache", None)
        if cache is None:
            cache = []
            for robot in rsuite_inner.robots:
                entry = {
                    "arm_idxs": torch.as_tensor(
                        robot._ref_joint_pos_indexes, dtype=torch.long, device=qpos_t.device
                    ),
                    "arm_init": torch.as_tensor(
                        np.asarray(robot.init_qpos, dtype=np.float32), device=qpos_t.device
                    ),
                    "noise_type": robot.initialization_noise.get("type", "gaussian"),
                    "noise_mag": float(robot.initialization_noise.get("magnitude", 0.0) or 0.0),
                }
                grip_idxs = getattr(robot, "_ref_gripper_joint_pos_indexes", None)
                if grip_idxs is not None and robot.has_gripper and robot.gripper is not None:
                    entry["grip_idxs"] = torch.as_tensor(
                        grip_idxs, dtype=torch.long, device=qpos_t.device
                    )
                    entry["grip_init"] = torch.as_tensor(
                        np.asarray(robot.gripper.init_qpos, dtype=np.float32),
                        device=qpos_t.device,
                    )
                cache.append(entry)
            self._robot_qpos_cache = cache

        row_idxs = torch.as_tensor(idxs, dtype=torch.long, device=qpos_t.device)
        for entry in cache:
            arm = entry["arm_init"]
            if entry["noise_mag"] > 0.0:
                if entry["noise_type"] == "uniform":
                    noise = (torch.rand_like(arm) * 2.0 - 1.0) * entry["noise_mag"]
                else:
                    noise = torch.randn_like(arm) * entry["noise_mag"]
                arm = arm + noise
            qpos_t[row_idxs.unsqueeze(1), entry["arm_idxs"].unsqueeze(0)] = arm
            if "grip_idxs" in entry:
                qpos_t[row_idxs.unsqueeze(1), entry["grip_idxs"].unsqueeze(0)] = entry["grip_init"]

    def _obs_dict_to_tensordict(self, obs_dict: dict) -> TensorDict:
        """Materialise observations as a TensorDict on ``self.device``.

        When ``flatten_obs`` is True (RSL-RL mode), all keys in
        ``self._obs_keys`` are concatenated along the feature dim and stored
        under the ``"policy"`` key. When False (DPPO mode), each observation
        key becomes its own entry in the TensorDict, preserving original
        per-key shapes -- downstream consumers handle flattening themselves so
        per-key normalization stays meaningful.
        """
        per_key: dict[str, torch.Tensor] = {}
        for k in self._obs_keys:
            v = obs_dict[k]
            if not isinstance(v, torch.Tensor):
                v = torch.as_tensor(np.asarray(v, dtype=np.float32), device=self.device)
            v = v.float().to(self.device)
            if v.dim() == 1:
                v = v.unsqueeze(-1)
            per_key[k] = v
        if self.flatten_obs:
            flat = torch.cat([per_key[k] for k in self._obs_keys], dim=-1)
            return TensorDict({"policy": flat}, batch_size=[self.num_envs], device=self.device)
        return TensorDict(per_key, batch_size=[self.num_envs], device=self.device)
