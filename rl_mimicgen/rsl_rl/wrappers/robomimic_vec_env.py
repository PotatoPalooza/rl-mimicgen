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
from typing import Callable

import torch
import numpy as np
from tensordict import TensorDict

from rsl_rl.env import VecEnv


class _EnvCfg:
    """Small wrapper exposing a ``to_dict()`` method for RSL-RL's wandb writer.

    ``WandbSummaryWriter.store_config`` first tries ``env_cfg.to_dict()`` and
    falls back to ``dataclasses.asdict(env_cfg)`` — a plain dict fails both.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class RobomimicVecEnv(VecEnv):
    """Wraps a robomimic warp environment for use with RSL-RL."""

    def __init__(
        self,
        env,
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
    ):
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
                the cumulative environment step count. When it returns True and no
                recording is in progress, frames from env 0 start being captured and
                written to ``video_dir`` as an mp4 when env 0 ends its episode
                (early termination or horizon). Mirrors ``gym.wrappers.RecordVideo``'s
                ``step_trigger`` pattern. Requires the underlying robomimic env to
                have been created with ``render_offscreen=True``.
            video_dir: Directory to write mp4 files into. Defaults to the current
                working directory if ``video_trigger`` is provided without one.
            video_width, video_height, video_fps, video_camera: mp4 framing options.
                ``video_camera=None`` uses the env's first configured camera.
        """
        self.env = env
        self.clip_actions = clip_actions
        self.terminate_on_success = terminate_on_success

        # --- VecEnv required attributes ---
        self.num_envs: int = env.env.num_envs
        self.num_actions: int = env.action_dimension
        self.max_episode_length: int = horizon
        self.device: torch.device = torch.device(device)
        self.episode_length_buf: torch.Tensor = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # RSL-RL's wandb writer calls env_cfg.to_dict() before falling back to
        # dataclasses.asdict(); a plain dict fails both branches. Use a tiny
        # wrapper so the writer gets something usable.
        self.cfg = _EnvCfg(
            env_name=env.env.__class__.__name__,
            num_envs=self.num_envs,
            horizon=self.max_episode_length,
            clip_actions=clip_actions,
            terminate_on_success=terminate_on_success,
            obs_keys=list(obs_keys) if obs_keys is not None else None,
        )

        # --- Internal state ---
        self._obs_keys: list[str] | None = list(obs_keys) if obs_keys is not None else None
        self._cached_obs_td: TensorDict | None = None
        self._nan_reset_count: int = 0
        self._nan_total: int = 0
        self._nan_reported: bool = False
        # Episode-outcome counts accumulated since the last horizon log
        self._success_count: int = 0      # episodes in which success was achieved
        self._completion_count: int = 0   # total episodes that ended
        self._term_success_count: int = 0 # ended early due to success (and not diverged)
        self._term_nan_count: int = 0     # ended early due to divergence
        self._term_timeout_count: int = 0 # ended via horizon timeout

        # Per-episode reward accumulators (for /episode/return_mean)
        self._cur_return: torch.Tensor = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._return_buf: list[float] = []

        # Per-episode subtask tracking ("ever achieved during this episode").
        # Populated lazily on the first step where the underlying env exposes
        # _get_partial_task_metrics (Coffee, ThreePieceAssembly). Other tasks
        # just have a single "task" key already covered by success_rate.
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
        self._video_frames: list[np.ndarray] = []
        self._video_start_step: int = 0

        # Initial reset – also discovers observation keys
        self._full_reset()

    # ------------------------------------------------------------------
    # VecEnv interface
    # ------------------------------------------------------------------

    def get_observations(self) -> TensorDict:
        assert self._cached_obs_td is not None
        return self._cached_obs_td

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        # Check video trigger BEFORE stepping, so a fresh recording captures the
        # initial-state frame of env 0. Only fires when not already recording.
        if (
            self._video_trigger is not None
            and not self._video_active
            and self._video_trigger(self._video_step_count)
        ):
            self._video_active = True
            self._video_start_step = self._video_step_count
            self._video_frames = []
            print(f"[INFO] Video: started recording env 0 at step={self._video_step_count} "
                  f"(will flush when env 0's episode ends)")
        if self._video_active:
            self._video_frames.append(self._render_env0())

        obs_dict, reward, _, info = self.env.step(actions)
        self.episode_length_buf += 1
        self._video_step_count += 1

        # Detect per-env physics divergence (NaN qpos from solver non-convergence),
        # reset just those envs, and mark them done. Rare at small nworld but
        # unavoidable at large scale with f32 warp physics + aggressive actions.
        diverged = self._handle_divergence(obs_dict, actions)

        # Early termination on task success (does a keyframe-only reset of the
        # succeeded envs; object-placement randomisation only happens at the
        # horizon boundary via the full env.reset() below).
        succeeded = self._success_mask()
        if self.terminate_on_success:
            to_reset = succeeded & ~diverged  # diverged envs were already reset
            if to_reset.any():
                obs_dict = self._reset_envs(to_reset, obs_dict)

        # Reward -> (num_envs,) float tensor on device
        if not isinstance(reward, torch.Tensor):
            reward = torch.full((self.num_envs,), float(reward), dtype=torch.float32, device=self.device)
        reward = reward.float().to(self.device)
        if diverged.any():
            reward = reward.clone()
            reward[diverged] = 0.0

        # Accumulate undiscounted per-env return + per-subtask "ever achieved" flags.
        self._cur_return += reward
        self._update_subtask_ever()

        # Timeout check – all envs share the same horizon so they timeout together
        timeouts = self.episode_length_buf >= self.max_episode_length
        if self.terminate_on_success:
            early = succeeded | diverged
        else:
            early = diverged
        dones = (timeouts | early).long()

        # Finalize an in-progress video recording if env 0 just ended its episode.
        if self._video_active and bool(dones[0].item()):
            self._finalize_video()

        # Per-env episode accounting
        if early.any():
            # Each early-terminated env closed out an episode
            self.episode_length_buf[early] = 0
            self._completion_count += int(early.sum().item())
            succ_early = int((succeeded & ~diverged).sum().item())
            self._success_count += succ_early
            self._term_success_count += succ_early
            self._term_nan_count += int(diverged.sum().item())
            self._snapshot_completed(early)

        extras: dict = {"time_outs": timeouts & ~early}

        # Per-env timeout handling: reset only the envs that timed out (not the
        # ones that already early-terminated this step). This keeps episode
        # phases desynchronised across envs — otherwise any env hitting horizon
        # would drag the whole batch back to buf=0, defeating
        # init_at_random_ep_len.
        timed_out = timeouts & ~early
        if timed_out.any():
            final_succ = self.env.is_success()["task"]
            if isinstance(final_succ, torch.Tensor):
                self._success_count += int((final_succ.to(self.device).bool() & timed_out).sum().item())
            else:
                self._success_count += int(bool(final_succ)) * int(timed_out.sum().item())
            n_timeout = int(timed_out.sum().item())
            self._completion_count += n_timeout
            self._term_timeout_count += n_timeout
            self._snapshot_completed(timed_out)
            obs_dict = self._reset_envs(timed_out, obs_dict)
            self.episode_length_buf[timed_out] = 0

        # Flush the rolling episode-stats buffers any time at least one env
        # just closed an episode (early-term or timeout). With per-env resets
        # this fires often; RSL-RL's Logger averages across steps per iter.
        if early.any() or timed_out.any():
            n_done = max(self._completion_count, 1)
            extras["log"] = {
                "Episode_Reward/success_rate": self._success_count / n_done,
            }
            if self._return_buf:
                extras["log"]["Episode_Reward/return_mean"] = float(np.mean(self._return_buf))
                self._return_buf.clear()
            for k, buf in self._subtask_buf.items():
                if buf:
                    extras["log"][f"Episode_Reward/{k}_rate"] = float(np.mean(buf))
                    buf.clear()

            extras["log"]["Episode_Termination/success"] = self._term_success_count / n_done
            extras["log"]["Episode_Termination/time_out"] = self._term_timeout_count / n_done
            extras["log"]["Episode_Termination/nan_term"] = self._term_nan_count / n_done
            extras["log"]["Episode_Termination/completions"] = float(self._completion_count)
            if self._nan_reset_count > 0:
                extras["log"]["Episode_Termination/nan_resets"] = float(self._nan_reset_count)
                self._nan_reset_count = 0

            self._success_count = 0
            self._completion_count = 0
            self._term_success_count = 0
            self._term_nan_count = 0
            self._term_timeout_count = 0

        self._cached_obs_td = self._obs_dict_to_tensordict(obs_dict)
        return self._cached_obs_td, reward, dones, extras

    def reset(self) -> tuple[TensorDict, dict]:
        self._full_reset()
        return self._cached_obs_td, {}

    def seed(self, seed: int = -1) -> int:
        if seed >= 0:
            torch.manual_seed(seed)
            np.random.seed(seed)
        return seed

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _render_env0(self) -> np.ndarray:
        """Render env 0 from the warp sim at the configured framing. Returns a
        contiguous ``uint8`` RGB array (H, W, 3)."""
        sim = self.env.env.sim
        if getattr(sim, "_render_context_offscreen", None) is None:
            raise RuntimeError(
                "MjSimWarp has no offscreen render context; was the env created "
                "with render_offscreen=True? (train_rl.py enables this only when "
                "--video is passed.)"
            )
        frame = sim.render(
            width=self._video_width,
            height=self._video_height,
            camera_name=self._video_camera,
            env_idx=0,
        )
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
            print(f"[INFO] Video: wrote {len(frames)} frames → {path}")
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
            # Drop "task" — already covered by /episode/success_rate.
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
        self._return_buf.extend(self._cur_return[mask].cpu().tolist())
        self._cur_return[mask] = 0.0
        for k, ever in self._subtask_ever.items():
            self._subtask_buf[k].extend(ever[mask].float().cpu().tolist())
            ever[mask] = False

    def _full_reset(self):
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
        self._cur_return[:] = 0.0
        for ever in self._subtask_ever.values():
            ever[:] = False
        self._cached_obs_td = self._obs_dict_to_tensordict(obs_dict)

    def _find_diverged_envs(self, obs_dict: dict) -> torch.Tensor:
        """Return a (num_envs,) bool mask of envs with NaN in any policy obs key.

        Only the keys in ``self._obs_keys`` are checked — these are guaranteed
        to be per-env tensors. Other dict entries (e.g. robosuite's concatenated
        modality groups like ``"object-state"``) are not necessarily batched.
        """
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
        self._nan_reset_count += n_div
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
                      f"→ physics divergence (not an obs-extraction bug)")
            except Exception as e:
                print(f"  [could not read sim state: {e}]")
            a = actions[first_env].detach().cpu().tolist() if isinstance(actions, torch.Tensor) else list(actions[first_env])
            print(f"  last action[env={first_env}] = {[f'{x:+.3f}' for x in a]}")
            print(f"  → resetting these envs in-place, zero reward, marking done.")
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

    def _reset_envs(self, env_indices, obs_dict: dict | None = None) -> dict:
        """Per-env reset to a fresh random initial state (keyframe qpos/qvel +
        re-sampled object placements + task-specific reset hooks such as
        Coffee's hinge qpos, Kitchen's button qpos, etc.).

        Strategy: snapshot the warp sim state (qpos, qvel, time) for envs we
        want to *keep*, run the underlying robosuite env's soft reset (which
        re-randomises placements for ALL envs and re-runs each task's
        ``_reset_internal`` hooks), then restore the saved state for the
        kept envs. That leaves the envs in ``env_indices`` with fresh
        randomly-sampled initial states and every other env untouched —
        all without modifying the 7+ mimicgen task files to support
        per-env placement sampling.

        Args:
            env_indices: Bool ``(num_envs,)`` mask or iterable of env idxs
                to reset.
            obs_dict: If provided, overwrite the reset envs' slices in this
                dict with fresh observations and return it. If None, return
                a fresh full obs dict from the env.

        Returns:
            The obs dict (modified in place if one was passed).
        """
        import warp as wp
        rsuite_inner = self.env.env  # robosuite env (e.g. Coffee_D0)
        sim = rsuite_inner.sim

        # Normalise env_indices to a bool mask + list[int]
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
        keep_mask = ~reset_mask

        # Snapshot state for envs we're keeping. Warp tensors live on GPU
        # (same device as self.device when num_envs > 0); move mask to match.
        qpos = wp.to_torch(sim._warp_data.qpos)
        qvel = wp.to_torch(sim._warp_data.qvel)
        time = wp.to_torch(sim._warp_data.time)
        keep_on_sim = keep_mask.to(qpos.device)
        saved_qpos = qpos[keep_on_sim].clone()
        saved_qvel = qvel[keep_on_sim].clone()
        saved_time = time[keep_on_sim].clone()

        # Force the soft-reset path (avoid hard_reset which reloads the sim
        # and reallocates GPU buffers — prohibitively expensive if called
        # once per step). Soft reset runs each task's _reset_internal which
        # re-samples num_envs placements and writes them to all envs. We call
        # through the robomimic wrapper so ep_meta/obs-key mapping stay in sync.
        prev_hard = rsuite_inner.hard_reset
        rsuite_inner.hard_reset = False
        try:
            self.env.reset()
        finally:
            rsuite_inner.hard_reset = prev_hard

        # Restore the kept envs' sim state. The envs in reset_mask keep the
        # freshly-sampled state from the reset.
        qpos[keep_on_sim] = saved_qpos
        qvel[keep_on_sim] = saved_qvel
        time[keep_on_sim] = saved_time
        sim.forward()

        fresh = self.env.get_observation()
        if obs_dict is None:
            return fresh
        for k in list(obs_dict.keys()):
            if k in fresh:
                obs_dict[k] = fresh[k]
        return obs_dict

    def _obs_dict_to_tensordict(self, obs_dict: dict) -> TensorDict:
        """Flatten all observation keys into a single ``(num_envs, obs_dim)`` tensor
        stored under the ``"policy"`` key of a TensorDict."""
        tensors = []
        for k in self._obs_keys:
            v = obs_dict[k]
            if not isinstance(v, torch.Tensor):
                v = torch.as_tensor(np.asarray(v, dtype=np.float32), device=self.device)
            v = v.float().to(self.device)
            if v.dim() == 1:
                v = v.unsqueeze(-1)
            tensors.append(v)
        flat = torch.cat(tensors, dim=-1)  # (num_envs, obs_dim)
        return TensorDict({"policy": flat}, batch_size=[self.num_envs], device=self.device)
