"""VecEnv wrapper adapting a robomimic warp environment for RSL-RL's OnPolicyRunner.

All parallel warp envs share a fixed horizon. Individual envs can terminate
early on task success or physics divergence (NaN), in which case their warp
sim state is reset to the model keyframe (object placements are NOT
re-randomised until the next full horizon reset). At the horizon boundary,
``env.reset()`` is called which re-randomises every env's object placement.
"""

from __future__ import annotations

import torch
import numpy as np
from tensordict import TensorDict

from rsl_rl.env import VecEnv


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
        self.cfg: dict = {}

        # --- Internal state ---
        self._obs_keys: list[str] | None = list(obs_keys) if obs_keys is not None else None
        self._cached_obs_td: TensorDict | None = None
        self._nan_reset_count: int = 0
        self._nan_total: int = 0
        self._nan_reported: bool = False
        self._success_count: int = 0  # successful early-terminations since last log
        self._completion_count: int = 0  # all episode completions since last log

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

        obs_dict, reward, _, info = self.env.step(actions)
        self.episode_length_buf += 1

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
                obs_dict = self._reset_envs_inplace(to_reset, obs_dict)

        # Reward -> (num_envs,) float tensor on device
        if not isinstance(reward, torch.Tensor):
            reward = torch.full((self.num_envs,), float(reward), dtype=torch.float32, device=self.device)
        reward = reward.float().to(self.device)
        if diverged.any():
            reward = reward.clone()
            reward[diverged] = 0.0

        # Timeout check – all envs share the same horizon so they timeout together
        timeouts = self.episode_length_buf >= self.max_episode_length
        if self.terminate_on_success:
            early = succeeded | diverged
        else:
            early = diverged
        dones = (timeouts | early).long()

        # Per-env episode accounting
        if early.any():
            # Each early-terminated env closed out an episode
            self.episode_length_buf[early] = 0
            self._completion_count += int(early.sum().item())
            self._success_count += int((succeeded & ~diverged).sum().item())

        extras: dict = {"time_outs": timeouts & ~early}

        # Log episode stats on horizon boundary (all remaining envs roll over).
        if timeouts.any():
            # Envs that timed out without early-terminating also close an episode.
            remaining = timeouts & ~early
            if remaining.any():
                final_succ = self.env.is_success()["task"]
                if isinstance(final_succ, torch.Tensor):
                    self._success_count += int((final_succ.to(self.device).bool() & remaining).sum().item())
                else:
                    self._success_count += int(bool(final_succ)) * int(remaining.sum().item())
                self._completion_count += int(remaining.sum().item())

            if self._completion_count > 0:
                sr = self._success_count / self._completion_count
            else:
                sr = 0.0
            extras["log"] = {
                "/episode/success_rate": sr,
                "/episode/completions": float(self._completion_count),
            }
            if self._nan_reset_count > 0:
                extras["log"]["/episode/nan_resets"] = float(self._nan_reset_count)
                self._nan_reset_count = 0
            self._success_count = 0
            self._completion_count = 0

            # Full reset (re-randomises object placements for all envs)
            obs_dict = self.env.reset()
            self.episode_length_buf[:] = 0

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

        self._reset_envs_inplace(diverged, obs_dict)
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

    def _reset_envs_inplace(self, mask: torch.Tensor, obs_dict: dict) -> dict:
        """Reset the warp sim state for the envs selected by ``mask`` (to keyframe
        qpos/qvel; object placements are unchanged), then overwrite their slices
        in ``obs_dict`` with fresh observations. Modifies ``obs_dict`` in place
        and returns it.
        """
        idxs = mask.nonzero().flatten().cpu().tolist()
        sim = self.env.env.sim
        sim.reset(env_indices=idxs)
        sim.forward()
        fresh = self.env.get_observation()
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
