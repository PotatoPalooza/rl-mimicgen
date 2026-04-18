from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from rl_mimicgen.dppo.config.schema import DPPORunConfig
from rl_mimicgen.dppo.data import DPPODatasetBundle
from rl_mimicgen.dppo.model import CriticObs, DiffusionModel


@dataclass(slots=True)
class DPPORolloutSample:
    normalized_actions: np.ndarray
    actions: np.ndarray
    history: np.ndarray


class DiffusionPolicyAdapter(nn.Module):
    def __init__(
        self,
        config: DPPORunConfig,
        bundle: DPPODatasetBundle,
        checkpoint_path: str,
        deterministic: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.bundle = bundle
        self.device = torch.device(config.device)
        self.default_deterministic = deterministic

        self.actor = DiffusionModel(
            obs_dim=bundle.obs_dim,
            action_dim=bundle.action_dim,
            horizon_steps=config.diffusion.horizon_steps,
            cond_steps=config.diffusion.cond_steps,
            denoising_steps=config.diffusion.denoising_steps,
            predict_epsilon=config.diffusion.predict_epsilon,
            denoised_clip_value=config.diffusion.denoised_clip_value,
            time_dim=config.diffusion.time_dim,
            mlp_dims=config.diffusion.mlp_dims,
            residual_style=config.diffusion.residual_style,
            device=config.device,
        ).to(self.device)
        self.ema_actor = deepcopy(self.actor)
        self.value_net = CriticObs(cond_dim=bundle.obs_dim * config.diffusion.cond_steps).to(self.device)

        self.observation_horizon = int(config.diffusion.cond_steps)
        self.prediction_horizon = int(config.diffusion.horizon_steps)
        self.act_steps = int(config.diffusion.act_steps)
        self.min_logprob_denoising_std = 1e-3

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        actor_state, ema_state, critic_state = self._extract_checkpoint_state(checkpoint)
        self._load_actor_state_dict(actor_state)
        if ema_state is not None:
            self.ema_actor.load_state_dict(ema_state, strict=False)
        else:
            self.ema_actor.load_state_dict(self.actor.state_dict(), strict=False)
        if critic_state is not None:
            self.value_net.load_state_dict(critic_state, strict=False)
        self.actor.eval()
        self.value_net.train()

        self._rollout_obs_history: torch.Tensor | None = None
        self._train_action_queues: list[deque[torch.Tensor]] | None = None
        self._eval_action_queues: list[deque[torch.Tensor]] | None = None

    def _load_actor_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        missing_keys, unexpected_keys = self.actor.load_state_dict(state_dict, strict=False)
        allowed_missing = {
            "ddpm_logvar_clipped",
            "ddpm_mu_coef1",
            "ddpm_mu_coef2",
        }
        unexpected = set(unexpected_keys)
        missing = set(missing_keys)
        if unexpected:
            raise RuntimeError(f"Unexpected checkpoint keys: {sorted(unexpected)}")
        disallowed_missing = missing - allowed_missing
        if disallowed_missing:
            raise RuntimeError(f"Missing required checkpoint keys: {sorted(disallowed_missing)}")

    def _extract_checkpoint_state(
        self,
        checkpoint: dict[str, Any],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None, dict[str, torch.Tensor] | None]:
        if "actor" in checkpoint:
            actor_state = checkpoint["actor"]
            ema_state = checkpoint.get("ema_actor")
            critic_state = checkpoint.get("critic")
            return actor_state, ema_state, critic_state
        if "ema" in checkpoint or "model" in checkpoint:
            actor_state = checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"]
            return actor_state, None, None
        raise RuntimeError(f"Unsupported checkpoint format with keys: {sorted(checkpoint.keys())}")

    def actor_parameters(self) -> list[nn.Parameter]:
        return list(self.actor.parameters())

    def step_ema(self) -> None:
        decay = float(self.config.training.ema_decay)
        with torch.no_grad():
            for ema_param, param in zip(self.ema_actor.parameters(), self.actor.parameters(), strict=True):
                ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)

    def _flatten_obs_batch(self, obs: dict[str, np.ndarray | torch.Tensor]) -> np.ndarray:
        chunks = []
        batch_size = None
        for key in self.bundle.metadata["obs_keys"]:
            value = obs[key]
            if torch.is_tensor(value):
                value_np = value.detach().cpu().numpy()
            else:
                value_np = np.asarray(value)
            if value_np.ndim == 1:
                value_np = value_np[None, ...]
            batch_size = value_np.shape[0] if batch_size is None else batch_size
            chunks.append(value_np.reshape(batch_size, -1).astype(np.float32, copy=False))
        return np.concatenate(chunks, axis=1)

    def prepare_observation_batch(self, obs: dict[str, np.ndarray | torch.Tensor]) -> torch.Tensor:
        flat_obs = self._flatten_obs_batch(obs)
        normalized_obs = self.bundle.normalization.normalize_obs(flat_obs).astype(np.float32, copy=False)
        return torch.as_tensor(normalized_obs, dtype=torch.float32, device=self.device)

    def _advance_history(
        self,
        history: torch.Tensor | None,
        obs_batch: torch.Tensor,
        episode_starts: np.ndarray,
    ) -> torch.Tensor:
        batch_size = obs_batch.shape[0]
        reset_mask = torch.as_tensor(episode_starts, device=self.device, dtype=torch.bool)
        if history is None or history.shape[0] != batch_size:
            history = obs_batch.unsqueeze(1).repeat(1, self.observation_horizon, 1)
            return history
        history = history.clone()
        if self.observation_horizon > 1:
            history[:, :-1] = history[:, 1:].clone()
        history[:, -1] = obs_batch
        if reset_mask.any():
            history[reset_mask] = obs_batch[reset_mask].unsqueeze(1).repeat(1, self.observation_horizon, 1)
        return history

    def _ensure_action_queues(self, batch_size: int, training: bool) -> list[deque[torch.Tensor]]:
        queues = self._train_action_queues if training else self._eval_action_queues
        if queues is None or len(queues) != batch_size:
            queues = [deque() for _ in range(batch_size)]
            if training:
                self._train_action_queues = queues
            else:
                self._eval_action_queues = queues
        return queues

    def reset_rollout_state(self) -> None:
        self._rollout_obs_history = None
        self._train_action_queues = None
        self._eval_action_queues = None

    def clone_rollout_state(self) -> Any:
        return {
            "obs_history": None if self._rollout_obs_history is None else self._rollout_obs_history.detach().clone(),
            "train_action_queues": _clone_action_queues(self._train_action_queues),
            "eval_action_queues": _clone_action_queues(self._eval_action_queues),
        }

    def restore_rollout_state(self, state: Any) -> None:
        self._rollout_obs_history = None if state["obs_history"] is None else state["obs_history"].detach().clone()
        self._train_action_queues = _clone_action_queues(state.get("train_action_queues"))
        self._eval_action_queues = _clone_action_queues(state.get("eval_action_queues"))

    def predict_value(self, obs: dict[str, np.ndarray], goal: dict[str, np.ndarray] | None = None) -> np.ndarray:
        del goal
        obs_batch = self.prepare_observation_batch(obs)
        history = self._advance_history(self._rollout_obs_history, obs_batch, np.zeros(obs_batch.shape[0], dtype=bool))
        with torch.no_grad():
            values = self.value_net({"state": history}).view(-1)
        return values.detach().cpu().numpy()

    def _sample_sequences(
        self,
        obs_history: torch.Tensor,
        deterministic: bool,
        use_ema: bool = False,
        return_chain: bool = False,
    ):
        actor = self.ema_actor if use_ema else self.actor
        return actor({"state": obs_history}, deterministic=deterministic, return_chain=return_chain)

    def _compute_chain_log_probs(
        self,
        obs_history: torch.Tensor,
        chain_samples: torch.Tensor,
        chain_next_samples: torch.Tensor,
        chain_timesteps: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, chain_length, horizon_steps, action_dim = chain_samples.shape
        cond = {
            "state": obs_history.unsqueeze(1)
            .repeat(1, chain_length, 1, 1)
            .reshape(batch_size * chain_length, self.observation_horizon, self.bundle.obs_dim)
        }
        prev_flat = chain_samples.reshape(batch_size * chain_length, horizon_steps, action_dim)
        next_flat = chain_next_samples.reshape(batch_size * chain_length, horizon_steps, action_dim)
        timestep_flat = chain_timesteps.reshape(-1)
        mean, logvar = self.actor.p_mean_var(prev_flat, timestep_flat, cond)
        std = torch.exp(0.5 * logvar).clamp(min=self.min_logprob_denoising_std)
        log_probs = Normal(mean, std).log_prob(next_flat)
        return log_probs.reshape(batch_size, chain_length, horizon_steps, action_dim)

    def _chain_log_prob_subsample(
        self,
        obs_history: dict[str, torch.Tensor],
        goal_batch: dict[str, torch.Tensor] | None,
        chain_samples: torch.Tensor,
        chain_next_samples: torch.Tensor,
        chain_timesteps: torch.Tensor,
    ) -> torch.Tensor:
        del goal_batch
        return self._compute_chain_log_probs(
            obs_history=obs_history["state"],
            chain_samples=chain_samples,
            chain_next_samples=chain_next_samples,
            chain_timesteps=chain_timesteps,
        )

    def _start_new_decisions(
        self,
        obs_history: torch.Tensor,
        replan_ids: list[int],
    ) -> dict[str, Any] | None:
        if not replan_ids:
            return None
        replan_index = torch.as_tensor(replan_ids, device=self.device, dtype=torch.long)
        replan_obs_history = obs_history.index_select(0, replan_index)
        sample = self._sample_sequences(replan_obs_history, deterministic=False, return_chain=True)
        normalized_sequences = sample.trajectories
        action_sequences = torch.as_tensor(
            self.bundle.normalization.unnormalize_action(normalized_sequences.detach().cpu().numpy()),
            dtype=torch.float32,
            device=self.device,
        )
        chain = sample.chains
        chain_prev = chain[:, :-1]
        chain_next = chain[:, 1:]
        chain_length = chain_prev.shape[1]
        timestep_template = torch.arange(chain_length - 1, -1, -1, device=self.device, dtype=torch.long)
        chain_timesteps = timestep_template.unsqueeze(0).repeat(chain_prev.shape[0], 1)
        log_probs = self._compute_chain_log_probs(replan_obs_history, chain_prev, chain_next, chain_timesteps)
        values = self.value_net({"state": replan_obs_history}).view(-1)
        return {
            "env_indices": np.asarray(replan_ids, dtype=np.int64),
            "obs_history": {"state": replan_obs_history.detach().clone()},
            "goals": None,
            "chain_samples": chain_prev.detach().clone(),
            "chain_next_samples": chain_next.detach().clone(),
            "chain_timesteps": chain_timesteps.detach().clone(),
            "log_probs": log_probs.detach().clone(),
            "values": values.detach().clone(),
            "action_sequences": action_sequences.detach().clone(),
        }

    def act(
        self,
        obs: dict[str, np.ndarray],
        goal: dict[str, np.ndarray] | None,
        episode_starts: np.ndarray,
        clip_actions: bool = True,
    ):
        del goal
        self.actor.train()
        self.value_net.train()
        obs_batch = self.prepare_observation_batch(obs)
        obs_history = self._advance_history(self._rollout_obs_history, obs_batch, episode_starts)
        queues = self._ensure_action_queues(obs_batch.shape[0], training=True)
        reset_mask = np.asarray(episode_starts, dtype=bool)
        for env_idx in np.nonzero(reset_mask)[0].tolist():
            queues[env_idx].clear()

        replan_ids = [env_idx for env_idx, queue in enumerate(queues) if len(queue) == 0]
        replan_data = self._start_new_decisions(obs_history, replan_ids)
        if replan_data is not None:
            for local_idx, env_idx in enumerate(replan_ids):
                queues[env_idx].extend(replan_data["action_sequences"][local_idx, : self.act_steps].unbind(dim=0))

        env_actions = torch.stack([queue.popleft() for queue in queues], dim=0)
        completed_envs = np.asarray([env_idx for env_idx, queue in enumerate(queues) if len(queue) == 0], dtype=np.int64)
        if clip_actions:
            env_actions = env_actions.clamp(-1.0, 1.0)
        self._rollout_obs_history = obs_history.detach().clone()
        return env_actions.detach().cpu().numpy(), replan_data, completed_envs

    def act_deterministic(
        self,
        obs: dict[str, np.ndarray],
        goal: dict[str, np.ndarray] | None,
        episode_starts: np.ndarray,
        clip_actions: bool = True,
    ) -> np.ndarray:
        del goal
        self.actor.eval()
        self.value_net.eval()
        obs_batch = self.prepare_observation_batch(obs)
        obs_history = self._advance_history(self._rollout_obs_history, obs_batch, episode_starts)
        queues = self._ensure_action_queues(obs_batch.shape[0], training=False)
        reset_mask = np.asarray(episode_starts, dtype=bool)
        for env_idx in np.nonzero(reset_mask)[0].tolist():
            queues[env_idx].clear()

        replan_ids = [env_idx for env_idx, queue in enumerate(queues) if len(queue) == 0]
        if replan_ids:
            replan_index = torch.as_tensor(replan_ids, device=self.device, dtype=torch.long)
            replan_obs_history = obs_history.index_select(0, replan_index)
            sample = self._sample_sequences(replan_obs_history, deterministic=True, use_ema=True, return_chain=False)
            action_sequences = torch.as_tensor(
                self.bundle.normalization.unnormalize_action(sample.trajectories.detach().cpu().numpy()),
                dtype=torch.float32,
                device=self.device,
            )
            for local_idx, env_idx in enumerate(replan_ids):
                queues[env_idx].extend(action_sequences[local_idx, : self.act_steps].unbind(dim=0))

        env_actions = torch.stack([queue.popleft() for queue in queues], dim=0)
        if clip_actions:
            env_actions = env_actions.clamp(-1.0, 1.0)
        self._rollout_obs_history = obs_history.detach().clone()
        return env_actions.detach().cpu().numpy()

    def sample(self, obs: dict[str, np.ndarray]) -> DPPORolloutSample:
        obs_batch = {key: value[None, ...] for key, value in obs.items()}
        action = self.act_deterministic(obs=obs_batch, goal=None, episode_starts=np.ones(1, dtype=bool))[0]
        history = self._rollout_obs_history[0].detach().cpu().numpy()
        normalized_actions = self.bundle.normalization.normalize_action(action[None]).astype(np.float32, copy=False)
        return DPPORolloutSample(
            normalized_actions=normalized_actions,
            actions=action.astype(np.float32, copy=False),
            history=history,
        )


def _clone_action_queues(queues: list[deque[torch.Tensor]] | None) -> list[deque[torch.Tensor]] | None:
    if queues is None:
        return None
    return [deque([item.detach().clone() for item in queue]) for queue in queues]
