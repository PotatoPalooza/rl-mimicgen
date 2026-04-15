from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(slots=True)
class DiffusionRolloutBatch:
    observations: dict[str, torch.Tensor]
    goals: dict[str, torch.Tensor] | None
    chain_samples: torch.Tensor
    chain_next_samples: torch.Tensor
    chain_timesteps: torch.Tensor
    log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


@dataclass(slots=True)
class _PendingDecision:
    observations: dict[str, torch.Tensor]
    goals: dict[str, torch.Tensor] | None
    chain_samples: torch.Tensor
    chain_next_samples: torch.Tensor
    chain_timesteps: torch.Tensor
    log_probs: torch.Tensor
    value: torch.Tensor
    reward_sum: float = 0.0
    done: float = 0.0
    advantage: torch.Tensor | None = None
    return_: torch.Tensor | None = None


class DiffusionRolloutStorage:
    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        obs_shapes: dict[str, tuple[int, ...]],
        goal_shapes: dict[str, tuple[int, ...]] | None,
        action_dim: int,
        prediction_horizon: int,
        chain_length: int,
        device: torch.device,
    ) -> None:
        del rollout_steps, obs_shapes, goal_shapes, action_dim, prediction_horizon, chain_length
        self.num_envs = num_envs
        self.device = device
        self._completed: list[list[_PendingDecision]] = [[] for _ in range(num_envs)]
        self._pending: list[_PendingDecision | None] = [None for _ in range(num_envs)]

    def reset(self) -> None:
        self._completed = [[] for _ in range(self.num_envs)]
        self._pending = [None for _ in range(self.num_envs)]

    def start_decisions(
        self,
        env_indices: np.ndarray,
        obs_history: dict[str, torch.Tensor],
        goals: dict[str, torch.Tensor] | None,
        chain_samples: torch.Tensor,
        chain_next_samples: torch.Tensor,
        chain_timesteps: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        for local_idx, env_idx in enumerate(env_indices.tolist()):
            if self._pending[env_idx] is not None:
                raise RuntimeError(f"Pending diffusion decision already exists for env {env_idx}.")
            goal_history = None
            if goals is not None:
                goal_history = {key: value[local_idx].detach().clone() for key, value in goals.items()}
            self._pending[env_idx] = _PendingDecision(
                observations={key: value[local_idx].detach().clone() for key, value in obs_history.items()},
                goals=goal_history,
                chain_samples=chain_samples[local_idx].detach().clone(),
                chain_next_samples=chain_next_samples[local_idx].detach().clone(),
                chain_timesteps=chain_timesteps[local_idx].detach().clone(),
                log_probs=log_probs[local_idx].detach().clone(),
                value=values[local_idx].detach().clone(),
            )

    def accumulate_step(self, rewards: np.ndarray, dones: np.ndarray) -> None:
        for env_idx, pending in enumerate(self._pending):
            if pending is None:
                continue
            pending.reward_sum += float(rewards[env_idx])
            if dones[env_idx]:
                pending.done = 1.0

    def finalize_decisions(self, env_indices: np.ndarray) -> None:
        for env_idx in env_indices.tolist():
            pending = self._pending[env_idx]
            if pending is None:
                continue
            self._completed[env_idx].append(pending)
            self._pending[env_idx] = None

    def finalize_all_pending(self) -> None:
        env_indices = np.asarray([idx for idx, pending in enumerate(self._pending) if pending is not None], dtype=np.int64)
        if env_indices.size:
            self.finalize_decisions(env_indices)

    def compute_returns_and_advantages(self, last_value: np.ndarray, gamma: float, gae_lambda: float) -> None:
        self.finalize_all_pending()
        last_value_tensor = torch.as_tensor(last_value, dtype=torch.float32, device=self.device)
        for env_idx, records in enumerate(self._completed):
            next_advantage = torch.zeros((), dtype=torch.float32, device=self.device)
            next_value = last_value_tensor[env_idx]
            for record in reversed(records):
                non_terminal = 1.0 - record.done
                reward_tensor = torch.tensor(record.reward_sum, dtype=torch.float32, device=self.device)
                delta = reward_tensor + gamma * next_value * non_terminal - record.value
                next_advantage = delta + gamma * gae_lambda * non_terminal * next_advantage
                record.log_probs = record.log_probs.to(dtype=torch.float32, device=self.device)
                record.chain_timesteps = record.chain_timesteps.to(device=self.device, dtype=torch.long)
                record.value = record.value.to(dtype=torch.float32, device=self.device)
                record.chain_samples = record.chain_samples.to(dtype=torch.float32, device=self.device)
                record.chain_next_samples = record.chain_next_samples.to(dtype=torch.float32, device=self.device)
                record.reward_sum = float(reward_tensor.item())
                record.done = float(record.done)
                record.advantage = next_advantage.detach().clone()
                record.return_ = (next_advantage + record.value).detach().clone()
                next_value = record.value

    def as_batch(self) -> DiffusionRolloutBatch:
        flat_records = [record for env_records in self._completed for record in env_records]
        if not flat_records:
            raise RuntimeError("Diffusion rollout batch is empty.")

        advantages = torch.stack([record.advantage for record in flat_records], dim=0)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        goal_batch = None
        if flat_records[0].goals is not None:
            goal_batch = {
                key: torch.stack([record.goals[key] for record in flat_records], dim=0)
                for key in flat_records[0].goals
            }

        return DiffusionRolloutBatch(
            observations={
                key: torch.stack([record.observations[key] for record in flat_records], dim=0)
                for key in flat_records[0].observations
            },
            goals=goal_batch,
            chain_samples=torch.stack([record.chain_samples for record in flat_records], dim=0),
            chain_next_samples=torch.stack([record.chain_next_samples for record in flat_records], dim=0),
            chain_timesteps=torch.stack([record.chain_timesteps for record in flat_records], dim=0),
            log_probs=torch.stack([record.log_probs for record in flat_records], dim=0),
            returns=torch.stack([record.return_ for record in flat_records], dim=0),
            advantages=advantages,
            values=torch.stack([record.value for record in flat_records], dim=0),
            rewards=torch.tensor([record.reward_sum for record in flat_records], dtype=torch.float32, device=self.device),
            dones=torch.tensor([record.done for record in flat_records], dtype=torch.float32, device=self.device),
        )

    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError("Diffusion rollout storage checkpointing is not implemented for queue-based storage.")

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        del state_dict
        raise NotImplementedError("Diffusion rollout storage checkpoint loading is not implemented for queue-based storage.")
