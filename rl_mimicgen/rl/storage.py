from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(slots=True)
class RolloutBatch:
    observations: dict[str, torch.Tensor]
    actions: torch.Tensor
    log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    episode_starts: torch.Tensor


class RolloutStorage:
    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        obs_shapes: dict[str, tuple[int, ...]],
        action_dim: int,
        device: torch.device,
    ) -> None:
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.device = device
        self.observations = {
            key: torch.zeros((rollout_steps, num_envs, *shape), dtype=torch.float32, device=device)
            for key, shape in obs_shapes.items()
        }
        self.actions = torch.zeros((rollout_steps, num_envs, action_dim), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.episode_starts = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.returns = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.ptr = 0

    def add(
        self,
        obs: dict[str, np.ndarray],
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        episode_starts: np.ndarray,
        values: np.ndarray,
    ) -> None:
        for key, value in obs.items():
            self.observations[key][self.ptr] = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        self.log_probs[self.ptr] = torch.as_tensor(log_probs, dtype=torch.float32, device=self.device)
        self.rewards[self.ptr] = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        self.dones[self.ptr] = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        self.episode_starts[self.ptr] = torch.as_tensor(episode_starts, dtype=torch.float32, device=self.device)
        self.values[self.ptr] = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        self.ptr += 1

    def compute_returns_and_advantages(self, last_value: np.ndarray, gamma: float, gae_lambda: float) -> None:
        next_advantage = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        next_value = torch.as_tensor(last_value, dtype=torch.float32, device=self.device)
        for step in reversed(range(self.rollout_steps)):
            non_terminal = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * next_value * non_terminal - self.values[step]
            next_advantage = delta + gamma * gae_lambda * non_terminal * next_advantage
            self.advantages[step] = next_advantage
            self.returns[step] = self.advantages[step] + self.values[step]
            next_value = self.values[step]

    def as_batch(self) -> RolloutBatch:
        advantages = self.advantages.reshape(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        return RolloutBatch(
            observations={key: value for key, value in self.observations.items()},
            actions=self.actions.reshape(-1, self.actions.shape[-1]),
            log_probs=self.log_probs.reshape(-1),
            returns=self.returns.reshape(-1),
            advantages=advantages,
            episode_starts=self.episode_starts,
        )
