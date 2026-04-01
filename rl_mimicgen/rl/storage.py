from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(slots=True)
class RolloutBatch:
    observations: dict[str, torch.Tensor]
    goals: dict[str, torch.Tensor] | None
    actions: torch.Tensor
    log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    episode_starts: torch.Tensor
    initial_rnn_state: Any = None


class RolloutStorage:
    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        obs_shapes: dict[str, tuple[int, ...]],
        goal_shapes: dict[str, tuple[int, ...]] | None,
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
        self.goals = (
            {
                key: torch.zeros((rollout_steps, num_envs, *shape), dtype=torch.float32, device=device)
                for key, shape in goal_shapes.items()
            }
            if goal_shapes
            else {}
        )
        self.log_probs = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.episode_starts = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.returns = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.initial_rnn_state = None
        self.ptr = 0

    def reset(self) -> None:
        self.ptr = 0
        self.initial_rnn_state = None

    def set_initial_rnn_state(self, state: Any) -> None:
        self.initial_rnn_state = _clone_rnn_state(state)

    def add(
        self,
        obs: dict[str, np.ndarray],
        actions: np.ndarray,
        goals: dict[str, np.ndarray] | None,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        episode_starts: np.ndarray,
        values: np.ndarray,
    ) -> None:
        missing_obs_keys = [key for key in self.observations if key not in obs]
        if missing_obs_keys:
            raise KeyError(f"Missing observation keys for rollout storage: {missing_obs_keys}")
        for key in self.observations:
            self.observations[key][self.ptr] = torch.as_tensor(obs[key], dtype=torch.float32, device=self.device)
        if goals is not None:
            missing_goal_keys = [key for key in self.goals if key not in goals]
            if missing_goal_keys:
                raise KeyError(f"Missing goal keys for rollout storage: {missing_goal_keys}")
            for key in self.goals:
                self.goals[key][self.ptr] = torch.as_tensor(goals[key], dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        self.log_probs[self.ptr] = torch.as_tensor(log_probs, dtype=torch.float32, device=self.device)
        self.rewards[self.ptr] = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        self.dones[self.ptr] = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        self.episode_starts[self.ptr] = torch.as_tensor(episode_starts, dtype=torch.float32, device=self.device)
        self.values[self.ptr] = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        self.ptr += 1

    def compute_returns_and_advantages(self, last_value: np.ndarray, gamma: float, gae_lambda: float) -> None:
        if self.ptr != self.rollout_steps:
            raise RuntimeError(f"Expected full rollout of {self.rollout_steps} steps, got {self.ptr}.")
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
            goals={key: value for key, value in self.goals.items()} if self.goals else None,
            actions=self.actions,
            log_probs=self.log_probs.reshape(-1),
            returns=self.returns.reshape(-1),
            advantages=advantages,
            values=self.values.reshape(-1),
            rewards=self.rewards.reshape(-1),
            dones=self.dones.reshape(-1),
            episode_starts=self.episode_starts,
            initial_rnn_state=_clone_rnn_state(self.initial_rnn_state),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "observations": {key: value.detach().cpu() for key, value in self.observations.items()},
            "goals": {key: value.detach().cpu() for key, value in self.goals.items()},
            "actions": self.actions.detach().cpu(),
            "log_probs": self.log_probs.detach().cpu(),
            "rewards": self.rewards.detach().cpu(),
            "dones": self.dones.detach().cpu(),
            "episode_starts": self.episode_starts.detach().cpu(),
            "values": self.values.detach().cpu(),
            "advantages": self.advantages.detach().cpu(),
            "returns": self.returns.detach().cpu(),
            "initial_rnn_state": _state_to_cpu(self.initial_rnn_state),
            "ptr": self.ptr,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for key, value in state_dict["observations"].items():
            self.observations[key].copy_(value.to(device=self.device))
        for key, value in state_dict.get("goals", {}).items():
            self.goals[key].copy_(value.to(device=self.device))
        self.actions.copy_(state_dict["actions"].to(device=self.device))
        self.log_probs.copy_(state_dict["log_probs"].to(device=self.device))
        self.rewards.copy_(state_dict["rewards"].to(device=self.device))
        self.dones.copy_(state_dict["dones"].to(device=self.device))
        self.episode_starts.copy_(state_dict["episode_starts"].to(device=self.device))
        self.values.copy_(state_dict["values"].to(device=self.device))
        self.advantages.copy_(state_dict["advantages"].to(device=self.device))
        self.returns.copy_(state_dict["returns"].to(device=self.device))
        self.initial_rnn_state = _state_to_device(state_dict.get("initial_rnn_state"), self.device)
        self.ptr = int(state_dict["ptr"])


def _clone_rnn_state(state: Any) -> Any:
    if state is None:
        return None
    if isinstance(state, dict):
        return {key: _clone_rnn_state(value) for key, value in state.items()}
    if isinstance(state, tuple):
        return tuple(_clone_rnn_state(value) for value in state)
    if torch.is_tensor(state):
        return state.detach().clone()
    return state


def _state_to_cpu(state: Any) -> Any:
    if state is None:
        return None
    if isinstance(state, dict):
        return {key: _state_to_cpu(value) for key, value in state.items()}
    if isinstance(state, tuple):
        return tuple(_state_to_cpu(value) for value in state)
    if torch.is_tensor(state):
        return state.detach().cpu()
    return state


def _state_to_device(state: Any, device: torch.device) -> Any:
    if state is None:
        return None
    if isinstance(state, dict):
        return {key: _state_to_device(value, device) for key, value in state.items()}
    if isinstance(state, tuple):
        return tuple(_state_to_device(value, device) for value in state)
    if torch.is_tensor(state):
        return state.to(device=device)
    return state
