from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(slots=True)
class TransitionBatch:
    observations: dict[str, torch.Tensor]
    next_observations: dict[str, torch.Tensor]
    goals: dict[str, torch.Tensor] | None
    next_goals: dict[str, torch.Tensor] | None
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_shapes: dict[str, tuple[int, ...]],
        goal_shapes: dict[str, tuple[int, ...]] | None,
        action_dim: int,
    ) -> None:
        self.capacity = max(1, int(capacity))
        self.obs_shapes = obs_shapes
        self.goal_shapes = goal_shapes or {}
        self.action_dim = int(action_dim)

        self.observations = {
            key: np.zeros((self.capacity, *shape), dtype=np.float32)
            for key, shape in obs_shapes.items()
        }
        self.next_observations = {
            key: np.zeros((self.capacity, *shape), dtype=np.float32)
            for key, shape in obs_shapes.items()
        }
        self.goals = {
            key: np.zeros((self.capacity, *shape), dtype=np.float32)
            for key, shape in self.goal_shapes.items()
        }
        self.next_goals = {
            key: np.zeros((self.capacity, *shape), dtype=np.float32)
            for key, shape in self.goal_shapes.items()
        }
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add_batch(
        self,
        obs: dict[str, np.ndarray],
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: dict[str, np.ndarray],
        dones: np.ndarray,
        goals: dict[str, np.ndarray] | None = None,
        next_goals: dict[str, np.ndarray] | None = None,
    ) -> None:
        batch_size = int(actions.shape[0])
        for idx in range(batch_size):
            self._add_single(
                obs={key: obs[key][idx] for key in self.observations},
                action=actions[idx],
                reward=float(rewards[idx]),
                next_obs={key: next_obs[key][idx] for key in self.next_observations},
                done=float(dones[idx]),
                goal=None if goals is None else {key: goals[key][idx] for key in self.goals},
                next_goal=None if next_goals is None else {key: next_goals[key][idx] for key in self.next_goals},
            )

    def _add_single(
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        next_obs: dict[str, np.ndarray],
        done: float,
        goal: dict[str, np.ndarray] | None,
        next_goal: dict[str, np.ndarray] | None,
    ) -> None:
        insert_idx = self.ptr
        for key in self.observations:
            self.observations[key][insert_idx] = np.asarray(obs[key], dtype=np.float32)
            self.next_observations[key][insert_idx] = np.asarray(next_obs[key], dtype=np.float32)
        for key in self.goals:
            if goal is None or next_goal is None:
                self.goals[key][insert_idx].fill(0.0)
                self.next_goals[key][insert_idx].fill(0.0)
            else:
                self.goals[key][insert_idx] = np.asarray(goal[key], dtype=np.float32)
                self.next_goals[key][insert_idx] = np.asarray(next_goal[key], dtype=np.float32)

        self.actions[insert_idx] = np.asarray(action, dtype=np.float32)
        self.rewards[insert_idx] = float(reward)
        self.dones[insert_idx] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> TransitionBatch:
        if self.size == 0:
            raise RuntimeError("Replay buffer is empty.")
        sample_size = min(int(batch_size), self.size)
        indices = np.random.randint(0, self.size, size=sample_size)
        return TransitionBatch(
            observations={
                key: torch.as_tensor(value[indices], dtype=torch.float32, device=device)
                for key, value in self.observations.items()
            },
            next_observations={
                key: torch.as_tensor(value[indices], dtype=torch.float32, device=device)
                for key, value in self.next_observations.items()
            },
            goals=(
                {
                    key: torch.as_tensor(value[indices], dtype=torch.float32, device=device)
                    for key, value in self.goals.items()
                }
                if self.goals
                else None
            ),
            next_goals=(
                {
                    key: torch.as_tensor(value[indices], dtype=torch.float32, device=device)
                    for key, value in self.next_goals.items()
                }
                if self.next_goals
                else None
            ),
            actions=torch.as_tensor(self.actions[indices], dtype=torch.float32, device=device),
            rewards=torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=device),
            dones=torch.as_tensor(self.dones[indices], dtype=torch.float32, device=device),
        )
