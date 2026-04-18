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


@dataclass(slots=True)
class SequenceBatch:
    observations: dict[str, torch.Tensor]
    goals: dict[str, torch.Tensor] | None
    actions: torch.Tensor
    episode_starts: torch.Tensor
    mask: torch.Tensor


@dataclass(slots=True)
class TrajectoryRecord:
    observations: dict[str, np.ndarray]
    goals: dict[str, np.ndarray] | None
    actions: np.ndarray


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
        self.trajectories: list[TrajectoryRecord] = []

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

    def add_trajectory(
        self,
        obs: dict[str, np.ndarray],
        actions: np.ndarray,
        goals: dict[str, np.ndarray] | None = None,
    ) -> None:
        if actions.shape[0] == 0:
            return
        trajectory = TrajectoryRecord(
            observations={
                key: np.asarray(value, dtype=np.float32).copy()
                for key, value in obs.items()
                if key in self.observations
            },
            goals=(
                {
                    key: np.asarray(value, dtype=np.float32).copy()
                    for key, value in goals.items()
                    if key in self.goals
                }
                if goals is not None and self.goals
                else None
            ),
            actions=np.asarray(actions, dtype=np.float32).copy(),
        )
        self.trajectories.append(trajectory)
        if len(self.trajectories) > self.capacity:
            self.trajectories.pop(0)

    def can_sample_sequences(self, sequence_length: int) -> bool:
        required_length = max(1, int(sequence_length))
        return any(record.actions.shape[0] >= 1 for record in self.trajectories) and required_length > 0

    def sample_sequences(self, sequence_length: int, batch_size: int, device: torch.device) -> SequenceBatch:
        if not self.trajectories:
            raise RuntimeError("Replay buffer does not contain any stored trajectories.")

        seq_len = max(1, int(sequence_length))
        num_sequences = max(1, int(batch_size))
        candidate_indices = [idx for idx, record in enumerate(self.trajectories) if record.actions.shape[0] >= 1]
        if not candidate_indices:
            raise RuntimeError("Replay buffer does not contain any non-empty trajectories.")

        obs_batch = {
            key: np.zeros((seq_len, num_sequences, *shape), dtype=np.float32)
            for key, shape in self.obs_shapes.items()
        }
        goal_batch = {
            key: np.zeros((seq_len, num_sequences, *shape), dtype=np.float32)
            for key, shape in self.goal_shapes.items()
        }
        action_batch = np.zeros((seq_len, num_sequences, self.action_dim), dtype=np.float32)
        episode_starts = np.zeros((seq_len, num_sequences), dtype=bool)
        mask = np.zeros((seq_len, num_sequences), dtype=bool)

        sampled_traj_ids = np.random.choice(candidate_indices, size=num_sequences, replace=True)
        for column, traj_idx in enumerate(sampled_traj_ids):
            record = self.trajectories[int(traj_idx)]
            traj_len = int(record.actions.shape[0])
            start_idx = 0 if traj_len <= seq_len else int(np.random.randint(0, traj_len - seq_len + 1))
            actual_len = min(seq_len, traj_len - start_idx)
            end_idx = start_idx + actual_len

            for key in obs_batch:
                obs_batch[key][:actual_len, column] = record.observations[key][start_idx:end_idx]
            for key in goal_batch:
                if record.goals is not None and key in record.goals:
                    goal_batch[key][:actual_len, column] = record.goals[key][start_idx:end_idx]
            action_batch[:actual_len, column] = record.actions[start_idx:end_idx]
            episode_starts[0, column] = True
            mask[:actual_len, column] = True

        tensor_goals = None
        if goal_batch:
            tensor_goals = {
                key: torch.as_tensor(value, dtype=torch.float32, device=device)
                for key, value in goal_batch.items()
            }

        return SequenceBatch(
            observations={
                key: torch.as_tensor(value, dtype=torch.float32, device=device)
                for key, value in obs_batch.items()
            },
            goals=tensor_goals,
            actions=torch.as_tensor(action_batch, dtype=torch.float32, device=device),
            episode_starts=torch.as_tensor(episode_starts, dtype=torch.bool, device=device),
            mask=torch.as_tensor(mask, dtype=torch.bool, device=device),
        )
