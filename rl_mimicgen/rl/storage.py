from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RolloutBatch:
    observations: list[dict] = field(default_factory=list)
    goals: list[dict | None] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    episode_starts: list[bool] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    returns: np.ndarray | None = None
    advantages: np.ndarray | None = None

    def add(
        self,
        observation: dict,
        goal: dict | None,
        action: np.ndarray,
        reward: float,
        done: bool,
        episode_start: bool,
        value: float,
        log_prob: float,
    ) -> None:
        self.observations.append(observation)
        self.goals.append(goal)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.episode_starts.append(bool(episode_start))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))

    def finish(self, last_value: float, gamma: float, gae_lambda: float) -> None:
        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values + [float(last_value)], dtype=np.float32)
        dones = np.asarray(self.dones + [False], dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for step in reversed(range(len(rewards))):
            non_terminal = 1.0 - dones[step]
            delta = rewards[step] + gamma * values[step + 1] * non_terminal - values[step]
            gae = delta + gamma * gae_lambda * non_terminal * gae
            advantages[step] = gae

        returns = advantages + values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.advantages = advantages
        self.returns = returns

    def as_numpy(self) -> dict[str, np.ndarray]:
        if self.advantages is None or self.returns is None:
            raise RuntimeError("finish() must be called before reading rollout tensors")
        return {
            "actions": np.asarray(self.actions, dtype=np.float32),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "dones": np.asarray(self.dones, dtype=np.float32),
            "episode_starts": np.asarray(self.episode_starts, dtype=np.float32),
            "values": np.asarray(self.values, dtype=np.float32),
            "log_probs": np.asarray(self.log_probs, dtype=np.float32),
            "advantages": self.advantages.astype(np.float32),
            "returns": self.returns.astype(np.float32),
        }
