from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from rl_mimicgen.rl.awac import AWAC
from rl_mimicgen.rl.replay_buffer import ReplayBuffer


class _DummyQ(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(5, 1)

    def forward(self, obs_dict, acts, goal_dict=None):
        del goal_dict
        obs = obs_dict["obs"]
        return self.net(torch.cat([obs, acts], dim=-1))


class _DummyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = nn.Linear(3, 2)
        self.q_networks = nn.ModuleList([_DummyQ(), _DummyQ()])
        self.target_q_networks = nn.ModuleList([copy.deepcopy(q) for q in self.q_networks])

    def train(self, mode: bool = True):
        super().train(mode)
        return self

    def actor_parameters(self):
        return list(self.actor.parameters())

    def q_values(self, observations, actions, goals=None, use_target=False):
        del goals
        q_networks = self.target_q_networks if use_target else self.q_networks
        return [q_net(observations, actions).squeeze(-1) for q_net in q_networks]

    def min_q_value(self, observations, actions, goals=None, use_target=False):
        return torch.stack(self.q_values(observations, actions, goals=goals, use_target=use_target), dim=0).min(dim=0)[0]

    def approximate_v_value(self, observations, goals=None, use_target=False, num_action_samples=None):
        del goals, use_target, num_action_samples
        return torch.zeros(observations["obs"].shape[0], dtype=torch.float32, device=observations["obs"].device)

    def actor_log_prob_replay(self, observations, goals, actions):
        del goals
        mean_actions = self.actor(observations["obs"])
        log_probs = -((actions - mean_actions) ** 2).sum(dim=-1)
        entropy = torch.ones_like(log_probs) * 0.5
        return log_probs, entropy

    def soft_update_targets(self) -> None:
        for q_net, target_q_net in zip(self.q_networks, self.target_q_networks):
            for source_param, target_param in zip(q_net.parameters(), target_q_net.parameters()):
                target_param.data.copy_(0.5 * source_param.data + 0.5 * target_param.data)


def test_awac_update_reports_replay_weighting_and_q_metrics() -> None:
    device = torch.device("cpu")
    replay_buffer = ReplayBuffer(
        capacity=32,
        obs_shapes={"obs": (3,)},
        goal_shapes=None,
        action_dim=2,
    )
    replay_buffer.add_batch(
        obs={"obs": np.random.randn(8, 3).astype(np.float32)},
        actions=np.random.randn(8, 2).astype(np.float32),
        rewards=np.linspace(0.0, 1.0, num=8, dtype=np.float32),
        next_obs={"obs": np.random.randn(8, 3).astype(np.float32)},
        dones=np.zeros(8, dtype=np.float32),
    )

    policy = _DummyPolicy()
    algo = AWAC(
        policy=policy,
        batch_size=4,
        actor_learning_rate=1e-3,
        critic_learning_rate=1e-3,
        beta=0.5,
        max_weight=5.0,
        num_learning_epochs=1,
        num_mini_batches=1,
        device=device,
    )

    metrics = algo.update(replay_buffer)

    assert "weight_mean" in metrics
    assert "weight_std" in metrics
    assert "q_pred_mean" in metrics
    assert "target_q_mean" in metrics
    assert metrics["effective_num_minibatches"] == 1.0
    assert metrics["weight_mean"] > 0.0
    assert metrics["weight_max"] >= metrics["weight_min"]
