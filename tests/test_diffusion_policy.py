import numpy as np
import torch
import torch.nn as nn

from rl_mimicgen.rl.diffusion_policy import DiffusionOnlinePolicyAdapter


class _DummyModule(nn.Module):
    def forward(self, *args, **kwargs):
        raise NotImplementedError


def test_diffusion_policy_uses_live_actor_for_train_rollouts_and_ema_for_eval(monkeypatch) -> None:
    adapter = DiffusionOnlinePolicyAdapter.__new__(DiffusionOnlinePolicyAdapter)
    nn.Module.__init__(adapter)
    adapter.actor = _DummyModule()
    adapter.value_net = _DummyModule()
    adapter.device = torch.device("cpu")
    adapter.use_ema_for_evaluation = True
    adapter._rollout_obs_history = None

    obs_batch = {"obs": torch.zeros((2, 3), dtype=torch.float32)}
    goal_batch = {"goal": torch.ones((2, 1), dtype=torch.float32)}
    recorded_use_ema: list[bool] = []

    monkeypatch.setattr(adapter, "prepare_observation_batch", lambda obs: obs_batch)
    monkeypatch.setattr(adapter, "prepare_goal_batch", lambda goal: goal_batch)
    monkeypatch.setattr(adapter, "_advance_history", lambda history, obs, starts: obs)

    def fake_sample_action_chain(*, obs_history, goal_batch, deterministic, use_ema):
        del obs_history, goal_batch, deterministic
        recorded_use_ema.append(bool(use_ema))
        batch_size = 2
        return (
            torch.zeros((batch_size, 3), dtype=torch.float32),
            torch.zeros(batch_size, dtype=torch.float32),
            torch.zeros((batch_size, 1, 1, 3), dtype=torch.float32),
            torch.zeros((batch_size, 1, 1, 3), dtype=torch.float32),
            torch.zeros((batch_size, 1), dtype=torch.long),
        )

    monkeypatch.setattr(adapter, "_sample_action_chain", fake_sample_action_chain)

    adapter.act(
        obs={"obs": np.zeros((2, 3), dtype=np.float32)},
        goal={"goal": np.ones((2, 1), dtype=np.float32)},
        episode_starts=np.array([True, False]),
    )
    adapter.act_deterministic(
        obs={"obs": np.zeros((2, 3), dtype=np.float32)},
        goal={"goal": np.ones((2, 1), dtype=np.float32)},
        episode_starts=np.array([True, False]),
    )

    assert recorded_use_ema == [False, True]


def test_diffusion_policy_deterministic_eval_reuses_queued_actions(monkeypatch) -> None:
    adapter = DiffusionOnlinePolicyAdapter.__new__(DiffusionOnlinePolicyAdapter)
    nn.Module.__init__(adapter)
    adapter.actor = _DummyModule()
    adapter.value_net = _DummyModule()
    adapter.device = torch.device("cpu")
    adapter.action_horizon = 3
    adapter.use_ema_for_evaluation = True
    adapter._rollout_obs_history = None
    adapter._eval_action_queues = None

    obs_batch = {"obs": torch.zeros((1, 3), dtype=torch.float32)}
    sampled_sequences = [
        torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]], dtype=torch.float32),
        torch.tensor([[[4.0, 0.0], [5.0, 0.0], [6.0, 0.0]]], dtype=torch.float32),
    ]
    call_count = 0

    monkeypatch.setattr(adapter, "prepare_observation_batch", lambda obs: obs_batch)
    monkeypatch.setattr(adapter, "prepare_goal_batch", lambda goal: None)
    monkeypatch.setattr(adapter, "_advance_history", lambda history, obs, starts: obs)

    def fake_sample_action_sequence(*, obs_history, goal_batch, use_ema):
        del obs_history, goal_batch, use_ema
        nonlocal call_count
        sequence = sampled_sequences[call_count]
        call_count += 1
        return sequence

    monkeypatch.setattr(adapter, "_sample_action_sequence", fake_sample_action_sequence)

    actions = []
    for _ in range(4):
        action = adapter.act_deterministic(
            obs={"obs": np.zeros((1, 3), dtype=np.float32)},
            goal=None,
            episode_starts=np.array([False]),
            clip_actions=False,
        )
        actions.append(action[0, 0])

    assert actions == [1.0, 2.0, 3.0, 4.0]
    assert call_count == 2
