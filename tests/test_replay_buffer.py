from __future__ import annotations

import numpy as np
import torch

from rl_mimicgen.rl.replay_buffer import ReplayBuffer


def test_replay_buffer_add_and_sample_transition_batch() -> None:
    buffer = ReplayBuffer(
        capacity=16,
        obs_shapes={"obs": (4,)},
        goal_shapes={"goal": (2,)},
        action_dim=3,
    )
    buffer.add_batch(
        obs={"obs": np.ones((5, 4), dtype=np.float32)},
        actions=np.ones((5, 3), dtype=np.float32),
        rewards=np.arange(5, dtype=np.float32),
        next_obs={"obs": np.full((5, 4), 2.0, dtype=np.float32)},
        dones=np.zeros(5, dtype=np.float32),
        goals={"goal": np.ones((5, 2), dtype=np.float32)},
        next_goals={"goal": np.full((5, 2), 3.0, dtype=np.float32)},
    )

    batch = buffer.sample(batch_size=4, device=torch.device("cpu"))

    assert len(buffer) == 5
    assert batch.actions.shape == (4, 3)
    assert batch.observations["obs"].shape == (4, 4)
    assert batch.next_observations["obs"].shape == (4, 4)
    assert batch.goals is not None
    assert batch.goals["goal"].shape == (4, 2)
