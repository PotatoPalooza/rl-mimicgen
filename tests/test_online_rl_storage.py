import numpy as np
import torch

from rl_mimicgen.rl.storage import RolloutStorage


def test_rollout_storage_flattens_time_and_env_axes() -> None:
    storage = RolloutStorage(
        rollout_steps=2,
        num_envs=3,
        obs_shapes={"obs": (4,)},
        action_dim=2,
        device=torch.device("cpu"),
    )
    obs = {"obs": np.ones((3, 4), dtype=np.float32)}
    actions = np.ones((3, 2), dtype=np.float32)
    log_probs = np.zeros(3, dtype=np.float32)
    rewards = np.ones(3, dtype=np.float32)
    dones = np.zeros(3, dtype=np.float32)
    starts = np.zeros(3, dtype=np.float32)
    values = np.zeros(3, dtype=np.float32)

    storage.add(obs, actions, log_probs, rewards, dones, starts, values)
    storage.add(obs, actions, log_probs, rewards, dones, starts, values)
    storage.compute_returns_and_advantages(last_value=np.zeros(3, dtype=np.float32), gamma=0.99, gae_lambda=0.95)

    batch = storage.as_batch()

    assert batch.actions.shape == (6, 2)
    assert batch.log_probs.shape == (6,)
    assert batch.observations["obs"].shape == (2, 3, 4)
