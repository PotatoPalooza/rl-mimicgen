import numpy as np
import torch

from rl_mimicgen.rl.diffusion_storage import DiffusionRolloutStorage


def test_diffusion_rollout_storage_builds_decision_level_batch() -> None:
    storage = DiffusionRolloutStorage(
        rollout_steps=2,
        num_envs=3,
        obs_shapes={"obs": (4,)},
        goal_shapes=None,
        action_dim=2,
        prediction_horizon=5,
        chain_length=4,
        device=torch.device("cpu"),
    )
    obs_history = {"obs": torch.ones((2, 2, 4), dtype=torch.float32)}
    chain_samples = torch.ones((2, 4, 5, 2), dtype=torch.float32)
    chain_next_samples = torch.ones((2, 4, 5, 2), dtype=torch.float32)
    chain_timesteps = torch.tile(torch.arange(4, dtype=torch.long), (2, 1))
    log_probs = torch.zeros((2, 4, 2, 2), dtype=torch.float32)
    values = torch.zeros(2, dtype=torch.float32)

    storage.start_decisions(
        env_indices=np.array([0, 2], dtype=np.int64),
        obs_history=obs_history,
        goals=None,
        chain_samples=chain_samples,
        chain_next_samples=chain_next_samples,
        chain_timesteps=chain_timesteps,
        log_probs=log_probs,
        values=values,
    )
    storage.accumulate_step(
        rewards=np.array([1.0, 0.0, 2.0], dtype=np.float32),
        dones=np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    storage.finalize_decisions(np.array([2], dtype=np.int64))
    storage.accumulate_step(
        rewards=np.array([3.0, 0.0, 0.0], dtype=np.float32),
        dones=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    storage.finalize_decisions(np.array([0], dtype=np.int64))
    storage.compute_returns_and_advantages(last_value=np.zeros(3, dtype=np.float32), gamma=0.99, gae_lambda=0.95)

    batch = storage.as_batch()

    assert batch.observations["obs"].shape == (2, 2, 4)
    assert batch.chain_samples.shape == (2, 4, 5, 2)
    assert batch.chain_next_samples.shape == (2, 4, 5, 2)
    assert batch.chain_timesteps.shape == (2, 4)
    assert batch.log_probs.shape == (2, 4, 2, 2)
    assert batch.returns.shape == (2,)
    assert torch.allclose(batch.rewards, torch.tensor([4.0, 2.0], dtype=torch.float32), atol=1e-4)
