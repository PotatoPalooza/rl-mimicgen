from __future__ import annotations

from pathlib import Path

import numpy as np

from rl_mimicgen.dppo.data.dataset import DPPODatasetBundle, DPPONormalizationStats
from rl_mimicgen.dppo.data.sequence import DPPODiffusionDataset


def _make_bundle(obs: np.ndarray, actions: np.ndarray, traj_lengths: np.ndarray, demo_offsets: np.ndarray) -> DPPODatasetBundle:
    obs_dim = int(obs.shape[1])
    action_dim = int(actions.shape[1])
    return DPPODatasetBundle(
        bundle_dir=Path("/tmp/dppo_test_bundle"),
        metadata={
            "task": "coffee",
            "variant": "D0",
            "obs_keys": ["b_key", "a_key"],
            "num_demos": int(traj_lengths.shape[0]),
            "num_transitions": int(obs.shape[0]),
        },
        obs=obs,
        next_obs=obs.copy(),
        actions=actions,
        rewards=np.zeros(obs.shape[0], dtype=np.float32),
        dones=np.zeros(obs.shape[0], dtype=np.float32),
        traj_lengths=traj_lengths,
        demo_offsets=demo_offsets,
        normalization=DPPONormalizationStats(
            obs_mean=np.zeros(obs_dim, dtype=np.float32),
            obs_std=np.ones(obs_dim, dtype=np.float32),
            obs_min=np.zeros(obs_dim, dtype=np.float32),
            obs_max=np.full(obs_dim, 20.0, dtype=np.float32),
            obs_range=np.full(obs_dim, 20.0, dtype=np.float32),
            action_mean=np.zeros(action_dim, dtype=np.float32),
            action_std=np.ones(action_dim, dtype=np.float32),
            action_min=np.zeros(action_dim, dtype=np.float32),
            action_max=np.full(action_dim, 20.0, dtype=np.float32),
            action_range=np.full(action_dim, 20.0, dtype=np.float32),
        ),
    )


def test_dppo_normalization_round_trip_and_flatten_obs_respects_metadata_order() -> None:
    bundle = _make_bundle(
        obs=np.zeros((2, 2), dtype=np.float32),
        actions=np.zeros((2, 2), dtype=np.float32),
        traj_lengths=np.asarray([2], dtype=np.int32),
        demo_offsets=np.asarray([[0, 2]], dtype=np.int64),
    )
    actions = np.asarray([[5.0, 15.0]], dtype=np.float32)
    round_trip = bundle.normalization.unnormalize_action(bundle.normalization.normalize_action(actions))
    flattened = bundle.flatten_obs(
        {
            "a_key": np.asarray([3.0], dtype=np.float32),
            "b_key": np.asarray([1.0], dtype=np.float32),
        }
    )

    assert np.allclose(round_trip, actions)
    assert flattened.tolist() == [1.0, 3.0]


def test_dppo_sequence_dataset_respects_demo_boundaries_and_prefix_pads_history() -> None:
    obs = np.asarray([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0], [13.0]], dtype=np.float32)
    actions = np.asarray([[100.0], [101.0], [102.0], [200.0], [201.0], [202.0], [203.0]], dtype=np.float32)
    bundle = _make_bundle(
        obs=obs,
        actions=actions,
        traj_lengths=np.asarray([3, 4], dtype=np.int32),
        demo_offsets=np.asarray([[0, 3], [3, 7]], dtype=np.int64),
    )
    dataset = DPPODiffusionDataset(bundle=bundle, horizon_steps=2, cond_steps=3, normalize=False, device="cpu")

    first_demo_first = dataset[0]
    second_demo_first = dataset[2]

    assert len(dataset) == 5
    assert first_demo_first.conditions["state"].squeeze(-1).tolist() == [0.0, 0.0, 0.0]
    assert first_demo_first.actions.squeeze(-1).tolist() == [100.0, 101.0]
    assert second_demo_first.conditions["state"].squeeze(-1).tolist() == [10.0, 10.0, 10.0]
    assert second_demo_first.actions.squeeze(-1).tolist() == [200.0, 201.0]
