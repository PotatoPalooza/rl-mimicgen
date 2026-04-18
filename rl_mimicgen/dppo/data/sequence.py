from __future__ import annotations

from collections import namedtuple

import numpy as np
import torch

from rl_mimicgen.dppo.data.dataset import DPPODatasetBundle


DPPODiffusionBatch = namedtuple("DPPODiffusionBatch", "actions conditions")


def _normalize_to_unit_interval(values: np.ndarray, min_value: np.ndarray, value_range: np.ndarray) -> np.ndarray:
    return (2.0 * (values - min_value) / value_range) - 1.0


class DPPODiffusionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        bundle: DPPODatasetBundle,
        horizon_steps: int,
        cond_steps: int,
        normalize: bool = True,
        device: str = "cpu",
        max_demos: int | None = None,
    ) -> None:
        self.bundle = bundle
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps
        self.normalize = normalize
        self.device = torch.device(device)

        traj_lengths = bundle.traj_lengths[:max_demos] if max_demos is not None else bundle.traj_lengths
        demo_offsets = bundle.demo_offsets[: len(traj_lengths)]
        total_steps = int(traj_lengths.sum())

        obs = bundle.obs[:total_steps]
        actions = bundle.actions[:total_steps]
        if normalize:
            obs = _normalize_to_unit_interval(
                obs,
                bundle.normalization.obs_min,
                bundle.normalization.obs_range,
            )
            actions = _normalize_to_unit_interval(
                actions,
                bundle.normalization.action_min,
                bundle.normalization.action_range,
            )

        self.obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        self.indices = self._make_indices(traj_lengths=traj_lengths, demo_offsets=demo_offsets)

    def _make_indices(self, traj_lengths: np.ndarray, demo_offsets: np.ndarray) -> list[tuple[int, int]]:
        indices: list[tuple[int, int]] = []
        for traj_length, (traj_start, _) in zip(traj_lengths, demo_offsets.tolist(), strict=True):
            max_start = int(traj_start + traj_length - self.horizon_steps)
            indices.extend((start, start - int(traj_start)) for start in range(int(traj_start), max_start + 1))
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> DPPODiffusionBatch:
        start, num_before_start = self.indices[index]
        end = start + self.horizon_steps
        actions = self.actions[start:end]
        obs_history = self.obs[(start - num_before_start) : (start + 1)]
        state = torch.stack(
            [obs_history[max(num_before_start - t, 0)] for t in reversed(range(self.cond_steps))]
        )
        return DPPODiffusionBatch(actions=actions, conditions={"state": state})
