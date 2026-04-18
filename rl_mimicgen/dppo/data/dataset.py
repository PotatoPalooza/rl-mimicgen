from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class DPPONormalizationStats:
    obs_mean: np.ndarray
    obs_std: np.ndarray
    obs_min: np.ndarray
    obs_max: np.ndarray
    obs_range: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    action_min: np.ndarray
    action_max: np.ndarray
    action_range: np.ndarray

    @classmethod
    def load(cls, bundle_dir: str | Path) -> "DPPONormalizationStats":
        stats = np.load(Path(bundle_dir) / "normalization.npz")
        return cls(
            obs_mean=np.asarray(stats["obs_mean"], dtype=np.float32),
            obs_std=np.asarray(stats["obs_std"], dtype=np.float32),
            obs_min=np.asarray(stats["obs_min"], dtype=np.float32),
            obs_max=np.asarray(stats["obs_max"], dtype=np.float32),
            obs_range=np.asarray(stats["obs_range"], dtype=np.float32),
            action_mean=np.asarray(stats["action_mean"], dtype=np.float32),
            action_std=np.asarray(stats["action_std"], dtype=np.float32),
            action_min=np.asarray(stats["action_min"], dtype=np.float32),
            action_max=np.asarray(stats["action_max"], dtype=np.float32),
            action_range=np.asarray(stats["action_range"], dtype=np.float32),
        )

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return (2.0 * (obs - self.obs_min) / self.obs_range) - 1.0

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return (2.0 * (action - self.action_min) / self.action_range) - 1.0

    def unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        return ((action + 1.0) * 0.5 * self.action_range) + self.action_min


@dataclass(frozen=True, slots=True)
class DPPODatasetBundle:
    bundle_dir: Path
    metadata: dict
    obs: np.ndarray
    next_obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    traj_lengths: np.ndarray
    demo_offsets: np.ndarray
    normalization: DPPONormalizationStats

    @classmethod
    def load(cls, bundle_dir: str | Path) -> "DPPODatasetBundle":
        bundle_path = Path(bundle_dir)
        dataset = np.load(bundle_path / "dataset.npz")
        with open(bundle_path / "metadata.json", "r", encoding="utf-8") as file_obj:
            metadata = json.load(file_obj)
        return cls(
            bundle_dir=bundle_path,
            metadata=metadata,
            obs=np.asarray(dataset["obs"], dtype=np.float32),
            next_obs=np.asarray(dataset["next_obs"], dtype=np.float32),
            actions=np.asarray(dataset["actions"], dtype=np.float32),
            rewards=np.asarray(dataset["rewards"], dtype=np.float32),
            dones=np.asarray(dataset["dones"], dtype=np.float32),
            traj_lengths=np.asarray(dataset["traj_lengths"], dtype=np.int32),
            demo_offsets=np.asarray(dataset["demo_offsets"], dtype=np.int64),
            normalization=DPPONormalizationStats.load(bundle_path),
        )

    @property
    def obs_dim(self) -> int:
        return int(self.obs.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    def summary(self) -> str:
        return (
            f"bundle={self.bundle_dir} task={self.metadata['task']} variant={self.metadata['variant']} "
            f"demos={self.metadata['num_demos']} transitions={self.metadata['num_transitions']} "
            f"obs_dim={self.obs_dim} action_dim={self.action_dim}"
        )

    def flatten_obs(self, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
        chunks = []
        for key in self.metadata["obs_keys"]:
            value = np.asarray(obs_dict[key], dtype=np.float32).reshape(-1)
            chunks.append(value)
        return np.concatenate(chunks, axis=0)
