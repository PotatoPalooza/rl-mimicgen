"""On-device padded-sequence demo bank for DAPG.

Loads a robomimic-format HDF5 dataset once, chunks each demo into fixed-length
windows, and stores the result as ``(T, N_chunks, D)`` tensors so mini-batches
look exactly like what RSL-RL's recurrent update loop already consumes.

Chunks are non-overlapping full-length windows by default (``stride=seq_length``);
any trailing steps of a demo shorter than ``seq_length`` are dropped. This keeps
every chunk fully valid and lets the actor's RNN run over them without the
``masks``/``unpad_trajectories`` machinery, which is only well-defined when every
trajectory in a batch has the same length.
"""

from __future__ import annotations

import os

import h5py
import numpy as np
import torch
from tensordict import TensorDict


class DemoStorage:
    """Pre-loaded, on-device demo windows in ``(T, N, D)`` layout.

    Args:
        dataset_path: HDF5 file path (robomimic format: ``data/demo_i/{obs/<k>, actions}``).
        obs_keys: ordered list of obs keys whose values get concatenated per timestep.
            Must match the env-side ordering used by the policy.
        seq_length: window size (should match BC-RNN training; default 10).
        stride: step between consecutive window starts. Defaults to ``seq_length``
            (non-overlapping). Use 1 to replicate BC-RNN's training-time sliding
            window (costs ~seq_length× memory).
        filter_key: optional HDF5 ``mask/<key>`` demo-subset filter.
        device: tensor destination.
        dtype: numeric dtype for stored obs/actions (default float32).
    """

    def __init__(
        self,
        dataset_path: str,
        obs_keys: list[str],
        seq_length: int = 10,
        stride: int | None = None,
        filter_key: str | None = None,
        device: str | torch.device = "cuda:0",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"Demo dataset not found: {dataset_path}")

        self.dataset_path = dataset_path
        self.obs_keys = list(obs_keys)
        self.seq_length = int(seq_length)
        self.stride = int(stride) if stride is not None else self.seq_length
        self.device = torch.device(device)
        self.dtype = dtype

        obs_chunks: list[np.ndarray] = []
        act_chunks: list[np.ndarray] = []
        n_demos_used = 0
        n_demos_skipped_short = 0

        with h5py.File(dataset_path, "r") as f:
            if filter_key is not None:
                mask_key = f"mask/{filter_key}"
                if mask_key not in f:
                    raise KeyError(f"filter_key '{filter_key}' not present in {dataset_path}")
                demo_ids = [s.decode() if isinstance(s, bytes) else str(s) for s in f[mask_key][:]]
            else:
                demo_ids = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))

            for demo_id in demo_ids:
                demo = f[f"data/{demo_id}"]
                L = int(demo.attrs.get("num_samples", demo["actions"].shape[0]))
                if L < self.seq_length:
                    n_demos_skipped_short += 1
                    continue

                actions = np.asarray(demo["actions"][:L], dtype=np.float32)
                obs_parts = []
                for k in self.obs_keys:
                    v = np.asarray(demo[f"obs/{k}"][:L], dtype=np.float32)
                    if v.ndim == 1:
                        v = v[:, None]
                    elif v.ndim > 2:
                        v = v.reshape(L, -1)
                    obs_parts.append(v)
                obs_flat = np.concatenate(obs_parts, axis=-1)  # (L, obs_dim)

                last_start = L - self.seq_length
                for start in range(0, last_start + 1, self.stride):
                    end = start + self.seq_length
                    obs_chunks.append(obs_flat[start:end])
                    act_chunks.append(actions[start:end])
                n_demos_used += 1

        if not obs_chunks:
            raise RuntimeError(
                f"No usable demo chunks extracted from {dataset_path} "
                f"(seq_length={self.seq_length}, skipped {n_demos_skipped_short} short demos)."
            )

        # Stack into (T, N, D) with T as the first axis to match RSL-RL's
        # recurrent batch layout (see RolloutStorage.recurrent_mini_batch_generator).
        obs_np = np.stack(obs_chunks, axis=1)   # (T, N, obs_dim)
        act_np = np.stack(act_chunks, axis=1)   # (T, N, action_dim)

        self.demo_obs: torch.Tensor = torch.as_tensor(obs_np, dtype=dtype, device=self.device)
        self.demo_actions: torch.Tensor = torch.as_tensor(act_np, dtype=dtype, device=self.device)

        self.obs_dim: int = int(self.demo_obs.shape[-1])
        self.action_dim: int = int(self.demo_actions.shape[-1])
        self.num_chunks: int = int(self.demo_obs.shape[1])
        self.n_demos_used: int = n_demos_used
        self.n_demos_skipped_short: int = n_demos_skipped_short

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> tuple[TensorDict, torch.Tensor]:
        """Return a random batch of demo windows.

        Returns ``(obs_td, actions)`` where ``obs_td`` is a TensorDict with a
        single ``"policy"`` key shaped ``(T, batch_size, obs_dim)`` and
        ``actions`` is a tensor of shape ``(T, batch_size, action_dim)``.
        """
        idx = torch.randint(0, self.num_chunks, (batch_size,), device=self.device)
        obs = self.demo_obs.index_select(1, idx)    # (T, B, obs_dim)
        actions = self.demo_actions.index_select(1, idx)  # (T, B, action_dim)
        obs_td = TensorDict(
            {"policy": obs},
            batch_size=[self.seq_length, batch_size],
            device=self.device,
        )
        return obs_td, actions

    def memory_mb(self) -> float:
        return (self.demo_obs.numel() + self.demo_actions.numel()) * 4 / (1024 * 1024)
