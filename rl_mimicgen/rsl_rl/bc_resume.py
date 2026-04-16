"""Load a robomimic BC-RNN checkpoint (local or W&B) for RSL-RL warm-start.

The extracted :class:`BCResumeInfo` provides everything needed to configure an
RSL-RL ``RNNModel`` so that the BC-RNN backbone can be copied in:

* ``rnn_type``, ``rnn_hidden_dim``, ``rnn_num_layers`` — match BC's LSTM
* ``obs_keys`` — ordered obs list (BC concatenates obs in ``modalities.obs.low_dim`` order)
* ``env_meta`` — passed to ``EnvUtils.create_env_from_metadata``

The GMM head is intentionally not copied; the RL policy's Gaussian head is
learned from scratch on top of the pre-trained recurrent backbone.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch


@dataclass
class BCResumeInfo:
    rnn_type: str
    rnn_hidden_dim: int
    rnn_num_layers: int
    obs_keys: list[str]
    obs_dim: int
    action_dim: int
    env_meta: dict
    ckpt_path: str
    state_dict: dict[str, torch.Tensor]
    obs_modalities: dict
    """``config.observation.modalities`` dict from the BC run; needed to prime
    ``robomimic.utils.obs_utils`` before creating the env."""


def _parse_config(raw) -> dict:
    return json.loads(raw) if isinstance(raw, str) else raw


def fetch_wandb_checkpoint(run_ref: str, model_name: str, download_dir: str) -> str:
    """Download ``model_name`` from the given ``entity/project/run_id`` and
    return the local file path."""
    import wandb

    os.makedirs(download_dir, exist_ok=True)
    api = wandb.Api()
    run = api.run(run_ref)
    run_files = list(run.files())
    target = None
    for f in run_files:
        if os.path.basename(f.name) == model_name:
            target = f
            break
    if target is None:
        available = [f.name for f in run_files]
        raise FileNotFoundError(
            f"Model '{model_name}' not found in wandb run '{run_ref}'. "
            f"Available files: {available}"
        )
    downloaded = target.download(root=download_dir, replace=True)
    return os.path.abspath(downloaded.name)


def load_bc_checkpoint(path: str) -> BCResumeInfo:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = _parse_config(ckpt["config"])
    rnn_cfg = cfg["algo"]["rnn"]
    if not rnn_cfg["enabled"]:
        raise ValueError(
            f"Checkpoint {path} is not a BC-RNN policy (rnn.enabled=False). "
            "Only BC-RNN checkpoints can seed the RSL-RL RNN actor."
        )

    obs_keys = list(cfg["observation"]["modalities"]["obs"]["low_dim"])
    shape_meta = ckpt["shape_metadata"]
    obs_dim = 0
    for k in obs_keys:
        if k not in shape_meta["all_shapes"]:
            raise ValueError(
                f"BC obs key '{k}' missing from shape_metadata.all_shapes "
                f"(have: {list(shape_meta['all_shapes'])})"
            )
        obs_dim += int(shape_meta["all_shapes"][k][0])

    return BCResumeInfo(
        rnn_type=str(rnn_cfg["rnn_type"]).lower(),
        rnn_hidden_dim=int(rnn_cfg["hidden_dim"]),
        rnn_num_layers=int(rnn_cfg["num_layers"]),
        obs_keys=obs_keys,
        obs_dim=obs_dim,
        action_dim=int(shape_meta["ac_dim"]),
        env_meta=ckpt["env_metadata"],
        ckpt_path=path,
        state_dict=ckpt["model"]["nets"],
        obs_modalities=cfg["observation"]["modalities"],
    )


def copy_rnn_weights_into_actor(actor_model, bc_state_dict: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
    """Copy BC-RNN backbone weights into an RSL-RL ``RNNModel`` actor.

    Only parameters under ``policy.nets.rnn.nets.*`` are transferred; shapes
    must match (i.e. obs_dim, hidden_dim, num_layers align with BC). Returns
    ``(loaded_keys, skipped_keys)``.
    """
    src_prefix = "policy.nets.rnn.nets."
    rnn = actor_model.rnn.rnn
    target = rnn.state_dict()
    loaded: list[str] = []
    skipped: list[str] = []
    for k, tgt_tensor in target.items():
        src_k = src_prefix + k
        src = bc_state_dict.get(src_k)
        if src is None or src.shape != tgt_tensor.shape:
            skipped.append(k)
            continue
        target[k] = src.clone()
        loaded.append(k)
    rnn.load_state_dict(target)
    return loaded, skipped
