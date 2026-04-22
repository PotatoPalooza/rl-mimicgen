"""Load a robomimic BC-RNN checkpoint (local or W&B) for RSL-RL warm-start.

The extracted :class:`BCResumeInfo` carries everything needed to shape an
RSL-RL ``RNNModel`` so the entire BC policy -- RNN backbone, MLP trunk, and
distribution head -- can be copied over. That lets the RL actor start with a
policy identical (or near-identical) to the BC one, rather than just the
recurrent backbone.

Supports:

* ``BC_RNN`` -- MLP trunk + deterministic action head (tanh-squashed). Maps onto
  :class:`rl_mimicgen.rsl_rl.distributions.TanhGaussianDistribution`.
* ``BC_RNN_GMM`` -- MLP trunk + Mixture-of-Gaussians head. Maps onto
  :class:`rl_mimicgen.rsl_rl.distributions.GMMDistribution`.

Refuses transformer-based BC policies (``BC_Transformer*``) since RSL-RL's
``RNNModel`` can't consume those weights.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import torch


@dataclass
class BCResumeInfo:
    # -- RNN backbone --
    rnn_type: str
    rnn_hidden_dim: int
    rnn_num_layers: int

    # -- MLP trunk (between RNN and head). ``[]`` means no trunk -- rare but legal. --
    actor_layer_dims: list[int]

    # -- Obs / action shapes + wiring --
    obs_keys: list[str]
    obs_dim: int
    action_dim: int
    env_meta: dict
    obs_modalities: dict
    """``config.observation.modalities`` dict from the BC run; needed to prime
    ``robomimic.utils.obs_utils`` before creating the env."""

    # -- Head type + its hyperparameters --
    algo_name: str  # e.g. "bc", "bc_rnn", "bc_rnn_gmm"
    is_gmm: bool
    gmm_kwargs: dict = field(default_factory=dict)
    """Only populated when ``is_gmm``. Keys: ``num_modes``, ``min_std``,
    ``std_activation``, ``low_noise_eval``, ``use_tanh``. Matches the kwargs
    robomimic passes to ``RNNGMMActorNetwork``."""

    # -- Checkpoint state + provenance --
    ckpt_path: str = ""
    state_dict: dict[str, torch.Tensor] = field(default_factory=dict)

    # -- DAPG demo-data defaults (pulled from the BC training config) --
    dataset_path: str | None = None
    """Path to the HDF5 demonstrations the BC policy was trained on (first entry
    of ``config.train.data``). Used by DAPG to load demo transitions. ``None`` if
    the BC config's ``train.data`` field was empty or malformed."""
    seq_length: int = 10
    """BC-RNN training sequence length (``config.train.seq_length``), used as a
    default for DAPG demo chunking when not overridden."""


def _extract_dataset_path(cfg: dict) -> str | None:
    data = cfg.get("train", {}).get("data")
    if isinstance(data, str):
        return data
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            p = first.get("path")
            return p if isinstance(p, str) else None
    return None


def _parse_config(raw: str | dict) -> dict:
    return json.loads(raw) if isinstance(raw, str) else raw


def fetch_wandb_checkpoint(run_ref: str, model_name: str, download_dir: str) -> str:
    """Download ``model_name`` from the given ``entity/project/run_id``."""
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
    algo_name = str(cfg.get("algo_name", "")).lower()

    # We only support BC policies whose forward pass is LSTM -> MLP -> head.
    # Transformer / Diffusion BC variants don't fit the RSL-RL RNNModel shape.
    if "transformer" in algo_name:
        raise ValueError(
            f"BC checkpoint {path} uses algo='{algo_name}' (transformer-based). "
            "RSL-RL's RNNModel can only warm-start from LSTM-based BC policies "
            "(BC_RNN or BC_RNN_GMM). Retrain with algo.rnn.enabled=True."
        )
    if not cfg.get("algo", {}).get("rnn", {}).get("enabled", False):
        raise ValueError(
            f"Checkpoint {path} is not a BC-RNN policy (algo.rnn.enabled=False). "
            "Only recurrent BC checkpoints can seed the RSL-RL RNN actor."
        )

    rnn_cfg = cfg["algo"]["rnn"]
    actor_layer_dims = [int(d) for d in cfg["algo"].get("actor_layer_dims", []) or []]

    gmm_cfg = cfg["algo"].get("gmm", {}) or {}
    is_gmm = bool(gmm_cfg.get("enabled", False))
    gmm_kwargs: dict = {}
    if is_gmm:
        gmm_kwargs = {
            "num_modes": int(gmm_cfg.get("num_modes", 5)),
            "min_std": float(gmm_cfg.get("min_std", 1e-4)),
            "std_activation": str(gmm_cfg.get("std_activation", "softplus")),
            "low_noise_eval": bool(gmm_cfg.get("low_noise_eval", True)),
            "use_tanh": bool(gmm_cfg.get("use_tanh", False)),
        }

    # Order must match shape_metadata.all_shapes iteration (BC's encoder concat order),
    # not config.observation.modalities.obs.low_dim -- otherwise warm-start permutes inputs.
    shape_meta = ckpt["shape_metadata"]
    low_dim_set = set(cfg["observation"]["modalities"]["obs"]["low_dim"])
    obs_keys = [k for k in shape_meta["all_shapes"] if k in low_dim_set]
    missing = low_dim_set - set(obs_keys)
    if missing:
        raise ValueError(
            f"BC obs keys {sorted(missing)} are in config.low_dim but missing "
            f"from shape_metadata.all_shapes (have: {list(shape_meta['all_shapes'])})"
        )
    obs_dim = sum(int(shape_meta["all_shapes"][k][0]) for k in obs_keys)

    return BCResumeInfo(
        rnn_type=str(rnn_cfg["rnn_type"]).lower(),
        rnn_hidden_dim=int(rnn_cfg["hidden_dim"]),
        rnn_num_layers=int(rnn_cfg["num_layers"]),
        actor_layer_dims=actor_layer_dims,
        obs_keys=obs_keys,
        obs_dim=obs_dim,
        action_dim=int(shape_meta["ac_dim"]),
        env_meta=ckpt["env_metadata"],
        obs_modalities=cfg["observation"]["modalities"],
        algo_name=algo_name,
        is_gmm=is_gmm,
        gmm_kwargs=gmm_kwargs,
        ckpt_path=path,
        state_dict=ckpt["model"]["nets"],
        dataset_path=_extract_dataset_path(cfg),
        seq_length=int(cfg.get("train", {}).get("seq_length", 10) or 10),
    )


def build_actor_hidden_dims(bc_info: BCResumeInfo) -> tuple[list[int], str]:
    """Return ``(hidden_dims, activation)`` for the RSL-RL actor MLP.

    Mirrors BC's trunk layout so weight transfer is a direct mapping. When BC
    has no trunk (``actor_layer_dims=[]``), we can't set ``hidden_dims=[]``
    because RSL-RL's ``MLPModel`` requires at least one hidden layer, so we
    pad with a single layer of size ``rnn_hidden_dim`` and use identity
    activation. :func:`copy_bc_weights_into_actor` then initializes that
    padded layer as a pass-through (identity weight, zero bias), making the
    overall network behavior identical to BC's single-Linear decoder path.
    """
    if bc_info.actor_layer_dims:
        return list(bc_info.actor_layer_dims), "relu"
    return [bc_info.rnn_hidden_dim], "identity"


def build_distribution_cfg_from_bc(bc_info: BCResumeInfo, gaussian_init_std: float = 0.05) -> dict:
    """Return the ``actor.distribution_cfg`` dict that reproduces the BC head.

    * ``BC_RNN_GMM`` -> :class:`GMMDistribution` seeded with BC's GMM kwargs.
    * ``BC_RNN`` (non-GMM) -> :class:`TanhGaussianDistribution` with a small
      state-independent std. Matches BC's forward-time ``tanh(decoder_output)``;
      std is state-independent because BC is deterministic at train time.
    """
    if bc_info.is_gmm:
        return {
            "class_name": "rl_mimicgen.rsl_rl.modules.distributions:GMMDistribution",
            **bc_info.gmm_kwargs,
        }
    return {
        "class_name": "rl_mimicgen.rsl_rl.modules.distributions:TanhGaussianDistribution",
        "init_std": float(gaussian_init_std),
        "std_type": "scalar",
    }


def _copy_rnn(actor_rnn: torch.nn.Module, bc_sd: dict, loaded: list, skipped: list) -> None:
    """Copy BC's nn.LSTM state_dict under ``policy.nets.rnn.nets.*`` into RSL-RL's."""
    src_prefix = "policy.nets.rnn.nets."
    target = actor_rnn.state_dict()
    for k, tgt in target.items():
        src = bc_sd.get(src_prefix + k)
        if src is None or src.shape != tgt.shape:
            skipped.append(f"rnn.{k}")
            continue
        target[k] = src.clone()
        loaded.append(f"rnn.{k}")
    actor_rnn.load_state_dict(target)


def _copy_mlp_trunk(actor_mlp: torch.nn.Module, bc_sd: dict, bc_num_trunk: int, actor_num_trunk: int, loaded: list, skipped: list) -> None:
    """Copy BC's MLP trunk into RSL-RL's MLP, padding with identity layers if needed.

    RSL-RL's MLP is a ``Sequential`` with Linears at even indices. We copy
    BC's ``bc_num_trunk`` Linears into the corresponding RSL-RL indices. When
    ``actor_num_trunk > bc_num_trunk`` (BC has no trunk but RSL-RL needs a
    pad layer), the extra Linears are initialized as pass-throughs: identity
    weight + zero bias. This keeps the initial network output bit-exact to
    BC when paired with ``activation="identity"``.
    """
    import torch.nn as nn
    target = actor_mlp.state_dict()

    # BC -> RSL-RL direct copy for the shared trunk layers.
    for i in range(bc_num_trunk):
        idx = 2 * i
        for suffix in ("weight", "bias"):
            key = f"{idx}.{suffix}"
            src_key = f"policy.nets.mlp._model.{idx}.{suffix}"
            src = bc_sd.get(src_key)
            tgt = target.get(key)
            if src is None or tgt is None or src.shape != tgt.shape:
                skipped.append(
                    f"mlp.{key} (src_shape={None if src is None else tuple(src.shape)}, "
                    f"tgt_shape={None if tgt is None else tuple(tgt.shape)})"
                )
                continue
            target[key] = src.clone()
            loaded.append(f"mlp.{key}")

    # Identity-init the padded layers (only reached when BC has no trunk).
    for i in range(bc_num_trunk, actor_num_trunk):
        idx = 2 * i
        w_key, b_key = f"{idx}.weight", f"{idx}.bias"
        tgt_w = target.get(w_key)
        tgt_b = target.get(b_key)
        if tgt_w is None or tgt_b is None:
            skipped.append(f"mlp.{w_key} (pad layer missing in target)")
            continue
        if tgt_w.shape[0] != tgt_w.shape[1]:
            skipped.append(
                f"mlp.{w_key} (pad layer not square {tuple(tgt_w.shape)} -- can't identity-init)"
            )
            continue
        target[w_key] = torch.eye(tgt_w.shape[0], dtype=tgt_w.dtype)
        target[b_key] = torch.zeros_like(tgt_b)
        loaded.append(f"mlp.{w_key} (identity pad)")

    actor_mlp.load_state_dict(target)


def _copy_head(actor_mlp: torch.nn.Module, bc_sd: dict, bc_info: BCResumeInfo, actor_num_trunk: int, loaded: list, skipped: list) -> None:
    """Copy BC's decoder head into RSL-RL's final MLP Linear layer.

    RSL-RL builds a single Linear(hN, flat_out) as the final MLP layer. BC's
    decoder is a ``ModuleDict`` of separate Linear heads -- equivalent to a
    single Linear with rows concatenated along the output dim, which is what
    :class:`GMMDistribution` expects. For non-GMM, there's just one decoder
    head (``action``) that maps straight onto the final Linear.
    """
    # Index of the final Linear in RSL-RL's MLP: 2N where N = actor_num_trunk.
    final_idx = 2 * actor_num_trunk
    target = actor_mlp.state_dict()
    tgt_w = target.get(f"{final_idx}.weight")
    tgt_b = target.get(f"{final_idx}.bias")
    if tgt_w is None or tgt_b is None:
        skipped.append(f"head.{final_idx} (missing in target)")
        return

    if bc_info.is_gmm:
        # Stack order MUST match GMMDistribution._split: [mean, scale, logits].
        keys = ("mean", "scale", "logits")
    else:
        keys = ("action",)

    try:
        src_ws = [bc_sd[f"policy.nets.decoder.nets.{k}.weight"] for k in keys]
        src_bs = [bc_sd[f"policy.nets.decoder.nets.{k}.bias"] for k in keys]
    except KeyError as e:
        skipped.append(f"head (missing BC key {e})")
        return

    # Flatten each head's output axis (BC uses ``out.reshape(-1, num_modes, ac_dim)``
    # after the linear; weights are already shape ``(num_modes*ac_dim, hN)``).
    cat_w = torch.cat([w.reshape(-1, w.shape[-1]) for w in src_ws], dim=0)
    cat_b = torch.cat([b.reshape(-1) for b in src_bs], dim=0)

    if cat_w.shape != tgt_w.shape or cat_b.shape != tgt_b.shape:
        skipped.append(
            f"head.{final_idx} (shape mismatch: src=({tuple(cat_w.shape)},{tuple(cat_b.shape)}) "
            f"vs tgt=({tuple(tgt_w.shape)},{tuple(tgt_b.shape)}))"
        )
        return

    target[f"{final_idx}.weight"] = cat_w.clone()
    target[f"{final_idx}.bias"] = cat_b.clone()
    actor_mlp.load_state_dict(target)
    loaded.extend([f"head.{k}" for k in keys])


def reset_gmm_scale_head(
    actor_model: torch.nn.Module,
    bc_info: BCResumeInfo,
    target_std: float,
) -> None:
    """Re-initialize the GMM scale head to produce ``target_std`` per-component at init.

    GMM has no state-independent std; its component scales are MLP outputs
    ``softplus(raw) + min_std``. For exploration-noise control at RL start, we
    zero the scale-head weights (state-independent at init) and set the scale-
    head bias to ``log(exp(target_std - min_std) - 1)`` so ``softplus(bias) +
    min_std == target_std``. Weights remain ``nn.Parameter``s; gradient descent
    re-learns state-dependent scale as training proceeds.

    Must be called AFTER :func:`copy_bc_weights_into_actor`, otherwise BC's
    scale-head copy overwrites this reset.
    """
    if not bc_info.is_gmm:
        raise ValueError("reset_gmm_scale_head only valid for GMM BC heads.")

    num_modes = int(bc_info.gmm_kwargs.get("num_modes", 5))
    min_std = float(bc_info.gmm_kwargs.get("min_std", 1e-4))
    ac_dim = int(bc_info.action_dim)
    if target_std <= min_std:
        raise ValueError(f"target_std={target_std} must exceed min_std={min_std}.")

    # Locate final Linear (head) in actor_model.mlp: last Linear in the module list.
    final_linear: torch.nn.Linear | None = None
    for m in actor_model.mlp:
        if isinstance(m, torch.nn.Linear):
            final_linear = m
    if final_linear is None:
        raise RuntimeError("reset_gmm_scale_head: no Linear found in actor.mlp")

    # Scale slice in the stacked final-layer output: rows [n_modes*ac_dim : 2*n_modes*ac_dim].
    s0 = num_modes * ac_dim
    s1 = 2 * num_modes * ac_dim
    raw = math.log(math.exp(target_std - min_std) - 1.0)
    with torch.no_grad():
        final_linear.weight[s0:s1, :].zero_()
        final_linear.bias[s0:s1].fill_(raw)


def copy_bc_weights_into_actor(
    actor_model: torch.nn.Module,
    bc_info: BCResumeInfo,
) -> tuple[list[str], list[str]]:
    """Copy the full BC policy (RNN + MLP trunk + head) into an RSL-RL ``RNNModel``.

    Requires that the RSL-RL actor was built with ``hidden_dims`` and
    ``activation`` from :func:`build_actor_hidden_dims` and a distribution
    config from :func:`build_distribution_cfg_from_bc`. When BC has no trunk
    (``actor_layer_dims=[]``), the padded passthrough Linear is identity-init'd
    so the combined network reproduces BC's decoder-from-RNN-output path.

    Returns ``(loaded_keys, skipped_keys)`` for diagnostics. If the final head
    shape doesn't match, the head copy is skipped rather than raising -- the
    RNN + trunk still transfer.
    """
    loaded: list[str] = []
    skipped: list[str] = []
    bc_sd = bc_info.state_dict

    # Infer actor trunk size from the MLP structure (Linears at even indices,
    # minus the final Linear which is the head).
    actor_linears = sum(1 for m in actor_model.mlp if isinstance(m, torch.nn.Linear))
    actor_num_trunk = max(0, actor_linears - 1)
    bc_num_trunk = len(bc_info.actor_layer_dims)

    _copy_rnn(actor_model.rnn.rnn, bc_sd, loaded, skipped)
    _copy_mlp_trunk(actor_model.mlp, bc_sd, bc_num_trunk, actor_num_trunk, loaded, skipped)
    _copy_head(actor_model.mlp, bc_sd, bc_info, actor_num_trunk, loaded, skipped)
    return loaded, skipped


def copy_rnn_weights_into_actor(actor_model: torch.nn.Module, bc_state_dict: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
    """Backwards-compatible RNN-only copy.

    Prefer :func:`copy_bc_weights_into_actor`, which transfers the full BC
    policy. This helper is kept for callers that only need the recurrent
    backbone (e.g. when the MLP / head shapes intentionally don't match).
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
