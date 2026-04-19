"""Direct A/B: identical BC rollout at two different MjSimWarp buffer caps.

Rolls the same BC policy for the same horizon under (njmax, naconmax) =
(500, 15) vs (3500, 60). Reports NaN-divergence count, per-substep average
qpos/qvel magnitude, and partial-task success rates for each.

Purpose: decide whether shrinking the warp buffer caps causes physical
instability (qpos NaN, jitter, explosion) even when no overflow is reported.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _sub in ("robomimic", "robosuite", "mimicgen"):
    sys.path.insert(0, str(_REPO_ROOT / "resources" / _sub))

os.environ["ROBOSUITE_WARP_GRAPH"] = "0"

import robomimic.utils.env_utils as EnvUtils  # noqa: E402
import robomimic.utils.obs_utils as ObsUtils  # noqa: E402
from rsl_rl.models.rnn_model import RNNModel  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from rl_mimicgen.rsl_rl import (  # noqa: E402
    RobomimicVecEnv,
    build_actor_hidden_dims,
    build_distribution_cfg_from_bc,
    copy_bc_weights_into_actor,
    load_bc_checkpoint,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bc_checkpoint", type=str, required=True)
    p.add_argument("--num_envs", type=int, default=50)  # matches BC rollout n=50
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--warmup_steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_actor(bc_info, obs_td, obs_groups, device):
    hidden_dims, activation = build_actor_hidden_dims(bc_info)
    dist_cfg = build_distribution_cfg_from_bc(bc_info, gaussian_init_std=0.05)
    actor = RNNModel(
        obs=obs_td, obs_groups=obs_groups, obs_set="actor",
        output_dim=bc_info.action_dim, hidden_dims=hidden_dims,
        activation=activation, obs_normalization=False,
        distribution_cfg=dist_cfg,
        rnn_type=bc_info.rnn_type,
        rnn_hidden_dim=bc_info.rnn_hidden_dim,
        rnn_num_layers=bc_info.rnn_num_layers,
    )
    loaded, skipped = copy_bc_weights_into_actor(actor, bc_info)
    if skipped:
        print(f"[WARN] skipped: {skipped}", file=sys.stderr)
    actor.to(device)
    actor.eval()
    return actor


@torch.no_grad()
def run_one(bc_info, njmax: int, naconmax: int, args, device) -> dict:
    # Force buffer caps via env_meta.env_kwargs.
    env_meta = dict(bc_info.env_meta)
    env_meta["env_kwargs"] = dict(env_meta.get("env_kwargs", {}))
    env_meta["env_kwargs"]["njmax_per_env"] = int(njmax)
    env_meta["env_kwargs"]["naconmax_per_env"] = int(naconmax)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=False,
        use_image_obs=False, use_warp=True, num_envs=args.num_envs,
    )
    vec_env = RobomimicVecEnv(
        env=env, horizon=args.horizon, device="cuda:0",
        obs_keys=bc_info.obs_keys,
    )
    obs_td = vec_env.get_observations()
    actor = build_actor(bc_info, obs_td, {"actor": ["policy"], "critic": ["policy"]}, device)

    for _ in range(args.warmup_steps):
        action = actor.forward(obs_td, stochastic_output=False)
        obs_td, _, dones, _ = vec_env.step(action)
        actor.reset(dones.bool())

    obs_td, _ = vec_env.reset()
    actor.reset(torch.ones(args.num_envs, dtype=torch.bool, device=device))

    ever_succ = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    partial_ever: dict[str, torch.Tensor] = {}
    partial_fn = getattr(vec_env.env.env, "_get_partial_task_metrics", None)
    # Track mean abs qvel across the rollout as an "explosion" signal.
    qvel_mean_per_step: list[float] = []
    qvel_max_per_step: list[float] = []

    sim = vec_env.env.env.sim
    for _ in range(args.horizon):
        action = actor.forward(obs_td, stochastic_output=False)
        obs_td, _, dones, _ = vec_env.step(action)
        sim = vec_env.env.env.sim
        qvel = sim._warp_data.qvel.numpy()  # (nworld, nv)
        qvel_mean_per_step.append(float(np.nanmean(np.abs(qvel))))
        qvel_max_per_step.append(float(np.nanmax(np.abs(qvel))))
        succ = vec_env.env.is_success().get("task")
        if succ is not None:
            if not isinstance(succ, torch.Tensor):
                succ = torch.as_tensor(np.asarray(succ), device=device)
            ever_succ |= succ.to(device).bool().view(-1)
        if partial_fn is not None:
            try:
                metrics = partial_fn()
            except Exception:
                metrics = None
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    if not (isinstance(v, torch.Tensor) and v.shape[0] == args.num_envs):
                        continue
                    if k not in partial_ever:
                        partial_ever[k] = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
                    partial_ever[k] |= v.to(device).bool()
        actor.reset(dones.bool())

    result = {
        "caps": {"njmax_per_env": njmax, "naconmax_per_env": naconmax},
        "num_envs": args.num_envs,
        "horizon": args.horizon,
        "nan_total": int(vec_env._nan_total),
        "success_rate": float(ever_succ.float().mean().item()),
        "partial_rates": {k: float(v.float().mean().item()) for k, v in partial_ever.items()},
        "qvel_abs_mean_overall": float(np.mean(qvel_mean_per_step)),
        "qvel_abs_max_overall": float(np.max(qvel_max_per_step)),
        "qvel_abs_mean_p95_step": float(np.percentile(qvel_mean_per_step, 95)),
    }
    vec_env.close()
    return result


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")

    bc_info = load_bc_checkpoint(args.bc_checkpoint)
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=[bc_info.obs_modalities])

    results = []
    for njmax, naconmax in [(3500, 60), (500, 15)]:
        print(f"\n===== caps njmax={njmax} naconmax={naconmax} =====", flush=True)
        r = run_one(bc_info, njmax, naconmax, args, device)
        results.append(r)
        print(json.dumps(r, indent=2))

    print("\n========== A/B SUMMARY ==========")
    fmt = "{:<20} {:>12} {:>12}"
    print(fmt.format("metric", "3500/60", "500/15"))
    print("-" * 48)
    keys = [
        ("nan_total", "nan_total"),
        ("success_rate", "success_rate"),
        ("qvel |mean|", "qvel_abs_mean_overall"),
        ("qvel |max|", "qvel_abs_max_overall"),
    ]
    for label, key in keys:
        print(fmt.format(label, f"{results[0][key]:.4g}", f"{results[1][key]:.4g}"))
    print("\npartial rates:")
    all_keys = set()
    for r in results:
        all_keys |= set(r["partial_rates"].keys())
    for k in sorted(all_keys):
        a = results[0]["partial_rates"].get(k, float("nan"))
        b = results[1]["partial_rates"].get(k, float("nan"))
        print(fmt.format(k, f"{a:.3f}", f"{b:.3f}"))


if __name__ == "__main__":
    main()
