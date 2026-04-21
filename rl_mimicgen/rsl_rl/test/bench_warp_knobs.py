"""Benchmark warp-sim speed-vs-accuracy knobs by rolling out a BC policy.

For a given knob configuration (set via env vars read by
``robosuite.utils.binding_utils.MjSimWarp`` + a CLI flag for the physics
timestep), this script:

* loads a BC-RNN checkpoint,
* builds a warp vec-env at ``num_envs`` (default 4096),
* rolls the (deterministic) BC policy for ``horizon`` steps,
* times each ``env.step`` (cuda-synced), and
* records each env's "ever succeeded" flag during the rollout.

Prints a one-line JSON result. Designed to be run once per configuration
(separate subprocess) by ``bench_run.py``.
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

import robomimic.utils.env_utils as EnvUtils  # noqa: E402
import robomimic.utils.obs_utils as ObsUtils  # noqa: E402
import robosuite.macros as rs_macros  # noqa: E402
from rsl_rl.models.rnn_model import RNNModel  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from rl_mimicgen.rsl_rl import (  # noqa: E402
    RobomimicVecEnv,
    build_actor_hidden_dims,
    build_distribution_cfg_from_bc,
    copy_bc_weights_into_actor,
    load_bc_checkpoint,
)
from rl_mimicgen.rsl_rl.warp_buffer_sizes import resolve_warp_buffer_sizes  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bc_checkpoint", type=str, required=True)
    p.add_argument("--num_envs", type=int, default=4096)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--warmup_steps", type=int, default=10,
                   help="Untimed steps before timing begins (JIT, first-call warp compiles).")
    p.add_argument("--physics_timestep", type=float, default=None,
                   help="If set, override robosuite.macros.SIMULATION_TIMESTEP "
                        "BEFORE creating the env. Default 0.002.")
    p.add_argument("--config_name", type=str, default="baseline")
    p.add_argument("--output_json", type=str, default=None,
                   help="Append result as a JSON line to this file.")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_actor_from_bc(bc_info, obs_td: TensorDict, obs_groups: dict, device: torch.device) -> RNNModel:
    hidden_dims, activation = build_actor_hidden_dims(bc_info)
    dist_cfg = build_distribution_cfg_from_bc(bc_info, gaussian_init_std=0.05)
    actor = RNNModel(
        obs=obs_td,
        obs_groups=obs_groups,
        obs_set="actor",
        output_dim=bc_info.action_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        obs_normalization=False,
        distribution_cfg=dist_cfg,
        rnn_type=bc_info.rnn_type,
        rnn_hidden_dim=bc_info.rnn_hidden_dim,
        rnn_num_layers=bc_info.rnn_num_layers,
    )
    loaded, skipped = copy_bc_weights_into_actor(actor, bc_info)
    if skipped:
        print(f"[WARN] BC warm-start skipped: {skipped}", file=sys.stderr)
    print(f"[INFO] BC warm-start: copied {len(loaded)} tensors", file=sys.stderr)
    actor.to(device)
    actor.eval()
    return actor


@torch.no_grad()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    if args.physics_timestep is not None:
        rs_macros.SIMULATION_TIMESTEP = float(args.physics_timestep)
        print(f"[INFO] SIMULATION_TIMESTEP overridden -> {rs_macros.SIMULATION_TIMESTEP}", file=sys.stderr)

    bc_info = load_bc_checkpoint(args.bc_checkpoint)
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=[bc_info.obs_modalities])

    env_meta = bc_info.env_meta
    warp_caps = resolve_warp_buffer_sizes(env_meta.get("env_name")) or {}
    if warp_caps:
        env_meta.setdefault("env_kwargs", {}).update(warp_caps)

    print(
        f"[INFO] Creating warp env: num_envs={args.num_envs} horizon={args.horizon} "
        f"env={env_meta.get('env_name')}  knobs={{"
        f"TOL_CLAMP={os.environ.get('ROBOSUITE_WARP_TOLERANCE_CLAMP','0')} "
        f"SOLVER_ITERS={os.environ.get('ROBOSUITE_WARP_SOLVER_ITERS','default')} "
        f"LS_ITERS={os.environ.get('ROBOSUITE_WARP_LS_ITERS','default')} "
        f"CONE={os.environ.get('ROBOSUITE_WARP_CONE','xml-default')} "
        f"GRAPH={os.environ.get('ROBOSUITE_WARP_GRAPH','0')} "
        f"TIMESTEP={rs_macros.SIMULATION_TIMESTEP} "
        f"CAPS={warp_caps or '(class defaults)'}}}",
        file=sys.stderr,
    )
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
        use_warp=True,
        num_envs=args.num_envs,
    )
    vec_env = RobomimicVecEnv(
        env=env,
        horizon=args.horizon,
        device=args.device,
        obs_keys=bc_info.obs_keys,
    )

    obs_td = vec_env.get_observations()
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = build_actor_from_bc(bc_info, obs_td, obs_groups, device)

    ever_succeeded = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    partial_fn = getattr(vec_env.env.env, "_get_partial_task_metrics", None)
    partial_ever: dict[str, torch.Tensor] = {}

    # --- warmup (untimed): compile warp kernels + prime caches ---
    for _ in range(args.warmup_steps):
        action = actor.forward(obs_td, stochastic_output=False)
        obs_td, _, dones, _ = vec_env.step(action)
        actor.reset(dones.bool())

    # Full reset so the timed rollout starts from a clean slate for all envs.
    obs_td, _ = vec_env.reset()
    actor.reset(torch.ones(args.num_envs, dtype=torch.bool, device=device))
    ever_succeeded[:] = False

    # --- timed rollout ---
    torch.cuda.synchronize()
    step_times_ms: list[float] = []
    t_total0 = time.perf_counter()
    for _ in range(args.horizon):
        action = actor.forward(obs_td, stochastic_output=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        obs_td, _, dones, _ = vec_env.step(action)
        torch.cuda.synchronize()
        step_times_ms.append((time.perf_counter() - t0) * 1e3)

        succ = vec_env.env.is_success().get("task")
        if succ is not None:
            if not isinstance(succ, torch.Tensor):
                succ = torch.as_tensor(np.asarray(succ), device=device)
            ever_succeeded |= succ.to(device).bool().view(-1)

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
    torch.cuda.synchronize()
    t_total = time.perf_counter() - t_total0

    sr = float(ever_succeeded.float().mean().item())
    step_arr = np.asarray(step_times_ms)
    partial_rates = {k: float(v.float().mean().item()) for k, v in partial_ever.items()}
    result = {
        "config_name": args.config_name,
        "num_envs": args.num_envs,
        "horizon": args.horizon,
        "success_rate": sr,
        "n_success": int(ever_succeeded.sum().item()),
        "partial_rates": partial_rates,
        "nan_total": int(vec_env._nan_total),
        "total_rollout_s": t_total,
        "step_ms_mean": float(step_arr.mean()),
        "step_ms_p50": float(np.percentile(step_arr, 50)),
        "step_ms_p95": float(np.percentile(step_arr, 95)),
        "physics_timestep": rs_macros.SIMULATION_TIMESTEP,
        "env_knobs": {
            "tolerance_clamp": os.environ.get("ROBOSUITE_WARP_TOLERANCE_CLAMP", "0"),
            "solver_iters": os.environ.get("ROBOSUITE_WARP_SOLVER_ITERS"),
            "ls_iters": os.environ.get("ROBOSUITE_WARP_LS_ITERS"),
            "cone": os.environ.get("ROBOSUITE_WARP_CONE"),
        },
    }
    print(json.dumps(result))
    if args.output_json:
        with open(args.output_json, "a") as f:
            f.write(json.dumps(result) + "\n")

    vec_env.close()


if __name__ == "__main__":
    main()
