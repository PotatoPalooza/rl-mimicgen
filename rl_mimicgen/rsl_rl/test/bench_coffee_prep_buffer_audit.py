"""Quick nacon/nefc overflow audit for CoffeePreparation under warp.

Standalone: no BC checkpoint or policy needed. Builds
``CoffeePreparation_D{0,1}`` via ``robosuite.make(use_warp=True)``, resets,
then rolls ``n_steps`` of **zero OSC deltas** (robot holds pose; mug settles
onto table). Each step reads ``sim._warp_data.nefc`` (per-world) and
``sim._warp_data.nacon[0]`` (total across worlds) and tallies peaks +
clipped-step counts.

Motivated by unstable-mug observations in DPPO pretrain eval videos: the
CoffeePreparation mug is a 32-geom BlenderObject convex decomposition
(``shapenet_core/mugs/3143a4ac/collision/``), which under warp+f32 can spike
``nacon`` past the default ``_NACONMAX_PER_ENV = 60`` cap. Overflow is
silent — dropped contact slots cause one-sided normals and the mug launches.

If ``nacon`` clips with zero actions at the target ``num_envs``, the fix
is a per-task entry in ``rl_mimicgen.rsl_rl.warp_buffer_sizes`` (e.g.
``coffee_preparation: {naconmax_per_env: 150}``), then re-run this script
to confirm the clip count drops to zero.

Usage::

    python -m rl_mimicgen.rsl_rl.test.bench_coffee_prep_buffer_audit \\
        --env_name CoffeePreparation_D0 --num_envs 256
    python -m rl_mimicgen.rsl_rl.test.bench_coffee_prep_buffer_audit \\
        --env_name CoffeePreparation_D0 --num_envs 1024 --random_actions
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _sub in ("robomimic", "robosuite", "mimicgen"):
    sys.path.insert(0, str(_REPO_ROOT / "resources" / _sub))

# Graph capture interferes with per-step .numpy() reads of nefc/nacon.
os.environ.setdefault("ROBOSUITE_WARP_GRAPH", "0")

import robosuite  # noqa: E402
import mimicgen  # noqa: F401,E402  (registers CoffeePreparation_D{0,1})
import mimicgen.envs.robosuite.coffee  # noqa: F401,E402

from rl_mimicgen.rsl_rl.warp_buffer_sizes import resolve_warp_buffer_sizes  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env_name", type=str, default="CoffeePreparation_D0")
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--n_steps", type=int, default=100)
    p.add_argument("--random_actions", action="store_true",
                   help="Use uniform-[-1,1] actions instead of zero deltas "
                        "(stress-tests gripper-mug contacts).")
    p.add_argument("--njmax_per_env", type=int, default=None,
                   help="Force njmax cap (otherwise use warp_buffer_sizes + class default).")
    p.add_argument("--naconmax_per_env", type=int, default=None,
                   help="Force naconmax cap (otherwise use warp_buffer_sizes + class default).")
    p.add_argument("--output_json", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _panda_osc_kwargs() -> dict:
    """OSC_POSE controller cfg matching the mimicgen Coffee datasets."""
    return dict(
        type="OSC_POSE",
        input_max=1, input_min=-1,
        output_max=[0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        output_min=[-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        kp=150, damping=1, impedance_mode="fixed",
        kp_limits=[0, 300], damping_limits=[0, 10],
        position_limits=None, orientation_limits=None,
        uncouple_pos_ori=True, control_delta=True,
        interpolation=None, ramp_ratio=0.2,
    )


def build_env(args: argparse.Namespace) -> object:
    caps = resolve_warp_buffer_sizes(args.env_name) or {}
    if args.njmax_per_env is not None:
        caps["njmax_per_env"] = args.njmax_per_env
    if args.naconmax_per_env is not None:
        caps["naconmax_per_env"] = args.naconmax_per_env
    return robosuite.make(
        args.env_name,
        robots="Panda",
        controller_configs=_panda_osc_kwargs(),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        horizon=1000,
        control_freq=20,
        reward_shaping=False,
        ignore_done=True,
        use_warp=True,
        num_envs=args.num_envs,
        **caps,
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = build_env(args)
    env.reset()
    # hard_reset=True (default) rebuilds sim on reset — re-fetch after reset + per-step.
    sim = env.sim
    njmax_cap = int(sim._effective_njmax_per_env)
    naconmax_per_env = int(sim._effective_naconmax_per_env)
    naconmax_total = naconmax_per_env * sim.num_envs

    print(f"[INFO] env={args.env_name} num_envs={sim.num_envs} "
          f"njmax_per_env={njmax_cap} naconmax_per_env={naconmax_per_env} "
          f"(total_naconmax={naconmax_total})", file=sys.stderr)

    # Action shape is (num_envs, action_dim) torch tensor on CUDA (see robot_env.py:_pre_action).
    act_zero = torch.zeros((sim.num_envs, env.action_dim), dtype=torch.float32, device="cuda")

    nefc_peak = 0
    nefc_peak_step = -1
    nefc_peak_world = -1
    nacon_peak = 0
    nacon_peak_step = -1
    nefc_samples: list[int] = []
    nacon_samples: list[int] = []
    nefc_clipped_steps = 0
    nacon_clipped_steps = 0

    torch.cuda.synchronize()
    for step_i in range(args.n_steps):
        if args.random_actions:
            action = torch.empty((sim.num_envs, env.action_dim), device="cuda").uniform_(-1.0, 1.0)
        else:
            action = act_zero
        env.step(action)
        sim = env.sim  # rebuilt on hard_reset (not expected here, but cheap to re-fetch).

        nefc = sim._warp_data.nefc.numpy()  # (nworld,)
        nacon = int(sim._warp_data.nacon.numpy()[0])
        max_nefc = int(nefc.max())
        max_nefc_world = int(nefc.argmax())
        nefc_samples.append(max_nefc)
        nacon_samples.append(nacon)

        if max_nefc > nefc_peak:
            nefc_peak = max_nefc
            nefc_peak_step = step_i
            nefc_peak_world = max_nefc_world
        if nacon > nacon_peak:
            nacon_peak = nacon
            nacon_peak_step = step_i
        if max_nefc >= njmax_cap:
            nefc_clipped_steps += 1
        if nacon >= naconmax_total:
            nacon_clipped_steps += 1

    torch.cuda.synchronize()

    nefc_arr = np.asarray(nefc_samples)
    nacon_arr = np.asarray(nacon_samples)
    nacon_peak_per_env = nacon_peak / sim.num_envs
    nacon_mean_per_env = float(nacon_arr.mean()) / sim.num_envs

    result = {
        "env_name": args.env_name,
        "num_envs": sim.num_envs,
        "n_steps": args.n_steps,
        "action_mode": "random" if args.random_actions else "zero",
        "caps": {"njmax_per_env": njmax_cap, "naconmax_per_env": naconmax_per_env,
                 "naconmax_total": naconmax_total},
        "nefc_per_world": {
            "peak": nefc_peak,
            "peak_step": nefc_peak_step,
            "peak_world": nefc_peak_world,
            "mean_of_per_step_max": float(nefc_arr.mean()),
            "p95_of_per_step_max": float(np.percentile(nefc_arr, 95)),
            "clipped_steps": nefc_clipped_steps,
        },
        "nacon_total": {
            "peak": nacon_peak,
            "peak_step": nacon_peak_step,
            "peak_per_env_equiv": nacon_peak_per_env,
            "mean_per_env_equiv": nacon_mean_per_env,
            "clipped_steps": nacon_clipped_steps,
        },
    }

    print("\n========== CoffeePreparation BUFFER AUDIT ==========")
    print(f"env={args.env_name}  num_envs={sim.num_envs}  n_steps={args.n_steps}  "
          f"mode={'random' if args.random_actions else 'zero'}")
    print(f"caps:  njmax_per_env={njmax_cap}  naconmax_per_env={naconmax_per_env}  "
          f"(total_naconmax={naconmax_total})")
    print(f"\nnefc per-world: peak={nefc_peak} (world {nefc_peak_world}, step {nefc_peak_step})  "
          f"util={nefc_peak/njmax_cap:.1%}  clipped_steps={nefc_clipped_steps}/{args.n_steps}")
    print(f"nacon total:    peak={nacon_peak} (step {nacon_peak_step})  "
          f"util={nacon_peak/naconmax_total:.1%}  "
          f"peak_per_env={nacon_peak_per_env:.1f}  mean_per_env={nacon_mean_per_env:.1f}  "
          f"clipped_steps={nacon_clipped_steps}/{args.n_steps}")
    print(f"\nraw JSON: {json.dumps(result)}")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
