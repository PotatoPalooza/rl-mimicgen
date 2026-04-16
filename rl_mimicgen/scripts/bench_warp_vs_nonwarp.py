"""Compare warp and non-warp rollout success rates for a given checkpoint.

Rolls out N episodes under each backend on the same policy/env/horizon and
reports per-backend success rate, return, and total wall time.

Usage:
    python -m rl_mimicgen.scripts.bench_warp_vs_nonwarp \
        --checkpoint runs/.../models/model_epoch_100.pth \
        --num_episodes 10 --horizon 400
"""

import argparse
import os
import sys
import time

import numpy as np
import torch


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _sub in ("robomimic", "robosuite", "mimicgen"):
    sys.path.insert(0, os.path.join(_REPO_ROOT, "resources", _sub))


import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.train_utils as TrainUtils


def run_warp_backend(policy, env_meta, num_envs, horizon):
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, render_offscreen=False, use_image_obs=False,
        use_warp=True, num_envs=num_envs,
    )
    policy.use_warp = True
    TrainUtils.WARP_EXPLOSION_STEPS = 0
    TrainUtils.WARP_EXPLOSION_ENVS = 0

    t0 = time.time()
    results = TrainUtils.run_warp_rollout(
        policy=policy, env=env, horizon=horizon,
        use_goals=False, terminate_on_success=False,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    succ = float(np.mean([r["Success_Rate"] for r in results]))
    ret = float(np.mean([r["Return"] for r in results]))
    return {
        "num_episodes": len(results),
        "success_rate": succ,
        "mean_return": ret,
        "elapsed_s": elapsed,
        "explosion_steps": TrainUtils.WARP_EXPLOSION_STEPS,
        "explosion_env_steps": TrainUtils.WARP_EXPLOSION_ENVS,
    }


def run_sequential_backend(policy, env_meta, num_episodes, horizon):
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, render_offscreen=False, use_image_obs=False,
        use_warp=False, num_envs=1,
    )
    policy.use_warp = False

    t0 = time.time()
    succs, rets = [], []
    for _ in range(num_episodes):
        r = TrainUtils.run_rollout(
            policy=policy, env=env, horizon=horizon,
            use_goals=False, terminate_on_success=False,
        )
        succs.append(r["Success_Rate"])
        rets.append(r["Return"])
    elapsed = time.time() - t0
    return {
        "num_episodes": num_episodes,
        "success_rate": float(np.mean(succs)),
        "mean_return": float(np.mean(rets)),
        "elapsed_s": elapsed,
        "explosion_steps": 0,
        "explosion_env_steps": 0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num_episodes", type=int, default=10,
                   help="Number of episodes to roll out per backend (= warp num_envs).")
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--skip_nonwarp", action="store_true",
                   help="Only run warp backend (faster iteration).")
    p.add_argument("--skip_warp", action="store_true",
                   help="Only run sequential backend.")
    args = p.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    print(f"Loading policy from {args.checkpoint}")
    policy, ckpt = FileUtils.policy_from_checkpoint(
        ckpt_path=args.checkpoint, device=torch.device(args.device), verbose=False,
    )
    env_meta = ckpt["env_metadata"]

    rows = []
    if not args.skip_warp:
        print(f"\n=== warp (num_envs={args.num_episodes}, horizon={args.horizon}) ===")
        rows.append(("warp", run_warp_backend(policy, env_meta, args.num_episodes, args.horizon)))
        r = rows[-1][1]
        print(f"  success={r['success_rate']:.2f}  return={r['mean_return']:.3f}  "
              f"elapsed={r['elapsed_s']:.1f}s  explosions={r['explosion_steps']}")

    if not args.skip_nonwarp:
        print(f"\n=== sequential (num_episodes={args.num_episodes}, horizon={args.horizon}) ===")
        rows.append(("sequential", run_sequential_backend(policy, env_meta, args.num_episodes, args.horizon)))
        r = rows[-1][1]
        print(f"  success={r['success_rate']:.2f}  return={r['mean_return']:.3f}  elapsed={r['elapsed_s']:.1f}s")

    print("\n" + "~" * 80)
    print(f"{'backend':12s}  {'episodes':>8s}  {'success':>7s}  {'return':>7s}  {'elapsed':>8s}  {'expl':>5s}")
    print("-" * 80)
    for label, r in rows:
        print(f"{label:12s}  {r['num_episodes']:8d}  {r['success_rate']:7.2f}  "
              f"{r['mean_return']:7.3f}  {r['elapsed_s']:8.1f}s  {r['explosion_steps']:5d}")
    print("~" * 80)


if __name__ == "__main__":
    main()
