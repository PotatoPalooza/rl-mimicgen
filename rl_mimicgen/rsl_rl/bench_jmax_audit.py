"""Audit per-world ``nefc`` and total ``nacon`` peak usage during a rollout.

Used to size ``_NJMAX_PER_ENV`` / ``_NACONMAX_PER_ENV`` in
``resources/robosuite/robosuite/utils/binding_utils.py``. Current defaults
(3500 / 60) are conservative; shrinking them frees GPU memory so CUDA graph
capture fits at 4096 envs (see CLAUDE.md "CUDA graph capture" notes).

Approach:
  * Load a BC policy + warp vec-env (same path as ``bench_warp_knobs.py``).
  * Pre-seed random per-env episode lengths so timeouts — and therefore
    ``_reset_envs`` snapshot-restores — spread across steps, matching the RL
    training behaviour with ``init_at_random_ep_len=True``.
  * Roll out ``horizon`` steps. After each step, read
    ``sim._warp_data.nefc.numpy().max()`` (per-world efc peak) and
    ``sim._warp_data.nacon.numpy()[0]`` (total contacts across all worlds).
  * Report observed peaks, current caps, and recommended new caps
    (observed × ``--headroom``, rounded up).

Graph capture is force-disabled during the audit to avoid confusing the
``.numpy()`` reads with captured-kernel state.

Example::

    python -m rl_mimicgen.rsl_rl.bench_jmax_audit \\
        --bc_checkpoint runs/coffee_d0_low_dim/.../models/model_2000.pth \\
        --num_envs 2048 --horizon 400
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _sub in ("robomimic", "robosuite", "mimicgen"):
    sys.path.insert(0, str(_REPO_ROOT / "resources" / _sub))

# Force graph capture off so per-step .numpy() reads are straightforward.
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
    p.add_argument("--num_envs", type=int, default=2048)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--headroom", type=float, default=1.3,
                   help="Multiplier applied to observed peak for recommended cap.")
    p.add_argument("--njmax_per_env", type=int, default=None,
                   help="Force this njmax_per_env cap for the audit (injects into "
                        "env_meta.env_kwargs). Default: use whatever env_meta already has.")
    p.add_argument("--naconmax_per_env", type=int, default=None,
                   help="Force this naconmax_per_env cap for the audit.")
    p.add_argument("--output_json", type=str, default=None)
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


def _round_up(x: float, step: int) -> int:
    return int(math.ceil(x / step) * step)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    bc_info = load_bc_checkpoint(args.bc_checkpoint)
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=[bc_info.obs_modalities])

    # Force buffer caps into env_meta.env_kwargs if requested (simulates what
    # a BC config's experiment.env_meta_update_dict.env_kwargs would do).
    ek = bc_info.env_meta.setdefault("env_kwargs", {})
    if args.njmax_per_env is not None:
        ek["njmax_per_env"] = int(args.njmax_per_env)
    if args.naconmax_per_env is not None:
        ek["naconmax_per_env"] = int(args.naconmax_per_env)

    print(
        f"[INFO] audit: num_envs={args.num_envs} horizon={args.horizon} "
        f"env={bc_info.env_meta.get('env_name')}",
        file=sys.stderr,
    )
    env = EnvUtils.create_env_from_metadata(
        env_meta=bc_info.env_meta,
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
    # NOTE: robosuite rebuilds `sim` on every `env.reset()` (hard_reset=True
    # path destroys MjSimWarp and allocates a fresh one), and RobomimicVecEnv's
    # snapshot-restore reset path ultimately calls env.reset() too. So we must
    # re-fetch sim after every vec_env.step() rather than caching it.
    sim = vec_env.env.env.sim
    nworld = sim.num_envs
    njmax_cap = int(sim._effective_njmax_per_env)
    naconmax_cap_per_env = int(sim._effective_naconmax_per_env)
    naconmax_cap_total = naconmax_cap_per_env * nworld

    obs_td = vec_env.get_observations()
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = build_actor_from_bc(bc_info, obs_td, obs_groups, device)

    # --- warmup (untimed): JIT-compile warp kernels, prime caches ---
    for _ in range(args.warmup_steps):
        action = actor.forward(obs_td, stochastic_output=False)
        obs_td, _, dones, _ = vec_env.step(action)
        actor.reset(dones.bool())

    # Full reset so the audit rollout starts from a clean slate, then pre-seed
    # random episode lengths so per-env timeouts — and their snapshot-restore
    # resets — spread across the rollout (matches init_at_random_ep_len=True
    # from training).
    obs_td, _ = vec_env.reset()
    actor.reset(torch.ones(args.num_envs, dtype=torch.bool, device=device))
    vec_env.episode_length_buf.copy_(
        torch.randint(0, args.horizon, (args.num_envs,), device=device, dtype=torch.long)
    )

    # --- audited rollout ---
    peak_nefc_per_world: int = 0
    peak_nefc_step: int = -1
    peak_nefc_world: int = -1
    peak_nacon_total: int = 0
    peak_nacon_step: int = -1
    nefc_samples: list[int] = []
    nacon_samples: list[int] = []
    n_resets_total: int = 0
    # Count steps where at least one env's nefc ~saturated the cap. mujoco-warp
    # silently clips nefc to njmax on overflow → constraints get dropped →
    # physics can explode. If this count is non-zero for a chosen njmax, the
    # cap is too low.
    n_nefc_clipped_steps: int = 0
    n_nacon_clipped_steps: int = 0

    torch.cuda.synchronize()
    t_total0 = time.perf_counter()
    for step_i in range(args.horizon):
        action = actor.forward(obs_td, stochastic_output=False)
        obs_td, _, dones, _ = vec_env.step(action)

        n_resets_total += int(dones.sum().item())

        sim = vec_env.env.env.sim  # may have been rebuilt inside vec_env.step
        nefc = sim._warp_data.nefc.numpy()
        nacon = int(sim._warp_data.nacon.numpy()[0])
        max_nefc = int(nefc.max())
        max_nefc_world = int(nefc.argmax())
        nefc_samples.append(max_nefc)
        nacon_samples.append(nacon)

        if max_nefc > peak_nefc_per_world:
            peak_nefc_per_world = max_nefc
            peak_nefc_step = step_i
            peak_nefc_world = max_nefc_world
        if nacon > peak_nacon_total:
            peak_nacon_total = nacon
            peak_nacon_step = step_i
        if max_nefc >= int(sim._effective_njmax_per_env):
            n_nefc_clipped_steps += 1
        if nacon >= int(sim._effective_naconmax_per_env) * nworld:
            n_nacon_clipped_steps += 1

        actor.reset(dones.bool())
    torch.cuda.synchronize()
    t_total = time.perf_counter() - t_total0

    # --- compile report ---
    nefc_arr = np.asarray(nefc_samples)
    nacon_arr = np.asarray(nacon_samples)
    nacon_per_env_mean = float(nacon_arr.mean()) / nworld
    nacon_per_env_peak = peak_nacon_total / nworld

    rec_njmax = _round_up(peak_nefc_per_world * args.headroom, 100)
    rec_naconmax_per_env = max(1, _round_up(nacon_per_env_peak * args.headroom, 1))

    njmax_util = peak_nefc_per_world / njmax_cap
    naconmax_util = peak_nacon_total / naconmax_cap_total

    result = {
        "num_envs": nworld,
        "horizon": args.horizon,
        "headroom": args.headroom,
        "n_resets_total": n_resets_total,
        "n_nefc_clipped_steps": n_nefc_clipped_steps,
        "n_nacon_clipped_steps": n_nacon_clipped_steps,
        "rollout_s": t_total,
        "current_caps": {
            "NJMAX_PER_ENV": njmax_cap,
            "NACONMAX_PER_ENV": naconmax_cap_per_env,
            "NACONMAX_TOTAL": naconmax_cap_total,
        },
        "nefc_per_world": {
            "peak": peak_nefc_per_world,
            "peak_at_step": peak_nefc_step,
            "peak_world_idx": peak_nefc_world,
            "mean_of_per_step_max": float(nefc_arr.mean()),
            "p95_of_per_step_max": float(np.percentile(nefc_arr, 95)),
            "utilization_vs_cap": njmax_util,
        },
        "nacon_total": {
            "peak": peak_nacon_total,
            "peak_at_step": peak_nacon_step,
            "peak_per_env_equiv": nacon_per_env_peak,
            "mean_per_env_equiv": nacon_per_env_mean,
            "utilization_vs_cap": naconmax_util,
        },
        "recommended": {
            "NJMAX_PER_ENV": rec_njmax,
            "NACONMAX_PER_ENV": rec_naconmax_per_env,
        },
    }

    print("\n========== JMAX / NACONMAX AUDIT ==========")
    print(f"num_envs={nworld}  horizon={args.horizon}  resets_during_rollout={n_resets_total}")
    print(f"nefc-clipped steps: {n_nefc_clipped_steps}/{args.horizon}   "
          f"nacon-clipped steps: {n_nacon_clipped_steps}/{args.horizon}")
    print(f"\ncurrent caps:  NJMAX_PER_ENV={njmax_cap}  "
          f"NACONMAX_PER_ENV={naconmax_cap_per_env}  "
          f"(NACONMAX total = {naconmax_cap_total})")
    print("\nobserved peaks:")
    print(f"  per-world nefc: peak={peak_nefc_per_world} "
          f"(world {peak_nefc_world}, step {peak_nefc_step})  "
          f"util={njmax_util:.1%}  p95={float(np.percentile(nefc_arr, 95)):.0f}")
    print(f"  total   nacon: peak={peak_nacon_total} "
          f"(step {peak_nacon_step})  util={naconmax_util:.1%}  "
          f"per-env peak={nacon_per_env_peak:.1f}  mean={nacon_per_env_mean:.1f}")
    print(f"\nrecommended (observed x {args.headroom}):")
    print(f"  _NJMAX_PER_ENV   = {rec_njmax}   (was {njmax_cap})")
    print(f"  _NACONMAX_PER_ENV = {rec_naconmax_per_env}   (was {naconmax_cap_per_env})")
    print("\nraw JSON:")
    print(json.dumps(result))

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.output_json}")

    vec_env.close()


if __name__ == "__main__":
    main()
