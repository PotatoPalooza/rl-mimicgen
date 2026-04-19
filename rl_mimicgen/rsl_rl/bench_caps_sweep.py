"""Sweep MjSimWarp buffer caps vs fidelity + VRAM at fixed ``num_envs``.

For each ``(njmax_per_env, naconmax_per_env)`` pair in ``--configs``, spawns a
fresh worker subprocess (so warp JIT-kernel caches from earlier configs don't
inflate later VRAM measurements), rolls a deterministic BC policy for
``--horizon`` steps, and records fidelity (task SR + partial "ever achieved"
rates) plus perf (step_ms, NaN, qvel magnitude, torch peak mem). A parent-side
``nvidia-smi`` sampler runs alongside each worker for peak VRAM + GPU
utilisation.

Designed for small-batch fidelity comparison (128 / 256 envs) — at that size
absolute VRAM savings from shrinking caps are small, but the point is to
measure *fidelity degradation* in isolation from VRAM pressure. Projects per-
config VRAM to 4096 envs (linear upper bound) for planning.

Usage::

    python -m rl_mimicgen.rsl_rl.bench_caps_sweep \\
        --bc_checkpoint runs/coffee_d0_low_dim/.../models/model_2000.pth \\
        --num_envs 256 \\
        --output_json bench_out/caps_sweep_coffee_d0_256.json

The defaults sweep njmax_per_env from 3500 (class default) down to 150 (~peak
observed for Coffee_D0) and scale naconmax_per_env roughly proportionally.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _sub in ("robomimic", "robosuite", "mimicgen"):
    sys.path.insert(0, str(_REPO_ROOT / "resources" / _sub))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bc_checkpoint", type=str, required=True)
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--warmup_steps", type=int, default=5)
    p.add_argument(
        "--configs",
        type=str,
        default="3500,60;1500,40;750,25;500,15;300,10;200,7;150,6",
        help="Semicolon-separated 'njmax,naconmax' pairs.",
    )
    p.add_argument("--cpu_baseline", type=str, default=None,
                   help="Optional CPU baseline JSON (with success_rate + partial_rates) for Δ-columns.")
    p.add_argument("--output_json", type=str, default=None)
    p.add_argument("--gpu_sample_ms", type=int, default=100,
                   help="nvidia-smi sampling interval during each worker run.")
    p.add_argument("--seed", type=int, default=0)
    # Worker mode (internal — one config per subprocess).
    p.add_argument("--worker_njmax", type=int, default=None)
    p.add_argument("--worker_naconmax", type=int, default=None)
    return p.parse_args()


def parse_configs(s: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        nj, na = chunk.split(",")
        out.append((int(nj), int(na)))
    return out


# ----------------------------- worker -----------------------------

def run_worker(args: argparse.Namespace) -> None:
    """Single-config rollout. Prints one JSON line on stdout."""
    os.environ.setdefault("ROBOSUITE_WARP_GRAPH", "0")

    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils
    from rsl_rl.models.rnn_model import RNNModel

    from rl_mimicgen.rsl_rl import (
        RobomimicVecEnv,
        build_actor_hidden_dims,
        build_distribution_cfg_from_bc,
        copy_bc_weights_into_actor,
        load_bc_checkpoint,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")

    njmax = int(args.worker_njmax)
    naconmax = int(args.worker_naconmax)

    bc_info = load_bc_checkpoint(args.bc_checkpoint)
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=[bc_info.obs_modalities])

    env_meta = dict(bc_info.env_meta)
    env_meta["env_kwargs"] = dict(env_meta.get("env_kwargs", {}))
    env_meta["env_kwargs"]["njmax_per_env"] = njmax
    env_meta["env_kwargs"]["naconmax_per_env"] = naconmax

    print(
        f"[worker] njmax={njmax} naconmax={naconmax} num_envs={args.num_envs} "
        f"horizon={args.horizon} env={env_meta.get('env_name')}",
        file=sys.stderr,
    )
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=False,
        use_image_obs=False, use_warp=True, num_envs=args.num_envs,
    )
    vec_env = RobomimicVecEnv(
        env=env, horizon=args.horizon, device="cuda:0",
        obs_keys=bc_info.obs_keys,
    )
    obs_td = vec_env.get_observations()

    hidden_dims, activation = build_actor_hidden_dims(bc_info)
    dist_cfg = build_distribution_cfg_from_bc(bc_info, gaussian_init_std=0.05)
    actor = RNNModel(
        obs=obs_td, obs_groups={"actor": ["policy"], "critic": ["policy"]},
        obs_set="actor", output_dim=bc_info.action_dim,
        hidden_dims=hidden_dims, activation=activation,
        obs_normalization=False, distribution_cfg=dist_cfg,
        rnn_type=bc_info.rnn_type, rnn_hidden_dim=bc_info.rnn_hidden_dim,
        rnn_num_layers=bc_info.rnn_num_layers,
    )
    _, skipped = copy_bc_weights_into_actor(actor, bc_info)
    if skipped:
        print(f"[WARN] BC warm-start skipped: {skipped}", file=sys.stderr)
    actor.to(device)
    actor.eval()

    # Warmup (untimed): JIT warp kernels, prime caches.
    for _ in range(args.warmup_steps):
        action = actor.forward(obs_td, stochastic_output=False)
        obs_td, _, dones, _ = vec_env.step(action)
        actor.reset(dones.bool())

    obs_td, _ = vec_env.reset()
    actor.reset(torch.ones(args.num_envs, dtype=torch.bool, device=device))

    torch.cuda.reset_peak_memory_stats()

    sim = vec_env.env.env.sim
    nworld = sim.num_envs
    njmax_cap = int(sim._effective_njmax_per_env)
    naconmax_cap_per_env = int(sim._effective_naconmax_per_env)
    naconmax_cap_total = naconmax_cap_per_env * nworld

    partial_fn = getattr(vec_env.env.env, "_get_partial_task_metrics", None)
    partial_ever: dict[str, torch.Tensor] = {}
    ever_succ = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    qvel_mean_per_step: list[float] = []
    qvel_max_per_step: list[float] = []
    step_times_ms: list[float] = []
    n_nefc_clipped_steps = 0
    n_nacon_clipped_steps = 0
    peak_nefc = 0
    peak_nacon = 0

    torch.cuda.synchronize()
    t_total0 = time.perf_counter()
    for _ in range(args.horizon):
        action = actor.forward(obs_td, stochastic_output=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        obs_td, _, dones, _ = vec_env.step(action)
        torch.cuda.synchronize()
        step_times_ms.append((time.perf_counter() - t0) * 1e3)

        sim = vec_env.env.env.sim  # may have been rebuilt during hard_reset
        nefc = sim._warp_data.nefc.numpy()
        nacon = int(sim._warp_data.nacon.numpy()[0])
        qvel = sim._warp_data.qvel.numpy()
        qvel_mean_per_step.append(float(np.nanmean(np.abs(qvel))))
        qvel_max_per_step.append(float(np.nanmax(np.abs(qvel))))
        max_nefc = int(nefc.max())
        peak_nefc = max(peak_nefc, max_nefc)
        peak_nacon = max(peak_nacon, nacon)
        if max_nefc >= njmax_cap:
            n_nefc_clipped_steps += 1
        if nacon >= naconmax_cap_total:
            n_nacon_clipped_steps += 1

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
    torch.cuda.synchronize()
    t_total = time.perf_counter() - t_total0

    step_arr = np.asarray(step_times_ms)
    qvel_mean_arr = np.asarray(qvel_mean_per_step)
    qvel_max_arr = np.asarray(qvel_max_per_step)

    result = {
        "njmax_per_env": njmax,
        "naconmax_per_env": naconmax,
        "num_envs": args.num_envs,
        "horizon": args.horizon,
        "nworld": nworld,
        "njmax_cap": njmax_cap,
        "naconmax_cap_per_env": naconmax_cap_per_env,
        "peak_nefc_per_world": peak_nefc,
        "peak_nacon_total": peak_nacon,
        "n_nefc_clipped_steps": n_nefc_clipped_steps,
        "n_nacon_clipped_steps": n_nacon_clipped_steps,
        "nan_total": int(vec_env._nan_total),
        "success_rate": float(ever_succ.float().mean().item()),
        "partial_rates": {k: float(v.float().mean().item()) for k, v in partial_ever.items()},
        "qvel_abs_mean_overall": float(qvel_mean_arr.mean()),
        "qvel_abs_max_overall": float(qvel_max_arr.max()),
        "qvel_abs_mean_p95_step": float(np.percentile(qvel_mean_arr, 95)),
        "total_rollout_s": t_total,
        "step_ms_mean": float(step_arr.mean()),
        "step_ms_p50": float(np.percentile(step_arr, 50)),
        "step_ms_p95": float(np.percentile(step_arr, 95)),
        "torch_peak_mem_mb": float(torch.cuda.max_memory_allocated() / 1e6),
    }
    # Single JSON line on stdout — parent greps for it.
    print(json.dumps(result))
    vec_env.close()


# ----------------------------- sweep driver -----------------------------

def start_gpu_sampler(sample_path: Path, interval_ms: int) -> subprocess.Popen | None:
    """Background nvidia-smi sampler. Returns None if nvidia-smi isn't on PATH."""
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        f = open(sample_path, "w")
        return subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
                "-lms", str(interval_ms),
            ],
            stdout=f, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return None


def parse_gpu_samples(sample_path: Path) -> dict:
    utils: list[float] = []
    mems: list[float] = []
    try:
        for line in sample_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                utils.append(float(parts[0]))
                mems.append(float(parts[1]))
            except ValueError:
                continue
    except Exception:
        pass
    if not utils:
        return {"gpu_util_mean": None, "gpu_util_p95": None,
                "gpu_mem_peak_mb": None, "gpu_mem_mean_mb": None, "n_samples": 0}
    u = np.asarray(utils)
    m = np.asarray(mems)
    return {
        "gpu_util_mean": float(u.mean()),
        "gpu_util_p95": float(np.percentile(u, 95)),
        "gpu_mem_peak_mb": float(m.max()),
        "gpu_mem_mean_mb": float(m.mean()),
        "n_samples": int(len(u)),
    }


def run_sweep(args: argparse.Namespace) -> None:
    configs = parse_configs(args.configs)

    cpu_baseline = None
    if args.cpu_baseline:
        cpu_baseline = json.loads(Path(args.cpu_baseline).read_text())

    results: list[dict] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="caps_sweep_"))
    print(f"[sweep] tmp_dir={tmp_dir}", flush=True)

    for njmax, naconmax in configs:
        print(f"\n===== njmax={njmax} naconmax={naconmax} =====", flush=True)
        sample_path = tmp_dir / f"gpu_{njmax}_{naconmax}.csv"
        sampler = start_gpu_sampler(sample_path, args.gpu_sample_ms)
        time.sleep(0.2)  # let sampler emit a few baseline rows

        cmd = [
            sys.executable, "-m", "rl_mimicgen.rsl_rl.bench_caps_sweep",
            "--bc_checkpoint", args.bc_checkpoint,
            "--num_envs", str(args.num_envs),
            "--horizon", str(args.horizon),
            "--warmup_steps", str(args.warmup_steps),
            "--seed", str(args.seed),
            "--worker_njmax", str(njmax),
            "--worker_naconmax", str(naconmax),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())

        if sampler is not None:
            sampler.terminate()
            try:
                sampler.wait(timeout=3)
            except subprocess.TimeoutExpired:
                sampler.kill()

        if proc.returncode != 0:
            tail_err = "\n".join(proc.stderr.splitlines()[-25:]) if proc.stderr else "(no stderr)"
            print(f"[ERROR] worker rc={proc.returncode}\n{tail_err}", flush=True)
            result = {
                "njmax_per_env": njmax,
                "naconmax_per_env": naconmax,
                "error": tail_err,
            }
        else:
            json_lines = [l for l in proc.stdout.splitlines() if l.startswith("{") and l.rstrip().endswith("}")]
            if not json_lines:
                print(f"[ERROR] no JSON in worker stdout\nstdout tail: {proc.stdout[-500:]}", flush=True)
                result = {"njmax_per_env": njmax, "naconmax_per_env": naconmax, "error": "no_json"}
            else:
                try:
                    result = json.loads(json_lines[-1])
                except json.JSONDecodeError as e:
                    result = {"njmax_per_env": njmax, "naconmax_per_env": naconmax, "error": f"bad_json: {e}"}

        result["gpu_samples"] = parse_gpu_samples(sample_path)
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

    _print_summary(args, configs, results, cpu_baseline)

    if args.output_json:
        out = {
            "args": {k: v for k, v in vars(args).items() if not k.startswith("worker_")},
            "configs": [{"njmax_per_env": nj, "naconmax_per_env": na} for nj, na in configs],
            "results": results,
            "cpu_baseline": cpu_baseline,
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"\n[wrote {args.output_json}]")


def _print_summary(
    args: argparse.Namespace,
    configs: list[tuple[int, int]],
    results: list[dict],
    cpu_baseline: dict | None,
) -> None:
    print("\n" + "=" * 120)
    print(f"CAPS SWEEP SUMMARY  (num_envs={args.num_envs}, horizon={args.horizon})")
    print("=" * 120)
    header = (
        "{:>5} {:>4} | {:>5} {:>5} {:>5} {:>5} | {:>4} {:>4} {:>5} | "
        "{:>5} {:>6} | {:>6} {:>7} {:>5}"
    )
    print(header.format(
        "njmax", "nacn",
        "SR", "grasp", "rim", "inst",
        "clpJ", "clpA", "nan",
        "qvMx", "step_ms",
        "tchMB", "gpuMB", "util%",
    ))
    print("-" * 120)
    for r in results:
        if "error" in r:
            print(f"{r['njmax_per_env']:>5} {r['naconmax_per_env']:>4} | ERROR: {r['error'][:80]}")
            continue
        pr = r.get("partial_rates", {})
        gs = r.get("gpu_samples", {}) or {}
        print(header.format(
            r["njmax_per_env"], r["naconmax_per_env"],
            f"{r['success_rate']:.3f}",
            f"{pr.get('grasp', float('nan')):.3f}",
            f"{pr.get('rim', float('nan')):.3f}",
            f"{pr.get('insertion', float('nan')):.3f}",
            r.get("n_nefc_clipped_steps", 0),
            r.get("n_nacon_clipped_steps", 0),
            r.get("nan_total", 0),
            f"{r.get('qvel_abs_max_overall', 0):.1f}",
            f"{r['step_ms_mean']:.1f}",
            f"{r['torch_peak_mem_mb']:.0f}",
            f"{(gs.get('gpu_mem_peak_mb') or 0):.0f}",
            f"{(gs.get('gpu_util_mean') or 0):.0f}",
        ))
    print()
    print("legend: SR=task success rate; grasp/rim/insertion=partial 'ever achieved' rates;")
    print("        clpJ=#steps where max nefc hit njmax cap; clpA=#steps where total nacon hit cap;")
    print("        qvMx=peak |qvel| across rollout (>100 ≈ physical explosion);")
    print("        tchMB=torch.cuda.max_memory_allocated; gpuMB/util%=nvidia-smi peak memory / mean util.")

    if cpu_baseline is not None:
        cpu_sr = cpu_baseline.get("success_rate", float("nan"))
        cpu_partial = cpu_baseline.get("partial_rates", {})
        print(f"\n--- Δ vs CPU baseline (CPU SR={cpu_sr:.3f}, partial={cpu_partial}) ---")
        for r in results:
            if "error" in r:
                continue
            pr = r.get("partial_rates", {})
            print(
                f"  njmax={r['njmax_per_env']:>5} nacon={r['naconmax_per_env']:>3}: "
                f"ΔSR={r['success_rate']-cpu_sr:+.3f}  "
                f"Δrim={pr.get('rim', 0) - cpu_partial.get('rim', 0):+.3f}  "
                f"Δinsert={pr.get('insertion', 0) - cpu_partial.get('insertion', 0):+.3f}"
            )

    # VRAM projection to 4096 envs (linear upper bound — warp buffers scale
    # linearly in num_envs; kernel/torch overhead is ~fixed).
    print(f"\n--- VRAM projection from {args.num_envs} -> 4096 envs (linear upper bound) ---")
    scale = 4096 / args.num_envs
    for r in results:
        if "error" in r:
            continue
        gs = r.get("gpu_samples") or {}
        peak = gs.get("gpu_mem_peak_mb")
        if peak is None:
            continue
        print(
            f"  njmax={r['njmax_per_env']:>5} nacon={r['naconmax_per_env']:>3}: "
            f"{peak:>6.0f} MB  ->  ~{peak*scale:>6.0f} MB projected"
        )


def main() -> None:
    args = parse_args()
    if args.worker_njmax is not None and args.worker_naconmax is not None:
        run_worker(args)
    else:
        run_sweep(args)


if __name__ == "__main__":
    main()
