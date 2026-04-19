"""Sweep per-step warp-sim time across num_envs.

Launches ``rl_mimicgen.rsl_rl.test.bench_warp_knobs`` once per num_envs value,
each in its own subprocess (so warp re-JITs against the new world count),
with the same production settings ``train_rl.py`` uses: CUDA graph capture
on, per-task buffer caps applied automatically via
``resolve_warp_buffer_sizes``.

Reports ``step_ms``, an iteration-time proxy (``step_ms * num_steps_per_env``),
and env-steps/s so sub-linear scaling shows up. Sample output::

    num_envs   step_ms    iter_s   env_steps/s     SR     rim  insert
         50       38.1      0.91         1312   0.38   0.94   0.64
        128       51.2      1.23         2500   0.40   0.95   0.67
       ...

Example::

    python -m rl_mimicgen.rsl_rl.test.bench_num_envs \\
        --bc_checkpoint runs/robomimic_rl/.../bc_ckpt/model_2000.pth
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_NUM_ENVS: tuple[int, ...] = (50, 128, 256, 512, 1024, 2048, 4096)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bc_checkpoint", type=str, required=True)
    p.add_argument("--num_envs_list", type=str, default=None,
                   help="Comma-separated num_envs values. Default: "
                        f"{','.join(str(n) for n in DEFAULT_NUM_ENVS)}.")
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--num_steps_per_env", type=int, default=24,
                   help="Multiplier for the iter-time proxy (matches ppo_cfg.json).")
    p.add_argument("--no_graph_capture", action="store_true",
                   help="Disable CUDA graph capture for the whole sweep. "
                        "Default: graph capture on (mirrors train_rl.py).")
    p.add_argument("--output_json", type=str,
                   default=str(Path("runs") / "bench_num_envs_results.jsonl"))
    return p.parse_args()


def run_one(num_envs: int, args: argparse.Namespace) -> dict | None:
    env = os.environ.copy()
    env["ROBOSUITE_WARP_GRAPH"] = "0" if args.no_graph_capture else "1"
    # Clear unrelated knob env-vars so shell state doesn't leak into the run.
    for leak in (
        "ROBOSUITE_WARP_TOLERANCE_CLAMP",
        "ROBOSUITE_WARP_SOLVER_ITERS",
        "ROBOSUITE_WARP_LS_ITERS",
        "ROBOSUITE_WARP_CONE",
    ):
        env.pop(leak, None)

    cmd = [
        sys.executable, "-m", "rl_mimicgen.rsl_rl.test.bench_warp_knobs",
        "--bc_checkpoint", args.bc_checkpoint,
        "--num_envs", str(num_envs),
        "--horizon", str(args.horizon),
        "--warmup_steps", str(args.warmup_steps),
        "--config_name", f"num_envs_{num_envs}",
        "--output_json", args.output_json,
    ]

    print(f"\n========== num_envs={num_envs} "
          f"(graph_capture={'off' if args.no_graph_capture else 'on'}) ==========",
          flush=True)
    print(f"  cmd: {' '.join(cmd)}", flush=True)

    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        print(f"[ERROR] num_envs={num_envs} failed (rc={proc.returncode})")
        print(proc.stdout[-2000:])
        return None
    last_json_line = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            last_json_line = line
    if last_json_line is None:
        print(f"[ERROR] num_envs={num_envs} produced no JSON result")
        print(proc.stdout[-2000:])
        return None
    return json.loads(last_json_line)


def fmt_table(rows: list[dict], num_steps_per_env: int) -> str:
    if not rows:
        return "(no results)"
    header = (f"{'num_envs':>8} {'step_ms':>9} {'iter_s':>9} "
              f"{'env_steps/s':>12} {'SR':>6} {'rim':>6} {'insert':>7} {'nan':>6}")
    lines = [header, "-" * len(header)]
    for r in rows:
        n = r["num_envs"]
        step_ms = r["step_ms_mean"]
        iter_s = step_ms * num_steps_per_env / 1e3
        esps = n * 1e3 / step_ms  # env-steps per second
        partial = r.get("partial_rates", {}) or {}
        rim = partial.get("rim", float("nan"))
        ins = partial.get("insertion", float("nan"))
        lines.append(
            f"{n:>8d} "
            f"{step_ms:>9.1f} "
            f"{iter_s:>9.2f} "
            f"{esps:>12.0f} "
            f"{r['success_rate']:>6.3f} "
            f"{rim:>6.3f} "
            f"{ins:>7.3f} "
            f"{r.get('nan_total', 0):>6d}"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.num_envs_list:
        num_envs_values = [int(x) for x in args.num_envs_list.split(",") if x.strip()]
    else:
        num_envs_values = list(DEFAULT_NUM_ENVS)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    open(args.output_json, "w").close()  # truncate

    results: list[dict] = []
    for n in num_envs_values:
        r = run_one(n, args)
        if r is not None:
            results.append(r)

    print("\n\n========= SUMMARY =========")
    graph_tag = "off" if args.no_graph_capture else "on"
    print(f"horizon={args.horizon}  num_steps_per_env={args.num_steps_per_env}  "
          f"graph_capture={graph_tag}\n")
    print(fmt_table(results, args.num_steps_per_env))
    print(f"\nFull results: {args.output_json}")


if __name__ == "__main__":
    main()
