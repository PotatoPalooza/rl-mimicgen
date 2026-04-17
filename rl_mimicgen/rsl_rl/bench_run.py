"""Runner: benchmark each warp-knob individually vs baseline.

Launches ``rl_mimicgen.rsl_rl.bench_warp_knobs`` once per configuration in
its own subprocess (so warp kernels recompile cleanly against the new model
options each time), collects the JSON-line results, and prints a summary
table.

Default matrix (each row toggles ONE knob relative to the baseline):

    baseline            -- current defaults
    tol_clamp           -- accept mujoco-warp's 1e-6 tolerance floor
    timestep_4ms        -- SIMULATION_TIMESTEP=0.004 (substeps 25 -> 13)
    cone_pyramidal      -- opt.cone=pyramidal
    solver_20_ls5       -- opt.iterations=20, opt.ls_iterations=5

Example::

    python -m rl_mimicgen.rsl_rl.bench_run \
        --bc_checkpoint runs/coffee_d0_low_dim/.../models/model_2000.pth \
        --num_envs 4096 --horizon 400
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


CONFIGS: list[dict] = [
    {"name": "baseline", "env": {}, "timestep": None},
    {"name": "tol_clamp", "env": {"ROBOSUITE_WARP_TOLERANCE_CLAMP": "1"}, "timestep": None},
    {"name": "timestep_4ms", "env": {}, "timestep": 0.004},
    {"name": "cone_pyramidal", "env": {"ROBOSUITE_WARP_CONE": "pyramidal"}, "timestep": None},
    {"name": "solver_20_ls5",
     "env": {"ROBOSUITE_WARP_SOLVER_ITERS": "20", "ROBOSUITE_WARP_LS_ITERS": "5"},
     "timestep": None},
    {"name": "graph_capture", "env": {"ROBOSUITE_WARP_GRAPH": "1"}, "timestep": None},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bc_checkpoint", type=str, required=True)
    p.add_argument("--num_envs", type=int, default=4096)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--output_json", type=str,
                   default=str(Path("runs") / "warp_bench_results.jsonl"))
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated config names to run (default: all).")
    return p.parse_args()


def run_one(cfg: dict, args: argparse.Namespace) -> dict | None:
    env = os.environ.copy()
    for k, v in cfg["env"].items():
        env[k] = v
    # Unset any knob env-vars not in this cfg, so inherited shell state
    # doesn't silently leak across configs.
    for leak in (
        "ROBOSUITE_WARP_TOLERANCE_CLAMP",
        "ROBOSUITE_WARP_SOLVER_ITERS",
        "ROBOSUITE_WARP_LS_ITERS",
        "ROBOSUITE_WARP_CONE",
        "ROBOSUITE_WARP_GRAPH",
    ):
        if leak not in cfg["env"] and leak in env:
            env.pop(leak)

    cmd = [
        sys.executable, "-m", "rl_mimicgen.rsl_rl.bench_warp_knobs",
        "--bc_checkpoint", args.bc_checkpoint,
        "--num_envs", str(args.num_envs),
        "--horizon", str(args.horizon),
        "--warmup_steps", str(args.warmup_steps),
        "--config_name", cfg["name"],
        "--output_json", args.output_json,
    ]
    if cfg["timestep"] is not None:
        cmd += ["--physics_timestep", str(cfg["timestep"])]

    print(f"\n========== {cfg['name']} ==========", flush=True)
    print(f"  env-overrides: {cfg['env'] or '(none)'}  timestep={cfg['timestep'] or 'default'}")
    print(f"  cmd: {' '.join(cmd)}", flush=True)

    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    # Forward child's logs to our stderr so warp-module messages stay visible.
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        print(f"[ERROR] {cfg['name']} failed (rc={proc.returncode})")
        print(proc.stdout[-2000:])
        return None
    # Child prints exactly one JSON line as its final stdout line.
    last_json_line = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            last_json_line = line
    if last_json_line is None:
        print(f"[ERROR] {cfg['name']} produced no JSON result")
        print(proc.stdout[-2000:])
        return None
    return json.loads(last_json_line)


def fmt_table(rows: list[dict]) -> str:
    if not rows:
        return "(no results)"
    header = (f"{'config':<18} {'step_ms':>9} {'speedup':>9} "
              f"{'SR':>7} {'rim':>7} {'insert':>7} {'nan':>6}")
    base = next((r for r in rows if r["config_name"] == "baseline"), rows[0])
    base_step = base["step_ms_mean"]
    lines = [header, "-" * len(header)]
    for r in rows:
        spd = base_step / r["step_ms_mean"]
        partial = r.get("partial_rates", {}) or {}
        rim = partial.get("rim", float("nan"))
        ins = partial.get("insertion", float("nan"))
        lines.append(
            f"{r['config_name']:<18} "
            f"{r['step_ms_mean']:>9.1f} "
            f"{spd:>8.2f}x "
            f"{r['success_rate']:>7.3f} "
            f"{rim:>7.3f} "
            f"{ins:>7.3f} "
            f"{r.get('nan_total', 0):>6d}"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    only = set(args.only.split(",")) if args.only else None
    configs = [c for c in CONFIGS if (only is None or c["name"] in only)]
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    # Truncate so this invocation's results aren't mixed with old runs.
    open(args.output_json, "w").close()

    results: list[dict] = []
    for cfg in configs:
        r = run_one(cfg, args)
        if r is not None:
            results.append(r)

    print("\n\n========= SUMMARY =========")
    print(f"num_envs={args.num_envs} horizon={args.horizon}\n")
    print(fmt_table(results))
    print(f"\nFull results: {args.output_json}")


if __name__ == "__main__":
    main()
