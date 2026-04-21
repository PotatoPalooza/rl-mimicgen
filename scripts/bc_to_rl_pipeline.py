"""Sequential BC→RL pipeline for robomimic RNN policies on MimicGen tasks.

Runs, in order:

1. BC training via ``paper_bc_one_task.py`` with async warp rollouts at a
   higher env count (default 256) for low-variance success-rate signals.
2. Scans the generated BC output directory for the best-SR checkpoint
   (robomimic's async path only writes a ``model_epoch_<N>.pth`` snapshot
   when a new best rollout is seen — we pick the one with the highest
   ``variable_state.best_success_rate``).
3. RL fine-tuning via ``rl_mimicgen.rsl_rl.train_rl`` with DAPG + warp +
   2048 envs, warm-started from that BC checkpoint.

BC and RL runs share a wandb ``group`` tag so they link in the UI. The
BC run uses project ``<task>_<modality>``; the RL run uses
``<task>_<modality>_<algo>`` (auto-derived by ``train_rl.py``).

Usage::

    python scripts/bc_to_rl_pipeline.py --task coffee --variant D0

Pass ``--skip-bc --bc-output-dir <path>`` to reuse an existing BC run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = WORKSPACE_DIR / "runs"

# Pipeline-level defaults baked in (deliberately not CLI flags to keep the
# entry point minimal). Override via --bc_extra / --rl_extra if needed:
#   --bc_extra '--rollout-num-envs 128'
#   --rl_extra '--num_envs 1024'
MODALITY = "low_dim"
BC_ROLLOUT_NUM_ENVS = 256  # low-variance SR signal for paper_bc_one_task
RL_NUM_ENVS = 2048


@dataclass
class PipelineSummary:
    group: str
    task: str
    variant: str
    modality: str
    algo: str
    bc_output_dir: str
    bc_best_checkpoint: str
    bc_best_success_rate: float
    bc_best_epoch: int
    rl_log_dir: Optional[str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, help="MimicGen paper task (e.g. coffee, square).")
    p.add_argument("--variant", required=True, help="Dataset variant (D0, D1, D2, O1, ...).")
    p.add_argument("--algo", default="dapg", choices=("ppo", "dapg"),
                   help="RL algorithm for the second stage. Default: dapg.")
    p.add_argument("--rl_max_iterations", type=int, default=15000,
                   help="RL learning iterations. Default: 15000 (~20 h at 2048 envs / 4.7s per iter).")
    p.add_argument("--mujoco-gl", dest="mujoco_gl", default=None,
                   choices=("glx", "egl", "osmesa"),
                   help="MuJoCo GL backend (exported as MUJOCO_GL for the BC + RL "
                        "subprocesses). Use `osmesa` on WSL.")

    # Shared infra.
    p.add_argument("--run_root", type=Path, default=DEFAULT_RUN_ROOT,
                   help="Shared run output root. Default: <workspace>/runs.")
    p.add_argument("--group", type=str, default=None,
                   help="Wandb group. Default: <variant> (e.g. d0).")

    # Reuse an existing BC run.
    p.add_argument("--skip_bc", action="store_true",
                   help="Skip BC training and reuse --bc_output_dir.")
    p.add_argument("--bc_output_dir", type=Path, default=None,
                   help="Existing BC output dir (contains models/). Required with --skip_bc.")

    # Passthroughs.
    p.add_argument("--bc_extra", type=str, default="",
                   help="Extra args appended to paper_bc_one_task.py (shell-split).")
    p.add_argument("--rl_extra", type=str, default="",
                   help="Extra args appended to train_rl.py (shell-split).")

    return p.parse_args()


def _experiment_name(task: str, variant: str, modality: str) -> str:
    # Matches paper_bc_one_task.rewrite_training_output_dirs: <task>_<variant>_<modality>.
    return f"{task.lower()}_{variant.lower()}_{modality}"


def _latest_bc_run_dir(run_root: Path, experiment_name: str) -> Path:
    """Pick the most-recent timestamped run under ``<run_root>/<experiment_name>/``."""
    parent = run_root / experiment_name
    if not parent.is_dir():
        raise FileNotFoundError(f"No BC experiment dir at {parent}")
    candidates = [p for p in parent.iterdir() if p.is_dir() and (p / "models").is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No timestamped runs with models/ under {parent}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _best_bc_checkpoint(models_dir: Path) -> tuple[Path, float, int]:
    """Scan ``model_epoch_*.pth`` for the highest best_success_rate.

    Robomimic's async-rollout path writes a snapshot only when a new best
    (return or SR) is seen — the snapshot's ``variable_state.best_success_rate``
    is the SR that triggered the save.
    """
    candidates = sorted(models_dir.glob("model_epoch_*.pth"))
    if not candidates:
        raise FileNotFoundError(
            f"No model_epoch_*.pth under {models_dir}. Was async_rollouts enabled?"
        )
    best_path: Optional[Path] = None
    best_sr = -1.0
    best_epoch = -1
    for ckpt_path in candidates:
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        var_state = blob.get("variable_state") or {}
        sr_dict = var_state.get("best_success_rate") or {}
        sr = max(sr_dict.values()) if sr_dict else -1.0
        epoch = int(var_state.get("epoch", -1))
        if sr > best_sr or (sr == best_sr and epoch > best_epoch):
            best_sr = sr
            best_epoch = epoch
            best_path = ckpt_path
    assert best_path is not None
    return best_path, float(best_sr), best_epoch


def _run(cmd: list[str], *, env: Optional[dict[str, str]] = None, cwd: Optional[Path] = None) -> int:
    print(f"\n>>> {' '.join(cmd)}\n", flush=True)
    proc = subprocess.run(cmd, env=env, cwd=str(cwd) if cwd else None)
    return proc.returncode


def _run_bc(args: argparse.Namespace, group: str) -> Path:
    cmd = [
        sys.executable, "-m", "rl_mimicgen.mimicgen.paper_bc_one_task",
        "--task", args.task,
        "--variant", args.variant,
        "--modality", MODALITY,
        "--algo", "bc",
        "--warp",
        "--logger", "wandb",
        "--wandb-group", group,
        "--async-rollouts",
        "--rollout-num-envs", str(BC_ROLLOUT_NUM_ENVS),
        "--run-root", str(args.run_root.resolve()),
    ]
    if args.bc_extra:
        import shlex
        cmd += shlex.split(args.bc_extra)
    rc = _run(cmd, cwd=WORKSPACE_DIR)
    if rc != 0:
        raise SystemExit(f"BC training failed with exit code {rc}; halting before RL.")
    return _latest_bc_run_dir(args.run_root.resolve(), _experiment_name(args.task, args.variant, MODALITY))


def _run_rl(args: argparse.Namespace, group: str, bc_checkpoint: Path) -> Optional[Path]:
    rl_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cmd = [
        sys.executable, "-m", "rl_mimicgen.rsl_rl.train_rl",
        "--bc_checkpoint", str(bc_checkpoint),
        "--num_envs", str(RL_NUM_ENVS),
        "--max_iterations", str(args.rl_max_iterations),
        "--logger", "wandb",
        "--wandb_group", group,
        "--run_name", group,
    ]
    if args.algo == "dapg":
        cmd.append("--dapg")
    if args.rl_extra:
        import shlex
        cmd += shlex.split(args.rl_extra)

    # Snapshot runs/robomimic_rl/ pre-run so we can identify the new dir after.
    rl_parent = args.run_root.resolve() / "robomimic_rl"
    before = {p.name for p in rl_parent.glob("*")} if rl_parent.is_dir() else set()

    rc = _run(cmd, cwd=WORKSPACE_DIR)
    if rc != 0:
        raise SystemExit(f"RL training failed with exit code {rc}.")

    # Best-effort locate the just-created RL run dir (contains our group as run_name suffix).
    if rl_parent.is_dir():
        new_dirs = sorted(
            (p for p in rl_parent.glob(f"*_{group}") if p.is_dir() and p.name not in before),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if new_dirs:
            return new_dirs[0]
    return None


def main() -> int:
    args = parse_args()

    if args.mujoco_gl:
        os.environ["MUJOCO_GL"] = args.mujoco_gl

    if args.skip_bc and args.bc_output_dir is None:
        raise SystemExit("--skip_bc requires --bc_output_dir.")

    group = args.group or args.variant.lower()
    print(f"[pipeline] group = {group}")

    # Stage 1: BC.
    if args.skip_bc:
        assert args.bc_output_dir is not None
        bc_run_dir = args.bc_output_dir.resolve()
        print(f"[pipeline] skipping BC, using {bc_run_dir}")
    else:
        t0 = time.time()
        bc_run_dir = _run_bc(args, group)
        print(f"[pipeline] BC done in {time.time() - t0:.1f}s; output at {bc_run_dir}")

    models_dir = bc_run_dir / "models"
    best_ckpt, best_sr, best_epoch = _best_bc_checkpoint(models_dir)
    print(f"[pipeline] best BC checkpoint: {best_ckpt} (SR={best_sr:.3f}, epoch={best_epoch})")

    # Stage 2: RL.
    t0 = time.time()
    rl_log_dir = _run_rl(args, group, best_ckpt)
    print(f"[pipeline] RL done in {time.time() - t0:.1f}s; log dir = {rl_log_dir or '(unknown)'}")

    # Summary.
    summary = PipelineSummary(
        group=group,
        task=args.task,
        variant=args.variant,
        modality=MODALITY,
        algo=args.algo,
        bc_output_dir=str(bc_run_dir),
        bc_best_checkpoint=str(best_ckpt),
        bc_best_success_rate=best_sr,
        bc_best_epoch=best_epoch,
        rl_log_dir=str(rl_log_dir) if rl_log_dir else None,
    )
    summary_dir = args.run_root.resolve() / "logs" / "bc_to_rl" / group
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")
    print(f"[pipeline] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
