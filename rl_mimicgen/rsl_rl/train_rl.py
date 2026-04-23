"""Train an RL policy on a robomimic warp environment using RSL-RL (PPO).

The RL actor is an :class:`rsl_rl.models.RNNModel` (LSTM+MLP+Gaussian head)
warm-started from a pre-trained robomimic BC-RNN policy. Only the BC RNN
backbone is transferred; the Gaussian head and critic are learned from scratch.

Usage (local BC checkpoint)::

    python -m rl_mimicgen.rsl_rl.train_rl --bc_checkpoint runs/coffee_d0_low_dim/.../models/model_450.pth

Usage (wandb source)::

    python -m rl_mimicgen.rsl_rl.train_rl \
        --wandb_run user/proj/abc123 --wandb_model model_450.pth
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

# EGL is the right backend for Linux + NVIDIA; osmesa for WSL/CPU-only.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

# Add robomimic / robosuite / mimicgen to path
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _sub in ("robomimic", "robosuite", "mimicgen"):
    sys.path.insert(0, str(_REPO_ROOT / "resources" / _sub))

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.train_utils import suppress_warp_kernel_warnings

from rsl_rl.runners import OnPolicyRunner

from rl_mimicgen.rsl_rl import (
    BCResumeInfo,
    RobomimicVecEnv,
    build_actor_hidden_dims,
    build_distribution_cfg_from_bc,
    copy_bc_weights_into_actor,
    fetch_wandb_checkpoint,
    load_bc_checkpoint,
    reset_gmm_scale_head,
)
from rl_mimicgen.rsl_rl.warp_buffer_sizes import resolve_warp_buffer_sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL on robomimic env with RSL-RL.")

    # -- BC warm-start source (exactly one required) --
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--bc_checkpoint", type=str, default=None,
                     help="Path to a local robomimic BC-RNN checkpoint (.pth).")
    src.add_argument("--wandb_run", type=str, default=None,
                     help="W&B run reference 'entity/project/run_id' for BC ckpt.")
    parser.add_argument("--wandb_model", type=str, default=None,
                        help="Checkpoint filename within the wandb run (e.g. 'model_450.pth'). "
                             "Required with --wandb_run.")
    parser.add_argument("--no_warm_start", action="store_true",
                        help="Skip copying BC RNN weights into the RL actor "
                             "(env meta and RNN config are still taken from the ckpt).")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to an RL checkpoint (.pt saved by the runner) to "
                             "resume from. Restores actor/critic/optimizer and "
                             "current_learning_iteration. Implies --no_warm_start. "
                             "Still requires --bc_checkpoint/--wandb_run for env "
                             "metadata and actor architecture.")
    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help="W&B run id to resume logging into. Sets WANDB_RUN_ID + "
                             "WANDB_RESUME=allow before the runner inits wandb. Use "
                             "with --resume_checkpoint to continue the same run end-to-end.")

    # -- Environment --
    parser.add_argument("--num_envs", type=int, default=None,
                        help="Number of parallel warp environments. Overrides the value "
                             "in ppo_cfg.json / dapg_cfg.json if set (default there: 4096).")
    parser.add_argument("--horizon", type=int, default=400,
                        help="Maximum episode length (steps).")
    parser.add_argument("--physics_timestep", type=float, default=None,
                        help="Override robosuite.macros.SIMULATION_TIMESTEP "
                             "(default 0.002). Larger values proportionally "
                             "reduce substeps per env.step (25 substeps at "
                             "2ms -> 13 at 4ms) -- biggest sim-perf knob, but "
                             "trades fidelity (see CLAUDE.md warp perf notes). "
                             "Overrides ppo_cfg.json / dapg_cfg.json's "
                             "physics_timestep if set.")
    parser.add_argument("--njmax_per_env", type=int, default=None,
                        help="Override MjSimWarp per-world nefc cap. Default: "
                             "look up from rl_mimicgen.rsl_rl.warp_buffer_sizes."
                             "PER_TASK_WARP_BUFFER_SIZES by env_name, falling "
                             "back to the class default (3500). Shrinking frees "
                             "GPU memory but can select different JIT-specialised "
                             "warp kernels -- see CLAUDE.md.")
    parser.add_argument("--naconmax_per_env", type=int, default=None,
                        help="Override MjSimWarp per-env contact-buffer slice "
                             "(total naconmax = this x num_envs). Same fallback "
                             "chain as --njmax_per_env.")
    parser.add_argument("--graph_capture", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable CUDA graph capture in MjSimWarp.step "
                             "(ROBOSUITE_WARP_GRAPH=1). ~1.89x speedup at 2048 envs, "
                             "accuracy-neutral. At 4096 envs needs ~1 GB headroom -- "
                             "relies on the per-task buffer caps (e.g. 500/15 for "
                             "Coffee) to fit. Pass --no-graph_capture to disable.")

    # -- RL training --
    parser.add_argument("--max_iterations", type=int, default=None,
                        help="Total PPO iterations. Overrides ppo_cfg.json if set "
                             "(default there: 30000).")
    parser.add_argument("--num_steps_per_env", type=int, default=None,
                        help="Rollout steps per env per PPO update. Overrides the value "
                             "in ppo_cfg.json if set (default there: 24).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_interval", type=int, default=None,
                        help="Iterations between checkpoints. Overrides ppo_cfg.json.")

    # -- PPO / DAPG config files --
    _default_ppo_cfg = os.path.join(_REPO_ROOT, "rl_mimicgen", "rsl_rl", "config", "ppo_cfg.json")
    _default_dapg_cfg = os.path.join(_REPO_ROOT, "rl_mimicgen", "rsl_rl", "config", "dapg_cfg.json")
    parser.add_argument("--ppo_cfg", type=str, default=_default_ppo_cfg,
                        help=f"Path to PPO config JSON (default: {_default_ppo_cfg}). "
                             "Contains PPO algorithm hyperparameters, network architecture, "
                             "and obs-group wiring. BC-RNN fields (rnn_type, rnn_hidden_dim, "
                             "rnn_num_layers) are injected at runtime from the BC checkpoint. "
                             "Ignored when --dapg is set.")

    # -- DAPG (demo-augmented PPO; Rajeswaran et al. 2017) --
    parser.add_argument("--dapg", action="store_true",
                        help="Swap PPO for DAPG: each mini-batch also pays a demo "
                             "log-likelihood cost with a decaying weight "
                             "w_k = lambda0 * lambda1^iter * max|A|.")
    parser.add_argument("--dapg_cfg", type=str, default=_default_dapg_cfg,
                        help=f"Path to DAPG config JSON (default: {_default_dapg_cfg}). "
                             "Used instead of --ppo_cfg when --dapg is set.")
    parser.add_argument("--demo_dataset", type=str, default=None,
                        help="Path to HDF5 demonstrations for DAPG. Defaults to the "
                             "dataset path recorded in the BC checkpoint config.")
    parser.add_argument("--dapg_lambda0", type=float, default=None,
                        help="Initial DAPG weight scale. Overrides dapg_cfg.json.")
    parser.add_argument("--dapg_lambda1", type=float, default=None,
                        help="Per-iteration DAPG weight decay. Overrides dapg_cfg.json.")
    parser.add_argument("--dapg_batch_size", type=int, default=None,
                        help="Demo chunks per minibatch. Overrides dapg_cfg.json.")
    parser.add_argument("--demo_seq_length", type=int, default=None,
                        help="Demo chunk length. Defaults to BC-RNN train.seq_length "
                             "from the checkpoint.")

    # -- Env/actor overrides (commonly tuned) --
    parser.add_argument("--clip_actions", type=float, default=None,
                        help="Per-dim action clip. Default: no clipping. Setting a value "
                             "below 1.0 throws away action-range the BC policy was trained "
                             "to use; only set this if you're seeing warp f32 solver "
                             "divergence at large num_envs.")
    parser.add_argument("--init_noise_std", type=float, default=None,
                        help="Initial std for the Tanh-Gaussian head (non-GMM BC only). "
                             "Default: 0.05. Ignored when the BC policy is GMM.")

    # -- Profiling --
    parser.add_argument("--profile", action="store_true",
                        help="Print per-phase wallclock timings (act, env.step, "
                             "process_env_step, compute_returns, update) every "
                             "save_interval iterations. Uses cuda.synchronize().")
    parser.add_argument("--torch_profile", action="store_true",
                        help="Dump a torch.profiler trace over iterations 2-4 to "
                             "<log_dir>/trace.json (viewable in chrome://tracing "
                             "or tensorboard). Adds ~one-iter overhead.")

    # -- Logging --
    parser.add_argument("--experiment_name", type=str, default="robomimic_rl")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--logger", type=str, default="tensorboard",
                        choices=["tensorboard", "wandb"])
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="W&B project name. Defaults to '{task}_{obs_type}_ppo' "
                             "derived from the BC env_meta (e.g. 'coffee_low_dim_ppo').")
    parser.add_argument("--wandb_group", type=str, default=None,
                        help="W&B run group. Plumbed as WANDB_RUN_GROUP env var so rsl_rl's "
                             "wandb.init picks it up. Useful for tying a BC+RL pipeline together.")

    # -- Video logging --
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=None,
                        help="Log an eval-rollout video of env 0 at (approximately) each "
                             "video_interval env-steps, beginning at the start of the first "
                             "episode on/after that boundary. Enables offscreen rendering on "
                             "the warp env. If unset, uses ppo_cfg.json / dapg_cfg.json's "
                             "video.enabled (default true). Pass --no-video to disable. "
                             "Framing (width/height/fps/camera) lives in the 'video' config block.")
    parser.add_argument("--video_interval", type=int, default=None,
                        help="Minimum learning iterations between recorded clips (internally "
                             "multiplied by num_steps_per_env). Recording is armed once the "
                             "interval elapses, and capture starts at the next env-0 episode "
                             "boundary. Defaults to video.interval in the config (100 iters), "
                             "or save_interval if neither is set.")

    return parser.parse_args()


def _install_profiling_hooks(
    vec_env: Any,
    runner: Any,
    device: torch.device,
    report_every: int,
    torch_profiler: Any | None = None,
) -> None:
    """Wrap env/alg methods with cuda-synced timers; print totals every `report_every`
    calls to alg.update (one per PPO iter). If torch_profiler is provided, also
    step it once per update.
    """
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    state = {"iter": 0}
    use_cuda = device.type == "cuda"

    def _wrap(obj: Any, name: str, label: str) -> None:
        orig = getattr(obj, name)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = orig(*args, **kwargs)
            if use_cuda:
                torch.cuda.synchronize()
            totals[label] += time.perf_counter() - t0
            counts[label] += 1
            return out

        setattr(obj, name, wrapped)

    _wrap(vec_env, "step", "env.step")
    _wrap(runner.alg, "act", "alg.act")
    _wrap(runner.alg, "process_env_step", "alg.process_env_step")
    _wrap(runner.alg, "compute_returns", "alg.compute_returns")

    orig_update = runner.alg.update

    def timed_update(*args: Any, **kwargs: Any) -> Any:
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = orig_update(*args, **kwargs)
        if use_cuda:
            torch.cuda.synchronize()
        totals["alg.update"] += time.perf_counter() - t0
        counts["alg.update"] += 1
        state["iter"] += 1
        if torch_profiler is not None:
            torch_profiler.step()
        if state["iter"] % report_every == 0:
            total = sum(totals.values())
            print(f"\n[PROFILE] over last {report_every} iters (total {total:.2f}s, "
                  f"{total / report_every:.3f}s/iter):")
            for label in ["alg.act", "env.step", "alg.process_env_step",
                          "alg.compute_returns", "alg.update"]:
                t = totals[label]
                pct = 100.0 * t / total if total > 0 else 0.0
                n = counts[label]
                per = t / n if n else 0.0
                print(f"  {label:<25s} {t:7.2f}s  ({pct:5.1f}%)  n={n:<6d} {per*1000:7.2f} ms/call")
            totals.clear()
            counts.clear()
        return out

    runner.alg.update = timed_update


def _install_train_metrics_hook(runner: Any) -> None:
    """After each ``alg.update``, flush ``alg._train_metrics`` under ``Train/*``.

    Metrics (kl_max, kl_mean, clip_frac, ratio_max) are computed by DAPG's
    per-minibatch loop to diagnose off-policy drift. Adaptive LR only polices
    *mean* KL; tails (max KL / clip fraction / ratio blowup) are what correlate
    with SR oscillation on precision tasks. No-op for algorithms that don't
    populate ``_train_metrics``.
    """
    orig_update = runner.alg.update
    state = {"iter": int(getattr(runner, "current_learning_iteration", 0))}

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        out = orig_update(*args, **kwargs)
        metrics = getattr(runner.alg, "_train_metrics", None)
        writer = getattr(getattr(runner, "logger", None), "writer", None)
        if metrics and writer is not None:
            it = state["iter"]
            for k, v in metrics.items():
                writer.add_scalar(f"Train/{k}", float(v), it)
        state["iter"] += 1
        return out

    runner.alg.update = wrapped


def _default_wandb_project(env_meta: dict, obs_modalities: dict, algo: str = "ppo") -> str:
    """Derive a default wandb project like 'coffee_low_dim_ppo' from BC metadata.

    Strips mimicgen dataset variant suffixes (e.g. '_D0', '_d1') from the env
    name and picks 'image' vs 'low_dim' from the observation modalities.
    """
    env_name = str(env_meta.get("env_name", "robomimic")).lower()
    env_name = re.sub(r"_d\d+$", "", env_name)
    has_rgb = bool(obs_modalities.get("obs", {}).get("rgb"))
    obs_type = "image" if has_rgb else "low_dim"
    return f"{env_name}_{obs_type}_{algo}"


def _resolve_bc_source(args: argparse.Namespace, download_dir: str) -> BCResumeInfo:
    if args.bc_checkpoint is not None:
        return load_bc_checkpoint(args.bc_checkpoint)
    if args.wandb_model is None:
        raise SystemExit("--wandb_run requires --wandb_model")
    local_path = fetch_wandb_checkpoint(args.wandb_run, args.wandb_model, download_dir)
    print(f"[INFO] Downloaded wandb checkpoint to: {local_path}")
    return load_bc_checkpoint(local_path)


def _build_train_cfg(args: argparse.Namespace, bc_info: BCResumeInfo) -> dict[str, Any]:
    """Load PPO or DAPG config JSON, inject BC-derived RNN fields and runtime
    settings, and apply CLI overrides. Returns the dict passed to OnPolicyRunner.
    """
    cfg_path = args.dapg_cfg if args.dapg else args.ppo_cfg
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    # Mirror BC actor shape so weight transfer is a structural copy; JSON fields are fallbacks.
    actor = cfg["actor"]
    actor["rnn_type"] = bc_info.rnn_type
    actor["rnn_hidden_dim"] = bc_info.rnn_hidden_dim
    actor["rnn_num_layers"] = bc_info.rnn_num_layers
    hidden_dims, activation = build_actor_hidden_dims(bc_info)
    actor["hidden_dims"] = hidden_dims
    actor["activation"] = activation
    # Precedence for distribution_cfg: CLI --init_noise_std > JSON > BC-seeded defaults.
    # Architectural keys (class_name, num_modes) are dropped — they're tied to the
    # BC head. Remaining JSON keys layer onto the BC-seeded dict; keys not valid for
    # the current head class are warned and skipped. For GMM, `init_std` drives a
    # post-transfer scale-head reset instead (stashed on args for main() to apply).
    json_dist_cfg = dict(actor.get("distribution_cfg", {}))
    json_dist_cfg.pop("class_name", None)
    if "num_modes" in json_dist_cfg:
        print(f"[WARN] Ignoring distribution_cfg.num_modes={json_dist_cfg.pop('num_modes')} "
              f"from JSON; tied to BC MLP output width.")
    resolved_init_std = (
        args.init_noise_std
        if args.init_noise_std is not None
        else json_dist_cfg.pop("init_std", None)
    )
    actor["distribution_cfg"] = build_distribution_cfg_from_bc(
        bc_info,
        gaussian_init_std=resolved_init_std if resolved_init_std is not None else 0.05,
    )
    # Filter remaining JSON keys by what the current head class accepts.
    _GMM_KEYS = {"min_std", "std_activation", "low_noise_eval", "use_tanh"}
    _GAUSSIAN_KEYS = {"std_type"}
    accepted = _GMM_KEYS if bc_info.is_gmm else _GAUSSIAN_KEYS
    for k, v in list(json_dist_cfg.items()):
        if k in accepted:
            actor["distribution_cfg"][k] = v
        else:
            print(f"[WARN] distribution_cfg.{k}={v} ignored "
                  f"(not valid for {'GMM' if bc_info.is_gmm else 'TanhGaussian'} head).")
    # Side-channel: for GMM, apply init_std as a scale-head reset post-BC-transfer.
    args._resolved_gmm_init_std = resolved_init_std if bc_info.is_gmm else None

    # Runtime fields (always CLI-driven).
    cfg["seed"] = args.seed
    cfg["device"] = args.device
    cfg["experiment_name"] = args.experiment_name
    cfg["run_name"] = args.run_name
    cfg["logger"] = args.logger
    cfg["wandb_project"] = args.wandb_project

    # Optional CLI overrides (only apply if user passed the flag).
    if args.max_iterations is not None:
        cfg["max_iterations"] = args.max_iterations
    if args.num_steps_per_env is not None:
        cfg["num_steps_per_env"] = args.num_steps_per_env
    if args.num_envs is not None:
        cfg["num_envs"] = args.num_envs
    if args.save_interval is not None:
        cfg["save_interval"] = args.save_interval
    # DAPG-specific: inject demo dataset info + apply CLI overrides.
    if args.dapg:
        alg = cfg["algorithm"]
        demo_path = args.demo_dataset or bc_info.dataset_path
        if demo_path is None:
            raise SystemExit(
                "DAPG requires a demonstration dataset. Pass --demo_dataset or "
                "use a BC checkpoint whose config records train.data."
            )
        alg["demo_dataset_path"] = demo_path
        alg["demo_obs_keys"] = list(bc_info.obs_keys)
        if alg.get("demo_seq_length") is None:
            alg["demo_seq_length"] = bc_info.seq_length
        if args.demo_seq_length is not None:
            alg["demo_seq_length"] = args.demo_seq_length
        if args.dapg_lambda0 is not None:
            alg["dapg_lambda0"] = args.dapg_lambda0
        if args.dapg_lambda1 is not None:
            alg["dapg_lambda1"] = args.dapg_lambda1
        if args.dapg_batch_size is not None:
            alg["dapg_batch_size"] = args.dapg_batch_size

    return cfg


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log_root = os.path.abspath(os.path.join("runs", args.experiment_name))
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.run_name:
        run_stamp += f"_{args.run_name}"
    log_dir = os.path.join(log_root, run_stamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging to: {log_dir}")

    bc_info = _resolve_bc_source(args, download_dir=os.path.join(log_dir, "bc_ckpt"))
    print(f"[INFO] BC-RNN config: type={bc_info.rnn_type} hidden_dim={bc_info.rnn_hidden_dim} "
          f"num_layers={bc_info.rnn_num_layers} obs_dim={bc_info.obs_dim} action_dim={bc_info.action_dim}")
    print(f"[INFO] BC obs order: {bc_info.obs_keys}")

    if args.wandb_project is None:
        algo_tag = "dapg" if args.dapg else "ppo"
        args.wandb_project = _default_wandb_project(bc_info.env_meta, bc_info.obs_modalities, algo_tag)
        if args.logger == "wandb":
            print(f"[INFO] W&B project (auto): {args.wandb_project}")

    if args.wandb_group and args.logger == "wandb":
        os.environ["WANDB_RUN_GROUP"] = args.wandb_group
        print(f"[INFO] W&B group: {args.wandb_group}")

    if args.wandb_run_id and args.logger == "wandb":
        os.environ["WANDB_RUN_ID"] = args.wandb_run_id
        os.environ.setdefault("WANDB_RESUME", "allow")
        print(f"[INFO] W&B resume: id={args.wandb_run_id} mode={os.environ['WANDB_RESUME']}")

    train_cfg = _build_train_cfg(args, bc_info)
    args.num_envs = int(train_cfg.get("num_envs", 4096))
    cfg_path_used = args.dapg_cfg if args.dapg else args.ppo_cfg
    print(f"[INFO] Loaded {'DAPG' if args.dapg else 'PPO'} config from: {cfg_path_used}")
    dist_cfg = train_cfg["actor"]["distribution_cfg"]
    dist_tag = dist_cfg["class_name"].rsplit(":", 1)[-1]
    noise_tag = (
        f"num_modes={dist_cfg['num_modes']} min_std={dist_cfg.get('min_std')}" if bc_info.is_gmm
        else f"init_std={dist_cfg['init_std']} std_type={dist_cfg.get('std_type')}"
    )
    print(f"[INFO] Actor: hidden_dims={train_cfg['actor']['hidden_dims']} "
          f"dist={dist_tag} {noise_tag}")
    print(f"[INFO] num_steps_per_env={train_cfg['num_steps_per_env']}  "
          f"lr={train_cfg['algorithm']['learning_rate']}")
    if args.dapg:
        alg = train_cfg["algorithm"]
        print(f"[INFO] DAPG: demo={alg['demo_dataset_path']} "
              f"seq_length={alg['demo_seq_length']} batch={alg['dapg_batch_size']} "
              f"lambda0={alg['dapg_lambda0']} lambda1={alg['dapg_lambda1']}")

    # Must set SIMULATION_TIMESTEP pre-env-creation: robosuite reads it at compile
    # and propagates into each controller's timestep.
    cfg_timestep = train_cfg.get("physics_timestep")
    phys_ts = args.physics_timestep if args.physics_timestep is not None else cfg_timestep
    if phys_ts is not None:
        import robosuite.macros as _rs_macros
        _rs_macros.SIMULATION_TIMESTEP = float(phys_ts)
        print(f"[INFO] SIMULATION_TIMESTEP -> {_rs_macros.SIMULATION_TIMESTEP} "
              f"(substeps/env.step = {int(0.05 / float(phys_ts))})")

    # Prime robomimic's global obs-modality registry before creating the env.
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=[bc_info.obs_modalities])

    # Resolve video setting: CLI --video/--no-video overrides config.video.enabled.
    video_cfg = train_cfg.get("video", {}) or {}
    video_enabled = args.video if args.video is not None else bool(video_cfg.get("enabled", True))

    device = torch.device(args.device)
    env_meta = bc_info.env_meta
    cfg_env_kwargs: dict = train_cfg.get("env_kwargs", {}) or {}
    if cfg_env_kwargs:
        env_meta.setdefault("env_kwargs", {}).update(cfg_env_kwargs)

    # Per-task MjSimWarp buffer-size overrides. Table -> CLI override order.
    # CLI values (if set) win over the table; task is inferred from env_name.
    warp_caps = resolve_warp_buffer_sizes(env_meta.get("env_name")) or {}
    if args.njmax_per_env is not None:
        warp_caps["njmax_per_env"] = int(args.njmax_per_env)
    if args.naconmax_per_env is not None:
        warp_caps["naconmax_per_env"] = int(args.naconmax_per_env)
    if warp_caps:
        env_meta.setdefault("env_kwargs", {}).update(warp_caps)

    # Graph capture is read from env var inside MjSimWarp.__init__, so set it
    # before robosuite.make fires via create_env_from_metadata.
    os.environ["ROBOSUITE_WARP_GRAPH"] = "1" if args.graph_capture else "0"

    print(f"[INFO] Creating warp env (num_envs={args.num_envs}, horizon={args.horizon}, "
          f"env_kwargs={cfg_env_kwargs}, warp_caps={warp_caps or '(class defaults)'}, "
          f"graph_capture={args.graph_capture})")
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=video_enabled,
        use_image_obs=False,
        use_warp=True,
        num_envs=args.num_envs,
    )

    video_trigger: Optional[Callable[[int], bool]] = None
    if video_enabled:
        if args.video_interval is not None:
            video_interval_iters = args.video_interval
        elif video_cfg.get("interval") is not None:
            video_interval_iters = int(video_cfg["interval"])
        else:
            video_interval_iters = train_cfg["save_interval"]
        video_interval_iters = max(1, video_interval_iters)
        video_interval = video_interval_iters * train_cfg["num_steps_per_env"]
        video_trigger = lambda s: s % video_interval == 0  # noqa: E731
        print(f"[INFO] Video: arming every {video_interval_iters} learning iterations "
              f"({video_interval} env steps); each clip starts at the next env-0 episode boundary")

    vec_env = RobomimicVecEnv(
        env=env,
        horizon=args.horizon,
        device=args.device,
        clip_actions=args.clip_actions,
        obs_keys=bc_info.obs_keys,
        video_trigger=video_trigger,
        video_dir=log_dir,
        video_width=int(video_cfg.get("width", 256)),
        video_height=int(video_cfg.get("height", 256)),
        video_fps=int(video_cfg.get("fps", 20)),
        video_camera=video_cfg.get("camera"),
    )

    obs_td = vec_env.get_observations()
    obs_dim = obs_td["policy"].shape[-1]
    if obs_dim != bc_info.obs_dim:
        raise RuntimeError(
            f"Env obs_dim ({obs_dim}) does not match BC obs_dim ({bc_info.obs_dim}). "
            "Env metadata / obs_keys mismatch."
        )
    print(f"[INFO] obs_dim={obs_dim}  action_dim={vec_env.num_actions}  num_envs={vec_env.num_envs}")

    runner = OnPolicyRunner(vec_env, train_cfg, log_dir=log_dir, device=args.device)

    if args.resume_checkpoint:
        print("[INFO] --resume_checkpoint given; skipping BC warm-start.")
    elif not args.no_warm_start:
        loaded, skipped = copy_bc_weights_into_actor(runner.alg.actor, bc_info)
        print(f"[INFO] BC warm-start: copied {len(loaded)} params (RNN + MLP + head), "
              f"skipped {len(skipped)}")
        if skipped:
            print(f"[WARN] Skipped (shape mismatch or missing): {skipped}")
        # GMM has no state-independent std; apply init_std (if set) as a scale-head
        # reset so the learnable scale starts near target instead of BC-learned ~0.05.
        gmm_init = getattr(args, "_resolved_gmm_init_std", None)
        if gmm_init is not None and bc_info.is_gmm:
            reset_gmm_scale_head(runner.alg.actor, bc_info, float(gmm_init))
            print(f"[INFO] GMM scale head reset: per-component init_std={gmm_init} "
                  f"(weights zeroed, bias={math.log(math.exp(gmm_init - float(bc_info.gmm_kwargs.get('min_std', 1e-4))) - 1.0):.4f})")
        runner.alg.actor.to(device)
    else:
        print("[INFO] BC warm-start disabled (--no_warm_start); actor initialized randomly.")

    if args.resume_checkpoint:
        resume_path = os.path.abspath(args.resume_checkpoint)
        runner.load(resume_path)
        print(f"[INFO] Resumed from {resume_path} at iter {runner.current_learning_iteration}")

    _install_train_metrics_hook(runner)

    print(f"[INFO] Starting {'DAPG' if args.dapg else 'PPO'}: "
          f"{train_cfg['max_iterations']} iters, "
          f"{train_cfg['num_steps_per_env']} steps/env/update, {args.num_envs} envs")

    torch_prof = None
    if args.torch_profile:
        trace_path = os.path.join(log_dir, "trace.json")
        torch_prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=lambda p: (p.export_chrome_trace(trace_path),
                                      print(f"[PROFILE] torch trace -> {trace_path}")),
        )
        torch_prof.start()

    if args.profile or torch_prof is not None:
        report_every = args.save_interval if args.save_interval else train_cfg.get("save_interval", 50)
        _install_profiling_hooks(vec_env, runner, device, report_every=max(1, int(report_every)),
                                 torch_profiler=torch_prof)
        print(f"[INFO] Profiling enabled (report every {report_every} iters)")

    try:
        with suppress_warp_kernel_warnings():
            runner.learn(num_learning_iterations=train_cfg["max_iterations"], init_at_random_ep_len=True)
    finally:
        if torch_prof is not None:
            torch_prof.stop()

    print("[INFO] Training complete.")
    vec_env.close()


if __name__ == "__main__":
    main()
