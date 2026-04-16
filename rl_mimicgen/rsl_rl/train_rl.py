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

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

# Add robomimic / robosuite / mimicgen to path
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _sub in ("robomimic", "robosuite", "mimicgen"):
    sys.path.insert(0, str(_REPO_ROOT / "resources" / _sub))

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

from rsl_rl.runners import OnPolicyRunner

from rl_mimicgen.rsl_rl import (
    RobomimicVecEnv,
    build_actor_hidden_dims,
    build_distribution_cfg_from_bc,
    copy_bc_weights_into_actor,
    fetch_wandb_checkpoint,
    load_bc_checkpoint,
)


def parse_args():
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

    # -- Environment --
    parser.add_argument("--num_envs", type=int, default=50,
                        help="Number of parallel warp environments.")
    parser.add_argument("--horizon", type=int, default=400,
                        help="Maximum episode length (steps).")

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

    # -- Logging --
    parser.add_argument("--experiment_name", type=str, default="robomimic_rl")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--logger", type=str, default="tensorboard",
                        choices=["tensorboard", "wandb"])
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="W&B project name. Defaults to '{task}_{obs_type}_ppo' "
                             "derived from the BC env_meta (e.g. 'coffee_low_dim_ppo').")

    # -- Video logging --
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=None,
                        help="Log an eval-rollout video of env 0 at (approximately) each "
                             "video_interval env-steps, beginning at the start of the first "
                             "episode on/after that boundary. Enables offscreen rendering on "
                             "the warp env. If unset, uses ppo_cfg.json / dapg_cfg.json's "
                             "video.enabled (default true). Pass --no-video to disable. "
                             "Framing (width/height/fps/camera) lives in the 'video' config block.")
    parser.add_argument("--video_interval", type=int, default=None,
                        help="Minimum env-steps between recorded clips. Recording is armed once "
                             "the interval elapses, and capture starts at the next env-0 episode "
                             "boundary. Defaults to video.interval in the config, or "
                             "num_steps_per_env * save_interval if neither is set.")

    return parser.parse_args()


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


def _resolve_bc_source(args, download_dir: str):
    if args.bc_checkpoint is not None:
        return load_bc_checkpoint(args.bc_checkpoint)
    if args.wandb_model is None:
        raise SystemExit("--wandb_run requires --wandb_model")
    local_path = fetch_wandb_checkpoint(args.wandb_run, args.wandb_model, download_dir)
    print(f"[INFO] Downloaded wandb checkpoint to: {local_path}")
    return load_bc_checkpoint(local_path)


def _build_train_cfg(args, bc_info) -> dict:
    """Load PPO or DAPG config JSON, inject BC-derived RNN fields and runtime
    settings, and apply CLI overrides. Returns the dict passed to OnPolicyRunner.
    """
    cfg_path = args.dapg_cfg if args.dapg else args.ppo_cfg
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    # BC-derived: shape the whole actor (RNN + MLP + head) to mirror the BC
    # policy, so the weight transfer below is a structural copy rather than a
    # random-init warm-start. The JSON file's actor fields act as fallbacks
    # for anything BC doesn't pin down.
    actor = cfg["actor"]
    actor["rnn_type"] = bc_info.rnn_type
    actor["rnn_hidden_dim"] = bc_info.rnn_hidden_dim
    actor["rnn_num_layers"] = bc_info.rnn_num_layers
    hidden_dims, activation = build_actor_hidden_dims(bc_info)
    actor["hidden_dims"] = hidden_dims
    actor["activation"] = activation
    actor["distribution_cfg"] = build_distribution_cfg_from_bc(
        bc_info,
        gaussian_init_std=args.init_noise_std if args.init_noise_std is not None else 0.05,
    )

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
    if args.save_interval is not None:
        cfg["save_interval"] = args.save_interval
    if args.init_noise_std is not None and not bc_info.is_gmm:
        # GMM has no state-independent std; CLI override only applies to the
        # Tanh-Gaussian fallback used for non-GMM BC policies.
        cfg["actor"]["distribution_cfg"]["init_std"] = args.init_noise_std

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


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # 1. Log directory (also used as the BC-ckpt download cache)
    # ------------------------------------------------------------------
    log_root = os.path.abspath(os.path.join("runs", args.experiment_name))
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.run_name:
        run_stamp += f"_{args.run_name}"
    log_dir = os.path.join(log_root, run_stamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging to: {log_dir}")

    # ------------------------------------------------------------------
    # 2. Load BC-RNN source (local or wandb) -> config + weights
    # ------------------------------------------------------------------
    bc_info = _resolve_bc_source(args, download_dir=os.path.join(log_dir, "bc_ckpt"))
    print(f"[INFO] BC-RNN config: type={bc_info.rnn_type} hidden_dim={bc_info.rnn_hidden_dim} "
          f"num_layers={bc_info.rnn_num_layers} obs_dim={bc_info.obs_dim} action_dim={bc_info.action_dim}")
    print(f"[INFO] BC obs order: {bc_info.obs_keys}")

    if args.wandb_project is None:
        algo_tag = "dapg" if args.dapg else "ppo"
        args.wandb_project = _default_wandb_project(bc_info.env_meta, bc_info.obs_modalities, algo_tag)
        if args.logger == "wandb":
            print(f"[INFO] W&B project (auto): {args.wandb_project}")

    # ------------------------------------------------------------------
    # 3. Build RSL-RL training config (ppo_cfg.json / dapg_cfg.json + BC fields + CLI)
    # ------------------------------------------------------------------
    train_cfg = _build_train_cfg(args, bc_info)
    cfg_path_used = args.dapg_cfg if args.dapg else args.ppo_cfg
    print(f"[INFO] Loaded {'DAPG' if args.dapg else 'PPO'} config from: {cfg_path_used}")
    dist_cfg = train_cfg["actor"]["distribution_cfg"]
    dist_tag = dist_cfg["class_name"].rsplit(":", 1)[-1]
    noise_tag = (
        f"num_modes={dist_cfg['num_modes']}" if bc_info.is_gmm
        else f"init_std={dist_cfg['init_std']}"
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

    # ------------------------------------------------------------------
    # 4. Create robomimic warp environment (from BC env metadata)
    # ------------------------------------------------------------------
    # Prime robomimic's global obs-modality registry before creating the env.
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=[bc_info.obs_modalities])

    # Resolve video setting: CLI --video/--no-video overrides config.video.enabled.
    video_cfg = train_cfg.get("video", {}) or {}
    video_enabled = args.video if args.video is not None else bool(video_cfg.get("enabled", True))

    device = torch.device(args.device)
    env_meta = bc_info.env_meta
    # Config-driven env kwargs (e.g. extra_randomization, tipover_prob). Only coffee-family
    # envs accept them; other robosuite envs would reject unknown kwargs.
    cfg_env_kwargs: dict = train_cfg.get("env_kwargs", {}) or {}
    if cfg_env_kwargs and "coffee" in str(env_meta.get("env_name", "")).lower():
        env_meta.setdefault("env_kwargs", {}).update(cfg_env_kwargs)
    print(f"[INFO] Creating warp env (num_envs={args.num_envs}, horizon={args.horizon}, "
          f"env_kwargs={cfg_env_kwargs})")
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
            video_interval = args.video_interval
        elif video_cfg.get("interval") is not None:
            video_interval = int(video_cfg["interval"])
        else:
            video_interval = train_cfg["num_steps_per_env"] * train_cfg["save_interval"]
        video_interval = max(1, video_interval)
        video_trigger = lambda s: s % video_interval == 0  # noqa: E731
        print(f"[INFO] Video: arming every {video_interval} env steps; "
              f"each clip starts at the next env-0 episode boundary")

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

    # ------------------------------------------------------------------
    # 5. Runner + BC warm-start
    # ------------------------------------------------------------------
    runner = OnPolicyRunner(vec_env, train_cfg, log_dir=log_dir, device=args.device)

    if not args.no_warm_start:
        loaded, skipped = copy_bc_weights_into_actor(runner.alg.actor, bc_info)
        print(f"[INFO] BC warm-start: copied {len(loaded)} params (RNN + MLP + head), "
              f"skipped {len(skipped)}")
        if skipped:
            print(f"[WARN] Skipped (shape mismatch or missing): {skipped}")
        runner.alg.actor.to(device)
    else:
        print("[INFO] BC warm-start disabled (--no_warm_start); actor initialized randomly.")

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    print(f"[INFO] Starting {'DAPG' if args.dapg else 'PPO'}: "
          f"{train_cfg['max_iterations']} iters, "
          f"{train_cfg['num_steps_per_env']} steps/env/update, {args.num_envs} envs")
    runner.learn(num_learning_iterations=train_cfg["max_iterations"], init_at_random_ep_len=True)

    print("[INFO] Training complete.")
    vec_env.close()


if __name__ == "__main__":
    main()
