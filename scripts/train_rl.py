"""Train an RL policy on a robomimic warp environment using RSL-RL (PPO).

The RL actor is an :class:`rsl_rl.models.RNNModel` (LSTM+MLP+Gaussian head)
warm-started from a pre-trained robomimic BC-RNN policy. Only the BC RNN
backbone is transferred; the Gaussian head and critic are learned from scratch.

Usage (local BC checkpoint)::

    python scripts/train_rl.py --bc_checkpoint runs/coffee_d0_low_dim/.../models/model_450.pth

Usage (wandb source)::

    python scripts/train_rl.py \
        --wandb_run user/proj/abc123 --wandb_model model_450.pth
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch

# Add robomimic / robosuite / mimicgen to path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("robomimic", "robosuite", "mimicgen"):
    sys.path.insert(0, os.path.join(_REPO_ROOT, "resources", _sub))

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

from rsl_rl.runners import OnPolicyRunner

from rl_mimicgen.rsl_rl import (
    RobomimicVecEnv,
    copy_rnn_weights_into_actor,
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

    # -- PPO config file --
    _default_ppo_cfg = os.path.join(_REPO_ROOT, "rl_mimicgen", "rsl_rl", "ppo_cfg.json")
    parser.add_argument("--ppo_cfg", type=str, default=_default_ppo_cfg,
                        help=f"Path to PPO config JSON (default: {_default_ppo_cfg}). "
                             "Contains PPO algorithm hyperparameters, network architecture, "
                             "and obs-group wiring. BC-RNN fields (rnn_type, rnn_hidden_dim, "
                             "rnn_num_layers) are injected at runtime from the BC checkpoint.")

    # -- Env/actor overrides (commonly tuned) --
    parser.add_argument("--clip_actions", type=float, default=0.5,
                        help="Per-dim action clip (default: 0.5). Lower reduces the chance "
                             "of warp f32 solver divergence at large num_envs by keeping "
                             "OSC-pose deltas away from boundary saturation.")
    parser.add_argument("--init_noise_std", type=float, default=None,
                        help="Initial Gaussian-head std. Overrides ppo_cfg.json's "
                             "actor.distribution_cfg.init_std if set.")

    # -- Logging --
    parser.add_argument("--experiment_name", type=str, default="robomimic_rl")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--logger", type=str, default="tensorboard",
                        choices=["tensorboard", "wandb"])
    parser.add_argument("--wandb_project", type=str, default="robomimic_rl")

    return parser.parse_args()


def _resolve_bc_source(args, download_dir: str):
    if args.bc_checkpoint is not None:
        return load_bc_checkpoint(args.bc_checkpoint)
    if args.wandb_model is None:
        raise SystemExit("--wandb_run requires --wandb_model")
    local_path = fetch_wandb_checkpoint(args.wandb_run, args.wandb_model, download_dir)
    print(f"[INFO] Downloaded wandb checkpoint to: {local_path}")
    return load_bc_checkpoint(local_path)


def _build_train_cfg(args, bc_info) -> dict:
    """Load the PPO config JSON, inject BC-derived RNN fields and runtime
    settings, and apply CLI overrides. Returns the dict passed to OnPolicyRunner.
    """
    with open(args.ppo_cfg, "r") as f:
        cfg = json.load(f)

    # BC-derived: shape the actor's RNN to match the checkpoint.
    cfg["actor"]["rnn_type"] = bc_info.rnn_type
    cfg["actor"]["rnn_hidden_dim"] = bc_info.rnn_hidden_dim
    cfg["actor"]["rnn_num_layers"] = bc_info.rnn_num_layers

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
    if args.init_noise_std is not None:
        cfg["actor"]["distribution_cfg"]["init_std"] = args.init_noise_std

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

    # ------------------------------------------------------------------
    # 3. Create robomimic warp environment (from BC env metadata)
    # ------------------------------------------------------------------
    # Prime robomimic's global obs-modality registry before creating the env.
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=[bc_info.obs_modalities])

    device = torch.device(args.device)
    print(f"[INFO] Creating warp env (num_envs={args.num_envs}, horizon={args.horizon})")
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
        clip_actions=args.clip_actions,
        obs_keys=bc_info.obs_keys,
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
    # 4. Build RSL-RL training config (ppo_cfg.json + BC fields + CLI overrides)
    # ------------------------------------------------------------------
    train_cfg = _build_train_cfg(args, bc_info)
    print(f"[INFO] Loaded PPO config from: {args.ppo_cfg}")
    print(f"[INFO] num_steps_per_env={train_cfg['num_steps_per_env']}  "
          f"init_noise_std={train_cfg['actor']['distribution_cfg']['init_std']}  "
          f"lr={train_cfg['algorithm']['learning_rate']}")

    # ------------------------------------------------------------------
    # 5. Runner + BC warm-start
    # ------------------------------------------------------------------
    runner = OnPolicyRunner(vec_env, train_cfg, log_dir=log_dir, device=args.device)

    if not args.no_warm_start:
        loaded, skipped = copy_rnn_weights_into_actor(runner.alg.actor, bc_info.state_dict)
        print(f"[INFO] BC warm-start: copied {len(loaded)} RNN params, skipped {len(skipped)}")
        if skipped:
            print(f"[WARN] Skipped (shape mismatch or missing): {skipped}")
        runner.alg.actor.to(device)
    else:
        print("[INFO] BC warm-start disabled (--no_warm_start); actor initialized randomly.")

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    print(f"[INFO] Starting PPO: {train_cfg['max_iterations']} iters, "
          f"{train_cfg['num_steps_per_env']} steps/env/update, {args.num_envs} envs")
    runner.learn(num_learning_iterations=train_cfg["max_iterations"], init_at_random_ep_len=False)

    print("[INFO] Training complete.")
    vec_env.close()


if __name__ == "__main__":
    main()
