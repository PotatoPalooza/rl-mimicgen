import argparse

from rl_mimicgen.rl import OnlineRLConfig, OnlineRLTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online PPO / DPPO fine-tuning for robomimic checkpoints.")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file.")
    parser.add_argument("--algorithm", type=str, choices=["ppo", "awac", "dppo"], default=None, help="Online RL algorithm to run.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a robomimic checkpoint.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to write logs and checkpoints.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel robosuite environments.")
    parser.add_argument("--total_updates", type=int, default=None, help="Number of PPO updates to run.")
    parser.add_argument("--rollout_steps", type=int, default=None, help="Number of policy time steps to collect per rollout update.")
    parser.add_argument("--actor_lr", type=float, default=None, help="Actor learning rate.")
    parser.add_argument("--value_lr", type=float, default=None, help="Critic learning rate.")
    parser.add_argument("--entropy_coef", type=float, default=None, help="Entropy bonus coefficient.")
    parser.add_argument(
        "--critic_warmup_updates",
        type=int,
        default=None,
        help="Number of initial PPO updates that train only the critic.",
    )
    parser.add_argument(
        "--actor_freeze_env_steps",
        type=int,
        default=None,
        help="Freeze BC actor updates for the first N collected environment steps.",
    )
    parser.add_argument(
        "--eval_video_path",
        type=str,
        default=None,
        help="Relative path pattern for eval videos, e.g. videos/eval_update_{update:04d}.mp4.",
    )
    parser.add_argument("--eval_num_envs", type=int, default=None, help="Number of parallel environments to use during evaluation.")
    parser.add_argument("--demo_coef", type=float, default=None, help="Initial demo BC loss weight.")
    parser.add_argument("--demo_decay", type=float, default=None, help="Per-update decay applied to demo BC loss.")
    parser.add_argument("--diffusion_num_inference_timesteps", type=int, default=None, help="Override the number of DDPM denoising steps used during diffusion-policy online RL.")
    parser.add_argument("--diffusion_use_ema", action="store_true", help="Use the diffusion checkpoint EMA weights for deterministic evaluation. PPO training rollouts always use the live actor weights.")
    parser.add_argument("--residual", action="store_true", help="Enable residual policy fine-tuning on top of the BC checkpoint.")
    parser.add_argument("--residual_scale", type=float, default=None, help="Scale factor applied to residual actions before adding to the frozen BC action.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Increase run visibility by logging every update, evaluating every update, "
            "and writing eval videos by default."
        ),
    )
    return parser.parse_args()


def apply_cli_overrides(args: argparse.Namespace) -> OnlineRLConfig:
    config = OnlineRLConfig.from_json(args.config) if args.config is not None else OnlineRLConfig()
    if args.algorithm is not None:
        config.algorithm = args.algorithm
    if args.checkpoint is not None:
        config.checkpoint_path = args.checkpoint
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.num_envs is not None:
        config.num_envs = args.num_envs
    if args.total_updates is not None:
        config.total_updates = args.total_updates
    if args.rollout_steps is not None:
        config.rollout_steps = args.rollout_steps
    if args.actor_lr is not None:
        config.optimizer.actor_lr = args.actor_lr
    if args.value_lr is not None:
        config.optimizer.value_lr = args.value_lr
    if args.entropy_coef is not None:
        config.ppo.entropy_coef = args.entropy_coef
    if args.critic_warmup_updates is not None:
        config.ppo.critic_warmup_updates = args.critic_warmup_updates
    if args.actor_freeze_env_steps is not None:
        config.ppo.actor_freeze_env_steps = args.actor_freeze_env_steps
    if args.eval_video_path is not None:
        config.evaluation.video_path = args.eval_video_path
    if args.eval_num_envs is not None:
        config.evaluation.num_envs = args.eval_num_envs
    if args.demo_coef is not None:
        config.demo.coef = args.demo_coef
    if args.demo_decay is not None:
        config.demo.decay = args.demo_decay
    if args.diffusion_num_inference_timesteps is not None:
        config.diffusion.num_inference_timesteps = args.diffusion_num_inference_timesteps
    if args.diffusion_use_ema:
        config.diffusion.use_ema = True
    if args.residual:
        config.residual.enabled = True
    if args.residual_scale is not None:
        config.residual.scale = args.residual_scale
    if args.debug:
        config.log_every_n_updates = 1
        config.evaluation.enabled = True
        config.evaluation.every_n_updates = 1
        if config.evaluation.video_path is None:
            config.evaluation.video_path = "videos/debug_eval_update_{update:04d}.mp4"
    return config


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(args)

    trainer = OnlineRLTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
