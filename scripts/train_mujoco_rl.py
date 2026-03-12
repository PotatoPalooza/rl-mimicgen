import argparse
import sys
from pathlib import Path

from rl_mimicgen.rl import OnlineRLConfig, OnlineRLTrainer


def parse_bool_or_auto(value: str):
    lowered = value.lower()
    if lowered == "auto":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ValueError(f"Expected one of auto|true|false, got: {value}")


def build_config(args: argparse.Namespace) -> OnlineRLConfig:
    if args.config is not None:
        config = OnlineRLConfig.from_json(args.config)
    else:
        config = OnlineRLConfig()

    if args.checkpoint is not None:
        config.checkpoint_path = args.checkpoint
    if not config.checkpoint_path:
        raise ValueError("--checkpoint is required unless it is present in --config")

    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.device is not None:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed
    if args.total_updates is not None:
        config.total_updates = args.total_updates
    if args.rollout_steps is not None:
        config.rollout_steps = args.rollout_steps
    if args.save_every is not None:
        config.save_every_n_updates = args.save_every

    if args.actor_lr is not None:
        config.optimizer.actor_lr = args.actor_lr
    if args.value_lr is not None:
        config.optimizer.value_lr = args.value_lr
    if args.max_grad_norm is not None:
        config.optimizer.max_grad_norm = args.max_grad_norm

    if args.update_epochs is not None:
        config.ppo.update_epochs = args.update_epochs
    if args.clip_ratio is not None:
        config.ppo.clip_ratio = args.clip_ratio
    if args.target_kl is not None:
        config.ppo.target_kl = args.target_kl
    if args.init_log_std is not None:
        config.ppo.init_log_std = args.init_log_std

    if args.demo_coef is not None:
        config.demo.coef = args.demo_coef
    if args.demo_decay is not None:
        config.demo.decay = args.demo_decay
    if args.demo_batch_size is not None:
        config.demo.batch_size = args.demo_batch_size
    if args.demo_dataset is not None:
        config.demo.dataset_path = args.demo_dataset
    if args.disable_demo:
        config.demo.enabled = False

    if args.eval_episodes is not None:
        config.evaluation.episodes = args.eval_episodes
    if args.eval_every is not None:
        config.evaluation.every_n_updates = args.eval_every
    if args.render_eval:
        config.evaluation.render = True
    if args.save_eval_video and args.eval_video is None:
        config.evaluation.video_path = "eval_videos/update_{update:04d}.mp4"
    if args.eval_video is not None:
        config.evaluation.video_path = args.eval_video

    if args.env_name is not None:
        config.robosuite.env_name = args.env_name
    if args.horizon is not None:
        config.robosuite.horizon = args.horizon
    if args.reward_shaping is not None:
        config.robosuite.reward_shaping = parse_bool_or_auto(args.reward_shaping)
    if args.render_train:
        config.robosuite.render_train = True
    if args.camera_name is not None:
        config.robosuite.camera_name = args.camera_name

    return config


class TeeStream:
    def __init__(self, stream, log_path: Path):
        self.stream = stream
        self.log_file = open(log_path, "a", encoding="utf-8")

    def write(self, data: str) -> int:
        self.stream.write(data)
        self.log_file.write(data)
        return len(data)

    def flush(self) -> None:
        self.stream.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.stream.isatty()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a robomimic / MimicGen BC policy with MuJoCo rollouts.")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config path.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a robomimic policy checkpoint.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for logs and checkpoints.")
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cpu or cuda.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--total-updates", type=int, default=None, help="Number of policy updates.")
    parser.add_argument("--rollout-steps", type=int, default=None, help="Number of environment steps per update.")
    parser.add_argument("--save-every", type=int, default=None, help="Save every N updates.")
    parser.add_argument("--actor-lr", type=float, default=None, help="Actor learning rate.")
    parser.add_argument("--value-lr", type=float, default=None, help="Value learning rate.")
    parser.add_argument("--max-grad-norm", type=float, default=None, help="Gradient clipping norm.")
    parser.add_argument("--update-epochs", type=int, default=None, help="PPO epochs per rollout.")
    parser.add_argument("--clip-ratio", type=float, default=None, help="PPO clip ratio.")
    parser.add_argument("--target-kl", type=float, default=None, help="Approximate KL early stop threshold.")
    parser.add_argument("--init-log-std", type=float, default=None, help="Initial log std for deterministic BC policies.")
    parser.add_argument("--demo-coef", type=float, default=None, help="Initial demo loss coefficient.")
    parser.add_argument("--demo-decay", type=float, default=None, help="Per-update multiplicative decay on demo loss.")
    parser.add_argument("--demo-batch-size", type=int, default=None, help="Demo batch size.")
    parser.add_argument("--demo-dataset", type=str, default=None, help="Override demo dataset path.")
    parser.add_argument("--disable-demo", action="store_true", help="Disable demo regularization.")
    parser.add_argument("--eval-episodes", type=int, default=None, help="Evaluation episodes.")
    parser.add_argument("--eval-every", type=int, default=None, help="Evaluate every N updates.")
    parser.add_argument(
        "--save-eval-video",
        action="store_true",
        help="Save per-eval videos under OUTPUT_DIR/eval_videos instead of live rendering.",
    )
    parser.add_argument("--eval-video", type=str, default=None, help="Optional relative path for eval video output.")
    parser.add_argument("--render-eval", action="store_true", help="Render evaluation rollouts on screen.")
    parser.add_argument("--render-train", action="store_true", help="Render training rollouts on screen.")
    parser.add_argument("--env-name", type=str, default=None, help="Optional environment override.")
    parser.add_argument("--horizon", type=int, default=None, help="Optional rollout horizon override.")
    parser.add_argument(
        "--reward-shaping",
        type=str,
        default=None,
        help="Override env reward shaping: auto, true, or false.",
    )
    parser.add_argument("--camera-name", type=str, default=None, help="Camera name for evaluation video rendering.")
    args = parser.parse_args()

    config = build_config(args)
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    sys.stdout = TeeStream(sys.stdout, log_path)
    sys.stderr = TeeStream(sys.stderr, log_path)

    trainer = OnlineRLTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
