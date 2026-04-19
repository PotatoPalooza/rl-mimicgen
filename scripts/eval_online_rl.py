#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from rl_mimicgen.rl import OnlineRLConfig, OnlineRLTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved robomimic / online-RL policy checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="Path to an online RL JSON config file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the saved policy checkpoint to evaluate.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write eval outputs.")
    parser.add_argument("--episodes", type=int, default=None, help="Number of evaluation episodes.")
    parser.add_argument("--eval-num-envs", type=int, default=None, help="Number of eval environments. Use 1 when writing video.")
    parser.add_argument("--seed", type=int, default=None, help="Evaluation seed override.")
    parser.add_argument("--render", action="store_true", help="Render the eval environment onscreen.")
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Relative video path inside output-dir, for example videos/eval.mp4.",
    )
    return parser.parse_args()


def resolve_eval_checkpoint(checkpoint_path: str, output_dir: Path) -> tuple[str, dict | None, dict | None, Path | None]:
    artifact = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(artifact, dict):
        return checkpoint_path, None, None, None

    if "algo_name" in artifact:
        return checkpoint_path, None, None, None

    if {"actor_state", "config", "value_net"}.issubset(artifact.keys()):
        trainer_config = artifact.get("config", {})
        if not isinstance(trainer_config, dict):
            raise ValueError(f"Trainer checkpoint is missing a usable config: {checkpoint_path}")
        base_checkpoint_path = trainer_config.get("checkpoint_path")
        if not isinstance(base_checkpoint_path, str) or not base_checkpoint_path:
            raise ValueError(f"Trainer checkpoint is missing checkpoint_path in config: {checkpoint_path}")
        return base_checkpoint_path, None, artifact, None

    if artifact.get("mode") != "residual":
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    base_checkpoint_path = artifact.get("base_checkpoint_path")
    if isinstance(base_checkpoint_path, str) and os.path.exists(base_checkpoint_path):
        return base_checkpoint_path, artifact, None

    base_checkpoint = artifact.get("base_checkpoint")
    if not isinstance(base_checkpoint, dict):
        raise ValueError(f"Residual policy artifact is missing a usable base checkpoint: {checkpoint_path}")

    synthesized_path = output_dir / "_eval_base_checkpoint.pth"
    torch.save(base_checkpoint, synthesized_path)
    return str(synthesized_path), artifact, None, synthesized_path


def load_residual_weights(trainer: OnlineRLTrainer, artifact: dict) -> None:
    trainer.policy.residual_enabled = True
    trainer.policy.residual_scale = float(artifact["residual_scale"])
    trainer.policy.actor.load_state_dict(artifact["residual_actor_state"])

    residual_log_std = artifact.get("residual_log_std")
    if trainer.policy.learned_log_std is not None and residual_log_std is not None:
        trainer.policy.learned_log_std.data.copy_(residual_log_std.to(device=trainer.device))


def load_trainer_checkpoint_weights(trainer: OnlineRLTrainer, artifact: dict[str, Any]) -> None:
    trainer.policy.actor.load_state_dict(artifact["actor_state"])
    trainer.policy.value_net.load_state_dict(artifact["value_net"])

    learned_log_std = artifact.get("learned_log_std")
    if trainer.policy.learned_log_std is not None and learned_log_std is not None:
        trainer.policy.learned_log_std.data.copy_(learned_log_std.to(device=trainer.device))


def main() -> None:
    args = parse_args()

    config = OnlineRLConfig.from_json(args.config)
    checkpoint_path = args.checkpoint
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_checkpoint, residual_artifact, trainer_artifact, synthesized_checkpoint = resolve_eval_checkpoint(checkpoint_path, output_dir)

    if trainer_artifact is not None:
        config = OnlineRLConfig(**trainer_artifact["config"])

    config.output_dir = args.output_dir
    config.total_updates = 0
    config.demo.enabled = False
    config.evaluation.enabled = True
    config.num_envs = 1
    config.rollout_steps = 1
    config.robosuite.parallel_envs = False

    if args.episodes is not None:
        config.evaluation.episodes = args.episodes
    if args.eval_num_envs is not None:
        config.evaluation.num_envs = args.eval_num_envs
    if args.seed is not None:
        config.seed = args.seed

    config.evaluation.render = bool(args.render)
    config.evaluation.video_path = args.video_path
    if args.video_path:
        config.evaluation.num_envs = 1

    config.checkpoint_path = resolved_checkpoint
    if residual_artifact is not None:
        config.residual.enabled = True
        config.residual.scale = float(residual_artifact["residual_scale"])
    if trainer_artifact is not None:
        config.residual.enabled = bool(trainer_artifact.get("residual_enabled", config.residual.enabled))

    trainer = OnlineRLTrainer(config)
    try:
        if residual_artifact is not None:
            load_residual_weights(trainer, residual_artifact)
        if trainer_artifact is not None:
            load_trainer_checkpoint_weights(trainer, trainer_artifact)
        metrics = trainer.evaluate(update=0)
    finally:
        trainer._close_env(trainer.env)
        trainer._close_env(trainer.eval_env)
        if synthesized_checkpoint is not None and synthesized_checkpoint.exists():
            synthesized_checkpoint.unlink()

    metrics_path = output_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "metrics_path": str(metrics_path), **metrics}, indent=2))


if __name__ == "__main__":
    main()
