#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from rl_mimicgen.rl import OnlineRLConfig, OnlineRLTrainer


EPOCH_PATTERN = re.compile(r"model_epoch_(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all saved BC checkpoints from a robomimic run using the RL-stack evaluation path."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to an online RL JSON config file that defines the evaluation runtime profile.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to a timestamped robomimic run directory, an experiment directory, or a models directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write sweep metrics and best-checkpoint artifacts.",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Override the number of eval episodes.")
    parser.add_argument("--eval-num-envs", type=int, default=None, help="Override the number of eval environments.")
    parser.add_argument("--seed", type=int, default=None, help="Evaluation seed override.")
    parser.add_argument("--device", type=str, default=None, help="Evaluation device override.")
    parser.add_argument(
        "--include-latest",
        action="store_true",
        help="Also evaluate last.pth if it exists alongside model_epoch_*.pth.",
    )
    return parser.parse_args()


def resolve_run_dir(path: Path) -> tuple[Path, Path]:
    path = path.expanduser().resolve()
    if path.name == "models" and path.is_dir():
        return path.parent, path
    if (path / "models").is_dir():
        return path, path / "models"
    candidates = sorted(child for child in path.iterdir() if child.is_dir() and (child / "models").is_dir())
    if not candidates:
        raise FileNotFoundError(f"Could not resolve a robomimic run directory from {path}")
    return candidates[-1], candidates[-1] / "models"


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = EPOCH_PATTERN.search(path.stem)
    if match:
        return int(match.group(1)), path.name
    if path.name == "last.pth":
        return 10**12, path.name
    return 10**12 + 1, path.name


def format_success_suffix(success_rate: float) -> str:
    return f"{success_rate:.4f}"


def evaluate_checkpoint(config: OnlineRLConfig, checkpoint: Path, output_dir: Path) -> dict[str, float]:
    config.checkpoint_path = str(checkpoint)
    config.output_dir = str(output_dir)
    config.total_updates = 0
    config.demo.enabled = False
    config.evaluation.enabled = True
    config.num_envs = 1
    config.rollout_steps = 1
    config.robosuite.parallel_envs = config.evaluation.num_envs > 1

    trainer = OnlineRLTrainer(config)
    try:
        metrics = trainer.evaluate(update=0)
    finally:
        trainer._close_env(trainer.env)
        trainer._close_env(trainer.eval_env)
    return metrics


def clone_eval_config(args: argparse.Namespace) -> OnlineRLConfig:
    config = OnlineRLConfig.from_json(args.config)
    if args.episodes is not None:
        config.evaluation.episodes = args.episodes
    if args.eval_num_envs is not None:
        config.evaluation.num_envs = args.eval_num_envs
    if args.seed is not None:
        config.seed = args.seed
    if args.device is not None:
        config.device = args.device
    config.evaluation.render = False
    config.evaluation.video_path = None
    return config


def main() -> None:
    args = parse_args()
    eval_config = clone_eval_config(args)

    run_dir, models_dir = resolve_run_dir(Path(args.run_dir))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = sorted(models_dir.glob("model_epoch_*.pth"), key=checkpoint_sort_key)
    if args.include_latest:
        latest = models_dir / "last.pth"
        if latest.exists():
            checkpoints.append(latest)
    if not checkpoints:
        raise FileNotFoundError(f"No model_epoch_*.pth checkpoints found under {models_dir}")

    jsonl_path = output_dir / "checkpoint_metrics.jsonl"
    csv_path = output_dir / "checkpoint_metrics.csv"
    best_path: Path | None = None
    best_metrics: dict[str, float] | None = None

    rows: list[dict[str, object]] = []
    with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
        for checkpoint in checkpoints:
            checkpoint_output_dir = output_dir / checkpoint.stem
            metrics = evaluate_checkpoint(eval_config, checkpoint, checkpoint_output_dir)
            epoch_match = EPOCH_PATTERN.search(checkpoint.stem)
            row = {
                "checkpoint": str(checkpoint),
                "checkpoint_name": checkpoint.name,
                "epoch": int(epoch_match.group(1)) if epoch_match else None,
                "success_rate": metrics["success_rate"],
                "return_mean": metrics["return_mean"],
                "length_mean": metrics["length_mean"],
                "evaluated_episodes": metrics["evaluated_episodes"],
            }
            rows.append(row)
            jsonl_file.write(json.dumps(row) + "\n")
            jsonl_file.flush()
            print(
                f"[eval_bc_checkpoints] checkpoint={checkpoint.name} "
                f"success_rate={metrics['success_rate']:.4f} "
                f"return_mean={metrics['return_mean']:.4f} "
                f"length_mean={metrics['length_mean']:.1f}",
                flush=True,
            )
            if best_metrics is None or metrics["success_rate"] > best_metrics["success_rate"]:
                best_path = checkpoint
                best_metrics = metrics

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "checkpoint",
                "checkpoint_name",
                "epoch",
                "success_rate",
                "return_mean",
                "length_mean",
                "evaluated_episodes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    assert best_path is not None and best_metrics is not None
    manifest = {
        "run_dir": str(run_dir),
        "models_dir": str(models_dir),
        "best_checkpoint": str(best_path),
        "config": str(Path(args.config).expanduser().resolve()),
        **best_metrics,
    }
    (output_dir / "best_checkpoint.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output_dir / "best_checkpoint.txt").write_text(str(best_path) + "\n", encoding="utf-8")
    link_path = output_dir / "best_checkpoint.pth"
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(best_path.resolve())
    scored_link_path = output_dir / f"best_checkpoint_success_{format_success_suffix(best_metrics['success_rate'])}.pth"
    if scored_link_path.exists() or scored_link_path.is_symlink():
        scored_link_path.unlink()
    scored_link_path.symlink_to(best_path.resolve())

    print(json.dumps({"output_dir": str(output_dir), **manifest}, indent=2), flush=True)


if __name__ == "__main__":
    main()
