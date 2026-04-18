#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl_mimicgen.dppo.config.schema import DPPORunConfig
from rl_mimicgen.dppo.data import DPPODatasetBundle
from rl_mimicgen.dppo.eval.eval_diffusion_agent import run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a sweep of DPPO checkpoints and select the best checkpoint.")
    parser.add_argument("--config", required=True, help="Path to the DPPO JSON config.")
    parser.add_argument("--checkpoint_dir", required=True, help="Directory containing state_*.pt checkpoints.")
    parser.add_argument("--output_dir", required=True, help="Directory for per-checkpoint metrics and best-checkpoint summary.")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per checkpoint evaluation.")
    parser.add_argument("--max_steps", type=int, default=None, help="Optional max episode length.")
    parser.add_argument("--every_n", type=int, default=1, help="Evaluate every Nth checkpoint by checkpoint index.")
    parser.add_argument("--start_index", type=int, default=None, help="Optional minimum checkpoint index.")
    parser.add_argument("--end_index", type=int, default=None, help="Optional maximum checkpoint index.")
    parser.add_argument(
        "--video_checkpoints",
        choices=("none", "best", "all"),
        default="none",
        help="Write no videos, one video for the best checkpoint, or one video per evaluated checkpoint.",
    )
    parser.add_argument("--video_name", default="eval.mp4", help="Video filename used under each checkpoint output directory.")
    return parser.parse_args()


def _checkpoint_index(path: Path) -> int | None:
    if not path.stem.startswith("state_"):
        return None
    suffix = path.stem.removeprefix("state_")
    return int(suffix) if suffix.isdigit() else None


def _collect_checkpoints(
    checkpoint_dir: Path,
    every_n: int,
    start_index: int | None,
    end_index: int | None,
) -> list[tuple[int, Path]]:
    selected: list[tuple[int, Path]] = []
    for path in sorted(checkpoint_dir.glob("state_*.pt")):
        index = _checkpoint_index(path)
        if index is None:
            continue
        if start_index is not None and index < start_index:
            continue
        if end_index is not None and index > end_index:
            continue
        if every_n > 1 and index % every_n != 0:
            continue
        selected.append((index, path))
    return selected


def _metric_rank(metrics: dict[str, float | str]) -> tuple[float, float, float]:
    return (
        float(metrics["success_rate"]),
        float(metrics["return_mean"]),
        -float(metrics["length_mean"]),
    )


def main() -> None:
    args = parse_args()
    config = DPPORunConfig.from_json(args.config)
    dataset = DPPODatasetBundle.load(config.dataset.bundle_dir)
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = _collect_checkpoints(
        checkpoint_dir=checkpoint_dir,
        every_n=max(1, args.every_n),
        start_index=args.start_index,
        end_index=args.end_index,
    )
    if not selected:
        raise FileNotFoundError(f"No checkpoints selected from {checkpoint_dir}")

    all_metrics: list[dict[str, float | str]] = []
    best_checkpoint_path: Path | None = None
    best_metrics: dict[str, float | str] | None = None

    for checkpoint_index, checkpoint_path in selected:
        checkpoint_output_dir = output_dir / f"state_{checkpoint_index}"
        checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
        video_path = None
        if args.video_checkpoints == "all":
            video_path = str(checkpoint_output_dir / args.video_name)
        metrics = run_evaluation(
            config=config,
            dataset=dataset,
            checkpoint_path=str(checkpoint_path),
            episodes=args.episodes,
            max_steps=args.max_steps,
            video_path=video_path,
        )
        metrics = {
            **metrics,
            "checkpoint_index": float(checkpoint_index),
        }
        all_metrics.append(metrics)
        with open(checkpoint_output_dir / "eval_metrics.json", "w", encoding="utf-8") as file_obj:
            json.dump(metrics, file_obj, indent=2)
        if best_metrics is None or _metric_rank(metrics) > _metric_rank(best_metrics):
            best_metrics = metrics
            best_checkpoint_path = checkpoint_path

    assert best_checkpoint_path is not None
    assert best_metrics is not None

    if args.video_checkpoints == "best":
        best_output_dir = output_dir / "best"
        best_output_dir.mkdir(parents=True, exist_ok=True)
        best_metrics = {
            **run_evaluation(
                config=config,
                dataset=dataset,
                checkpoint_path=str(best_checkpoint_path),
                episodes=args.episodes,
                max_steps=args.max_steps,
                video_path=str(best_output_dir / args.video_name),
            ),
            "checkpoint_index": float(_checkpoint_index(best_checkpoint_path) or -1),
        }

    summary = {
        "config": str(Path(args.config).expanduser().resolve()),
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(output_dir),
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "every_n": args.every_n,
        "num_evaluated": len(all_metrics),
        "best_checkpoint": str(best_checkpoint_path),
        "best_metrics": best_metrics,
    }
    with open(output_dir / "checkpoint_metrics.json", "w", encoding="utf-8") as file_obj:
        json.dump(all_metrics, file_obj, indent=2)
    with open(output_dir / "best_checkpoint.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
