#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RunData:
    name: str
    run_dir: Path
    metrics_path: Path
    config_path: Path | None
    rows: list[dict[str, Any]]
    config: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze RL training logs and export research-ready plots, tables, and CSVs."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Run directories, metrics.jsonl files, or parent directories containing runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis") / "rl_logs",
        help="Directory where plots, tables, and CSVs will be written.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Moving-average window for train curves.",
    )
    return parser.parse_args()


def discover_runs(inputs: list[str]) -> list[RunData]:
    metrics_files: dict[Path, Path] = {}
    for raw_input in inputs:
        path = Path(raw_input).expanduser().resolve()
        if path.is_file():
            if path.name != "metrics.jsonl":
                raise ValueError(f"Expected a metrics.jsonl file, got: {path}")
            metrics_files[path.parent] = path
            continue
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")
        direct_metrics = path / "metrics.jsonl"
        if direct_metrics.is_file():
            metrics_files[path] = direct_metrics
            continue
        for metrics_path in sorted(path.rglob("metrics.jsonl")):
            metrics_files[metrics_path.parent] = metrics_path

    runs: list[RunData] = []
    for run_dir, metrics_path in sorted(metrics_files.items(), key=lambda item: str(item[0])):
        rows = load_metrics(metrics_path)
        if not rows:
            continue
        config_path = run_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        runs.append(
            RunData(
                name=run_dir.name,
                run_dir=run_dir,
                metrics_path=metrics_path,
                config_path=config_path if config_path.is_file() else None,
                rows=rows,
                config=config,
            )
        )
    if not runs:
        raise ValueError("No non-empty metrics.jsonl files were found in the provided inputs.")
    return runs


def load_metrics(metrics_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {metrics_path}:{line_number}") from exc
            rows.append(row)
    rows.sort(key=lambda row: float(row.get("update", 0.0)))
    return rows


def collect_columns(rows: list[dict[str, Any]]) -> list[str]:
    columns = sorted({key for row in rows for key in row.keys()})
    preferred = ["update", "env_steps"]
    ordered = [key for key in preferred if key in columns]
    ordered.extend(key for key in columns if key not in ordered)
    return ordered


def write_metrics_csv(run: RunData, output_dir: Path) -> Path:
    path = output_dir / f"{run.name}_metrics.csv"
    columns = collect_columns(run.rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in run.rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return path


def get_series(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            values.append(np.nan)
        else:
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                values.append(np.nan)
    return np.asarray(values, dtype=float)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values.copy()
    result = np.full_like(values, np.nan, dtype=float)
    for idx in range(values.size):
        start = max(0, idx - window + 1)
        window_values = values[start : idx + 1]
        finite = window_values[np.isfinite(window_values)]
        if finite.size:
            result[idx] = float(np.mean(finite))
    return result


def plot_x(rows: list[dict[str, Any]]) -> np.ndarray:
    env_steps = get_series(rows, "env_steps")
    if np.isfinite(env_steps).any() and np.nanmax(env_steps) > 0:
        return env_steps
    return get_series(rows, "update")


def actual_eval_mask(rows: list[dict[str, Any]]) -> np.ndarray:
    if any("eval/was_run" in row for row in rows):
        return np.asarray([bool(row.get("eval/was_run", 0.0)) for row in rows], dtype=bool)

    presence_keys = ("eval/return_mean", "eval/length_mean", "eval/evaluated_episodes")
    if any(any(key in row for key in presence_keys) for row in rows):
        return np.asarray([any(key in row for key in presence_keys) for row in rows], dtype=bool)

    if any("eval/success_rate" in row for row in rows):
        return np.asarray([np.isfinite(safe_float(row.get("eval/success_rate"))) for row in rows], dtype=bool)

    return np.zeros(len(rows), dtype=bool)


def plot_run_dashboard(run: RunData, output_dir: Path, smooth_window: int) -> Path:
    rows = run.rows
    env_steps = plot_x(rows)
    updates = get_series(rows, "update")
    eval_mask = actual_eval_mask(rows)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(run.name)

    eval_success = get_series(rows, "eval/success_rate")
    axes[0, 0].plot(env_steps[eval_mask], eval_success[eval_mask], marker="o", linewidth=2)
    axes[0, 0].set_title("Evaluation Success Rate")
    axes[0, 0].set_xlabel("Environment steps" if "env_steps" in rows[0] else "Update")
    axes[0, 0].set_ylabel("Success rate")
    axes[0, 0].set_ylim(-0.05, 1.05)
    axes[0, 0].grid(alpha=0.3)

    eval_return = get_series(rows, "eval/return_mean")
    axes[0, 1].plot(env_steps[eval_mask], eval_return[eval_mask], marker="o", linewidth=2)
    axes[0, 1].set_title("Evaluation Return")
    axes[0, 1].set_xlabel("Environment steps" if "env_steps" in rows[0] else "Update")
    axes[0, 1].set_ylabel("Return")
    axes[0, 1].grid(alpha=0.3)

    train_success = get_series(rows, "success_rate_mean")
    axes[1, 0].plot(env_steps, train_success, alpha=0.25, linewidth=1, label="raw")
    axes[1, 0].plot(env_steps, moving_average(train_success, smooth_window), linewidth=2, label="smoothed")
    axes[1, 0].set_title("Train Success Rate")
    axes[1, 0].set_xlabel("Environment steps" if "env_steps" in rows[0] else "Update")
    axes[1, 0].set_ylabel("Success rate")
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend(frameon=False)

    policy_loss = get_series(rows, "policy_loss")
    value_loss = get_series(rows, "value_loss")
    axes[1, 1].plot(updates, policy_loss, linewidth=2, label="policy_loss")
    axes[1, 1].plot(updates, value_loss, linewidth=2, label="value_loss")
    axes[1, 1].set_title("Optimization")
    axes[1, 1].set_xlabel("Update")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend(frameon=False)

    path = output_dir / f"{run.name}_dashboard.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_comparison(runs: list[RunData], output_dir: Path, metric_key: str, title: str, filename: str, y_label: str) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    plotted = False
    uses_env_steps = False
    for run in runs:
        x_values = plot_x(run.rows)
        uses_env_steps = uses_env_steps or ("env_steps" in run.rows[0] and np.isfinite(get_series(run.rows, "env_steps")).any())
        metric = get_series(run.rows, metric_key)
        mask = actual_eval_mask(run.rows) if metric_key.startswith("eval/") else np.isfinite(metric)
        if not np.any(mask):
            continue
        plotted = True
        ax.plot(x_values[mask], metric[mask], marker="o" if metric_key.startswith("eval/") else None, linewidth=2, label=run.name)

    ax.set_title(title)
    ax.set_xlabel("Environment steps" if uses_env_steps else "Update")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3)
    if "success" in metric_key:
        ax.set_ylim(-0.05, 1.05)
    if plotted:
        ax.legend(frameon=False, fontsize=9)

    path = output_dir / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def last_finite(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(finite[-1])


def summarize_run(run: RunData) -> dict[str, Any]:
    rows = run.rows
    last_row = rows[-1]
    eval_success = get_series(rows, "eval/success_rate")
    eval_return = get_series(rows, "eval/return_mean")
    eval_mask = actual_eval_mask(rows)

    best_eval_success = float(np.nanmax(eval_success[eval_mask])) if np.any(eval_mask) else math.nan
    best_eval_return = float(np.nanmax(eval_return[eval_mask])) if np.any(eval_mask) else math.nan
    best_eval_index = int(np.nanargmax(eval_success[eval_mask])) if np.any(eval_mask) else -1
    best_eval_update = math.nan
    if np.any(eval_mask):
        eval_updates = get_series(rows, "update")[eval_mask]
        best_eval_update = float(eval_updates[best_eval_index])

    residual_cfg = run.config.get("residual", {})
    optimizer_cfg = run.config.get("optimizer", {})
    evaluation_cfg = run.config.get("evaluation", {})

    return {
        "run_name": run.name,
        "run_dir": str(run.run_dir),
        "updates": int(safe_float(last_row.get("update", 0.0)) + 1),
        "final_env_steps": int(safe_float(last_row.get("env_steps", last_row.get("update", 0.0)))),
        "best_eval_success_rate": best_eval_success,
        "best_eval_return_mean": best_eval_return,
        "best_eval_update": best_eval_update,
        "last_eval_success_rate": last_finite(eval_success[eval_mask]),
        "last_eval_return_mean": last_finite(eval_return[eval_mask]),
        "last_train_success_rate": safe_float(last_row.get("success_rate_mean")),
        "last_train_return_mean": safe_float(last_row.get("episode_return_mean")),
        "residual_enabled": bool(residual_cfg.get("enabled", False)),
        "residual_scale": safe_float(residual_cfg.get("scale")),
        "actor_lr": safe_float(optimizer_cfg.get("actor_lr")),
        "value_lr": safe_float(optimizer_cfg.get("value_lr")),
        "eval_episodes": safe_float(evaluation_cfg.get("episodes")),
    }


def write_summary_table(summaries: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    csv_path = output_dir / "summary.csv"
    fieldnames = list(summaries[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    md_path = output_dir / "summary.md"
    lines = []
    lines.append("| " + " | ".join(fieldnames) + " |")
    lines.append("| " + " | ".join(["---"] * len(fieldnames)) + " |")
    for summary in summaries:
        cells = []
        for field in fieldnames:
            value = summary[field]
            if isinstance(value, float):
                cells.append("" if math.isnan(value) else f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def main() -> None:
    args = parse_args()
    runs = discover_runs(args.inputs)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
        }
    )

    summaries = []
    for run in runs:
        write_metrics_csv(run, output_dir)
        plot_run_dashboard(run, output_dir, smooth_window=args.smooth_window)
        summaries.append(summarize_run(run))

    summaries.sort(
        key=lambda summary: (
            -summary["best_eval_success_rate"] if not math.isnan(summary["best_eval_success_rate"]) else float("inf"),
            summary["run_name"],
        )
    )
    write_summary_table(summaries, output_dir)

    if len(runs) >= 1:
        plot_comparison(
            runs=runs,
            output_dir=output_dir,
            metric_key="eval/success_rate",
            title="Evaluation Success Rate",
            filename="comparison_eval_success_rate.png",
            y_label="Success rate",
        )
        plot_comparison(
            runs=runs,
            output_dir=output_dir,
            metric_key="eval/return_mean",
            title="Evaluation Return",
            filename="comparison_eval_return_mean.png",
            y_label="Return",
        )
        plot_comparison(
            runs=runs,
            output_dir=output_dir,
            metric_key="success_rate_mean",
            title="Train Success Rate",
            filename="comparison_train_success_rate.png",
            y_label="Success rate",
        )

    print(f"Analyzed {len(runs)} run(s). Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
