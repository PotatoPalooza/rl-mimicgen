import importlib.util
import json
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_rl_logs.py"
SPEC = importlib.util.spec_from_file_location("analyze_rl_logs", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

discover_runs = MODULE.discover_runs
summarize_run = MODULE.summarize_run


def _write_run(run_dir: Path, rows: list[dict], config: dict | None = None) -> None:
    run_dir.mkdir(parents=True)
    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    if config is not None:
        (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_discover_runs_recurses_and_loads_rows(tmp_path) -> None:
    rows = [
        {"update": 1, "env_steps": 20, "success_rate_mean": 0.5},
        {"update": 0, "env_steps": 10, "success_rate_mean": 0.25},
    ]
    _write_run(tmp_path / "logs" / "run_a", rows)

    runs = discover_runs([str(tmp_path / "logs")])

    assert len(runs) == 1
    assert runs[0].name == "run_a"
    assert runs[0].rows[0]["update"] == 0
    assert runs[0].rows[1]["update"] == 1


def test_summarize_run_uses_real_eval_points(tmp_path) -> None:
    rows = [
        {
            "update": 0,
            "env_steps": 100,
            "success_rate_mean": 0.1,
            "episode_return_mean": 0.2,
            "eval/was_run": 0.0,
            "eval/best_success_rate": 0.0,
        },
        {
            "update": 1,
            "env_steps": 200,
            "success_rate_mean": 0.3,
            "episode_return_mean": 0.4,
            "eval/was_run": 1.0,
            "eval/success_rate": 0.6,
            "eval/return_mean": 0.8,
            "eval/best_success_rate": 0.6,
        },
        {
            "update": 2,
            "env_steps": 300,
            "success_rate_mean": 0.5,
            "episode_return_mean": 0.7,
            "eval/was_run": 0.0,
            "eval/last_success_rate": 0.6,
            "eval/last_return_mean": 0.8,
            "eval/best_success_rate": 0.6,
        },
    ]
    config = {"residual": {"enabled": True, "scale": 0.2}, "optimizer": {"actor_lr": 1e-4, "value_lr": 1e-3}}
    run_dir = tmp_path / "run_eval"
    _write_run(run_dir, rows, config=config)
    run = discover_runs([str(run_dir)])[0]

    summary = summarize_run(run)

    assert summary["best_eval_success_rate"] == 0.6
    assert summary["best_eval_update"] == 1.0
    assert summary["last_eval_success_rate"] == 0.6
    assert summary["last_train_success_rate"] == 0.5
    assert summary["residual_enabled"] is True
