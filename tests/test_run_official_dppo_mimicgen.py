from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf


def _load_launcher_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_official_dppo_mimicgen.py"
    spec = importlib.util.spec_from_file_location("run_official_dppo_mimicgen", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


launcher = _load_launcher_module()


def test_finetune_checkpoint_metadata_reads_bc_success_from_sweep_copy(tmp_path: Path) -> None:
    log_root = tmp_path / "logs"
    run_dir = log_root / "sweep" / "coffee_d0" / "coffee_d0_sweep_bc" / "sweep_run"
    run_dir.mkdir(parents=True)
    checkpoint_path = run_dir / "best_checkpoint.pt"
    checkpoint_path.write_text("checkpoint", encoding="utf-8")
    summary_path = run_dir / "best_checkpoint.json"
    summary_path.write_text(
        json.dumps(
            {
                "best_checkpoint": str(tmp_path / "pretrain" / "state_150.pt"),
                "best_checkpoint_copy": str(checkpoint_path),
                "best_metrics": {
                    "success_rate": 0.65,
                    "checkpoint_path": str(tmp_path / "pretrain" / "state_150.pt"),
                },
            }
        ),
        encoding="utf-8",
    )

    metadata = launcher._finetune_checkpoint_metadata(checkpoint_path, log_root, "coffee_d0")

    assert metadata["source_checkpoint"] == checkpoint_path.as_posix()
    assert metadata["bc_success_rate"] == 0.65
    assert metadata["bc_success_source"] == summary_path.resolve().as_posix()


def test_finetune_checkpoint_metadata_matches_swept_original_checkpoint_path(tmp_path: Path) -> None:
    log_root = tmp_path / "logs"
    checkpoint_path = log_root / "pretrain" / "coffee_d0" / "run" / "checkpoint" / "state_150.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_text("checkpoint", encoding="utf-8")
    summary_path = log_root / "sweep" / "coffee_d0" / "coffee_d0_sweep_bc" / "sweep_run" / "best_checkpoint.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        json.dumps(
            {
                "best_checkpoint": str(checkpoint_path),
                "best_checkpoint_copy": str(summary_path.parent / "best_checkpoint.pt"),
                "best_metrics": {
                    "success_rate": 0.8,
                    "checkpoint_path": str(checkpoint_path),
                },
            }
        ),
        encoding="utf-8",
    )

    metadata = launcher._finetune_checkpoint_metadata(checkpoint_path, log_root, "coffee_d0")

    assert metadata["bc_success_rate"] == 0.8
    assert metadata["bc_success_source"] == summary_path.resolve().as_posix()


def test_finetune_checkpoint_metadata_falls_back_to_checkpoint_filename(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "policy_best_update_0010_success_0.5104.pth"
    checkpoint_path.write_text("checkpoint", encoding="utf-8")

    metadata = launcher._finetune_checkpoint_metadata(checkpoint_path, tmp_path / "logs", "coffee_d0")

    assert metadata["bc_success_rate"] == 0.5104
    assert metadata["bc_success_source"] == "checkpoint_filename"


def test_run_dppo_with_snapshot_accepts_string_env_for_pretrain(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "run_config.yaml"
    OmegaConf.save(OmegaConf.create({"env": "Coffee_D0"}), config_path)

    captured: dict[str, object] = {}

    def fake_subprocess_env(task_spec, *, for_video: bool = False, run_dir: Path | None = None):
        captured["for_video"] = for_video
        captured["env_run_dir"] = run_dir
        return {"WANDB_DIR": "dummy"}

    def fake_run(command, cwd=None, check=None, env=None):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["check"] = check
        captured["env"] = env

    monkeypatch.setattr(launcher, "_subprocess_env", fake_subprocess_env)
    monkeypatch.setattr(launcher.subprocess, "run", fake_run)

    launcher._run_dppo_with_snapshot(OmegaConf.create({"runtime": {}}), tmp_path, config_path)

    assert captured["for_video"] is False
    assert captured["env_run_dir"] == tmp_path
    assert captured["cwd"] == launcher.DPPO_ROOT
    assert captured["check"] is True
    assert captured["env"] == {"WANDB_DIR": "dummy"}


def test_run_dppo_with_snapshot_reads_save_video_from_mapping_env(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "run_config.yaml"
    OmegaConf.save(OmegaConf.create({"env": {"save_video": True}}), config_path)

    captured: dict[str, object] = {}

    def fake_subprocess_env(task_spec, *, for_video: bool = False, run_dir: Path | None = None):
        captured["for_video"] = for_video
        return {}

    monkeypatch.setattr(launcher, "_subprocess_env", fake_subprocess_env)
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: None)

    launcher._run_dppo_with_snapshot(OmegaConf.create({"runtime": {}}), tmp_path, config_path)

    assert captured["for_video"] is True
