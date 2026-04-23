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


def test_pretrain_completion_detection_prefers_best_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "pretrain" / "coffee_d0" / "coffee_d0_pre" / "run"
    run_dir.mkdir(parents=True)
    run_config = run_dir / "run_config.yaml"
    OmegaConf.save(OmegaConf.create({"train": {"n_epochs": 100}}), run_config)

    expected = run_dir / "best_checkpoint.pt"
    expected.write_text("checkpoint", encoding="utf-8")

    incomplete = run_dir / "checkpoint" / "state_100.pt"
    incomplete.parent.mkdir(parents=True, exist_ok=True)
    incomplete.write_text("checkpoint", encoding="utf-8")

    assert launcher._completed_pretrain_checkpoint_from_run_dir(run_dir) == expected


def test_pretrain_completion_detection_uses_final_epoch_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "pretrain" / "coffee_d0" / "coffee_d0_pre" / "run"
    run_dir.mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({"train": {"n_epochs": 20}}), run_dir / "run_config.yaml")
    completed_checkpoint = run_dir / "checkpoint" / "state_20.pt"
    completed_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    completed_checkpoint.write_text("checkpoint", encoding="utf-8")

    assert launcher._completed_pretrain_checkpoint_from_run_dir(run_dir) == completed_checkpoint


def test_latest_completed_pretrain_checkpoint_prefers_newest_run(tmp_path: Path) -> None:
    dataset_id = "coffee_d0"
    run_root = tmp_path / "logs" / "official_dppo" / "mimicgen"
    old_run_dir = run_root / "pretrain" / dataset_id / "old_run" / "20250101_000001_s0_aaa"
    new_run_dir = run_root / "pretrain" / dataset_id / "new_run" / "20260101_000001_s0_bbb"
    old_run_dir.mkdir(parents=True, exist_ok=True)
    new_run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(
        OmegaConf.create({"train": {"n_epochs": 5}}),
        old_run_dir / "run_config.yaml",
    )
    OmegaConf.save(
        OmegaConf.create({"train": {"n_epochs": 10}}),
        new_run_dir / "run_config.yaml",
    )

    (old_run_dir / "checkpoint" / "state_5.pt").parent.mkdir(parents=True, exist_ok=True)
    (old_run_dir / "checkpoint" / "state_5.pt").write_text("checkpoint", encoding="utf-8")
    (new_run_dir / "checkpoint" / "state_10.pt").parent.mkdir(parents=True, exist_ok=True)

    (new_run_dir / "checkpoint" / "state_5.pt").write_text("checkpoint", encoding="utf-8")

    assert (
        launcher._latest_completed_pretrain_checkpoint(run_root, dataset_id)
        == old_run_dir / "checkpoint" / "state_5.pt"
    )


def test_prepare_parser_accepts_auto_skip_pretrain_for_finetune_and_sweep() -> None:
    parser = launcher._prepare_parser()
    finetune_args = parser.parse_args(["finetune", "--task", "coffee_d0", "--auto-skip-pretrain"])
    assert finetune_args.auto_skip_pretrain is True

    sweep_args = parser.parse_args(["sweep", "--task", "coffee_d0", "--auto-skip-pretrain"])
    assert sweep_args.auto_skip_pretrain is True


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
