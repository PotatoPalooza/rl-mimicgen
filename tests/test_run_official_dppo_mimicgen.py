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


def test_pretrain_completion_detection_requires_final_epoch_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "pretrain" / "coffee_d0" / "coffee_d0_pre" / "run"
    run_dir.mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({"train": {"n_epochs": 20}}), run_dir / "run_config.yaml")
    (run_dir / "best_checkpoint.pt").write_text("checkpoint", encoding="utf-8")
    (run_dir / "checkpoint" / "state_10.pt").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoint" / "state_10.pt").write_text("checkpoint", encoding="utf-8")

    assert launcher._completed_pretrain_checkpoint_from_run_dir(run_dir) is None


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


def test_find_reusable_stage_run_prefers_latest_incomplete_sweep(tmp_path: Path) -> None:
    log_root = tmp_path / "logs"
    dataset_id = "coffee_d0"
    run_name = "coffee_d0_sweep_bc"
    completed_run = log_root / "sweep" / dataset_id / run_name / "20260423_000001_s42_old"
    resumable_run = log_root / "sweep" / dataset_id / run_name / "20260423_000002_s42_new"
    completed_run.mkdir(parents=True)
    resumable_run.mkdir(parents=True)

    (completed_run / "run_manifest.json").write_text(
        json.dumps({"stage": "sweep", "run_id": completed_run.name, "created_at": "2026-04-23T00:00:01"}),
        encoding="utf-8",
    )
    (completed_run / "best_checkpoint.json").write_text(json.dumps({"num_evaluated": 4}), encoding="utf-8")

    (resumable_run / "run_manifest.json").write_text(
        json.dumps({"stage": "sweep", "run_id": resumable_run.name, "created_at": "2026-04-23T00:00:02"}),
        encoding="utf-8",
    )
    (resumable_run / "run_state.json").write_text(
        json.dumps({"stage": "sweep", "status": "failed"}),
        encoding="utf-8",
    )

    reusable = launcher._find_reusable_stage_run(
        log_root=log_root,
        dataset_id=dataset_id,
        log_group="sweep",
        run_name=run_name,
        stage="sweep",
    )

    assert reusable is not None
    run_dir, manifest = reusable
    assert run_dir == resumable_run
    assert manifest["run_id"] == resumable_run.name


def test_snapshot_run_config_writes_running_run_state(tmp_path: Path) -> None:
    generated_config_path = tmp_path / "generated.yaml"
    generated_config_path.write_text("train:\n  n_epochs: 12\n", encoding="utf-8")
    task_spec_path = tmp_path / "task.yaml"
    task_spec_path.write_text("dataset:\n  path: coffee_d0.hdf5\n", encoding="utf-8")

    cfg = OmegaConf.create(
        {
            "seed": 42,
            "horizon_steps": 4,
            "denoising_steps": 20,
            "train": {"n_epochs": 12},
            "wandb": None,
        }
    )
    task_spec = OmegaConf.create({"dataset": {"path": "coffee_d0.hdf5"}})

    run_dir, _config_path, _manifest = launcher._snapshot_run_config(
        cfg=cfg,
        task_spec=task_spec,
        task_spec_path=task_spec_path,
        generated_config_path=generated_config_path,
        dataset_id="coffee_d0",
        stage="pretrain",
        log_root=tmp_path / "logs",
    )

    run_state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    assert run_state["stage"] == "pretrain"
    assert run_state["status"] == "running"
    assert run_state["target_progress"] == 12


def test_snapshot_run_config_sets_stable_wandb_resume_fields(tmp_path: Path) -> None:
    generated_config_path = tmp_path / "generated.yaml"
    generated_config_path.write_text("train:\n  n_epochs: 12\n", encoding="utf-8")
    task_spec_path = tmp_path / "task.yaml"
    task_spec_path.write_text("dataset:\n  path: coffee_d0.hdf5\n", encoding="utf-8")

    cfg = OmegaConf.create(
        {
            "seed": 7,
            "horizon_steps": 4,
            "denoising_steps": 20,
            "train": {"n_epochs": 12},
            "wandb": {"entity": "entity", "project": "project", "run": "placeholder"},
        }
    )
    task_spec = OmegaConf.create({"dataset": {"path": "coffee_d0.hdf5"}})

    run_dir, config_path, manifest = launcher._snapshot_run_config(
        cfg=cfg,
        task_spec=task_spec,
        task_spec_path=task_spec_path,
        generated_config_path=generated_config_path,
        dataset_id="coffee_d0",
        stage="pretrain",
        log_root=tmp_path / "logs",
    )

    saved_cfg = OmegaConf.load(config_path)
    assert saved_cfg.wandb.run == manifest["run_id"]
    assert saved_cfg.wandb.id == manifest["run_id"]
    assert saved_cfg.wandb.resume == "allow"
    assert saved_cfg.logdir == run_dir.as_posix()


def test_configure_stage_resume_prefers_latest_state_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "pretrain" / "coffee_d0" / "run" / "20260423_000001_s42_abc"
    checkpoint_dir = run_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True)
    (run_dir / "best_checkpoint.pt").write_text("best", encoding="utf-8")
    (checkpoint_dir / "state_10.pt").write_text("10", encoding="utf-8")
    (checkpoint_dir / "state_15.pt").write_text("15", encoding="utf-8")
    cfg = OmegaConf.create({})

    resume_checkpoint, details = launcher._configure_stage_resume(
        cfg,
        stage="finetune",
        reusable_run=(run_dir, {"run_id": run_dir.name}),
    )

    assert resume_checkpoint == checkpoint_dir / "state_15.pt"
    assert cfg.resume_path == (checkpoint_dir / "state_15.pt").as_posix()
    assert details["reused_run_dir"] == run_dir.as_posix()
    assert details["reused_run_id"] == run_dir.name
    assert details["resume_path"] == (checkpoint_dir / "state_15.pt").as_posix()
