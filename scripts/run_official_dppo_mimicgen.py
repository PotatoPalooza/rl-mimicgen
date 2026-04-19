from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from rl_mimicgen.adapters.dppo_lowdim import REPO_ROOT, materialize_mimicgen_lowdim_port


TASK_SPECS_ROOT = REPO_ROOT / "configs" / "mimicgen_tasks"
DPPO_ROOT = REPO_ROOT / "dppo"
DEFAULT_OUTPUT_ROOT = "runs/official_dppo_mimicgen"
DEFAULT_LOG_ROOT = "logs/official_dppo/mimicgen"
DEFAULT_SWEEP_EVAL_MODE = "bc"


def _register_resolvers() -> None:
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    if not OmegaConf.has_resolver("round_up"):
        OmegaConf.register_new_resolver("round_up", math.ceil)
    if not OmegaConf.has_resolver("round_down"):
        OmegaConf.register_new_resolver("round_down", math.floor)


def _resolve_repo_path(path_like: str | None, default: str | Path | None = None) -> Path | None:
    value = path_like if path_like is not None else default
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _load_task_spec(task: str) -> tuple[DictConfig, Path]:
    candidate = Path(task).expanduser()
    if candidate.suffix in {".yaml", ".yml"}:
        task_path = candidate if candidate.is_absolute() else (REPO_ROOT / candidate)
    else:
        task_path = TASK_SPECS_ROOT / f"{task}.yaml"
    if not task_path.exists():
        raise FileNotFoundError(f"Task spec not found: {task_path}")
    return OmegaConf.load(task_path), task_path.resolve()


def _task_dataset_id(task_spec: DictConfig) -> str:
    if "dataset_id" in task_spec and task_spec.dataset_id:
        return str(task_spec.dataset_id)
    return Path(task_spec.dataset.path).stem


def _materialize_from_task_spec(task_spec: DictConfig) -> dict[str, Any]:
    dataset_id = _task_dataset_id(task_spec)
    return materialize_mimicgen_lowdim_port(
        _resolve_repo_path(str(task_spec.dataset.path)),
        output_root=_resolve_repo_path(task_spec.get("materialize", {}).get("output_root"), DEFAULT_OUTPUT_ROOT),
        dataset_id=dataset_id,
        obs_keys=task_spec.get("materialize", {}).get("obs_keys"),
        max_demos=task_spec.get("materialize", {}).get("max_demos"),
        val_split=float(task_spec.get("materialize", {}).get("val_split", 0.0)),
        split_seed=int(task_spec.get("materialize", {}).get("split_seed", 0)),
        log_root=_resolve_repo_path(task_spec.get("logging", {}).get("root"), DEFAULT_LOG_ROOT),
        config_root=REPO_ROOT / "dppo" / "cfg" / "mimicgen" / "generated",
        wandb_entity=task_spec.get("wandb", {}).get("entity"),
    )


def _stage_config_filename(stage: str) -> str:
    return {
        "pretrain": "pre_diffusion_mlp.yaml",
        "finetune": "ft_ppo_diffusion_mlp.yaml",
        "eval-bc": "eval_bc_diffusion_mlp.yaml",
        "eval-rl-init": "eval_rl_init_diffusion_mlp.yaml",
    }[stage]


def _stage_config_key(stage: str) -> str:
    return {
        "pretrain": "pretrain",
        "finetune": "finetune",
        "eval-bc": "eval_bc",
        "eval-rl-init": "eval_rl_init",
    }[stage]


def _stage_log_group(stage: str) -> str:
    return {
        "pretrain": "pretrain",
        "finetune": "finetune",
        "eval-bc": "eval",
        "eval-rl-init": "eval",
    }[stage]


def _build_run_name(stage: str, cfg: DictConfig, dataset_id: str) -> str:
    if stage == "pretrain":
        return f"{dataset_id}_pre_diffusion_mlp_ta{cfg.horizon_steps}_td{cfg.denoising_steps}"
    if stage == "finetune":
        return (
            f"{dataset_id}_ft_diffusion_mlp_ta{cfg.horizon_steps}_td{cfg.denoising_steps}"
            f"_tdf{cfg.ft_denoising_steps}"
        )
    if stage == "eval-bc":
        return f"{dataset_id}_eval_bc_diffusion_mlp_ta{cfg.horizon_steps}_td{cfg.denoising_steps}"
    if stage == "eval-rl-init":
        return (
            f"{dataset_id}_eval_rl_init_diffusion_mlp_ta{cfg.horizon_steps}_td{cfg.denoising_steps}"
            f"_tdf{cfg.ft_denoising_steps}"
        )
    raise ValueError(f"Unknown stage: {stage}")


def _checkpoint_sort_key(path: Path) -> tuple[str, int]:
    run_dir = path.parent.parent
    run_stamp = run_dir.name
    checkpoint_name = path.stem
    try:
        checkpoint_step = int(checkpoint_name.split("_")[1])
    except (IndexError, ValueError):
        checkpoint_step = -1
    return run_stamp, checkpoint_step


def _latest_checkpoint_for_task(log_root: Path, dataset_id: str) -> Path:
    checkpoint_paths = sorted(
        log_root.glob(f"pretrain/{dataset_id}/**/checkpoint/state_*.pt"),
        key=_checkpoint_sort_key,
    )
    if not checkpoint_paths:
        raise FileNotFoundError(f"No pretrain checkpoints found for {dataset_id} under {log_root}")
    return checkpoint_paths[-1]


def _checkpoint_path_from_args(args: argparse.Namespace, log_root: Path, dataset_id: str) -> Path:
    checkpoint = getattr(args, "checkpoint", None)
    if checkpoint:
        return _resolve_repo_path(checkpoint)
    return _latest_checkpoint_for_task(log_root, dataset_id)


def _merge_stage_config(base_config_path: Path, task_spec: DictConfig, stage: str, extra_dotlist: list[str]) -> DictConfig:
    cfg = OmegaConf.load(base_config_path)
    stage_key = _stage_config_key(stage)
    stage_cfg = task_spec.get(stage_key, {})
    if "config" in stage_cfg and stage_cfg.config is not None:
        cfg = OmegaConf.merge(cfg, stage_cfg.config)
    if extra_dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(extra_dotlist))
    return cfg


def _snapshot_run_config(
    *,
    cfg: DictConfig,
    task_spec: DictConfig,
    task_spec_path: Path,
    generated_config_path: Path,
    dataset_id: str,
    stage: str,
    log_root: Path,
    source_checkpoint: Path | None = None,
) -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = _build_run_name(stage, cfg, dataset_id)
    run_dir = log_root / _stage_log_group(stage) / dataset_id / run_name / f"{timestamp}_{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg.name = run_name
    cfg.logdir = run_dir.as_posix()
    if "wandb" in cfg and cfg.wandb is not None:
        cfg.wandb.run = f"{timestamp.split('_')[1]}_{run_name}"

    config_path = run_dir / "run_config.yaml"
    OmegaConf.save(cfg, config_path)
    spec_snapshot_path = run_dir / "task_spec_snapshot.yaml"
    OmegaConf.save(task_spec, spec_snapshot_path)

    manifest = {
        "dataset_id": dataset_id,
        "stage": stage,
        "task_spec_path": task_spec_path.as_posix(),
        "generated_config_path": generated_config_path.as_posix(),
        "run_config_path": config_path.as_posix(),
        "run_dir": run_dir.as_posix(),
        "source_checkpoint": source_checkpoint.as_posix() if source_checkpoint is not None else None,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run_dir, config_path


def _subprocess_env(task_spec: DictConfig, *, for_video: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    runtime_cfg = task_spec.get("runtime", {})
    backend_key = "video_mujoco_gl" if for_video else "mujoco_gl"
    mujoco_gl = runtime_cfg.get(backend_key) or runtime_cfg.get("mujoco_gl")
    if mujoco_gl:
        env["MUJOCO_GL"] = str(mujoco_gl)
    return env


def _run_dppo_with_snapshot(task_spec: DictConfig, run_dir: Path, config_path: Path) -> None:
    command = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        str(DPPO_ROOT / "script" / "run.py"),
        f"--config-dir={run_dir.as_posix()}",
        "--config-name=run_config",
    ]
    subprocess.run(command, cwd=DPPO_ROOT, check=True, env=_subprocess_env(task_spec))


def _prepare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task-spec launcher for official DPPO on MimicGen.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Materialize artifacts and generated configs for a task.")
    prepare.add_argument("--task", required=True, help="Task id or path to a task spec YAML.")

    for stage_name in ("pretrain", "finetune", "eval-bc", "eval-rl-init"):
        stage = subparsers.add_parser(stage_name, help=f"Run {stage_name} through the official dppo launcher.")
        stage.add_argument("--task", required=True, help="Task id or path to a task spec YAML.")
        if stage_name in {"finetune", "eval-bc", "eval-rl-init"}:
            stage.add_argument("--checkpoint", default=None, help="Optional checkpoint path. Defaults to the latest pretrain checkpoint for the task.")
        stage.add_argument(
            "--set",
            dest="dotlist",
            action="append",
            default=[],
            help="Optional OmegaConf dotlist override applied to the run snapshot.",
        )

    sweep = subparsers.add_parser("sweep", help="Sweep saved checkpoints for a task using the configured eval mode.")
    sweep.add_argument("--task", required=True, help="Task id or path to a task spec YAML.")
    sweep.add_argument("--checkpoint-dir", default=None, help="Optional checkpoint directory to sweep. Defaults to the latest pretrain run checkpoint dir.")
    sweep.add_argument(
        "--eval-mode",
        choices=("bc", "rl-init"),
        default=None,
        help="Optional eval surface override. Defaults to the task spec sweep setting.",
    )
    sweep.add_argument(
        "--every-n",
        type=int,
        default=None,
        help="Optional checkpoint stride override for this sweep. Defaults to the task spec sweep.every_n value.",
    )
    sweep.add_argument(
        "--n-episodes",
        type=int,
        default=None,
        help="Optional completed-episode target override for this sweep. Defaults to the task spec eval/sweep setting.",
    )

    return parser


def _run_prepare(task_spec: DictConfig) -> None:
    materialized = _materialize_from_task_spec(task_spec)
    payload = {
        "dataset_id": materialized["spec"].dataset_id,
        "artifact_dir": materialized["artifact_dir"].as_posix(),
        "config_dir": materialized["config_dir"].as_posix(),
        "task_manifest": materialized["task_manifest"].as_posix(),
    }
    print(json.dumps(payload, indent=2))


def _run_stage(args: argparse.Namespace, stage: str) -> None:
    task_spec, task_spec_path = _load_task_spec(args.task)
    materialized = _materialize_from_task_spec(task_spec)
    dataset_id = materialized["spec"].dataset_id
    log_root = Path(materialized["log_root"])
    generated_config_path = materialized["configs"][_stage_config_filename(stage)]
    cfg = _merge_stage_config(generated_config_path, task_spec, stage, args.dotlist)

    source_checkpoint = None
    if stage in {"finetune", "eval-bc", "eval-rl-init"}:
        source_checkpoint = _checkpoint_path_from_args(args, log_root, dataset_id)
        cfg.base_policy_path = source_checkpoint.as_posix()

    run_dir, config_path = _snapshot_run_config(
        cfg=cfg,
        task_spec=task_spec,
        task_spec_path=task_spec_path,
        generated_config_path=generated_config_path,
        dataset_id=dataset_id,
        stage=stage,
        log_root=log_root,
        source_checkpoint=source_checkpoint,
    )
    _run_dppo_with_snapshot(task_spec, run_dir, config_path)


def _run_sweep(args: argparse.Namespace) -> None:
    task_spec, task_spec_path = _load_task_spec(args.task)
    materialized = _materialize_from_task_spec(task_spec)
    dataset_id = materialized["spec"].dataset_id
    log_root = Path(materialized["log_root"])

    sweep_cfg = task_spec.get("sweep", {})
    eval_mode = args.eval_mode or sweep_cfg.get("eval_mode", DEFAULT_SWEEP_EVAL_MODE)
    eval_stage = "eval-bc" if eval_mode == "bc" else "eval-rl-init"
    generated_config_path = materialized["configs"][_stage_config_filename(eval_stage)]
    cfg = _merge_stage_config(generated_config_path, task_spec, eval_stage, [])

    if args.checkpoint_dir is not None:
        checkpoint_dir = _resolve_repo_path(args.checkpoint_dir)
    else:
        checkpoint_dir = _latest_checkpoint_for_task(log_root, dataset_id).parent

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{dataset_id}_sweep_{eval_mode}"
    run_dir = log_root / "sweep" / dataset_id / run_name / f"{timestamp}_{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.logdir = run_dir.as_posix()

    eval_config_path = run_dir / "eval_run_config.yaml"
    OmegaConf.save(cfg, eval_config_path)
    OmegaConf.save(task_spec, run_dir / "task_spec_snapshot.yaml")
    manifest = {
        "dataset_id": dataset_id,
        "stage": "sweep",
        "eval_mode": eval_mode,
        "task_spec_path": task_spec_path.as_posix(),
        "generated_config_path": generated_config_path.as_posix(),
        "run_dir": run_dir.as_posix(),
        "checkpoint_dir": checkpoint_dir.as_posix(),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    command = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        str(DPPO_ROOT / "script" / "eval_checkpoint_sweep.py"),
        "--config-dir",
        run_dir.as_posix(),
        "--config-name",
        "eval_run_config",
        "--checkpoint-dir",
        checkpoint_dir.as_posix(),
        "--output-dir",
        run_dir.as_posix(),
        "--device",
        str(sweep_cfg.get("device", "cpu")),
        "--n-envs",
        str(sweep_cfg.get("n_envs", 8)),
        "--n-episodes",
        str(args.n_episodes if args.n_episodes is not None else sweep_cfg.get("n_episodes", cfg.get("n_episodes", 20))),
        "--n-steps",
        str(sweep_cfg.get("n_steps", materialized["spec"].horizon)),
        "--max-episode-steps",
        str(sweep_cfg.get("max_episode_steps", materialized["spec"].horizon)),
        "--every-n",
        str(args.every_n if args.every_n is not None else sweep_cfg.get("every_n", 1)),
        "--video-checkpoints",
        str(sweep_cfg.get("video_checkpoints", "none")),
        "--render-num",
        str(sweep_cfg.get("render_num", 1)),
    ]
    if sweep_cfg.get("skip_existing", True):
        command.append("--skip-existing")
    if sweep_cfg.get("copy_best_to"):
        command.extend(["--copy-best-to", str(Path(run_dir) / str(sweep_cfg.copy_best_to))])

    subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        env=_subprocess_env(task_spec, for_video=str(sweep_cfg.get("video_checkpoints", "none")) != "none"),
    )


def main() -> None:
    _register_resolvers()
    args = _prepare_parser().parse_args()
    if args.command == "prepare":
        task_spec, _ = _load_task_spec(args.task)
        _run_prepare(task_spec)
        return
    if args.command == "pretrain":
        _run_stage(args, "pretrain")
        return
    if args.command == "finetune":
        _run_stage(args, "finetune")
        return
    if args.command == "eval-bc":
        _run_stage(args, "eval-bc")
        return
    if args.command == "eval-rl-init":
        _run_stage(args, "eval-rl-init")
        return
    if args.command == "sweep":
        _run_sweep(args)
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
