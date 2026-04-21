from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
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


@dataclass(frozen=True)
class StageSpec:
    config_filename: str | None
    task_key: str | None
    log_group: str


STAGE_SPECS = {
    "pretrain": StageSpec(
        config_filename="pre_diffusion_mlp.yaml",
        task_key="pretrain",
        log_group="pretrain",
    ),
    "finetune": StageSpec(
        config_filename="ft_ppo_diffusion_mlp.yaml",
        task_key="finetune",
        log_group="finetune",
    ),
    "eval-bc": StageSpec(
        config_filename="eval_bc_diffusion_mlp.yaml",
        task_key="eval_bc",
        log_group="eval",
    ),
    "eval-rl-init": StageSpec(
        config_filename="eval_rl_init_diffusion_mlp.yaml",
        task_key="eval_rl_init",
        log_group="eval",
    ),
    "sweep-bc": StageSpec(
        config_filename="eval_bc_diffusion_mlp.yaml",
        task_key="eval_bc",
        log_group="sweep",
    ),
    "sweep-rl-init": StageSpec(
        config_filename="eval_rl_init_diffusion_mlp.yaml",
        task_key="eval_rl_init",
        log_group="sweep",
    ),
}


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


def _task_log_root(task_spec: DictConfig) -> Path:
    return _resolve_repo_path(task_spec.get("logging", {}).get("root"), DEFAULT_LOG_ROOT)


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
        log_root=_task_log_root(task_spec),
        config_root=REPO_ROOT / "dppo" / "cfg" / "mimicgen" / "generated",
        wandb_entity=task_spec.get("wandb", {}).get("entity"),
    )


def _stage_spec(stage: str) -> StageSpec:
    return STAGE_SPECS[stage]


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
    if stage == "sweep-bc":
        return f"{dataset_id}_sweep_bc"
    if stage == "sweep-rl-init":
        return f"{dataset_id}_sweep_rl_init"
    raise ValueError(f"Unknown stage: {stage}")


def _make_run_id(stage: str, dataset_id: str, seed: int, *, eval_mode: str | None = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    stage_token = stage.replace("-", "_")
    if eval_mode is not None:
        stage_token = f"{stage_token}_{eval_mode}"
    return f"{stage_token}_{dataset_id}_{timestamp}_s{seed}_{suffix}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def _checkpoint_sort_key(path: Path) -> tuple[str, int]:
    run_dir = path.parent.parent
    run_stamp = _run_dir_sort_token(run_dir)
    checkpoint_name = path.stem
    try:
        checkpoint_step = int(checkpoint_name.split("_")[1])
    except (IndexError, ValueError):
        checkpoint_step = -1
    return run_stamp, checkpoint_step


def _run_dir_sort_token(run_dir: Path) -> str:
    name = run_dir.name
    matches = re.findall(r"(\d{8}_\d{6}|\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", name)
    if matches:
        token = matches[-1]
        return token.replace("-", "")
    return name


def _latest_checkpoint_for_task(log_root: Path, dataset_id: str) -> Path:
    checkpoint_paths = sorted(
        log_root.glob(f"pretrain/{dataset_id}/**/checkpoint/state_*.pt"),
        key=_checkpoint_sort_key,
    )
    if not checkpoint_paths:
        raise FileNotFoundError(f"No pretrain checkpoints found for {dataset_id} under {log_root}")
    return checkpoint_paths[-1]


def _latest_checkpoint_in_run_dir(run_dir: Path) -> Path:
    best_checkpoint = run_dir / "best_checkpoint.pt"
    if best_checkpoint.exists():
        return best_checkpoint
    checkpoint_paths = sorted((run_dir / "checkpoint").glob("state_*.pt"), key=_checkpoint_sort_key)
    if checkpoint_paths:
        return checkpoint_paths[-1]
    raise FileNotFoundError(f"No checkpoint found under run directory: {run_dir}")


def _checkpoint_path_from_args(args: argparse.Namespace, log_root: Path, dataset_id: str) -> Path:
    checkpoint = getattr(args, "checkpoint", None)
    if checkpoint:
        return _resolve_repo_path(checkpoint)
    run_dir = getattr(args, "run_dir", None)
    if run_dir:
        return _latest_checkpoint_in_run_dir(_resolve_repo_path(run_dir))
    return _latest_checkpoint_for_task(log_root, dataset_id)


def _merge_stage_config(base_config_path: Path, task_spec: DictConfig, stage: str, extra_dotlist: list[str]) -> DictConfig:
    cfg = OmegaConf.load(base_config_path)
    stage_key = _stage_spec(stage).task_key
    stage_cfg = task_spec.get(stage_key, {})
    if "config" in stage_cfg and stage_cfg.config is not None:
        cfg = OmegaConf.merge(cfg, stage_cfg.config)
    if extra_dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(extra_dotlist))
    return cfg


def _maybe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _checkpoint_path_matches(target: Path, candidate: str | None) -> bool:
    if not candidate:
        return False
    return target == Path(candidate).expanduser().resolve()


def _bc_success_rate_from_summary(summary: dict[str, Any], checkpoint_path: Path) -> float | None:
    best_metrics = summary.get("best_metrics", {})
    if not isinstance(best_metrics, dict):
        return None
    if not any(
        (
            _checkpoint_path_matches(checkpoint_path, summary.get("best_checkpoint")),
            _checkpoint_path_matches(checkpoint_path, summary.get("best_checkpoint_copy")),
            _checkpoint_path_matches(checkpoint_path, best_metrics.get("checkpoint_path")),
        )
    ):
        return None
    success_rate = best_metrics.get("success_rate")
    if success_rate is None:
        return None
    return float(success_rate)


def _bc_success_rate_from_filename(checkpoint_path: Path) -> float | None:
    checkpoint_stem = checkpoint_path.stem
    success_match = re.search(r"_success_(\d+(?:\.\d+)?)$", checkpoint_stem)
    if success_match:
        return float(success_match.group(1))
    tagged_best_match = re.search(r"best_checkpoint_(\d+)_best$", checkpoint_stem)
    if tagged_best_match:
        return float(tagged_best_match.group(1)) / 100.0
    return None


def _candidate_bc_summary_paths(checkpoint_path: Path, log_root: Path, dataset_id: str) -> list[Path]:
    candidates: list[Path] = []
    local_candidates = (
        checkpoint_path.parent / "best_checkpoint.json",
        checkpoint_path.parent / "checkpoint_eval" / "best_checkpoint.json",
        checkpoint_path.parent.parent / "best_checkpoint.json",
        checkpoint_path.parent.parent / "checkpoint_eval" / "best_checkpoint.json",
    )
    seen: set[Path] = set()
    for candidate in local_candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            candidates.append(resolved)

    summary_paths = sorted(
        [
            *log_root.glob(f"sweep/{dataset_id}/**/best_checkpoint.json"),
            *log_root.glob(f"pretrain/{dataset_id}/**/best_checkpoint.json"),
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in summary_paths:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            candidates.append(resolved)
    return candidates


def _finetune_checkpoint_metadata(checkpoint_path: Path, log_root: Path, dataset_id: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source_checkpoint": checkpoint_path.as_posix(),
    }
    for summary_path in _candidate_bc_summary_paths(checkpoint_path, log_root, dataset_id):
        summary = _maybe_load_json(summary_path)
        if summary is None:
            continue
        success_rate = _bc_success_rate_from_summary(summary, checkpoint_path)
        if success_rate is None:
            continue
        metadata["bc_success_rate"] = success_rate
        metadata["bc_success_source"] = summary_path.as_posix()
        break
    else:
        success_rate = _bc_success_rate_from_filename(checkpoint_path)
        if success_rate is not None:
            metadata["bc_success_rate"] = success_rate
            metadata["bc_success_source"] = "checkpoint_filename"
    return metadata


def _seed_override_from_args_or_task_spec(
    args: argparse.Namespace,
    task_spec: DictConfig,
    *,
    config_seed: int,
    sweep_cfg: DictConfig | None = None,
) -> int:
    arg_seed = getattr(args, "seed", None)
    if arg_seed is not None:
        return int(arg_seed)
    if sweep_cfg is not None and sweep_cfg.get("seed", None) is not None:
        return int(sweep_cfg.seed)
    if task_spec.get("seed", None) is not None:
        return int(task_spec.seed)
    return int(config_seed)


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
    launcher_metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    spec = _stage_spec(stage)
    run_name = _build_run_name(stage, cfg, dataset_id)
    run_id = _make_run_id(stage, dataset_id, int(cfg.seed))
    created_at = datetime.now().isoformat(timespec="seconds")
    run_dir = log_root / spec.log_group / dataset_id / run_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg.name = run_name
    cfg.logdir = run_dir.as_posix()
    if "wandb" in cfg and cfg.wandb is not None:
        cfg.wandb.run = run_id

    config_path = run_dir / "run_config.yaml"
    OmegaConf.save(cfg, config_path)
    spec_snapshot_path = run_dir / "task_spec_snapshot.yaml"
    OmegaConf.save(task_spec, spec_snapshot_path)
    generated_snapshot_path = run_dir / "generated_config_snapshot.yaml"
    generated_snapshot_path.write_text(generated_config_path.read_text(encoding="utf-8"), encoding="utf-8")

    manifest = {
        "dataset_id": dataset_id,
        "stage": stage,
        "run_id": run_id,
        "run_name": run_name,
        "created_at": created_at,
        "task_spec_path": task_spec_path.as_posix(),
        "task_spec_snapshot_path": spec_snapshot_path.as_posix(),
        "generated_config_path": generated_config_path.as_posix(),
        "generated_config_snapshot_path": generated_snapshot_path.as_posix(),
        "run_config_path": config_path.as_posix(),
        "run_dir": run_dir.as_posix(),
        "artifacts_dir": (run_dir / "artifacts").as_posix(),
        "wandb_dir": (run_dir / "wandb").as_posix(),
        "source_checkpoint": source_checkpoint.as_posix() if source_checkpoint is not None else None,
        "launcher_metadata": launcher_metadata,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run_dir, config_path, manifest


def _subprocess_env(task_spec: DictConfig, *, for_video: bool = False, run_dir: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    runtime_cfg = task_spec.get("runtime", {})
    backend_key = "video_mujoco_gl" if for_video else "mujoco_gl"
    mujoco_gl = runtime_cfg.get(backend_key) or runtime_cfg.get("mujoco_gl")
    if mujoco_gl:
        env["MUJOCO_GL"] = str(mujoco_gl)
    if run_dir is not None:
        wandb_dir = run_dir / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        env["WANDB_DIR"] = wandb_dir.as_posix()
    return env


def _run_dppo_with_snapshot(task_spec: DictConfig, run_dir: Path, config_path: Path) -> None:
    run_cfg = OmegaConf.load(config_path)
    save_video = bool(run_cfg.get("env", {}).get("save_video", False))
    command = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        str(DPPO_ROOT / "script" / "run.py"),
        f"--config-dir={run_dir.as_posix()}",
        "--config-name=run_config",
    ]
    subprocess.run(
        command,
        cwd=DPPO_ROOT,
        check=True,
        env=_subprocess_env(task_spec, for_video=save_video, run_dir=run_dir),
    )


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
            stage.add_argument("--run-dir", default=None, help="Optional old run directory. Uses `best_checkpoint.pt` if present, otherwise the latest checkpoint under `checkpoint/`.")
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
    sweep.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional eval seed override for this sweep. Defaults to sweep.seed, then task seed, then the generated config seed.",
    )
    sweep.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Optional minimum checkpoint index. Defaults to the task spec sweep.start_index if set.",
    )
    sweep.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Optional maximum checkpoint index. Defaults to the task spec sweep.end_index if set.",
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
    stage_spec = _stage_spec(stage)
    generated_config_path = materialized["configs"][stage_spec.config_filename]
    cfg = _merge_stage_config(generated_config_path, task_spec, stage, args.dotlist)

    source_checkpoint = None
    launcher_metadata = None
    if stage in {"finetune", "eval-bc", "eval-rl-init"}:
        source_checkpoint = _checkpoint_path_from_args(args, log_root, dataset_id)
        cfg.base_policy_path = source_checkpoint.as_posix()
        if stage == "finetune":
            launcher_metadata = _finetune_checkpoint_metadata(source_checkpoint, log_root, dataset_id)
            cfg.launcher_metadata = OmegaConf.create(launcher_metadata)

    run_dir, config_path, _ = _snapshot_run_config(
        cfg=cfg,
        task_spec=task_spec,
        task_spec_path=task_spec_path,
        generated_config_path=generated_config_path,
        dataset_id=dataset_id,
        stage=stage,
        log_root=log_root,
        source_checkpoint=source_checkpoint,
        launcher_metadata=launcher_metadata,
    )
    _run_dppo_with_snapshot(task_spec, run_dir, config_path)


def _run_sweep(args: argparse.Namespace) -> None:
    task_spec, task_spec_path = _load_task_spec(args.task)
    materialized = _materialize_from_task_spec(task_spec)
    dataset_id = materialized["spec"].dataset_id
    log_root = Path(materialized["log_root"])

    sweep_cfg = task_spec.get("sweep", {})
    eval_mode = args.eval_mode or sweep_cfg.get("eval_mode", DEFAULT_SWEEP_EVAL_MODE)
    sweep_stage = "sweep-bc" if eval_mode == "bc" else "sweep-rl-init"
    eval_stage = "eval-bc" if eval_mode == "bc" else "eval-rl-init"
    sweep_stage_spec = _stage_spec(sweep_stage)
    generated_config_path = materialized["configs"][sweep_stage_spec.config_filename]
    cfg = _merge_stage_config(generated_config_path, task_spec, eval_stage, [])
    sweep_seed = _seed_override_from_args_or_task_spec(
        args,
        task_spec,
        config_seed=int(cfg.seed),
        sweep_cfg=sweep_cfg,
    )
    cfg.seed = sweep_seed

    if args.checkpoint_dir is not None:
        checkpoint_dir = _resolve_repo_path(args.checkpoint_dir)
    else:
        checkpoint_dir = _latest_checkpoint_for_task(log_root, dataset_id).parent

    run_name = _build_run_name(sweep_stage, cfg, dataset_id)
    run_id = _make_run_id("sweep", dataset_id, int(cfg.seed), eval_mode=eval_mode)
    created_at = datetime.now().isoformat(timespec="seconds")
    run_dir = log_root / sweep_stage_spec.log_group / dataset_id / run_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.logdir = run_dir.as_posix()

    eval_config_path = run_dir / "eval_run_config.yaml"
    OmegaConf.save(cfg, eval_config_path)
    OmegaConf.save(task_spec, run_dir / "task_spec_snapshot.yaml")
    generated_snapshot_path = run_dir / "generated_config_snapshot.yaml"
    generated_snapshot_path.write_text(generated_config_path.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = {
        "dataset_id": dataset_id,
        "stage": "sweep",
        "run_id": run_id,
        "run_name": run_name,
        "created_at": created_at,
        "eval_mode": eval_mode,
        "seed": sweep_seed,
        "task_spec_path": task_spec_path.as_posix(),
        "task_spec_snapshot_path": (run_dir / "task_spec_snapshot.yaml").as_posix(),
        "generated_config_path": generated_config_path.as_posix(),
        "generated_config_snapshot_path": generated_snapshot_path.as_posix(),
        "run_config_path": eval_config_path.as_posix(),
        "run_dir": run_dir.as_posix(),
        "artifacts_dir": run_dir.as_posix(),
        "wandb_dir": (run_dir / "wandb").as_posix(),
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
        "--seed",
        str(sweep_seed),
        "--n-envs",
        str(sweep_cfg.get("n_envs", 8)),
        "--n-episodes",
        str(args.n_episodes if args.n_episodes is not None else sweep_cfg.get("n_episodes", cfg.get("n_episodes", 40))),
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
    start_index = args.start_index if args.start_index is not None else sweep_cfg.get("start_index", None)
    end_index = args.end_index if args.end_index is not None else sweep_cfg.get("end_index", None)
    if start_index is not None:
        command.extend(["--start-index", str(start_index)])
    if end_index is not None:
        command.extend(["--end-index", str(end_index)])
    if sweep_cfg.get("skip_existing", True):
        command.append("--skip-existing")
    if sweep_cfg.get("copy_best_to"):
        command.extend(["--copy-best-to", str(Path(run_dir) / str(sweep_cfg.copy_best_to))])

    subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        env=_subprocess_env(task_spec, for_video=str(sweep_cfg.get("video_checkpoints", "none")) != "none", run_dir=run_dir),
    )
    best_summary_path = run_dir / "best_checkpoint.json"
    best_summary = json.loads(best_summary_path.read_text(encoding="utf-8")) if best_summary_path.exists() else {}
    print(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "stage": "sweep",
                "eval_mode": eval_mode,
                "seed": sweep_seed,
                "run_dir": run_dir.as_posix(),
                "run_manifest_path": (run_dir / "run_manifest.json").as_posix(),
                "best_checkpoint": best_summary.get("best_checkpoint"),
                "best_checkpoint_copy": best_summary.get("best_checkpoint_copy"),
                "best_checkpoint_summary_path": best_summary_path.as_posix(),
            },
            indent=2,
        )
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
