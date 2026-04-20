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
DPPO_ROOT = REPO_ROOT / "resources" / "dppo"
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


def _ensure_dataset(task_spec: DictConfig) -> None:
    """Download the mimicgen dataset referenced by ``task_spec`` if missing.

    The task spec's ``dataset.path`` is expected to live under a dataset-type
    directory (e.g. ``.../core/stack_d0.hdf5``). The parent directory name
    maps to a mimicgen ``DATASET_REGISTRY`` dataset type and the filename stem
    maps to a task; both are validated before issuing the HF download.
    """
    dataset_path = _resolve_repo_path(str(task_spec.dataset.path))
    if dataset_path is None or dataset_path.exists():
        return

    dataset_type = dataset_path.parent.name
    task_name = dataset_path.stem

    from mimicgen import DATASET_REGISTRY, HF_REPO_ID
    from mimicgen.utils import file_utils as FileUtils

    if dataset_type not in DATASET_REGISTRY:
        raise RuntimeError(
            f"Cannot auto-download: dataset file {dataset_path} is missing and "
            f"parent directory {dataset_type!r} is not a registered mimicgen "
            f"dataset type. Known types: {sorted(DATASET_REGISTRY.keys())}."
        )
    if task_name not in DATASET_REGISTRY[dataset_type]:
        raise RuntimeError(
            f"Cannot auto-download: task {task_name!r} not registered under "
            f"mimicgen dataset type {dataset_type!r}. Available: "
            f"{sorted(DATASET_REGISTRY[dataset_type].keys())}."
        )

    url = DATASET_REGISTRY[dataset_type][task_name]["url"]
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[run_official_dppo_mimicgen] dataset not found at {dataset_path}; "
        f"downloading huggingface://{HF_REPO_ID}/{url}",
        flush=True,
    )
    FileUtils.download_file_from_hf(
        repo_id=HF_REPO_ID,
        filename=url,
        download_dir=str(dataset_path.parent),
        check_overwrite=False,
    )
    if not dataset_path.exists():
        raise RuntimeError(
            f"Download reported success but dataset file is still missing: {dataset_path}"
        )


def _default_wandb_group(dataset_id: str) -> str:
    """Derive a default wandb group from the dataset id.

    Strips the task stem so variants group together across tasks: ``stack_d0`` →
    ``d0``, ``coffee_d1`` → ``d1``, ``mug_cleanup_o2`` → ``o2``. Falls back to
    the full id if there's no trailing ``_d<N>`` / ``_o<N>`` suffix.
    """
    match = re.search(r"_([doDO]\d+)$", dataset_id)
    return match.group(1).lower() if match else dataset_id


def _materialize_from_task_spec(task_spec: DictConfig) -> dict[str, Any]:
    _ensure_dataset(task_spec)
    dataset_id = _task_dataset_id(task_spec)
    wandb_cfg = task_spec.get("wandb", {}) or {}
    wandb_group = wandb_cfg.get("group") or _default_wandb_group(dataset_id)
    return materialize_mimicgen_lowdim_port(
        _resolve_repo_path(str(task_spec.dataset.path)),
        output_root=_resolve_repo_path(task_spec.get("materialize", {}).get("output_root"), DEFAULT_OUTPUT_ROOT),
        dataset_id=dataset_id,
        obs_keys=task_spec.get("materialize", {}).get("obs_keys"),
        max_demos=task_spec.get("materialize", {}).get("max_demos"),
        val_split=float(task_spec.get("materialize", {}).get("val_split", 0.0)),
        split_seed=int(task_spec.get("materialize", {}).get("split_seed", 0)),
        log_root=_task_log_root(task_spec),
        config_root=REPO_ROOT / "resources" / "dppo" / "cfg" / "mimicgen" / "generated",
        wandb_entity=wandb_cfg.get("entity"),
        wandb_group=wandb_group,
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


def _best_pretrain_checkpoint_for_task(log_root: Path, dataset_id: str) -> Path:
    """Pick the best pretrain checkpoint for ``dataset_id``.

    Walks pretrain run directories most-recent first and returns
    ``best_checkpoint.pt`` (written by async checkpoint-eval's ``copy_best_to``)
    if present, otherwise the latest ``state_*.pt`` in that run.
    """
    pretrain_root = log_root / "pretrain" / dataset_id
    run_dirs = sorted(
        {ckpt_dir.parent for ckpt_dir in pretrain_root.glob("**/checkpoint") if ckpt_dir.is_dir()},
        key=_run_dir_sort_token,
    )
    if not run_dirs:
        raise FileNotFoundError(f"No pretrain runs found for {dataset_id} under {log_root}")
    for run_dir in reversed(run_dirs):
        try:
            return _latest_checkpoint_in_run_dir(run_dir)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"No pretrain checkpoints found for {dataset_id} under {log_root}")


def _checkpoint_path_from_args(args: argparse.Namespace, log_root: Path, dataset_id: str) -> Path:
    checkpoint = getattr(args, "checkpoint", None)
    if checkpoint:
        return _resolve_repo_path(checkpoint)
    run_dir = getattr(args, "run_dir", None)
    if run_dir:
        return _latest_checkpoint_in_run_dir(_resolve_repo_path(run_dir))
    return _best_pretrain_checkpoint_for_task(log_root, dataset_id)


def _merge_stage_config(base_config_path: Path, task_spec: DictConfig, stage: str, extra_dotlist: list[str]) -> DictConfig:
    cfg = OmegaConf.load(base_config_path)
    stage_key = _stage_spec(stage).task_key
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
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run_dir, config_path, manifest


_RUNTIME_KEY_TO_ENV = {
    "warp_ccd_iterations": "ROBOSUITE_WARP_CCD_ITERATIONS",
    "warp_solver_iters": "ROBOSUITE_WARP_SOLVER_ITERS",
    "warp_ls_iters": "ROBOSUITE_WARP_LS_ITERS",
    "warp_cone": "ROBOSUITE_WARP_CONE",
    "warp_tolerance_clamp": "ROBOSUITE_WARP_TOLERANCE_CLAMP",
    "warp_graph": "ROBOSUITE_WARP_GRAPH",
    "warp_graph_warmup": "ROBOSUITE_WARP_GRAPH_WARMUP",
}


def _subprocess_env(task_spec: DictConfig, *, for_video: bool = False, run_dir: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    runtime_cfg = task_spec.get("runtime", {})
    backend_key = "video_mujoco_gl" if for_video else "mujoco_gl"
    mujoco_gl = runtime_cfg.get(backend_key) or runtime_cfg.get("mujoco_gl")
    if mujoco_gl:
        env["MUJOCO_GL"] = str(mujoco_gl)
    for runtime_key, env_name in _RUNTIME_KEY_TO_ENV.items():
        value = runtime_cfg.get(runtime_key)
        if value is not None:
            env[env_name] = str(value)
    if run_dir is not None:
        wandb_dir = run_dir / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        env["WANDB_DIR"] = wandb_dir.as_posix()
    # Plumb wandb group through env (upstream DPPO's wandb.init does not pass
    # group=; wandb SDK picks up WANDB_RUN_GROUP from the environment). Respect
    # a shell-level override if set; otherwise derive from the task spec.
    if not env.get("WANDB_RUN_GROUP"):
        wandb_cfg = task_spec.get("wandb", {}) or {}
        group = wandb_cfg.get("group") or _default_wandb_group(_task_dataset_id(task_spec))
        if group:
            env["WANDB_RUN_GROUP"] = str(group)
    return env


_WARP_NOISE_PATTERNS = (
    "Warning: opt.ccd_iterations",
    "Warning: EPA horizon",
)


def _run_dppo_with_snapshot(task_spec: DictConfig, run_dir: Path, config_path: Path) -> None:
    command = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        str(DPPO_ROOT / "script" / "run.py"),
        f"--config-dir={run_dir.as_posix()}",
        "--config-name=run_config",
    ]
    # Merge stderr into stdout so we can line-filter warp's device-printf
    # warnings (emitted via wp.printf from CUDA kernels, which cannot be
    # disabled at the warp level) without dropping legit stderr.
    proc = subprocess.Popen(
        command,
        cwd=DPPO_ROOT,
        env=_subprocess_env(task_spec, run_dir=run_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            if any(pat in line for pat in _WARP_NOISE_PATTERNS):
                continue
            sys.stdout.write(line)
            sys.stdout.flush()
    finally:
        returncode = proc.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)


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
    if stage in {"finetune", "eval-bc", "eval-rl-init"}:
        source_checkpoint = _checkpoint_path_from_args(args, log_root, dataset_id)
        cfg.base_policy_path = source_checkpoint.as_posix()

    run_dir, config_path, _ = _snapshot_run_config(
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
    sweep_stage = "sweep-bc" if eval_mode == "bc" else "sweep-rl-init"
    eval_stage = "eval-bc" if eval_mode == "bc" else "eval-rl-init"
    sweep_stage_spec = _stage_spec(sweep_stage)
    generated_config_path = materialized["configs"][sweep_stage_spec.config_filename]
    cfg = _merge_stage_config(generated_config_path, task_spec, eval_stage, [])

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
