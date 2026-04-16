from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

from rl_mimicgen.mimicgen.runtime_checks import check_torch_cuda_compatibility

WORKSPACE_DIR = Path(__file__).resolve().parents[2]
for repo_name in ("mimicgen", "robomimic", "robosuite", "robosuite-task-zoo"):
    repo_path = WORKSPACE_DIR / "resources" / repo_name
    if repo_path.exists():
        sys.path.insert(0, str(repo_path))

RELEASED_CORE_TASKS = {
    "stack": ["stack_d0", "stack_d1"],
    "stack_three": ["stack_three_d0", "stack_three_d1"],
    "square": ["square_d0", "square_d1", "square_d2"],
    "threading": ["threading_d0", "threading_d1", "threading_d2"],
    "three_piece_assembly": ["three_piece_assembly_d0", "three_piece_assembly_d1", "three_piece_assembly_d2"],
    "coffee": ["coffee_d0", "coffee_d1", "coffee_d2"],
    "coffee_preparation": ["coffee_preparation_d0", "coffee_preparation_d1"],
    "nut_assembly": ["nut_assembly_d0"],
    "pick_place": ["pick_place_d0"],
    "hammer_cleanup": ["hammer_cleanup_d0", "hammer_cleanup_d1"],
    "mug_cleanup": ["mug_cleanup_d0", "mug_cleanup_d1"],
    "kitchen": ["kitchen_d0", "kitchen_d1"],
}
TASK_VARIANTS = {
    "stack": ["D0", "D1"],
    "stack_three": ["D0", "D1"],
    "square": ["D0", "D1", "D2"],
    "threading": ["D0", "D1", "D2"],
    "three_piece_assembly": ["D0", "D1", "D2"],
    "coffee": ["D0", "D1", "D2"],
    "coffee_preparation": ["D0", "D1"],
    "nut_assembly": ["D0"],
    "pick_place": ["D0"],
    "hammer_cleanup": ["D0", "D1"],
    "mug_cleanup": ["D0", "D1", "O1", "O2"],
    "kitchen": ["D0", "D1"],
}
MODALITIES = ("low_dim", "image")


def load_mimicgen_stack() -> dict[str, object]:
    from mimicgen import DATASET_REGISTRY, HF_REPO_ID
    from mimicgen.scripts.generate_training_configs_for_public_datasets import generate_experiment_config
    from mimicgen.utils.file_utils import download_file_from_hf

    return {
        "DATASET_REGISTRY": DATASET_REGISTRY,
        "HF_REPO_ID": HF_REPO_ID,
        "generate_experiment_config": generate_experiment_config,
        "download_file_from_hf": download_file_from_hf,
    }


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no"}


SCRIPT_NAME = "paper_bc_one_task"


def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    dest = name.replace("-", "_")
    parser.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    parser.add_argument(f"--no-{name}", dest=dest, action="store_false", help=argparse.SUPPRESS)
    parser.set_defaults(**{dest: default})


@dataclass
class Config:
    task: str
    variants: Optional[list[str]]
    modalities: Optional[list[str]]
    workspace_dir: Path
    run_root: Path          # shared output root, e.g. <workspace>/runs
    stamp: str              # timestamp slug that scopes this invocation's metadata
    data_dir: Path
    download_datasets: bool
    run_training: bool
    dry_run: bool
    warp: bool
    logger: str
    wandb_project: Optional[str]
    wandb_run: Optional[str]
    wandb_model: Optional[str]
    resume: bool

    # Pipeline bookkeeping lives under runs/logs/<script>/<stamp>/; training
    # output goes directly under run_root (robomimic's get_exp_dir adds the
    # <experiment.name>/<timestamp> hierarchy beneath).
    @property
    def meta_dir(self) -> Path:
        return self.run_root / "logs" / SCRIPT_NAME / self.stamp

    @property
    def command_dir(self) -> Path:
        return self.meta_dir / "commands"

    @property
    def log_dir(self) -> Path:
        return self.meta_dir / "logs"

    @property
    def log_file(self) -> Path:
        return self.log_dir / "pipeline.log"

    @property
    def source_dir(self) -> Path:
        return self.data_dir / "source"

    @property
    def released_core_dir(self) -> Path:
        return self.data_dir / "core"

    @property
    def core_train_config_dir(self) -> Path:
        return self.meta_dir / "configs"

    @property
    def core_train_output_dir(self) -> Path:
        return self.run_root

    @property
    def core_train_commands(self) -> Path:
        return self.command_dir / "core_train_commands.txt"


class LoggerWriter:
    def __init__(self, logger: logging.Logger, level: int) -> None:
        self.logger = logger
        self.level = level
        self.buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line:
                self.logger.log(self.level, "%s", line)
        return len(text)

    def flush(self) -> None:
        if self.buffer:
            self.logger.log(self.level, "%s", self.buffer)
            self.buffer = ""


class Runner:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.cfg.command_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._build_logger()
        self._stack: Optional[dict[str, object]] = None

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger("rl-mimicgen-paper-bc-one-task")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.propagate = False
        fmt = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(fmt)
        logger.addHandler(stream)
        file_handler = logging.FileHandler(self.cfg.log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        return logger

    @property
    def stack(self) -> dict[str, object]:
        if self._stack is None:
            self._stack = load_mimicgen_stack()
        return self._stack

    @contextlib.contextmanager
    def capture_python_output(self) -> Iterable[None]:
        stdout_writer = LoggerWriter(self.logger, logging.INFO)
        stderr_writer = LoggerWriter(self.logger, logging.INFO)
        with contextlib.redirect_stdout(stdout_writer), contextlib.redirect_stderr(stderr_writer):
            yield
        stdout_writer.flush()
        stderr_writer.flush()

    def stage(self, name: str, fn: Callable[[], None]) -> None:
        self.logger.info("START %s", name)
        if self.cfg.dry_run:
            self.logger.info("DRY_RUN %s", name)
        else:
            fn()
        self.logger.info("DONE %s", name)

    def log_config(self) -> None:
        self.logger.info("Task: %s", self.cfg.task)
        self.logger.info("Variants: %s", ",".join(self.cfg.variants) if self.cfg.variants else "all")
        self.logger.info("Modalities: %s", ",".join(self.cfg.modalities) if self.cfg.modalities else "all")
        self.logger.info("Workspace dir: %s", self.cfg.workspace_dir)
        self.logger.info("Run root: %s", self.cfg.run_root)
        self.logger.info("Meta dir: %s", self.cfg.meta_dir)
        self.logger.info("Data dir: %s", self.cfg.data_dir)
        self.logger.info(
            "DOWNLOAD_DATASETS=%s RUN_TRAINING=%s DRY_RUN=%s WARP=%s LOGGER=%s",
            int(self.cfg.download_datasets),
            int(self.cfg.run_training),
            int(self.cfg.dry_run),
            int(self.cfg.warp),
            self.cfg.logger,
        )

    def run(self) -> None:
        self.log_config()
        if self.cfg.download_datasets:
            self.stage("download_released_core_datasets", self.download_released_core_datasets)
        self.stage("generate_core_training_commands", self.generate_core_training_commands)
        if self.cfg.run_training:
            self.stage(
                "verify_core_training_datasets", lambda: self.verify_training_inputs(self.cfg.core_train_commands)
            )
            self.stage("verify_training_environment", self.verify_training_environment)
            self.stage("run_core_training", lambda: self.run_command_file("core_train", self.cfg.core_train_commands))

    def download_released_core_datasets(self) -> None:
        dataset_registry = self.stack["DATASET_REGISTRY"]
        download_file_from_hf = self.stack["download_file_from_hf"]
        self.cfg.released_core_dir.mkdir(parents=True, exist_ok=True)
        with self.capture_python_output():
            for dataset_name in RELEASED_CORE_TASKS[self.cfg.task]:
                dataset_path = self.cfg.released_core_dir / f"{dataset_name}.hdf5"
                if dataset_path.exists():
                    self.logger.info("Released core dataset already exists at %s", dataset_path)
                    continue
                download_file_from_hf(
                    repo_id=self.stack["HF_REPO_ID"],
                    filename=dataset_registry["core"][dataset_name]["url"],
                    download_dir=str(self.cfg.released_core_dir),
                    check_overwrite=True,
                )

    def inject_warp_into_configs(self, config_paths: list[str]) -> None:
        for config_path in config_paths:
            path = Path(config_path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload.setdefault("experiment", {}).setdefault("rollout", {})["use_warp"] = True
            path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
            self.logger.info("Enabled use_warp in %s", config_path)

    def inject_wandb_into_configs(self, config_paths: list[str]) -> None:
        """Enable wandb logging in each generated training config.

        Defaults wandb_proj_name to ``<task>_<modality>`` (e.g. ``coffee_low_dim``)
        and wandb_group to the dataset variant (``d0``/``d1``/...), both derived
        from the generated config payload. ``--wandb-project`` overrides the
        project name.
        """
        for config_path in config_paths:
            path = Path(config_path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            modality = self.modality_from_payload(payload) or "low_dim"
            variant = self.variant_from_payload(payload)
            project = self.cfg.wandb_project or f"{self.cfg.task}_{modality}"
            group = variant.lower() if variant else None

            logging_cfg = payload.setdefault("experiment", {}).setdefault("logging", {})
            logging_cfg["log_wandb"] = True
            logging_cfg["wandb_proj_name"] = project
            logging_cfg["wandb_group"] = group

            path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
            self.logger.info(
                "Enabled wandb in %s (project=%s group=%s)", config_path, project, group,
            )

    def generate_core_training_commands(self) -> None:
        config_paths = self.generate_released_dataset_training_configs()
        config_paths = self.filter_training_configs(config_paths)
        if self.cfg.warp:
            self.inject_warp_into_configs(config_paths)
        if self.cfg.logger == "wandb":
            self.inject_wandb_into_configs(config_paths)
        if self.cfg.wandb_run is not None and len(config_paths) > 1:
            raise SystemExit(
                "--wandb_run identifies a single source run, but --variant/--modality selected "
                f"{len(config_paths)} configs. Narrow the selection down to one "
                "(e.g. --variant D0 --modality low_dim) before resuming from wandb."
            )
        if self.cfg.resume and self.cfg.wandb_run is not None:
            raise SystemExit("--resume and --wandb_run are mutually exclusive.")
        extra_args: list[str] = []
        if self.cfg.wandb_run is not None:
            if self.cfg.wandb_model is None:
                raise SystemExit("--wandb_run requires --wandb_model (e.g. model_50.pth)")
            extra_args = ["--wandb_run", self.cfg.wandb_run, "--wandb_model", self.cfg.wandb_model]
        if self.cfg.resume:
            extra_args.append("--resume")
        commands = [
            shlex.join(
                [sys.executable, "-m", "rl_mimicgen.mimicgen.train_robomimic", "--config", config_path]
                + extra_args
            )
            for config_path in config_paths
        ]
        self.write_command_file(self.cfg.core_train_commands, commands, "core training")

    def generate_released_dataset_training_configs(self) -> list[str]:
        generate_experiment_config = self.stack["generate_experiment_config"]
        config_paths: list[str] = []
        for dataset_name in RELEASED_CORE_TASKS[self.cfg.task]:
            for obs_modality in ("low_dim", "image"):
                _, config_path = generate_experiment_config(
                    base_exp_name="core",
                    base_config_dir=str(self.cfg.core_train_config_dir),
                    base_dataset_dir=str(self.cfg.data_dir),
                    base_output_dir=str(self.cfg.core_train_output_dir),
                    dataset_type="core",
                    task_name=dataset_name,
                    obs_modality=obs_modality,
                )
                config_paths.append(config_path)
        self.rewrite_training_output_dirs(config_paths)
        return config_paths

    def rewrite_training_output_dirs(self, config_paths: list[str]) -> None:
        for config_path in config_paths:
            path = Path(config_path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            experiment_name = payload.get("experiment", {}).get("name")
            if not isinstance(experiment_name, str) or not experiment_name:
                raise ValueError(f"Generated config missing experiment.name: {config_path}")
            train_cfg = payload.get("train")
            if not isinstance(train_cfg, dict):
                raise ValueError(f"Generated config missing train settings: {config_path}")
            # strip the generator's dataset-type prefix so experiment.name becomes
            # <task>_<variant>_<modality> (e.g. coffee_d0_low_dim); robomimic's
            # get_exp_dir will then produce runs/<experiment.name>/<timestamp>/...
            new_name = experiment_name
            for prefix in ("core_", "source_"):
                if new_name.startswith(prefix):
                    new_name = new_name[len(prefix):]
                    break
            payload["experiment"]["name"] = new_name
            train_cfg["output_dir"] = str(self.cfg.core_train_output_dir)
            path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
            self.logger.info(
                "Set training output dir for %s -> %s/%s/<stamp>/",
                experiment_name, self.cfg.core_train_output_dir, new_name,
            )

    def filter_training_configs(self, config_paths: list[str]) -> list[str]:
        if self.cfg.variants is None and self.cfg.modalities is None:
            return config_paths
        kept: list[str] = []
        for config_path in config_paths:
            payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
            if self.cfg.variants is not None:
                variant = self.variant_from_payload(payload)
                if variant not in set(self.cfg.variants):
                    continue
            if self.cfg.modalities is not None:
                modality = self.modality_from_payload(payload)
                if modality not in set(self.cfg.modalities):
                    continue
            kept.append(config_path)
        if not kept:
            raise ValueError(
                f"No training configs matched task={self.cfg.task}, "
                f"variants={self.cfg.variants or 'all'}, modalities={self.cfg.modalities or 'all'}."
            )
        self.logger.info("Filtered core training configs: kept %s of %s", len(kept), len(config_paths))
        return kept

    def variant_from_payload(self, payload: dict[str, object]) -> Optional[str]:
        dataset_path = payload.get("train", {}).get("data")
        if not isinstance(dataset_path, str):
            return None
        match = re.search(r"_(d\d|o\d)(?:[^a-z0-9]|$)", dataset_path.lower())
        if match is None:
            return None
        return match.group(1).upper()

    def modality_from_payload(self, payload: dict[str, object]) -> Optional[str]:
        experiment_name = payload.get("experiment", {}).get("name")
        if isinstance(experiment_name, str):
            if experiment_name.endswith("_low_dim"):
                return "low_dim"
            if experiment_name.endswith("_image"):
                return "image"
        obs = payload.get("observation", {}).get("modalities", {}).get("obs", {})
        if isinstance(obs, dict):
            rgb = obs.get("rgb")
            if isinstance(rgb, list) and rgb:
                return "image"
        return "low_dim"

    def verify_training_environment(self) -> None:
        check_torch_cuda_compatibility()

    def verify_training_inputs(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Training command file not found: {path}")
        commands = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not commands:
            self.logger.info("Skipping training input verification because %s is empty", path)
            return
        missing: list[Path] = []
        for command in commands:
            cmd = shlex.split(command)
            if "--config" not in cmd:
                continue
            config_path = Path(cmd[cmd.index("--config") + 1])
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            dataset_path = payload.get("train", {}).get("data")
            if isinstance(dataset_path, str):
                dataset_file = Path(dataset_path)
                if not dataset_file.exists():
                    missing.append(dataset_file)
        if missing:
            preview = ", ".join(str(path) for path in missing[:3])
            if len(missing) > 3:
                preview += f", ... ({len(missing)} missing total)"
            raise FileNotFoundError(
                f"Training requested but required datasets are missing. Expected dataset files such as: {preview}."
            )

    def write_command_file(self, path: Path, lines: list[str], label: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
        self.logger.info("Wrote %s %s commands to %s", len(lines), label, path)

    def run_command_file(self, label: str, path: Path) -> None:
        if not path.exists():
            self.logger.info("Skipping %s because %s does not exist", label, path)
            return
        commands = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not commands:
            self.logger.info("Skipping %s because %s is empty", label, path)
            return
        total = len(commands)
        for index, command in enumerate(commands, start=1):
            self.logger.info("RUN %s [%s/%s] %s", label, index, total, command)
            subprocess.run(
                shlex.split(command),
                cwd=self.cfg.workspace_dir,
                check=True,
                env=os.environ.copy(),
            )


def parse_args(argv: Optional[list[str]] = None) -> Config:
    workspace_dir = WORKSPACE_DIR
    task_choices = sorted(RELEASED_CORE_TASKS)
    default_run_root = workspace_dir / "runs"
    default_stamp = os.environ.get("RUN_STAMP", now_stamp())
    data_dir_env = os.environ.get("DATA_DIR")
    parser = argparse.ArgumentParser(description="Train BC for one task from released MimicGen core datasets.")
    parser.add_argument("--task", required=True, choices=task_choices, help="Paper task to train.")
    parser.add_argument(
        "--variant",
        dest="variants",
        action="append",
        help="Limit training to one or more dataset variants for this task, such as D0, D1, D2, O1, or O2.",
    )
    parser.add_argument(
        "--modality",
        dest="modalities",
        action="append",
        choices=MODALITIES,
        help="Limit training to one or more modalities.",
    )
    add_bool_arg(
        parser,
        "download-datasets",
        env_bool("DOWNLOAD_DATASETS", True),
        "Download released MimicGen core datasets for this task if they are missing.",
    )
    parser.add_argument("--run-root", type=Path, default=Path(os.environ.get("RUN_ROOT", str(default_run_root))))
    parser.add_argument("--stamp", default=default_stamp,
                        help="Timestamp slug for this invocation's metadata dir (default: now).")
    parser.add_argument("--data-dir", type=Path, default=Path(data_dir_env).resolve() if data_dir_env else None)
    add_bool_arg(parser, "run-training", env_bool("RUN_TRAINING", True), "Execute the training commands.")
    add_bool_arg(parser, "dry-run", env_bool("DRY_RUN", False), "Log stages without executing them.")
    add_bool_arg(
        parser,
        "warp",
        env_bool("WARP", False),
        "Enable MuJoCo Warp GPU-parallel rollouts in generated training configs.",
    )
    parser.add_argument(
        "--logger",
        choices=("tensorboard", "wandb"),
        default=os.environ.get("LOGGER", "tensorboard"),
        help="Experiment logger. 'wandb' enables wandb in generated training configs.",
    )
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT"),
        help="Override wandb project name (default: <task>_<modality>, e.g. coffee_low_dim).",
    )
    parser.add_argument(
        "--wandb_run",
        default=os.environ.get("WANDB_RUN"),
        help="Resume training from a checkpoint logged by this wandb run "
             "(format: entity/project/run_id). Requires --wandb_model and a single-config selection "
             "(narrow via --variant/--modality).",
    )
    parser.add_argument(
        "--wandb_model",
        default=os.environ.get("WANDB_MODEL"),
        help="Checkpoint filename within --wandb_run to fetch (e.g. model_50.pth).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=env_bool("RESUME", False),
        help="Resume core training from the latest checkpoint under the selected "
             "experiment dir (delegates to robomimic train.py's --resume).",
    )
    args = parser.parse_args(argv)
    run_root = args.run_root.resolve()
    data_dir = args.data_dir if args.data_dir is not None else run_root / "datasets"
    variants = sorted({value.upper() for value in args.variants}) if args.variants else None
    modalities = sorted(set(args.modalities)) if args.modalities else None
    if variants is not None:
        invalid_variants = sorted(set(variants) - set(TASK_VARIANTS[args.task]))
        if invalid_variants:
            raise SystemExit(
                f"Unsupported variants for task {args.task}: {', '.join(invalid_variants)}. "
                f"Choose from: {', '.join(TASK_VARIANTS[args.task])}."
            )
    return Config(
        task=args.task,
        variants=variants,
        modalities=modalities,
        workspace_dir=workspace_dir,
        run_root=run_root,
        stamp=args.stamp,
        data_dir=data_dir.resolve(),
        download_datasets=args.download_datasets,
        run_training=args.run_training,
        dry_run=args.dry_run,
        warp=args.warp,
        logger=args.logger,
        wandb_project=args.wandb_project,
        wandb_run=args.wandb_run,
        wandb_model=args.wandb_model,
        resume=args.resume,
    )


def main(argv: Optional[list[str]] = None) -> int:
    runner = Runner(parse_args(argv))
    try:
        runner.run()
    except Exception:
        runner.logger.exception("Pipeline failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
