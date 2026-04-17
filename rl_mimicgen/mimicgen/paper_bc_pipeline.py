from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import logging
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

from rl_mimicgen.diffusion_runtime import apply_runtime_profile_to_diffusion_payload, available_runtime_profiles
from rl_mimicgen.mimicgen.runtime_checks import check_torch_cuda_compatibility

WORKSPACE_DIR = Path(__file__).resolve().parents[2]
for repo_name in ("mimicgen", "robomimic", "robosuite", "robosuite-task-zoo"):
    repo_path = WORKSPACE_DIR / repo_name
    if repo_path.exists():
        sys.path.insert(0, str(repo_path))


SOURCE_DATASET_PREP_SPECS = [
    ("coffee", "MG_Coffee", "robosuite"),
    ("coffee_preparation", "MG_CoffeePreparation", "robosuite"),
    ("hammer_cleanup", "MG_HammerCleanup", "robosuite"),
    ("kitchen", "MG_Kitchen", "robosuite"),
    ("mug_cleanup", "MG_MugCleanup", "robosuite"),
    ("nut_assembly", "MG_NutAssembly", "robosuite"),
    ("pick_place", "MG_PickPlace", "robosuite"),
    ("square", "MG_Square", "robosuite"),
    ("stack", "MG_Stack", "robosuite"),
    ("stack_three", "MG_StackThree", "robosuite"),
    ("threading", "MG_Threading", "robosuite"),
    ("three_piece_assembly", "MG_ThreePieceAssembly", "robosuite"),
]

TASK_ZOO_DATASETS = {"hammer_cleanup", "kitchen"}
ALL_SOURCE_TASKS = sorted(dataset_name for dataset_name, _, _ in SOURCE_DATASET_PREP_SPECS)


def load_mimicgen_stack() -> dict[str, object]:
    from mimicgen import DATASET_REGISTRY, HF_REPO_ID
    from mimicgen.scripts import generate_core_configs
    from mimicgen.scripts import generate_core_training_configs
    from mimicgen.scripts import generate_training_configs_for_public_datasets
    from mimicgen.scripts import generate_robot_transfer_configs
    from mimicgen.scripts import generate_robot_transfer_training_configs
    from mimicgen.scripts.prepare_src_dataset import prepare_src_dataset
    from mimicgen.utils.file_utils import config_generator_to_script_lines
    from mimicgen.utils.file_utils import download_file_from_hf

    return {
        "DATASET_REGISTRY": DATASET_REGISTRY,
        "HF_REPO_ID": HF_REPO_ID,
        "generate_core_configs": generate_core_configs,
        "generate_core_training_configs": generate_core_training_configs,
        "generate_training_configs_for_public_datasets": generate_training_configs_for_public_datasets,
        "generate_robot_transfer_configs": generate_robot_transfer_configs,
        "generate_robot_transfer_training_configs": generate_robot_transfer_training_configs,
        "prepare_src_dataset": prepare_src_dataset,
        "config_generator_to_script_lines": config_generator_to_script_lines,
        "download_file_from_hf": download_file_from_hf,
    }


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no"}


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    dest = name.replace("-", "_")
    parser.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    parser.add_argument(f"--no-{name}", dest=dest, action="store_false", help=argparse.SUPPRESS)
    parser.set_defaults(**{dest: default})


@dataclass
class Config:
    workspace_dir: Path
    run_root: Path
    data_dir: Path
    num_traj: int
    guarantee: bool
    download_source: bool
    prepare_source: bool
    run_generation: bool
    run_training: bool
    training_data: str
    include_robot_transfer: bool
    tasks: Optional[list[str]]
    diffusion_runtime_profile: str | None
    dry_run: bool

    @property
    def command_dir(self) -> Path:
        return self.run_root / "commands"

    @property
    def log_dir(self) -> Path:
        return self.run_root / "logs"

    @property
    def log_file(self) -> Path:
        return self.log_dir / "pipeline.log"

    @property
    def core_config_dir(self) -> Path:
        return self.run_root / "core_configs"

    @property
    def core_dataset_dir(self) -> Path:
        return self.run_root / "core_datasets"

    @property
    def core_train_config_dir(self) -> Path:
        return self.run_root / "core_train_configs"

    @property
    def core_train_output_dir(self) -> Path:
        return self.run_root / "core_training_results"

    @property
    def robot_config_dir(self) -> Path:
        return self.run_root / "robot_configs"

    @property
    def robot_dataset_dir(self) -> Path:
        return self.run_root / "robot_datasets"

    @property
    def robot_train_config_dir(self) -> Path:
        return self.run_root / "robot_train_configs"

    @property
    def robot_train_output_dir(self) -> Path:
        return self.run_root / "robot_training_results"

    @property
    def core_dataset_commands(self) -> Path:
        return self.command_dir / "core_dataset_commands.txt"

    @property
    def core_train_commands(self) -> Path:
        return self.command_dir / "core_train_commands.txt"

    @property
    def robot_dataset_commands(self) -> Path:
        return self.command_dir / "robot_dataset_commands.txt"

    @property
    def robot_train_commands(self) -> Path:
        return self.command_dir / "robot_train_commands.txt"


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
    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.cfg.command_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._build_logger()
        self.stack = load_mimicgen_stack()

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger("rl-mimicgen-paper-bc")
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

    def log_config(self) -> None:
        self.logger.info("Workspace dir: %s", self.cfg.workspace_dir)
        self.logger.info("Run root: %s", self.cfg.run_root)
        self.logger.info("Data dir: %s", self.cfg.data_dir)
        self.logger.info("Tasks: %s", ",".join(self.cfg.tasks) if self.cfg.tasks else "all")
        self.logger.info(
            "NUM_TRAJ=%s GUARANTEE=%s RUN_GENERATION=%s RUN_TRAINING=%s TRAINING_DATA=%s ROBOT_TRANSFER=%s DIFFUSION_RUNTIME_PROFILE=%s DRY_RUN=%s",
            self.cfg.num_traj,
            int(self.cfg.guarantee),
            int(self.cfg.run_generation),
            int(self.cfg.run_training),
            self.cfg.training_data,
            int(self.cfg.include_robot_transfer),
            self.cfg.diffusion_runtime_profile or "none",
            int(self.cfg.dry_run),
        )

    def ensure_optional_dependencies(self) -> None:
        dataset_registry = self.stack["DATASET_REGISTRY"]
        relevant_tasks = set(self.cfg.tasks) if self.cfg.tasks else set(dataset_registry["source"].keys())
        needs_task_zoo = bool(relevant_tasks & TASK_ZOO_DATASETS)
        if needs_task_zoo and importlib.util.find_spec("robosuite_task_zoo") is None:
            required_tasks = ", ".join(sorted(TASK_ZOO_DATASETS))
            raise RuntimeError(
                "This pipeline includes task-zoo tasks "
                f"({required_tasks}) but robosuite_task_zoo is not installed in the active environment."
            )

    def run(self) -> None:
        self.log_config()
        self.ensure_optional_dependencies()
        if self.cfg.download_source:
            self.stage("download_source_datasets", self.download_missing_source_datasets)
        if self.cfg.prepare_source and self.cfg.run_generation:
            self.stage("prepare_source_datasets", self.prepare_source_datasets)
        elif self.cfg.prepare_source and not self.cfg.run_generation:
            self.logger.info("Skipping prepare_source_datasets because generation is disabled")
        self.stage("generate_core_dataset_commands", self.generate_core_dataset_commands)
        if self.cfg.include_robot_transfer:
            self.stage("generate_robot_dataset_commands", self.generate_robot_dataset_commands)
        if self.cfg.run_generation:
            self.stage("run_core_dataset_generation", lambda: self.run_command_file("core_dataset", self.cfg.core_dataset_commands))
            if self.cfg.include_robot_transfer:
                self.stage("run_robot_dataset_generation", lambda: self.run_command_file("robot_dataset", self.cfg.robot_dataset_commands))
        self.stage("generate_core_training_commands", self.generate_core_training_commands)
        if self.cfg.include_robot_transfer:
            self.stage("generate_robot_training_commands", self.generate_robot_training_commands)
        if self.cfg.run_training:
            self.stage("verify_core_training_datasets", lambda: self.verify_training_inputs(self.cfg.core_train_commands))
            self.stage("verify_training_environment", self.verify_training_environment)
            self.stage("run_core_training", lambda: self.run_command_file("core_train", self.cfg.core_train_commands))
            if self.cfg.include_robot_transfer:
                self.stage(
                    "verify_robot_training_datasets",
                    lambda: self.verify_training_inputs(self.cfg.robot_train_commands),
                )
                self.stage("run_robot_training", lambda: self.run_command_file("robot_train", self.cfg.robot_train_commands))

    def stage(self, name: str, fn: Callable[[], None]) -> None:
        self.logger.info("START %s", name)
        if self.cfg.dry_run:
            self.logger.info("DRY_RUN %s", name)
        else:
            fn()
        self.logger.info("DONE %s", name)

    @contextlib.contextmanager
    def capture_python_output(self) -> Iterable[None]:
        stdout_writer = LoggerWriter(self.logger, logging.INFO)
        stderr_writer = LoggerWriter(self.logger, logging.INFO)
        with contextlib.redirect_stdout(stdout_writer), contextlib.redirect_stderr(stderr_writer):
            yield
        stdout_writer.flush()
        stderr_writer.flush()

    @contextlib.contextmanager
    def temporary_module_attrs(self, module: object, **attrs: object) -> Iterable[None]:
        old_values = {name: getattr(module, name) for name in attrs}
        try:
            for name, value in attrs.items():
                setattr(module, name, value)
            yield
        finally:
            for name, value in old_values.items():
                setattr(module, name, value)

    def write_command_file(self, path: Path, lines: list[str], label: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
        self.logger.info("Wrote %s %s commands to %s", len(lines), label, path)

    def workspace_script(self, *parts: str) -> Path:
        return self.cfg.workspace_dir.joinpath(*parts)

    def dataset_generation_command_lines(self, lines: list[str]) -> list[str]:
        script_path = self.workspace_script("mimicgen", "mimicgen", "scripts", "generate_dataset.py")
        rewritten: list[str] = []
        for line in lines:
            cmd = shlex.split(line.strip())
            if len(cmd) < 2 or cmd[0] != "python":
                raise ValueError(f"Unexpected command format for dataset generation: {line.strip()}")
            cmd[0] = sys.executable
            cmd[1] = str(script_path)
            cmd.append("--auto-remove-exp")
            rewritten.append(shlex.join(cmd))
        return rewritten

    def training_command_lines(self, lines: list[str]) -> list[str]:
        rewritten: list[str] = []
        for line in lines:
            cmd = shlex.split(line.strip())
            if len(cmd) < 2 or cmd[0] != "python":
                raise ValueError(f"Unexpected command format for training: {line.strip()}")
            cmd[0] = sys.executable
            cmd[1:2] = ["-m", "rl_mimicgen.mimicgen.train_robomimic"]
            rewritten.append(shlex.join(cmd))
        return rewritten

    def source_tasks(self) -> list[str]:
        dataset_registry = self.stack["DATASET_REGISTRY"]
        all_tasks = sorted(dataset_registry["source"].keys())
        if self.cfg.tasks is None:
            return all_tasks
        selected = set(self.cfg.tasks)
        return [task for task in all_tasks if task in selected]

    def task_for_path(self, path_str: str) -> Optional[str]:
        dataset_registry = self.stack["DATASET_REGISTRY"]
        path = Path(os.path.expanduser(path_str))
        for part in path.parts:
            if part in dataset_registry["source"]:
                return part
        if path.stem in dataset_registry["source"]:
            return path.stem
        return None

    def command_task(self, line: str) -> Optional[str]:
        cmd = shlex.split(line.strip())
        if "--config" not in cmd:
            return None
        config_path = Path(cmd[cmd.index("--config") + 1])
        if not config_path.exists():
            return None
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        train_data = payload.get("train", {}).get("data")
        if isinstance(train_data, str):
            return self.task_for_path(train_data)
        source_data = payload.get("experiment", {}).get("source", {}).get("dataset_path")
        if isinstance(source_data, str):
            return self.task_for_path(source_data)
        return None

    def filter_command_lines(self, lines: list[str], label: str) -> list[str]:
        if self.cfg.tasks is None:
            return lines
        selected = set(self.cfg.tasks)
        kept: list[str] = []
        for line in lines:
            task = self.command_task(line)
            if task is None or task in selected:
                kept.append(line)
        self.logger.info("Filtered %s commands: kept %s of %s", label, len(kept), len(lines))
        return kept

    def download_missing_source_datasets(self) -> None:
        dataset_registry = self.stack["DATASET_REGISTRY"]
        hf_repo_id = self.stack["HF_REPO_ID"]
        download_file_from_hf = self.stack["download_file_from_hf"]
        source_dir = self.cfg.data_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        missing = [task for task in self.source_tasks() if not (source_dir / f"{task}.hdf5").exists()]
        if not missing:
            self.logger.info("All source datasets already exist under %s", source_dir)
            return
        with self.capture_python_output():
            for task in missing:
                download_file_from_hf(
                    repo_id=hf_repo_id,
                    filename=dataset_registry["source"][task]["url"],
                    download_dir=str(source_dir),
                    check_overwrite=True,
                )

    def prepare_source_datasets(self) -> None:
        prepare_src_dataset = self.stack["prepare_src_dataset"]
        with self.capture_python_output():
            for dataset_name, env_interface_name, env_interface_type in SOURCE_DATASET_PREP_SPECS:
                if self.cfg.tasks is not None and dataset_name not in set(self.cfg.tasks):
                    continue
                prepare_src_dataset(
                    dataset_path=str(self.cfg.data_dir / "source" / f"{dataset_name}.hdf5"),
                    env_interface_name=env_interface_name,
                    env_interface_type=env_interface_type,
                )

    def generate_core_dataset_commands(self) -> None:
        generate_core_configs = self.stack["generate_core_configs"]
        config_generator_to_script_lines = self.stack["config_generator_to_script_lines"]
        with self.temporary_module_attrs(
            generate_core_configs,
            CONFIG_DIR=str(self.cfg.core_config_dir),
            OUTPUT_FOLDER=str(self.cfg.core_dataset_dir),
            NUM_TRAJ=self.cfg.num_traj,
            GUARANTEE=self.cfg.guarantee,
            DEBUG=False,
        ):
            generators = generate_core_configs.make_generators(base_configs=generate_core_configs.BASE_CONFIGS)
            _, lines = config_generator_to_script_lines(generators, config_dir=str(self.cfg.core_config_dir))
        lines = self.filter_command_lines(self.dataset_generation_command_lines(lines), "core dataset")
        self.write_command_file(self.cfg.core_dataset_commands, lines, "core dataset")

    def generate_robot_dataset_commands(self) -> None:
        generate_robot_transfer_configs = self.stack["generate_robot_transfer_configs"]
        config_generator_to_script_lines = self.stack["config_generator_to_script_lines"]
        with self.temporary_module_attrs(
            generate_robot_transfer_configs,
            CONFIG_DIR=str(self.cfg.robot_config_dir),
            OUTPUT_FOLDER=str(self.cfg.robot_dataset_dir),
            NUM_TRAJ=self.cfg.num_traj,
            GUARANTEE=self.cfg.guarantee,
            DEBUG=False,
        ):
            generators = generate_robot_transfer_configs.make_generators(
                base_configs=generate_robot_transfer_configs.BASE_CONFIGS
            )
            _, lines = config_generator_to_script_lines(generators, config_dir=str(self.cfg.robot_config_dir))
        lines = self.filter_command_lines(self.dataset_generation_command_lines(lines), "robot dataset")
        self.write_command_file(self.cfg.robot_dataset_commands, lines, "robot dataset")

    def generate_core_training_commands(self) -> None:
        if self.cfg.training_data == "source":
            self.generate_source_training_commands()
            return
        generate_core_training_configs = self.stack["generate_core_training_configs"]
        config_generator_to_script_lines = self.stack["config_generator_to_script_lines"]
        generators = generate_core_training_configs.make_generators(
            base_config=generate_core_training_configs.BASE_CONFIG,
            dataset_dir=str(self.cfg.core_dataset_dir),
            output_dir=str(self.cfg.core_train_output_dir),
        )
        _, lines = config_generator_to_script_lines(generators, config_dir=str(self.cfg.core_train_config_dir))
        self.apply_diffusion_runtime_profile_to_command_configs(lines)
        lines = self.filter_command_lines(self.training_command_lines(lines), "core training")
        self.write_command_file(self.cfg.core_train_commands, lines, "core training")

    def generate_source_training_commands(self) -> None:
        generate_public_training_configs = self.stack["generate_training_configs_for_public_datasets"]
        config_paths: list[str] = []
        tasks = self.source_tasks()
        for task in tasks:
            for obs_modality in ("low_dim", "image"):
                _, config_path = generate_public_training_configs.generate_experiment_config(
                    base_exp_name="source",
                    base_config_dir=str(self.cfg.core_train_config_dir),
                    base_dataset_dir=str(self.cfg.data_dir),
                    base_output_dir=str(self.cfg.core_train_output_dir),
                    dataset_type="source",
                    task_name=task,
                    obs_modality=obs_modality,
                )
                config_paths.append(config_path)
        lines = [
            shlex.join([sys.executable, "-m", "rl_mimicgen.mimicgen.train_robomimic", "--config", config_path])
            for config_path in config_paths
        ]
        self.apply_diffusion_runtime_profile_to_command_configs(lines)
        self.write_command_file(self.cfg.core_train_commands, lines, "core training")

    def generate_robot_training_commands(self) -> None:
        generate_robot_transfer_training_configs = self.stack["generate_robot_transfer_training_configs"]
        config_generator_to_script_lines = self.stack["config_generator_to_script_lines"]
        generators = generate_robot_transfer_training_configs.make_generators(
            base_config=generate_robot_transfer_training_configs.BASE_CONFIG,
            dataset_dir=str(self.cfg.robot_dataset_dir),
            output_dir=str(self.cfg.robot_train_output_dir),
        )
        _, lines = config_generator_to_script_lines(generators, config_dir=str(self.cfg.robot_train_config_dir))
        self.apply_diffusion_runtime_profile_to_command_configs(lines)
        lines = self.filter_command_lines(self.training_command_lines(lines), "robot training")
        self.write_command_file(self.cfg.robot_train_commands, lines, "robot training")

    def apply_diffusion_runtime_profile_to_command_configs(self, lines: list[str]) -> None:
        if not self.cfg.diffusion_runtime_profile:
            return
        for line in lines:
            cmd = shlex.split(line.strip())
            if "--config" not in cmd:
                continue
            config_path = Path(cmd[cmd.index("--config") + 1])
            if not config_path.exists():
                continue
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            updated = apply_runtime_profile_to_diffusion_payload(payload, self.cfg.diffusion_runtime_profile)
            config_path.write_text(json.dumps(updated, indent=4) + "\n", encoding="utf-8")
            self.logger.info(
                "Applied diffusion runtime profile %s to %s",
                self.cfg.diffusion_runtime_profile,
                config_path,
            )

    def run_command_file(self, label: str, path: Path) -> None:
        if not path.exists():
            self.logger.info("Skipping %s because %s does not exist", label, path)
            return
        commands = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not commands:
            self.logger.info("Skipping %s because %s is empty", label, path)
            return
        total = len(commands)
        for idx, command in enumerate(commands, start=1):
            self.logger.info("RUN %s [%s/%s] %s", label, idx, total, command)
            if self.cfg.dry_run:
                continue
            subprocess.run(
                shlex.split(command),
                cwd=self.cfg.workspace_dir,
                check=True,
                env=os.environ.copy(),
            )

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
            expected_source = "download the source demonstrations" if self.cfg.training_data == "source" else "run the pipeline with generation enabled first"
            raise FileNotFoundError(
                "Training requested but required dataset files are missing. "
                f"Expected dataset files such as: {preview}. "
                f"{expected_source}, or point training at an existing dataset directory."
            )

    def verify_training_environment(self) -> None:
        check_torch_cuda_compatibility()


def parse_args(argv: Optional[list[str]] = None) -> Config:
    workspace_dir = WORKSPACE_DIR
    default_run_root = workspace_dir / "runs" / os.environ.get("RUN_NAME", f"paper_bc_{now_stamp()}")
    parser = argparse.ArgumentParser(description="Run a simplified MimicGen paper BC pipeline.")
    parser.add_argument("--run-root", type=Path, default=Path(os.environ.get("RUN_ROOT", str(default_run_root))))
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("DATA_DIR", str(workspace_dir / "datasets"))))
    parser.add_argument("--num-traj", type=int, default=int(os.environ.get("NUM_TRAJ", "1000")))
    parser.add_argument(
        "--task",
        dest="tasks",
        action="append",
        choices=ALL_SOURCE_TASKS,
        help="Limit the pipeline to one or more tasks. Pass multiple times to select several tasks.",
    )
    add_bool_arg(parser, "guarantee", env_bool("GUARANTEE", True), "Generate until the target number of successful trajectories is reached.")
    add_bool_arg(parser, "download-source", env_bool("DOWNLOAD_SOURCE", True), "Download missing source datasets before later stages.")
    add_bool_arg(parser, "prepare-source", env_bool("PREPARE_SOURCE", True), "Prepare source datasets with datagen metadata.")
    add_bool_arg(parser, "run-generation", env_bool("RUN_GENERATION", True), "Execute dataset generation commands.")
    add_bool_arg(parser, "run-training", env_bool("RUN_TRAINING", True), "Execute training commands.")
    parser.add_argument(
        "--training-data",
        choices=("generated", "source"),
        default=os.environ.get("TRAINING_DATA", "generated"),
        help="Which datasets training configs should target: generated MimicGen datasets or downloaded source demonstrations.",
    )
    parser.add_argument(
        "--diffusion-runtime-profile",
        choices=available_runtime_profiles(),
        default=os.environ.get("DIFFUSION_RUNTIME_PROFILE"),
        help="Optional shared diffusion runtime profile to apply to generated diffusion_policy training configs.",
    )
    add_bool_arg(parser, "include-robot-transfer", env_bool("INCLUDE_ROBOT_TRANSFER", False), "Include robot-transfer generation and training stages.")
    add_bool_arg(parser, "dry-run", env_bool("DRY_RUN", False), "Log stages without executing them.")
    args = parser.parse_args(argv)
    tasks = sorted(set(args.tasks)) if args.tasks else None
    return Config(
        workspace_dir=workspace_dir,
        run_root=args.run_root.resolve(),
        data_dir=args.data_dir.resolve(),
        num_traj=args.num_traj,
        guarantee=args.guarantee,
        download_source=args.download_source,
        prepare_source=args.prepare_source,
        run_generation=args.run_generation,
        run_training=args.run_training,
        training_data=args.training_data,
        include_robot_transfer=args.include_robot_transfer,
        tasks=tasks,
        diffusion_runtime_profile=args.diffusion_runtime_profile,
        dry_run=args.dry_run,
    )


def main(argv: Optional[list[str]] = None) -> int:
    Runner(parse_args(argv)).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
