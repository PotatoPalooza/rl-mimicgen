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

from rl_mimicgen.diffusion_runtime import apply_runtime_profile_to_diffusion_payload, available_runtime_profiles
from rl_mimicgen.mimicgen.runtime_checks import check_torch_cuda_compatibility

WORKSPACE_DIR = Path(__file__).resolve().parents[2]
for repo_name in ("mimicgen", "robomimic", "robosuite", "robosuite-task-zoo"):
    repo_path = WORKSPACE_DIR / repo_name
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
ALGORITHMS = ("bc", "diffusion_policy")


def load_mimicgen_stack() -> dict[str, object]:
    from mimicgen import DATASET_REGISTRY, HF_REPO_ID
    from mimicgen.scripts.generate_training_configs_for_public_datasets import modify_config_for_dataset
    from mimicgen.utils.file_utils import download_file_from_hf
    from robomimic.config import config_factory

    return {
        "DATASET_REGISTRY": DATASET_REGISTRY,
        "HF_REPO_ID": HF_REPO_ID,
        "config_factory": config_factory,
        "modify_config_for_dataset": modify_config_for_dataset,
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
    task: str
    variants: Optional[list[str]]
    modalities: Optional[list[str]]
    algorithms: list[str]
    diffusion_runtime_profile: str | None
    experiment_name: str | None
    workspace_dir: Path
    run_root: Path
    data_dir: Path
    download_datasets: bool
    run_training: bool
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
    def source_dir(self) -> Path:
        return self.data_dir / "source"

    @property
    def released_core_dir(self) -> Path:
        return self.data_dir / "core"

    @property
    def core_train_config_dir(self) -> Path:
        return self.run_root / "core_train_configs"

    @property
    def core_train_output_dir(self) -> Path:
        return self.run_root / "core_training_results"

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
        self.logger.info("Algorithms: %s", ",".join(self.cfg.algorithms))
        self.logger.info("Diffusion runtime profile: %s", self.cfg.diffusion_runtime_profile or "none")
        self.logger.info("Experiment name override: %s", self.cfg.experiment_name or "none")
        self.logger.info("Workspace dir: %s", self.cfg.workspace_dir)
        self.logger.info("Run root: %s", self.cfg.run_root)
        self.logger.info("Data dir: %s", self.cfg.data_dir)
        self.logger.info(
            "DOWNLOAD_DATASETS=%s RUN_TRAINING=%s DRY_RUN=%s",
            int(self.cfg.download_datasets),
            int(self.cfg.run_training),
            int(self.cfg.dry_run),
        )

    def run(self) -> None:
        self.log_config()
        if self.cfg.download_datasets:
            self.stage("download_released_core_datasets", self.download_released_core_datasets)
        self.stage("generate_core_training_commands", self.generate_core_training_commands)
        if self.cfg.run_training:
            self.stage("verify_core_training_datasets", lambda: self.verify_training_inputs(self.cfg.core_train_commands))
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

    def generate_core_training_commands(self) -> None:
        config_paths = self.generate_released_dataset_training_configs()
        config_paths = self.filter_training_configs(config_paths)
        commands = [
            shlex.join([sys.executable, "-m", "rl_mimicgen.mimicgen.train_robomimic", "--config", config_path])
            for config_path in config_paths
        ]
        self.write_command_file(self.cfg.core_train_commands, commands, "core training")

    def generate_released_dataset_training_configs(self) -> list[str]:
        config_paths: list[str] = []
        for dataset_name in RELEASED_CORE_TASKS[self.cfg.task]:
            for obs_modality in MODALITIES:
                for algo_name in self.cfg.algorithms:
                    config_path = self.generate_training_config(
                        dataset_name=dataset_name,
                        obs_modality=obs_modality,
                        algo_name=algo_name,
                    )
                    config_paths.append(config_path)
        self.rewrite_training_output_dirs(config_paths)
        return config_paths

    def generate_training_config(self, dataset_name: str, obs_modality: str, algo_name: str) -> str:
        config_factory = self.stack["config_factory"]
        modify_config_for_dataset = self.stack["modify_config_for_dataset"]
        config = config_factory(algo_name=algo_name)

        if algo_name == "bc":
            config = self.configure_bc(config=config, obs_modality=obs_modality)
            file_name = "bc_rnn.json"
        elif algo_name == "diffusion_policy":
            config = self.configure_diffusion_policy(config=config, obs_modality=obs_modality)
            file_name = "diffusion_policy.json"
        else:
            raise ValueError(f"Unsupported algorithm: {algo_name}")

        config = modify_config_for_dataset(
            config=config,
            dataset_type="core",
            task_name=dataset_name,
            obs_modality=obs_modality,
            base_dataset_dir=str(self.cfg.data_dir),
        )

        experiment_name = self.cfg.experiment_name
        if not experiment_name:
            experiment_name = (
                f"core_{dataset_name}_{obs_modality}"
                if algo_name == "bc"
                else f"core_{dataset_name}_{obs_modality}_{algo_name}"
            )
        with config.experiment.values_unlocked():
            config.experiment.name = experiment_name
        with config.train.values_unlocked():
            config.train.output_dir = os.path.join(
                str(self.cfg.core_train_output_dir),
                "core",
                dataset_name,
                obs_modality,
                algo_name,
                "trained_models",
            )

        dir_to_save = self.cfg.core_train_config_dir / "core" / dataset_name / obs_modality / algo_name
        dir_to_save.mkdir(parents=True, exist_ok=True)
        profile_suffix = ""
        if algo_name == "diffusion_policy" and self.cfg.diffusion_runtime_profile:
            profile_suffix = f"_{self.cfg.diffusion_runtime_profile}"
        json_path = dir_to_save / file_name.replace(".json", f"{profile_suffix}.json")
        config.dump(filename=str(json_path))
        if algo_name == "diffusion_policy" and self.cfg.diffusion_runtime_profile:
            self.apply_diffusion_runtime_profile(json_path)
        return str(json_path)

    def apply_diffusion_runtime_profile(self, json_path: Path) -> None:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        payload = apply_runtime_profile_to_diffusion_payload(payload, self.cfg.diffusion_runtime_profile)
        json_path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
        self.logger.info("Applied diffusion runtime profile %s to %s", self.cfg.diffusion_runtime_profile, json_path)

    def configure_bc(self, config, obs_modality: str):
        with config.train.values_unlocked():
            config.train.seq_length = 10

        with config.algo.values_unlocked():
            config.algo.rnn.enabled = True
            config.algo.rnn.horizon = 10
            config.algo.actor_layer_dims = ()
            config.algo.gmm.enabled = True
            config.algo.rnn.hidden_dim = 400

        if obs_modality == "low_dim":
            self.apply_low_dim_defaults(config)
            with config.algo.values_unlocked():
                config.algo.optim_params.policy.learning_rate.initial = 1e-3
        else:
            self.apply_image_defaults(config)
            with config.algo.values_unlocked():
                config.algo.optim_params.policy.learning_rate.initial = 1e-4
                config.algo.rnn.hidden_dim = 1000
        return config

    def configure_diffusion_policy(self, config, obs_modality: str):
        if obs_modality == "low_dim":
            self.apply_low_dim_defaults(config)
        else:
            self.apply_image_defaults(config)

        with config.experiment.values_unlocked():
            config.experiment.render_video = False

        with config.train.values_unlocked():
            config.train.hdf5_load_next_obs = False
            config.train.seq_length = int(config.algo.horizon.prediction_horizon)
            config.train.action_config["actions"]["normalization"] = "min_max"

        return config

    def apply_low_dim_defaults(self, config) -> None:
        with config.experiment.values_unlocked():
            config.experiment.save.enabled = True
            config.experiment.save.every_n_epochs = 200
            config.experiment.epoch_every_n_steps = 100
            config.experiment.validation_epoch_every_n_steps = 10
            config.experiment.rollout.enabled = True
            config.experiment.rollout.n = 50
            config.experiment.rollout.horizon = 400
            config.experiment.rollout.rate = 200
            config.experiment.rollout.warmstart = 0
            config.experiment.rollout.terminate_on_success = True

        with config.train.values_unlocked():
            config.train.num_data_workers = 0
            config.train.hdf5_cache_mode = "all"
            config.train.batch_size = 100
            config.train.num_epochs = 2000

        with config.observation.values_unlocked():
            config.observation.modalities.obs.low_dim = [
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "object",
            ]
            config.observation.modalities.obs.rgb = []

    def apply_image_defaults(self, config) -> None:
        with config.experiment.values_unlocked():
            config.experiment.save.enabled = True
            config.experiment.save.every_n_epochs = 20
            config.experiment.epoch_every_n_steps = 500
            config.experiment.validation_epoch_every_n_steps = 50
            config.experiment.rollout.enabled = True
            config.experiment.rollout.n = 50
            config.experiment.rollout.horizon = 400
            config.experiment.rollout.rate = 20
            config.experiment.rollout.warmstart = 0
            config.experiment.rollout.terminate_on_success = True

        with config.train.values_unlocked():
            config.train.num_data_workers = 2
            config.train.hdf5_cache_mode = "low_dim"
            config.train.batch_size = 16
            config.train.num_epochs = 600

        with config.observation.values_unlocked():
            config.observation.modalities.obs.low_dim = [
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
            ]
            config.observation.modalities.obs.rgb = [
                "agentview_image",
                "robot0_eye_in_hand_image",
            ]
            config.observation.modalities.goal.low_dim = []
            config.observation.modalities.goal.rgb = []
            config.observation.encoder.rgb.core_class = "VisualCore"
            config.observation.encoder.rgb.core_kwargs.feature_dimension = 64
            config.observation.encoder.rgb.core_kwargs.backbone_class = "ResNet18Conv"
            config.observation.encoder.rgb.core_kwargs.backbone_kwargs.pretrained = False
            config.observation.encoder.rgb.core_kwargs.backbone_kwargs.input_coord_conv = False
            config.observation.encoder.rgb.core_kwargs.pool_class = "SpatialSoftmax"
            config.observation.encoder.rgb.core_kwargs.pool_kwargs.num_kp = 32
            config.observation.encoder.rgb.core_kwargs.pool_kwargs.learnable_temperature = False
            config.observation.encoder.rgb.core_kwargs.pool_kwargs.temperature = 1.0
            config.observation.encoder.rgb.core_kwargs.pool_kwargs.noise_std = 0.0
            config.observation.encoder.rgb.obs_randomizer_class = "CropRandomizer"
            config.observation.encoder.rgb.obs_randomizer_kwargs.crop_height = 76
            config.observation.encoder.rgb.obs_randomizer_kwargs.crop_width = 76
            config.observation.encoder.rgb.obs_randomizer_kwargs.num_crops = 1
            config.observation.encoder.rgb.obs_randomizer_kwargs.pos_enc = False

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
            train_cfg["output_dir"] = str(self.cfg.core_train_output_dir)
            path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
            self.logger.info("Flattened training output dir for %s -> %s", experiment_name, self.cfg.core_train_output_dir)

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
                "Training requested but required datasets are missing. "
                f"Expected dataset files such as: {preview}."
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
    default_run_root = workspace_dir / "runs" / os.environ.get("RUN_NAME", f"paper_bc_one_task_{now_stamp()}")
    data_dir_env = os.environ.get("DATA_DIR")
    parser = argparse.ArgumentParser(description="Train MimicGen policies for one task from released core datasets.")
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
    parser.add_argument(
        "--algo",
        dest="algorithms",
        action="append",
        choices=ALGORITHMS,
        help="Training algorithm to generate configs for. Repeat to run multiple algorithms. Default: bc.",
    )
    parser.add_argument(
        "--diffusion-runtime-profile",
        choices=available_runtime_profiles(),
        default=os.environ.get("DIFFUSION_RUNTIME_PROFILE"),
        help="Optional shared diffusion runtime profile to apply to diffusion_policy BC configs.",
    )
    parser.add_argument(
        "--experiment-name",
        default=os.environ.get("EXPERIMENT_NAME"),
        help="Optional fixed robomimic experiment folder name. Requires exactly one variant, one modality, and one algorithm.",
    )
    add_bool_arg(
        parser,
        "download-datasets",
        env_bool("DOWNLOAD_DATASETS", True),
        "Download released MimicGen core datasets for this task if they are missing.",
    )
    parser.add_argument("--run-root", type=Path, default=Path(os.environ.get("RUN_ROOT", str(default_run_root))))
    parser.add_argument("--data-dir", type=Path, default=Path(data_dir_env).resolve() if data_dir_env else None)
    add_bool_arg(parser, "run-training", env_bool("RUN_TRAINING", True), "Execute the training commands.")
    add_bool_arg(parser, "dry-run", env_bool("DRY_RUN", False), "Log stages without executing them.")
    args = parser.parse_args(argv)
    run_root = args.run_root.resolve()
    data_dir = args.data_dir if args.data_dir is not None else run_root.parent / "datasets"
    variants = sorted({value.upper() for value in args.variants}) if args.variants else None
    modalities = sorted(set(args.modalities)) if args.modalities else None
    algorithms = sorted(set(args.algorithms)) if args.algorithms else ["bc"]
    if args.experiment_name is not None:
        if variants is None or len(variants) != 1 or modalities is None or len(modalities) != 1 or len(algorithms) != 1:
            raise SystemExit(
                "--experiment-name requires exactly one variant, one modality, and one algorithm "
                "so the BC output folder name is unambiguous."
            )
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
        algorithms=algorithms,
        diffusion_runtime_profile=args.diffusion_runtime_profile,
        experiment_name=args.experiment_name,
        workspace_dir=workspace_dir,
        run_root=run_root,
        data_dir=data_dir.resolve(),
        download_datasets=args.download_datasets,
        run_training=args.run_training,
        dry_run=args.dry_run,
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
