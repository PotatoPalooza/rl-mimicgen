from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Script to train a robomimic policy.")
parser.add_argument(
    "--config",
    type=str,
    default=None,
    help="(optional) path to a config json that will be used to override the default settings. If omitted, default "
    "settings are used. This is the preferred way to run experiments.",
)
parser.add_argument(
    "--task",
    type=str,
    default=None,
    help="(optional) Isaac Lab task name. Necessary if sim is set to isaac.",
)
parser.add_argument(
    "--algo",
    type=str,
    help="(optional) name of algorithm to run. Only needs to be provided if --config is not provided",
)
parser.add_argument(
    "--name",
    type=str,
    default=None,
    help="(optional) if provided, override the experiment name defined in the config",
)
parser.add_argument(
    "--dataset",
    type=str,
    default=None,
    help="(optional) if provided, override the dataset path defined in the config",
)
parser.add_argument(
    "--debug", action="store_true", help="set this flag to run a quick training run for debugging purposes"
)
parser.add_argument(
    "--resume", action="store_true", default=False, help="set this flag to resume training from latest checkpoint"
)
parser.add_argument(
    "--sim",
    type=str,
    choices=["mujoco", "isaac"],
    default="mujoco",
    help="(optional) specify the simulator to use for training. Defaults to mujoco.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.sim == "isaac":
    args_cli.headless = True
    args_cli.enable_cameras = True
    # suppress logs
    if not hasattr(args_cli, "kit_args"):
        args_cli.kit_args = ""
    args_cli.kit_args += " --/log/level=error"
    args_cli.kit_args += " --/log/fileLogLevel=error"
    args_cli.kit_args += " --/log/outputStreamLevel=error"

    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import robomimic.utils.torch_utils as TorchUtils  # noqa: E402
import torch  # noqa: E402
from robomimic.config import Config, config_factory  # noqa: E402
from robomimic.scripts.train import train  # noqa: E402

if args_cli.sim == "mujoco":
    WORKSPACE_DIR = Path(__file__).resolve().parents[2]
    for repo_name in ("mimicgen", "robomimic", "robosuite", "robosuite-task-zoo"):
        repo_path = WORKSPACE_DIR / repo_name
        if repo_path.exists():
            sys.path.insert(0, str(repo_path))

    import mimicgen  # noqa: F401
elif args_cli.sim == "isaac":
    import rl_mimicgen.tasks  # noqa: F401 E402

    ISAAC_TASKS = {"lift": "Franka-Lift-v0", "coffee": "Franka-Coffee-v0"}


def prepare_config() -> tuple[Config, torch.device]:
    if args_cli.config is not None:
        ext_cfg = json.load(open(args_cli.config, "r"))
        config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        config = config_factory(args_cli.algo)

    if args_cli.dataset is not None:
        config.train.data = [{"path": args_cli.dataset}]

    if args_cli.name is not None:
        config.experiment.name = args_cli.name

    # overrides for Isaac environments
    if args_cli.sim == "isaac":
        if args_cli.task is None:
            raise ValueError("--task must be set for Isaac environments.")
        config.experiment.env_meta_update_dict = {"env_name": args_cli.task, "type": 4}
        config.experiment.use_torch = True

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # maybe modify config for debugging purposes
    if args_cli.debug:
        # shrink length of training to test whether this run is likely to crash
        config.unlock()
        config.lock_keys()

        # train and validate (if enabled) for 3 gradient steps, for 2 epochs
        config.experiment.epoch_every_n_steps = 3
        config.experiment.validation_epoch_every_n_steps = 3
        config.train.num_epochs = 2

        # if rollouts are enabled, try 2 rollouts at end of each epoch, with 10 environment steps
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10

        # send output to a temporary directory
        config.train.output_dir = "/tmp/tmp_trained_models"

    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()

    return config, device


def main() -> int:
    config, device = prepare_config()
    train(config, device=device, resume=args_cli.resume, auto_remove_exp_dir=True)
    return 0


if __name__ == "__main__":
    exit_code = main()

    raise SystemExit(exit_code)
