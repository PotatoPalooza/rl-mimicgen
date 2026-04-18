from __future__ import annotations

import argparse

from rl_mimicgen.dppo.config.schema import DPPORunConfig
from rl_mimicgen.dppo.data import DPPODatasetBundle
from rl_mimicgen.dppo.envs import make_mimicgen_lowdim_env


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Online DPPO fine-tuning entry point.")
    parser.add_argument("--config", required=True, help="Path to the DPPO JSON config.")
    parser.add_argument("--checkpoint", default=None, help="Offline checkpoint to fine-tune.")
    parser.add_argument("--smoke_env_reset", action="store_true", help="Create the low-dim env and run a reset during dry-run validation.")
    parser.add_argument("--dry_run", action="store_true", help="Load config and exit without fine-tuning.")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    config = DPPORunConfig.from_json(args.config)
    dataset = DPPODatasetBundle.load(config.dataset.bundle_dir)
    if args.dry_run:
        print(
            f"Loaded finetune config for task={config.task} variant={config.variant} "
            f"checkpoint={args.checkpoint or '<none>'}"
        )
        print(dataset.summary())
        if args.smoke_env_reset:
            env = make_mimicgen_lowdim_env(task=config.task, variant=config.variant)
            obs = env.reset()
            print(f"env_reset obs_keys={sorted(obs.keys())} action_dim={env.action_dim} horizon={env.horizon}")
            env.close()
        return
    raise NotImplementedError("Online DPPO fine-tuning is implemented in a later milestone.")


if __name__ == "__main__":
    main()
