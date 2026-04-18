from __future__ import annotations

import argparse

from rl_mimicgen.dppo.config.schema import DPPORunConfig


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Online DPPO fine-tuning entry point.")
    parser.add_argument("--config", required=True, help="Path to the DPPO JSON config.")
    parser.add_argument("--checkpoint", default=None, help="Offline checkpoint to fine-tune.")
    parser.add_argument("--dry_run", action="store_true", help="Load config and exit without fine-tuning.")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    config = DPPORunConfig.from_json(args.config)
    if args.dry_run:
        print(
            f"Loaded finetune config for task={config.task} variant={config.variant} "
            f"checkpoint={args.checkpoint or '<none>'}"
        )
        return
    raise NotImplementedError("Online DPPO fine-tuning is implemented in a later milestone.")


if __name__ == "__main__":
    main()
