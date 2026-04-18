from __future__ import annotations

import argparse

from rl_mimicgen.dppo.config.schema import DPPORunConfig


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an offline or fine-tuned DPPO diffusion checkpoint.")
    parser.add_argument("--config", required=True, help="Path to the DPPO JSON config.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to evaluate.")
    parser.add_argument("--dry_run", action="store_true", help="Load config and exit without evaluation.")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    config = DPPORunConfig.from_json(args.config)
    if args.dry_run:
        print(
            f"Loaded eval config for task={config.task} variant={config.variant} "
            f"checkpoint={args.checkpoint or '<none>'}"
        )
        return
    raise NotImplementedError("Offline evaluation is implemented in a later milestone.")


if __name__ == "__main__":
    main()
