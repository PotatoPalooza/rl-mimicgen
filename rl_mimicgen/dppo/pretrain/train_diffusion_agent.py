from __future__ import annotations

import argparse

from rl_mimicgen.dppo.config.schema import DPPORunConfig


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline diffusion pretraining entry point for the DPPO stack.")
    parser.add_argument("--config", required=True, help="Path to the DPPO JSON config.")
    parser.add_argument("--dry_run", action="store_true", help="Load config and exit without training.")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    config = DPPORunConfig.from_json(args.config)
    if args.dry_run:
        print(
            f"Loaded pretrain config for task={config.task} variant={config.variant} "
            f"dataset={config.dataset.source_hdf5}"
        )
        return
    raise NotImplementedError("Offline pretraining is implemented in a later milestone.")


if __name__ == "__main__":
    main()
