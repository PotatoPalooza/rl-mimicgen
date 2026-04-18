from __future__ import annotations

import argparse
from pathlib import Path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a MimicGen HDF5 dataset into a DPPO-ready bundle.")
    parser.add_argument("--source_hdf5", required=True, help="Path to the MimicGen / robomimic HDF5 dataset.")
    parser.add_argument("--output_dir", required=True, help="Directory where the converted dataset bundle will be written.")
    parser.add_argument("--task", required=True, help="Task name, for example coffee.")
    parser.add_argument("--variant", required=True, help="Task variant, for example D0.")
    parser.add_argument("--dry_run", action="store_true", help="Validate inputs and print the planned conversion without writing files.")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    source = Path(args.source_hdf5)
    output_dir = Path(args.output_dir)

    if not source.exists():
        raise FileNotFoundError(f"Dataset not found: {source}")

    if args.dry_run:
        print(f"Validated dataset {source} for task={args.task} variant={args.variant}; output_dir={output_dir}")
        return

    raise NotImplementedError("Dataset conversion is implemented in the next milestone.")


if __name__ == "__main__":
    main()
