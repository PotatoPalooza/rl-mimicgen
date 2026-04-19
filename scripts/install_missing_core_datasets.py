#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_mimicgen.mimicgen.paper_bc_one_task import RELEASED_CORE_TASKS, load_mimicgen_stack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install only the missing released MimicGen core datasets."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "datasets",
        help="Dataset root containing the core/ directory. Default: runs/datasets",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the missing datasets without downloading them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    core_dir = data_dir / "core"
    core_dir.mkdir(parents=True, exist_ok=True)

    expected = sorted({name for names in RELEASED_CORE_TASKS.values() for name in names})
    missing = [name for name in expected if not (core_dir / f"{name}.hdf5").exists()]

    print(f"Core dataset directory: {core_dir}")
    print(f"Expected released datasets: {len(expected)}")
    print(f"Missing datasets: {len(missing)}")
    for name in missing:
        print(f"  - {name}")

    if not missing or args.dry_run:
        return 0

    stack = load_mimicgen_stack()
    dataset_registry = stack["DATASET_REGISTRY"]
    download_file_from_hf = stack["download_file_from_hf"]

    for name in missing:
        print(f"Downloading {name}.hdf5")
        download_file_from_hf(
            repo_id=stack["HF_REPO_ID"],
            filename=dataset_registry["core"][name]["url"],
            download_dir=str(core_dir),
            check_overwrite=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
