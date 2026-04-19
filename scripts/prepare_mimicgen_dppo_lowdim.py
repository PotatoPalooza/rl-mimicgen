from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl_mimicgen.adapters.dppo_lowdim import (
    inspect_mimicgen_lowdim_dataset,
    materialize_mimicgen_lowdim_port,
    verify_mimicgen_lowdim_env_reset,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize official-DPPO low-dim assets and configs from a MimicGen HDF5 dataset."
    )
    parser.add_argument("--dataset", required=True, help="Path to a MimicGen HDF5 dataset, for example runs/datasets/core/coffee_d0.hdf5.")
    parser.add_argument("--dataset-id", default=None, help="Optional dataset identifier override. Defaults to the HDF5 stem.")
    parser.add_argument("--output-root", default="runs/official_dppo_mimicgen", help="Root directory for generated DPPO-ready artifacts.")
    parser.add_argument("--log-root", default=None, help="Optional root directory for generated official-DPPO logdirs.")
    parser.add_argument("--obs-key", action="append", dest="obs_keys", default=None, help="Repeat to pin a low-dim observation key order.")
    parser.add_argument("--max-demos", type=int, default=None, help="Optional demo cap for preparing a smaller dataset bundle.")
    parser.add_argument("--val-split", type=float, default=0.0, help="Trajectory-level validation split fraction.")
    parser.add_argument("--split-seed", type=int, default=0, help="Random seed for the train/val trajectory split.")
    parser.add_argument("--verify-env-reset", action="store_true", help="Instantiate the robomimic env from generated metadata and run a reset.")
    parser.add_argument("--wandb-entity", default=None, help="Optional wandb entity (team or user). If omitted, the entity from your wandb login default is used.")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    materialized = materialize_mimicgen_lowdim_port(
        args.dataset,
        output_root=args.output_root,
        dataset_id=args.dataset_id,
        obs_keys=args.obs_keys,
        max_demos=args.max_demos,
        val_split=args.val_split,
        split_seed=args.split_seed,
        log_root=args.log_root,
        wandb_entity=args.wandb_entity,
    )
    spec = materialized["spec"]
    payload = {
        "dataset_id": spec.dataset_id,
        "source_hdf5": spec.source_hdf5.as_posix(),
        "artifact_dir": materialized["artifact_dir"].as_posix(),
        "config_dir": materialized["config_dir"].as_posix(),
        "obs_dim": spec.obs_dim,
        "action_dim": spec.action_dim,
        "horizon": spec.horizon,
        "low_dim_keys": list(spec.low_dim_keys),
    }
    if args.verify_env_reset:
        payload["env_reset"] = verify_mimicgen_lowdim_env_reset(
            inspect_mimicgen_lowdim_dataset(
                args.dataset,
                dataset_id=args.dataset_id,
                obs_keys=args.obs_keys,
                max_demos=args.max_demos,
            ),
            env_meta_path=Path(materialized["artifact_dir"]) / "env_meta.json",
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
