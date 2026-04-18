from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def _sorted_demo_keys(data_group: h5py.Group) -> list[str]:
    return sorted(data_group.keys(), key=lambda key: int(key.split("_")[1]))


def _default_obs_keys(obs_group: h5py.Group) -> list[str]:
    return sorted(key for key in obs_group.keys() if "image" not in key.lower())


def _build_obs_matrix(obs_group: h5py.Group, obs_keys: list[str]) -> tuple[np.ndarray, dict[str, list[int]], dict[str, list[int]]]:
    chunks: list[np.ndarray] = []
    obs_shapes: dict[str, list[int]] = {}
    obs_slices: dict[str, list[int]] = {}
    cursor = 0
    for key in obs_keys:
        value = np.asarray(obs_group[key][:], dtype=np.float32)
        obs_shapes[key] = list(value.shape[1:])
        flat_value = value.reshape(value.shape[0], -1)
        chunks.append(flat_value)
        next_cursor = cursor + flat_value.shape[1]
        obs_slices[key] = [cursor, next_cursor]
        cursor = next_cursor
    if not chunks:
        raise ValueError("No observation keys selected for conversion.")
    return np.concatenate(chunks, axis=1), obs_shapes, obs_slices


def _safe_std(array: np.ndarray) -> np.ndarray:
    std = array.std(axis=0)
    std[std < 1e-6] = 1.0
    return std.astype(np.float32, copy=False)


def _safe_range(min_value: np.ndarray, max_value: np.ndarray) -> np.ndarray:
    value_range = max_value - min_value
    value_range[value_range < 1e-6] = 1.0
    return value_range.astype(np.float32, copy=False)


def convert_dataset(
    source_hdf5: Path,
    output_dir: Path,
    task: str,
    variant: str,
    obs_keys: list[str] | None = None,
    max_demos: int | None = None,
    write_output: bool = True,
) -> dict[str, object]:
    with h5py.File(source_hdf5, "r") as file_obj:
        data_group = file_obj["data"]
        demo_keys = _sorted_demo_keys(data_group)
        if max_demos is not None:
            demo_keys = demo_keys[:max_demos]
        if not demo_keys:
            raise ValueError(f"No demos found in {source_hdf5}")

        sample_obs_group = data_group[demo_keys[0]]["obs"]
        selected_obs_keys = list(obs_keys) if obs_keys else _default_obs_keys(sample_obs_group)

        obs_chunks: list[np.ndarray] = []
        next_obs_chunks: list[np.ndarray] = []
        action_chunks: list[np.ndarray] = []
        done_chunks: list[np.ndarray] = []
        reward_chunks: list[np.ndarray] = []
        traj_lengths: list[int] = []
        demo_offsets: list[list[int]] = []
        total_steps = 0
        obs_shapes: dict[str, list[int]] | None = None
        obs_slices: dict[str, list[int]] | None = None

        for demo_key in demo_keys:
            demo_group = data_group[demo_key]
            obs_matrix, current_shapes, current_slices = _build_obs_matrix(demo_group["obs"], selected_obs_keys)
            actions = np.asarray(demo_group["actions"][:], dtype=np.float32)
            if actions.shape[0] != obs_matrix.shape[0]:
                raise ValueError(f"Action / observation length mismatch in {demo_key}")

            next_obs = np.concatenate([obs_matrix[1:], obs_matrix[-1:]], axis=0)
            rewards = np.asarray(demo_group["rewards"][:], dtype=np.float32) if "rewards" in demo_group else np.zeros(actions.shape[0], dtype=np.float32)
            dones = np.asarray(demo_group["dones"][:], dtype=np.float32) if "dones" in demo_group else np.zeros(actions.shape[0], dtype=np.float32)
            dones[-1] = 1.0

            obs_chunks.append(obs_matrix)
            next_obs_chunks.append(next_obs)
            action_chunks.append(actions)
            reward_chunks.append(rewards)
            done_chunks.append(dones)
            traj_lengths.append(actions.shape[0])
            demo_offsets.append([total_steps, total_steps + actions.shape[0]])
            total_steps += actions.shape[0]
            obs_shapes = current_shapes
            obs_slices = current_slices

        obs = np.concatenate(obs_chunks, axis=0)
        next_obs = np.concatenate(next_obs_chunks, axis=0)
        actions = np.concatenate(action_chunks, axis=0)
        rewards = np.concatenate(reward_chunks, axis=0)
        dones = np.concatenate(done_chunks, axis=0)

        env_metadata = json.loads(data_group.attrs["env_args"])

    metadata = {
        "task": task,
        "variant": variant,
        "source_hdf5": str(source_hdf5),
        "env_metadata": env_metadata,
        "obs_keys": selected_obs_keys,
        "obs_shapes": obs_shapes,
        "obs_slices": obs_slices,
        "num_demos": len(demo_keys),
        "num_transitions": int(obs.shape[0]),
        "obs_dim": int(obs.shape[1]),
        "action_dim": int(actions.shape[1]),
        "demo_keys": demo_keys,
    }
    if write_output:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_dir / "dataset.npz",
            obs=obs,
            next_obs=next_obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            traj_lengths=np.asarray(traj_lengths, dtype=np.int32),
            demo_offsets=np.asarray(demo_offsets, dtype=np.int64),
        )
        np.savez_compressed(
            output_dir / "normalization.npz",
            obs_mean=obs.mean(axis=0).astype(np.float32),
            obs_std=_safe_std(obs),
            obs_min=obs.min(axis=0).astype(np.float32),
            obs_max=obs.max(axis=0).astype(np.float32),
            obs_range=_safe_range(obs.min(axis=0), obs.max(axis=0)),
            action_mean=actions.mean(axis=0).astype(np.float32),
            action_std=_safe_std(actions),
            action_min=actions.min(axis=0).astype(np.float32),
            action_max=actions.max(axis=0).astype(np.float32),
            action_range=_safe_range(actions.min(axis=0), actions.max(axis=0)),
        )
        with open(output_dir / "metadata.json", "w", encoding="utf-8") as file_obj:
            json.dump(metadata, file_obj, indent=2)
    return metadata


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a MimicGen HDF5 dataset into a DPPO-ready bundle.")
    parser.add_argument("--source_hdf5", required=True, help="Path to the MimicGen / robomimic HDF5 dataset.")
    parser.add_argument("--output_dir", required=True, help="Directory where the converted dataset bundle will be written.")
    parser.add_argument("--task", required=True, help="Task name, for example coffee.")
    parser.add_argument("--variant", required=True, help="Task variant, for example D0.")
    parser.add_argument("--obs_key", action="append", dest="obs_keys", default=None, help="Observation key to include. Repeat to pin a custom low-dim key order.")
    parser.add_argument("--max_demos", type=int, default=None, help="Optional limit on the number of demos to convert for smoke tests.")
    parser.add_argument("--dry_run", action="store_true", help="Validate inputs and print the planned conversion without writing files.")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    source = Path(args.source_hdf5)
    output_dir = Path(args.output_dir)

    if not source.exists():
        raise FileNotFoundError(f"Dataset not found: {source}")

    metadata = convert_dataset(
        source_hdf5=source,
        output_dir=output_dir,
        task=args.task,
        variant=args.variant,
        obs_keys=args.obs_keys,
        max_demos=args.max_demos,
        write_output=not args.dry_run,
    )

    if args.dry_run:
        print(
            f"Validated dataset {source} for task={args.task} variant={args.variant}; "
            f"{metadata['num_demos']} demos, {metadata['num_transitions']} transitions, "
            f"obs_dim={metadata['obs_dim']}, action_dim={metadata['action_dim']}"
        )
        return

    print(
        f"Wrote DPPO dataset bundle to {output_dir} with "
        f"{metadata['num_demos']} demos and {metadata['num_transitions']} transitions."
    )


if __name__ == "__main__":
    main()
