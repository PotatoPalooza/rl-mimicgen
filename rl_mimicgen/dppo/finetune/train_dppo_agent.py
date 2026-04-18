from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rl_mimicgen.dppo.config.schema import DPPORunConfig
from rl_mimicgen.dppo.data import DPPODatasetBundle
from rl_mimicgen.dppo.envs import make_mimicgen_lowdim_env
from rl_mimicgen.dppo.policy import DiffusionPolicyAdapter


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Online DPPO fine-tuning entry point.")
    parser.add_argument("--config", required=True, help="Path to the DPPO JSON config.")
    parser.add_argument("--checkpoint", default=None, help="Offline checkpoint to fine-tune.")
    parser.add_argument("--rollout_steps", type=int, default=16, help="Number of environment steps to collect for the bootstrap rollout.")
    parser.add_argument("--output_dir", default=None, help="Directory to write rollout artifacts.")
    parser.add_argument("--smoke_env_reset", action="store_true", help="Create the low-dim env and run a reset during dry-run validation.")
    parser.add_argument("--dry_run", action="store_true", help="Load config and exit without fine-tuning.")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    config = DPPORunConfig.from_json(args.config)
    dataset = DPPODatasetBundle.load(config.dataset.bundle_dir)
    checkpoint_path = args.checkpoint or config.checkpoint_path
    if args.dry_run:
        print(
            f"Loaded finetune config for task={config.task} variant={config.variant} "
            f"checkpoint={checkpoint_path or '<none>'}"
        )
        print(dataset.summary())
        if args.smoke_env_reset:
            env = make_mimicgen_lowdim_env(task=config.task, variant=config.variant)
            obs = env.reset()
            print(f"env_reset obs_keys={sorted(obs.keys())} action_dim={env.action_dim} horizon={env.horizon}")
            env.close()
        return
    if checkpoint_path is None:
        raise ValueError("Fine-tuning bootstrap requires --checkpoint or config.checkpoint_path.")

    env = make_mimicgen_lowdim_env(task=config.task, variant=config.variant)
    policy = DiffusionPolicyAdapter(config=config, bundle=dataset, checkpoint_path=checkpoint_path, deterministic=False)
    obs = env.reset()
    policy.reset(obs)

    records: dict[str, list[np.ndarray | float | int]] = {
        "obs": [],
        "actions": [],
        "normalized_actions": [],
        "rewards": [],
        "dones": [],
    }
    collected_steps = 0
    while collected_steps < args.rollout_steps:
        rollout = policy.sample(obs)
        action_chunk = rollout.actions[: config.diffusion.act_steps]
        normalized_chunk = rollout.normalized_actions[: config.diffusion.act_steps]
        for action, normalized_action in zip(action_chunk, normalized_chunk, strict=True):
            records["obs"].append(dataset.flatten_obs(obs).astype(np.float32, copy=False))
            records["actions"].append(action.astype(np.float32, copy=False))
            records["normalized_actions"].append(normalized_action.astype(np.float32, copy=False))
            obs, reward, done, _info = env.step(action.astype(np.float32, copy=False))
            records["rewards"].append(float(reward))
            records["dones"].append(int(done))
            collected_steps += 1
            if done:
                obs = env.reset()
                policy.reset(obs)
            if collected_steps >= args.rollout_steps:
                break
    env.close()

    rollout_metrics = {
        "rollout_steps": collected_steps,
        "reward_sum": float(np.sum(records["rewards"])) if records["rewards"] else 0.0,
        "done_count": int(np.sum(records["dones"])) if records["dones"] else 0,
        "checkpoint": checkpoint_path,
    }
    print(json.dumps(rollout_metrics, indent=2))

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_dir / "bootstrap_rollout.npz",
            obs=np.asarray(records["obs"], dtype=np.float32),
            actions=np.asarray(records["actions"], dtype=np.float32),
            normalized_actions=np.asarray(records["normalized_actions"], dtype=np.float32),
            rewards=np.asarray(records["rewards"], dtype=np.float32),
            dones=np.asarray(records["dones"], dtype=np.int32),
        )
        with open(output_dir / "rollout_metrics.json", "w", encoding="utf-8") as file_obj:
            json.dump(rollout_metrics, file_obj, indent=2)


if __name__ == "__main__":
    main()
