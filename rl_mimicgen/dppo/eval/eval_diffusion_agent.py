from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rl_mimicgen.dppo.config.schema import DPPORunConfig
from rl_mimicgen.dppo.data import DPPODatasetBundle
from rl_mimicgen.dppo.envs import make_mimicgen_lowdim_env
from rl_mimicgen.dppo.policy import DiffusionPolicyAdapter


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an offline or fine-tuned DPPO diffusion checkpoint.")
    parser.add_argument("--config", required=True, help="Path to the DPPO JSON config.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to evaluate.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to evaluate.")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum environment steps per episode.")
    parser.add_argument("--output_dir", default=None, help="Directory to write eval metrics.")
    parser.add_argument("--smoke_env_reset", action="store_true", help="Create the low-dim env and run a reset during dry-run validation.")
    parser.add_argument("--dry_run", action="store_true", help="Load config and exit without evaluation.")
    return parser


def run_evaluation(
    config: DPPORunConfig,
    dataset: DPPODatasetBundle,
    checkpoint_path: str,
    actor_override: torch.nn.Module | None = None,
    episodes: int = 1,
    max_steps: int | None = None,
) -> dict[str, float]:
    env = make_mimicgen_lowdim_env(task=config.task, variant=config.variant)
    policy = DiffusionPolicyAdapter(config=config, bundle=dataset, checkpoint_path=checkpoint_path, deterministic=True)
    if actor_override is not None:
        policy.ema_actor.load_state_dict(actor_override.state_dict(), strict=False)
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    successes: list[float] = []
    effective_max_steps = max_steps or env.horizon
    for episode_index in range(episodes):
        obs = env.reset()
        total_reward = 0.0
        success = False
        done = False
        step_count = 0
        episode_starts = np.ones(1, dtype=bool)
        while not done and step_count < effective_max_steps:
            obs_batch = {key: value[None, ...] for key, value in obs.items()}
            action = policy.act_deterministic(obs=obs_batch, goal=None, episode_starts=episode_starts)[0]
            obs, reward, done, info = env.step(action.astype(np.float32, copy=False))
            total_reward += reward
            success = success or bool(info.get("success", False))
            step_count += 1
            episode_starts[0] = bool(done)
        episode_returns.append(float(total_reward))
        episode_lengths.append(step_count)
        successes.append(float(success))
        print(f"episode={episode_index + 1} return={total_reward:.6f} length={step_count} success={int(success)}")
    env.close()
    return {
        "episodes": float(episodes),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "return_mean": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "length_mean": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "checkpoint": checkpoint_path,
    }


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    config = DPPORunConfig.from_json(args.config)
    dataset = DPPODatasetBundle.load(config.dataset.bundle_dir)
    checkpoint_path = args.checkpoint or config.checkpoint_path
    if args.dry_run:
        print(
            f"Loaded eval config for task={config.task} variant={config.variant} "
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
        raise ValueError("Evaluation requires --checkpoint or config.checkpoint_path.")

    metrics = run_evaluation(
        config=config,
        dataset=dataset,
        checkpoint_path=checkpoint_path,
        episodes=args.episodes,
        max_steps=args.max_steps,
    )
    print(json.dumps(metrics, indent=2))
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "eval_metrics.json", "w", encoding="utf-8") as file_obj:
            json.dump(metrics, file_obj, indent=2)


if __name__ == "__main__":
    main()
