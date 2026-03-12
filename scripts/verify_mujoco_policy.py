import argparse
import math
import multiprocessing as mp
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

import mimicgen  # noqa: F401
import robomimic.utils.env_utils as EnvUtils

from rl_mimicgen.rl.policy import load_policy_bundle


def parse_bool_or_auto(value: str):
    lowered = value.lower()
    if lowered == "auto":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ValueError(f"Expected one of auto|true|false, got: {value}")


@dataclass
class WorkerConfig:
    checkpoint_path: str
    device: str
    env_name: str | None
    horizon: int | None
    reward_shaping: bool | None
    terminate_on_success: bool
    seed: int


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _build_env(worker_cfg: WorkerConfig, bundle: Any):
    env_meta = bundle.ckpt_dict["env_metadata"].copy()
    env_meta["env_kwargs"] = dict(env_meta["env_kwargs"])
    if worker_cfg.env_name is not None:
        env_meta["env_name"] = worker_cfg.env_name
    if worker_cfg.reward_shaping is not None:
        env_meta["env_kwargs"]["reward_shaping"] = worker_cfg.reward_shaping

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_meta["env_name"],
        render=False,
        render_offscreen=False,
        use_image_obs=bundle.shape_meta.get("use_images", False),
        use_depth_obs=bundle.shape_meta.get("use_depths", False),
    )
    return EnvUtils.wrap_env_from_config(env, config=bundle.config)


def _env_horizon(worker_cfg: WorkerConfig, bundle: Any) -> int:
    if worker_cfg.horizon is not None:
        return worker_cfg.horizon
    return int(bundle.config.experiment.rollout.horizon)


def _get_goal(bundle: Any, env) -> dict | None:
    if not bundle.config.use_goals:
        return None
    return env.get_goal()


def _close_env(env) -> None:
    close_fn = getattr(env, "close", None)
    if callable(close_fn):
        close_fn()


def _run_worker(worker_cfg: WorkerConfig, episodes: int, worker_id: int) -> dict[str, Any]:
    _seed_everything(worker_cfg.seed + worker_id)
    device = _resolve_device(worker_cfg.device)
    bundle = load_policy_bundle(worker_cfg.checkpoint_path, device)
    env = _build_env(worker_cfg, bundle)
    horizon = _env_horizon(worker_cfg, bundle)

    returns = []
    lengths = []
    successes = []
    try:
        for _ in range(episodes):
            obs = env.reset()
            goal = _get_goal(bundle, env)
            bundle.rollout_policy.start_episode()
            total_reward = 0.0
            success = False
            episode_length = horizon

            for step_idx in range(horizon):
                action = bundle.rollout_policy(ob=obs, goal=goal)
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                success = success or bool(env.is_success()["task"])
                goal = _get_goal(bundle, env)
                if done or (worker_cfg.terminate_on_success and success):
                    episode_length = step_idx + 1
                    break

            returns.append(float(total_reward))
            lengths.append(float(episode_length))
            successes.append(float(success))
    finally:
        _close_env(env)

    return {
        "episodes": episodes,
        "successes": int(sum(successes)),
        "return_mean": float(np.mean(returns)) if returns else 0.0,
        "length_mean": float(np.mean(lengths)) if lengths else 0.0,
        "returns": returns,
        "lengths": lengths,
        "success_flags": successes,
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    phat = successes / total
    denom = 1.0 + (z * z) / total
    center = (phat + (z * z) / (2.0 * total)) / denom
    radius = (z / denom) * math.sqrt((phat * (1.0 - phat) / total) + (z * z) / (4.0 * total * total))
    return max(0.0, center - radius), min(1.0, center + radius)


def split_episodes(total_episodes: int, workers: int) -> list[int]:
    workers = max(1, min(workers, total_episodes))
    base = total_episodes // workers
    remainder = total_episodes % workers
    return [base + (1 if idx < remainder else 0) for idx in range(workers)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a robomimic / MimicGen checkpoint with parallel MuJoCo rollouts.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a robomimic policy checkpoint.")
    parser.add_argument("--episodes", type=int, default=50, help="Total number of verification episodes.")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device per worker, e.g. cpu or cuda.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--env-name", type=str, default=None, help="Optional environment override.")
    parser.add_argument("--horizon", type=int, default=None, help="Optional rollout horizon override.")
    parser.add_argument(
        "--reward-shaping",
        type=str,
        default=None,
        help="Override env reward shaping: auto, true, or false.",
    )
    parser.add_argument(
        "--terminate-on-success",
        dest="terminate_on_success",
        action="store_true",
        help="Stop an episode immediately after task success.",
    )
    parser.add_argument(
        "--no-terminate-on-success",
        dest="terminate_on_success",
        action="store_false",
        help="Continue episodes until done or horizon even after success.",
    )
    parser.set_defaults(terminate_on_success=True)
    args = parser.parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    worker_cfg = WorkerConfig(
        checkpoint_path=args.checkpoint,
        device=args.device,
        env_name=args.env_name,
        horizon=args.horizon,
        reward_shaping=parse_bool_or_auto(args.reward_shaping) if args.reward_shaping is not None else None,
        terminate_on_success=args.terminate_on_success,
        seed=args.seed,
    )

    allocations = split_episodes(args.episodes, args.workers)
    ctx = mp.get_context("spawn")

    results = []
    with ProcessPoolExecutor(max_workers=len(allocations), mp_context=ctx) as executor:
        futures = [
            executor.submit(_run_worker, worker_cfg, episodes, worker_id)
            for worker_id, episodes in enumerate(allocations)
            if episodes > 0
        ]
        for future in futures:
            results.append(future.result())

    total_episodes = sum(result["episodes"] for result in results)
    total_successes = sum(result["successes"] for result in results)
    all_returns = [value for result in results for value in result["returns"]]
    all_lengths = [value for result in results for value in result["lengths"]]
    lower, upper = wilson_interval(total_successes, total_episodes)

    print(f"checkpoint={args.checkpoint}")
    print(f"episodes={total_episodes}")
    print(f"workers={len(allocations)}")
    print(f"successes={total_successes}")
    print(f"success_rate={total_successes / total_episodes:.4f}")
    print(f"success_rate_ci95_wilson=[{lower:.4f}, {upper:.4f}]")
    print(f"return_mean={float(np.mean(all_returns)) if all_returns else 0.0:.4f}")
    print(f"length_mean={float(np.mean(all_lengths)) if all_lengths else 0.0:.2f}")


if __name__ == "__main__":
    main()
