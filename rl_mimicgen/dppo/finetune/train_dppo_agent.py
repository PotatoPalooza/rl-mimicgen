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
from rl_mimicgen.rl.diffusion_storage import DiffusionRolloutStorage
from rl_mimicgen.rl.dppo import DiffusionPPO


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Online DPPO fine-tuning entry point.")
    parser.add_argument("--config", required=True, help="Path to the DPPO JSON config.")
    parser.add_argument("--checkpoint", default=None, help="Offline checkpoint to fine-tune.")
    parser.add_argument("--rollout_steps", type=int, default=None, help="Number of environment steps to collect for the rollout update.")
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
    rollout_target = args.rollout_steps or config.online.rollout_steps
    storage = DiffusionRolloutStorage(
        rollout_steps=rollout_target,
        num_envs=1,
        obs_shapes={"state": (config.diffusion.cond_steps, dataset.obs_dim)},
        goal_shapes=None,
        action_dim=dataset.action_dim,
        prediction_horizon=config.diffusion.horizon_steps,
        chain_length=config.diffusion.denoising_steps,
        device=torch.device(config.device),
    )
    algorithm = DiffusionPPO(
        policy=policy,
        actor_learning_rate=config.online.actor_learning_rate,
        critic_learning_rate=config.online.critic_learning_rate,
        weight_decay=config.online.weight_decay,
        num_learning_epochs=config.online.update_epochs,
        num_mini_batches=config.online.num_minibatches,
        clip_param=config.online.clip_ratio,
        value_loss_coef=config.online.value_loss_coef,
        gamma_denoising=config.online.gamma_denoising,
        act_steps=config.diffusion.act_steps,
        max_grad_norm=config.online.max_grad_norm,
        target_kl=config.online.target_kl,
        device=config.device,
    )
    algorithm.train_mode()

    output_dir = Path(args.output_dir or config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    obs = env.reset()
    episode_starts = np.ones(1, dtype=bool)
    storage.reset()
    rollout_env_steps = 0
    rewards_log: list[float] = []
    done_count = 0
    while rollout_env_steps < rollout_target:
        obs_batch = {key: value[None, ...] for key, value in obs.items()}
        env_actions, replan_data, completed_envs = policy.act(
            obs=obs_batch,
            goal=None,
            episode_starts=episode_starts,
            clip_actions=True,
        )
        if replan_data is not None:
            storage.start_decisions(
                env_indices=replan_data["env_indices"],
                obs_history=replan_data["obs_history"],
                goals=replan_data["goals"],
                chain_samples=replan_data["chain_samples"],
                chain_next_samples=replan_data["chain_next_samples"],
                chain_timesteps=replan_data["chain_timesteps"],
                log_probs=replan_data["log_probs"],
                values=replan_data["values"],
            )
        next_obs, reward, done, _info = env.step(env_actions[0].astype(np.float32, copy=False))
        reward_arr = np.asarray([reward], dtype=np.float32)
        done_arr = np.asarray([float(done)], dtype=np.float32)
        storage.accumulate_step(rewards=reward_arr, dones=done_arr)
        finalize_envs = np.union1d(completed_envs, np.asarray([0], dtype=np.int64) if done else np.asarray([], dtype=np.int64))
        if finalize_envs.size:
            storage.finalize_decisions(finalize_envs)
        rewards_log.append(float(reward))
        rollout_env_steps += 1
        if done:
            done_count += 1
            obs = env.reset()
            episode_starts = np.ones(1, dtype=bool)
        else:
            obs = next_obs
            episode_starts = np.zeros(1, dtype=bool)
    last_values = policy.predict_value(obs={key: value[None, ...] for key, value in obs.items()})
    storage.compute_returns_and_advantages(
        last_value=last_values,
        gamma=config.online.gamma,
        gae_lambda=config.online.gae_lambda,
    )
    batch = storage.as_batch()
    algorithm.set_total_env_steps(rollout_env_steps)
    update_metrics = algorithm.update(batch)
    env.close()

    rollout_metrics = {
        "rollout_steps": rollout_env_steps,
        "reward_sum": float(np.sum(rewards_log)) if rewards_log else 0.0,
        "done_count": int(done_count),
        "checkpoint": checkpoint_path,
        "update": update_metrics,
    }
    print(json.dumps(rollout_metrics, indent=2))

    torch.save(
        {
            "actor": policy.actor.state_dict(),
            "ema_actor": policy.ema_actor.state_dict(),
            "critic": policy.value_net.state_dict(),
            "optimizer": algorithm.save(),
            "config": config.to_dict(),
        },
        checkpoint_dir / "state_1.pt",
    )
    np.savez_compressed(
        output_dir / "bootstrap_rollout.npz",
        returns=batch.returns.detach().cpu().numpy(),
        advantages=batch.advantages.detach().cpu().numpy(),
        values=batch.values.detach().cpu().numpy(),
        rewards=batch.rewards.detach().cpu().numpy(),
        dones=batch.dones.detach().cpu().numpy(),
    )
    with open(output_dir / "rollout_metrics.json", "w", encoding="utf-8") as file_obj:
        json.dump(rollout_metrics, file_obj, indent=2)


if __name__ == "__main__":
    main()
