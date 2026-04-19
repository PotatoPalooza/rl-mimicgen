from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import torch

from rl_mimicgen.dppo.config.schema import DPPORunConfig
from rl_mimicgen.dppo.data import DPPODatasetBundle
from rl_mimicgen.dppo.envs import build_dppo_vec_env, make_mimicgen_lowdim_env
from rl_mimicgen.dppo.eval.eval_diffusion_agent import run_evaluation
from rl_mimicgen.dppo.online import DiffusionPPO, DiffusionRolloutStorage
from rl_mimicgen.dppo.policy import DiffusionPolicyAdapter


def _build_vec_env(config: DPPORunConfig):
    return build_dppo_vec_env(
        task=config.task,
        variant=config.variant,
        use_warp=config.use_warp,
        num_envs=config.num_envs,
        device=config.device,
        clip_actions=config.clip_actions,
        warp_njmax_per_env=config.warp.njmax_per_env,
        warp_naconmax_per_env=config.warp.naconmax_per_env,
        warp_graph_capture=config.warp.graph_capture,
        physics_timestep=config.warp.physics_timestep,
    )


def _load_checkpoint_payload(checkpoint_path: str, device: torch.device) -> dict[str, object]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected checkpoint payload to be a dict, found {type(payload).__name__}.")
    return payload


def _parse_checkpoint_update_index(checkpoint_path: str) -> int | None:
    match = re.fullmatch(r"state_(\d+)", Path(checkpoint_path).stem)
    if match is None:
        return None
    return int(match.group(1))


def _normalize_config_value(value: object) -> object:
    if isinstance(value, list):
        return tuple(value)
    return value


def _assert_checkpoint_matches_config(
    run_config: DPPORunConfig,
    checkpoint_payload: dict[str, object],
) -> None:
    checkpoint_config_payload = checkpoint_payload.get("config")
    if checkpoint_config_payload is None:
        return
    if not isinstance(checkpoint_config_payload, dict):
        raise RuntimeError(f"Expected checkpoint config to be a dict, found {type(checkpoint_config_payload).__name__}.")

    expected_pairs = {
        "task": (run_config.task, checkpoint_config_payload.get("task")),
        "variant": (run_config.variant, checkpoint_config_payload.get("variant")),
        "dataset.task": (run_config.dataset.task, checkpoint_config_payload.get("dataset", {}).get("task")),
        "dataset.variant": (run_config.dataset.variant, checkpoint_config_payload.get("dataset", {}).get("variant")),
        "diffusion.horizon_steps": (
            run_config.diffusion.horizon_steps,
            checkpoint_config_payload.get("diffusion", {}).get("horizon_steps"),
        ),
        "diffusion.act_steps": (
            run_config.diffusion.act_steps,
            checkpoint_config_payload.get("diffusion", {}).get("act_steps"),
        ),
        "diffusion.cond_steps": (
            run_config.diffusion.cond_steps,
            checkpoint_config_payload.get("diffusion", {}).get("cond_steps"),
        ),
        "diffusion.denoising_steps": (
            run_config.diffusion.denoising_steps,
            checkpoint_config_payload.get("diffusion", {}).get("denoising_steps"),
        ),
        "diffusion.predict_epsilon": (
            run_config.diffusion.predict_epsilon,
            checkpoint_config_payload.get("diffusion", {}).get("predict_epsilon"),
        ),
        "diffusion.denoised_clip_value": (
            run_config.diffusion.denoised_clip_value,
            checkpoint_config_payload.get("diffusion", {}).get("denoised_clip_value"),
        ),
        "diffusion.time_dim": (
            run_config.diffusion.time_dim,
            checkpoint_config_payload.get("diffusion", {}).get("time_dim"),
        ),
        "diffusion.mlp_dims": (
            tuple(run_config.diffusion.mlp_dims),
            _normalize_config_value(checkpoint_config_payload.get("diffusion", {}).get("mlp_dims")),
        ),
        "diffusion.residual_style": (
            run_config.diffusion.residual_style,
            checkpoint_config_payload.get("diffusion", {}).get("residual_style"),
        ),
    }
    mismatches = [
        f"{key}: config={expected!r} checkpoint={actual!r}"
        for key, (expected, actual) in expected_pairs.items()
        if expected != actual
    ]
    if mismatches:
        raise RuntimeError("Checkpoint/config mismatch:\n" + "\n".join(mismatches))


def _validate_resume_checkpoint(
    checkpoint_payload: dict[str, object],
    checkpoint_path: str,
) -> None:
    required_keys = {"actor", "ema_actor", "critic", "optimizer"}
    missing_keys = sorted(key for key in required_keys if key not in checkpoint_payload)
    if missing_keys:
        raise RuntimeError(
            f"--resume requires a local finetune checkpoint with keys {sorted(required_keys)}; "
            f"{checkpoint_path} is missing {missing_keys}."
        )
    optimizer_state = checkpoint_payload["optimizer"]
    if not isinstance(optimizer_state, dict):
        raise RuntimeError(f"Expected optimizer state dict in {checkpoint_path}, found {type(optimizer_state).__name__}.")
    update_step = int(optimizer_state.get("update_step", 0))
    checkpoint_index = _parse_checkpoint_update_index(checkpoint_path)
    if checkpoint_index is not None and checkpoint_index != update_step:
        raise RuntimeError(
            f"Resume checkpoint filename/update_step mismatch for {checkpoint_path}: "
            f"filename={checkpoint_index} optimizer.update_step={update_step}."
        )


def _load_existing_metrics(
    metrics_path: Path,
    resume: bool,
    max_update_index: int | None = None,
) -> list[dict[str, object]]:
    if not resume or not metrics_path.exists():
        return []
    with open(metrics_path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected list payload in {metrics_path}, found {type(payload).__name__}.")
    if max_update_index is None:
        return payload
    return [entry for entry in payload if int(entry.get("update_index", -1)) <= max_update_index]


def _restore_algorithm_state(
    algorithm: DiffusionPPO,
    checkpoint_payload: dict[str, object],
) -> tuple[int, int]:
    optimizer_state = checkpoint_payload.get("optimizer")
    if optimizer_state is None:
        return 0, 0
    if not isinstance(optimizer_state, dict):
        raise RuntimeError(f"Expected optimizer state dict, found {type(optimizer_state).__name__}.")
    algorithm.load(optimizer_state)
    return algorithm.update_step, algorithm.total_env_steps


def _save_training_checkpoint(
    checkpoint_path: Path,
    policy: DiffusionPolicyAdapter,
    algorithm: DiffusionPPO,
    config: DPPORunConfig,
) -> None:
    torch.save(
        {
            "actor": policy.actor.state_dict(),
            "ema_actor": policy.ema_actor.state_dict(),
            "critic": policy.value_net.state_dict(),
            "optimizer": algorithm.save(),
            "config": config.to_dict(),
        },
        checkpoint_path,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Online DPPO fine-tuning entry point.")
    parser.add_argument("--config", required=True, help="Path to the DPPO JSON config.")
    parser.add_argument("--checkpoint", default=None, help="Offline checkpoint to fine-tune.")
    parser.add_argument("--rollout_steps", type=int, default=None, help="Number of environment steps to collect for the rollout update.")
    parser.add_argument("--total_updates", type=int, default=1, help="Number of rollout / update iterations to run.")
    parser.add_argument("--eval_every", type=int, default=1, help="Run evaluation every N updates.")
    parser.add_argument("--output_dir", default=None, help="Directory to write rollout artifacts.")
    parser.add_argument("--resume", action="store_true", help="Resume from a local finetune checkpoint and append metrics in the output dir.")
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

    device = torch.device(config.device)
    checkpoint_payload = _load_checkpoint_payload(checkpoint_path, device=device)
    _assert_checkpoint_matches_config(config, checkpoint_payload)
    if args.resume:
        _validate_resume_checkpoint(checkpoint_payload, checkpoint_path)

    policy = DiffusionPolicyAdapter(config=config, bundle=dataset, checkpoint_path=checkpoint_path, deterministic=False)
    rollout_target = args.rollout_steps or config.online.rollout_steps
    num_envs = int(config.num_envs)
    storage = DiffusionRolloutStorage(
        rollout_steps=rollout_target,
        num_envs=num_envs,
        obs_shapes={"state": (config.diffusion.cond_steps, dataset.obs_dim)},
        goal_shapes=None,
        action_dim=dataset.action_dim,
        prediction_horizon=config.diffusion.horizon_steps,
        chain_length=config.diffusion.denoising_steps,
        device=device,
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
        device=device,
    )
    algorithm.train_mode()

    output_dir = Path(args.output_dir or config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "rollout_metrics.json"

    completed_updates, total_env_steps = _restore_algorithm_state(algorithm, checkpoint_payload) if args.resume else (0, 0)
    all_metrics: list[dict[str, object]] = _load_existing_metrics(metrics_path, resume=args.resume, max_update_index=completed_updates)
    if completed_updates >= args.total_updates:
        print(
            f"resume_noop completed_updates={completed_updates} total_updates={args.total_updates} "
            f"checkpoint={checkpoint_path}"
        )
        if not metrics_path.exists():
            with open(metrics_path, "w", encoding="utf-8") as file_obj:
                json.dump(all_metrics, file_obj, indent=2)
        return

    env = _build_vec_env(config)
    obs = env.reset()
    episode_starts = np.ones(num_envs, dtype=bool)
    last_batch = None
    for update_index in range(completed_updates + 1, args.total_updates + 1):
        storage.reset()
        rollout_env_steps = 0
        rewards_log: list[float] = []
        done_count = 0
        while rollout_env_steps < rollout_target:
            env_actions, replan_data, completed_envs = policy.act(
                obs=obs,
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
            actions_t = torch.as_tensor(env_actions, dtype=torch.float32, device=device)
            next_obs, reward_t, done_t, _extras = env.step(actions_t)
            reward_np = reward_t.detach().cpu().numpy().astype(np.float32, copy=False)
            done_np = done_t.detach().cpu().numpy().astype(bool, copy=False)
            storage.accumulate_step(rewards=reward_np, dones=done_np.astype(np.float32))
            done_indices = np.nonzero(done_np)[0].astype(np.int64, copy=False)
            finalize_envs = np.union1d(completed_envs, done_indices)
            if finalize_envs.size:
                storage.finalize_decisions(finalize_envs)
            rewards_log.append(float(reward_np.sum()))
            rollout_env_steps += num_envs
            total_env_steps += num_envs
            done_count += int(done_np.sum())
            obs = next_obs
            episode_starts = done_np.copy()

        last_values = policy.predict_value(obs=obs)
        storage.compute_returns_and_advantages(
            last_value=last_values,
            gamma=config.online.gamma,
            gae_lambda=config.online.gae_lambda,
        )
        batch = storage.as_batch()
        last_batch = batch
        algorithm.set_total_env_steps(total_env_steps)
        update_metrics = algorithm.update(batch)
        metrics: dict[str, object] = {
            "update_index": update_index,
            "rollout_steps": rollout_env_steps,
            "reward_sum": float(np.sum(rewards_log)) if rewards_log else 0.0,
            "done_count": int(done_count),
            "checkpoint": checkpoint_path,
            "update": update_metrics,
        }
        if update_index % args.eval_every == 0:
            eval_metrics = run_evaluation(
                config=config,
                dataset=dataset,
                checkpoint_path=checkpoint_path,
                actor_override=policy.ema_actor,
                episodes=1,
                max_steps=min(env.horizon, rollout_target),
            )
            metrics["eval"] = eval_metrics
        all_metrics.append(metrics)
        print(json.dumps(metrics, indent=2))
        _save_training_checkpoint(checkpoint_dir / f"state_{update_index}.pt", policy=policy, algorithm=algorithm, config=config)
        with open(metrics_path, "w", encoding="utf-8") as file_obj:
            json.dump(all_metrics, file_obj, indent=2)

    env.close()
    if last_batch is not None:
        np.savez_compressed(
            output_dir / "bootstrap_rollout.npz",
            returns=last_batch.returns.detach().cpu().numpy(),
            advantages=last_batch.advantages.detach().cpu().numpy(),
            values=last_batch.values.detach().cpu().numpy(),
            rewards=last_batch.rewards.detach().cpu().numpy(),
            dones=last_batch.dones.detach().cpu().numpy(),
        )
    if not metrics_path.exists():
        with open(metrics_path, "w", encoding="utf-8") as file_obj:
            json.dump(all_metrics, file_obj, indent=2)


if __name__ == "__main__":
    main()
