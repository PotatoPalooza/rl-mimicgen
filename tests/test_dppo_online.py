from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from rl_mimicgen.dppo.config.schema import DPPODiffusionConfig, DPPODatasetConfig, DPPOOnlineConfig, DPPORunConfig, DPPOTrainConfig
from rl_mimicgen.dppo.data.dataset import DPPODatasetBundle, DPPONormalizationStats
from rl_mimicgen.dppo.finetune.train_dppo_agent import (
    _assert_checkpoint_matches_config,
    _load_checkpoint_payload,
    _load_existing_metrics,
    _restore_algorithm_state,
    _save_training_checkpoint,
    _validate_resume_checkpoint,
)
from rl_mimicgen.dppo.model import DiffusionModel
from rl_mimicgen.dppo.online import DiffusionPPO, DiffusionRolloutStorage
from rl_mimicgen.dppo.policy import DiffusionPolicyAdapter


def _make_bundle() -> DPPODatasetBundle:
    obs = np.zeros((8, 3), dtype=np.float32)
    actions = np.zeros((8, 2), dtype=np.float32)
    zeros_obs = np.zeros(3, dtype=np.float32)
    ones_obs = np.ones(3, dtype=np.float32)
    zeros_action = np.zeros(2, dtype=np.float32)
    ones_action = np.ones(2, dtype=np.float32)
    return DPPODatasetBundle(
        bundle_dir=Path("/tmp/dppo_test_bundle"),
        metadata={"task": "coffee", "variant": "D0", "obs_keys": ["obs"], "num_demos": 1, "num_transitions": int(obs.shape[0])},
        obs=obs,
        next_obs=obs.copy(),
        actions=actions,
        rewards=np.zeros(obs.shape[0], dtype=np.float32),
        dones=np.zeros(obs.shape[0], dtype=np.float32),
        traj_lengths=np.asarray([obs.shape[0]], dtype=np.int32),
        demo_offsets=np.asarray([0], dtype=np.int64),
        normalization=DPPONormalizationStats(
            obs_mean=zeros_obs,
            obs_std=ones_obs,
            obs_min=-ones_obs,
            obs_max=ones_obs,
            obs_range=2.0 * ones_obs,
            action_mean=zeros_action,
            action_std=ones_action,
            action_min=-ones_action,
            action_max=ones_action,
            action_range=2.0 * ones_action,
        ),
    )


def _make_config() -> DPPORunConfig:
    return DPPORunConfig(
        task="coffee",
        variant="D0",
        device="cpu",
        num_envs=2,
        dataset=DPPODatasetConfig(task="coffee", variant="D0"),
        diffusion=DPPODiffusionConfig(
            horizon_steps=4,
            act_steps=2,
            cond_steps=1,
            denoising_steps=5,
            time_dim=8,
            mlp_dims=(32, 32),
            residual_style=False,
        ),
        training=DPPOTrainConfig(ema_decay=0.9),
        online=DPPOOnlineConfig(
            update_epochs=2,
            num_minibatches=1,
            actor_learning_rate=1e-3,
            critic_learning_rate=1e-3,
            gamma_denoising=0.95,
        ),
    )


def _make_checkpoint(tmp_path, config: DPPORunConfig, bundle: DPPODatasetBundle) -> str:
    actor = DiffusionModel(
        obs_dim=bundle.obs_dim,
        action_dim=bundle.action_dim,
        horizon_steps=config.diffusion.horizon_steps,
        cond_steps=config.diffusion.cond_steps,
        denoising_steps=config.diffusion.denoising_steps,
        predict_epsilon=config.diffusion.predict_epsilon,
        denoised_clip_value=config.diffusion.denoised_clip_value,
        time_dim=config.diffusion.time_dim,
        mlp_dims=config.diffusion.mlp_dims,
        residual_style=config.diffusion.residual_style,
        device=config.device,
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model": deepcopy(actor.state_dict()), "config": config.to_dict()}, checkpoint_path)
    return str(checkpoint_path)


def test_dppo_rollout_storage_finalizes_pending_records_and_normalizes_advantages() -> None:
    storage = DiffusionRolloutStorage(
        rollout_steps=2,
        num_envs=2,
        obs_shapes={"state": (1, 3)},
        goal_shapes=None,
        action_dim=2,
        prediction_horizon=4,
        chain_length=3,
        device=torch.device("cpu"),
    )
    obs_history = {"state": torch.ones((2, 1, 3), dtype=torch.float32)}
    chain_samples = torch.ones((2, 3, 4, 2), dtype=torch.float32)
    chain_next_samples = torch.full((2, 3, 4, 2), 2.0, dtype=torch.float32)
    chain_timesteps = torch.tile(torch.arange(3, dtype=torch.long), (2, 1))
    log_probs = torch.zeros((2, 3, 2, 2), dtype=torch.float32)
    values = torch.tensor([0.5, -0.25], dtype=torch.float32)

    storage.start_decisions(
        env_indices=np.asarray([0, 1], dtype=np.int64),
        obs_history=obs_history,
        goals=None,
        chain_samples=chain_samples,
        chain_next_samples=chain_next_samples,
        chain_timesteps=chain_timesteps,
        log_probs=log_probs,
        values=values,
    )
    storage.accumulate_step(
        rewards=np.asarray([1.0, 2.0], dtype=np.float32),
        dones=np.asarray([0.0, 1.0], dtype=np.float32),
    )
    storage.finalize_decisions(np.asarray([1], dtype=np.int64))
    storage.accumulate_step(
        rewards=np.asarray([3.0, 0.0], dtype=np.float32),
        dones=np.asarray([1.0, 0.0], dtype=np.float32),
    )
    storage.compute_returns_and_advantages(last_value=np.asarray([0.0, 0.0], dtype=np.float32), gamma=0.99, gae_lambda=0.95)

    batch = storage.as_batch()

    assert batch.observations["state"].shape == (2, 1, 3)
    assert batch.chain_samples.shape == (2, 3, 4, 2)
    assert batch.rewards.tolist() == [4.0, 2.0]
    assert batch.dones.tolist() == [1.0, 1.0]
    assert torch.isclose(batch.advantages.mean(), torch.tensor(0.0), atol=1e-6)


def test_dppo_algorithm_updates_actor_and_critic_from_synthetic_batch(tmp_path) -> None:
    config = _make_config()
    bundle = _make_bundle()
    checkpoint_path = _make_checkpoint(tmp_path, config, bundle)
    policy = DiffusionPolicyAdapter(config=config, bundle=bundle, checkpoint_path=checkpoint_path, deterministic=False)
    algorithm = DiffusionPPO(
        policy=policy,
        actor_learning_rate=config.online.actor_learning_rate,
        critic_learning_rate=config.online.critic_learning_rate,
        num_learning_epochs=config.online.update_epochs,
        num_mini_batches=config.online.num_minibatches,
        clip_param=config.online.clip_ratio,
        value_loss_coef=config.online.value_loss_coef,
        gamma_denoising=config.online.gamma_denoising,
        act_steps=config.diffusion.act_steps,
        max_grad_norm=config.online.max_grad_norm,
        device=config.device,
    )
    algorithm.set_total_env_steps(1)

    observations = {"state": torch.tensor([[[0.2, -0.1, 0.4]], [[-0.3, 0.5, 0.1]]], dtype=torch.float32)}
    chain_samples = torch.tensor(
        [
            [
                [[0.10, -0.20], [0.30, 0.40], [0.20, -0.10], [0.00, 0.10]],
                [[0.05, -0.10], [0.20, 0.10], [0.15, -0.05], [0.02, 0.08]],
                [[0.01, -0.05], [0.10, 0.05], [0.08, -0.02], [0.01, 0.03]],
                [[0.00, 0.00], [0.05, 0.02], [0.04, 0.01], [0.00, 0.01]],
            ],
            [
                [[-0.20, 0.10], [-0.10, 0.20], [0.05, 0.15], [0.02, -0.03]],
                [[-0.15, 0.08], [-0.08, 0.10], [0.03, 0.12], [0.01, -0.02]],
                [[-0.10, 0.05], [-0.05, 0.06], [0.02, 0.08], [0.00, -0.01]],
                [[-0.05, 0.02], [-0.02, 0.03], [0.01, 0.04], [0.00, 0.00]],
            ],
        ],
        dtype=torch.float32,
    )
    chain_next_samples = chain_samples * 0.9
    chain_timesteps = torch.tensor([[3, 2, 1, 0], [3, 2, 1, 0]], dtype=torch.long)
    with torch.no_grad():
        log_probs = policy._chain_log_prob_subsample(
            obs_history=observations,
            goal_batch=None,
            chain_samples=chain_samples,
            chain_next_samples=chain_next_samples,
            chain_timesteps=chain_timesteps,
        )
    storage = DiffusionRolloutStorage(
        rollout_steps=1,
        num_envs=2,
        obs_shapes={"state": (1, bundle.obs_dim)},
        goal_shapes=None,
        action_dim=bundle.action_dim,
        prediction_horizon=config.diffusion.horizon_steps,
        chain_length=config.diffusion.denoising_steps - 1,
        device=torch.device(config.device),
    )
    storage.start_decisions(
        env_indices=np.asarray([0, 1], dtype=np.int64),
        obs_history=observations,
        goals=None,
        chain_samples=chain_samples,
        chain_next_samples=chain_next_samples,
        chain_timesteps=chain_timesteps,
        log_probs=log_probs,
        values=torch.tensor([0.0, 0.0], dtype=torch.float32),
    )
    storage.accumulate_step(
        rewards=np.asarray([1.0, 0.5], dtype=np.float32),
        dones=np.asarray([1.0, 1.0], dtype=np.float32),
    )
    storage.compute_returns_and_advantages(last_value=np.asarray([0.0, 0.0], dtype=np.float32), gamma=0.99, gae_lambda=0.95)
    batch = storage.as_batch()

    actor_before = deepcopy(next(policy.actor.parameters()).detach())
    critic_before = deepcopy(next(policy.value_net.parameters()).detach())
    metrics = algorithm.update(batch)

    assert metrics["effective_num_minibatches"] >= 1.0
    assert metrics["value"] >= 0.0
    assert not torch.allclose(next(policy.actor.parameters()).detach(), actor_before)
    assert not torch.allclose(next(policy.value_net.parameters()).detach(), critic_before)


def test_dppo_algorithm_slice_batch_preserves_full_horizon_when_act_steps_is_shorter(tmp_path) -> None:
    config = _make_config()
    bundle = _make_bundle()
    checkpoint_path = _make_checkpoint(tmp_path, config, bundle)
    policy = DiffusionPolicyAdapter(config=config, bundle=bundle, checkpoint_path=checkpoint_path, deterministic=False)
    algorithm = DiffusionPPO(
        policy=policy,
        act_steps=config.diffusion.act_steps,
        device=config.device,
    )
    storage = DiffusionRolloutStorage(
        rollout_steps=1,
        num_envs=1,
        obs_shapes={"state": (1, bundle.obs_dim)},
        goal_shapes=None,
        action_dim=bundle.action_dim,
        prediction_horizon=config.diffusion.horizon_steps,
        chain_length=config.diffusion.denoising_steps - 1,
        device=torch.device(config.device),
    )
    storage.start_decisions(
        env_indices=np.asarray([0], dtype=np.int64),
        obs_history={"state": torch.zeros((1, 1, bundle.obs_dim), dtype=torch.float32)},
        goals=None,
        chain_samples=torch.zeros((1, config.diffusion.denoising_steps - 1, config.diffusion.horizon_steps, bundle.action_dim), dtype=torch.float32),
        chain_next_samples=torch.zeros((1, config.diffusion.denoising_steps - 1, config.diffusion.horizon_steps, bundle.action_dim), dtype=torch.float32),
        chain_timesteps=torch.zeros((1, config.diffusion.denoising_steps - 1), dtype=torch.long),
        log_probs=torch.zeros((1, config.diffusion.denoising_steps - 1, config.diffusion.horizon_steps, bundle.action_dim), dtype=torch.float32),
        values=torch.zeros(1, dtype=torch.float32),
    )
    storage.accumulate_step(rewards=np.asarray([1.0], dtype=np.float32), dones=np.asarray([1.0], dtype=np.float32))
    storage.compute_returns_and_advantages(last_value=np.asarray([0.0], dtype=np.float32), gamma=0.99, gae_lambda=0.95)

    sliced = algorithm._slice_batch(storage.as_batch(), torch.tensor([0], dtype=torch.long))

    assert sliced["chain_prev"].shape[2] == config.diffusion.horizon_steps
    assert sliced["chain_next"].shape[2] == config.diffusion.horizon_steps


def test_dppo_finetune_helpers_resume_optimizer_state_and_metrics(tmp_path) -> None:
    config = _make_config()
    bundle = _make_bundle()
    checkpoint_path = _make_checkpoint(tmp_path, config, bundle)
    policy = DiffusionPolicyAdapter(config=config, bundle=bundle, checkpoint_path=checkpoint_path, deterministic=False)
    algorithm = DiffusionPPO(
        policy=policy,
        actor_learning_rate=config.online.actor_learning_rate,
        critic_learning_rate=config.online.critic_learning_rate,
        num_learning_epochs=config.online.update_epochs,
        num_mini_batches=config.online.num_minibatches,
        clip_param=config.online.clip_ratio,
        value_loss_coef=config.online.value_loss_coef,
        gamma_denoising=config.online.gamma_denoising,
        act_steps=config.diffusion.act_steps,
        max_grad_norm=config.online.max_grad_norm,
        device=config.device,
    )
    algorithm.set_total_env_steps(17)
    algorithm.update_step = 3

    output_dir = tmp_path / "resume_run"
    output_dir.mkdir()
    metrics_path = output_dir / "rollout_metrics.json"
    metrics_path.write_text('[{"update_index": 1}, {"update_index": 4}]', encoding="utf-8")
    resumed_checkpoint = output_dir / "state_3.pt"
    _save_training_checkpoint(resumed_checkpoint, policy=policy, algorithm=algorithm, config=config)

    resumed_policy = DiffusionPolicyAdapter(config=config, bundle=bundle, checkpoint_path=str(resumed_checkpoint), deterministic=False)
    resumed_algorithm = DiffusionPPO(
        policy=resumed_policy,
        actor_learning_rate=config.online.actor_learning_rate,
        critic_learning_rate=config.online.critic_learning_rate,
        num_learning_epochs=config.online.update_epochs,
        num_mini_batches=config.online.num_minibatches,
        clip_param=config.online.clip_ratio,
        value_loss_coef=config.online.value_loss_coef,
        gamma_denoising=config.online.gamma_denoising,
        act_steps=config.diffusion.act_steps,
        max_grad_norm=config.online.max_grad_norm,
        device=config.device,
    )

    payload = _load_checkpoint_payload(str(resumed_checkpoint), device=torch.device(config.device))
    completed_updates, total_env_steps = _restore_algorithm_state(resumed_algorithm, payload)
    resumed_metrics = _load_existing_metrics(metrics_path, resume=True, max_update_index=completed_updates)
    fresh_metrics = _load_existing_metrics(metrics_path, resume=False)

    assert completed_updates == 3
    assert total_env_steps == 17
    assert resumed_algorithm.update_step == 3
    assert resumed_algorithm.total_env_steps == 17
    assert resumed_metrics == [{"update_index": 1}]
    assert fresh_metrics == []


def test_dppo_resume_validation_rejects_offline_checkpoint(tmp_path) -> None:
    config = _make_config()
    bundle = _make_bundle()
    checkpoint_path = _make_checkpoint(tmp_path, config, bundle)
    payload = _load_checkpoint_payload(checkpoint_path, device=torch.device(config.device))

    with pytest.raises(RuntimeError, match="--resume requires a local finetune checkpoint"):
        _validate_resume_checkpoint(payload, checkpoint_path)


def test_dppo_resume_validation_rejects_filename_update_step_mismatch(tmp_path) -> None:
    config = _make_config()
    bundle = _make_bundle()
    checkpoint_path = _make_checkpoint(tmp_path, config, bundle)
    policy = DiffusionPolicyAdapter(config=config, bundle=bundle, checkpoint_path=checkpoint_path, deterministic=False)
    algorithm = DiffusionPPO(policy=policy, device=config.device)
    algorithm.update_step = 2

    mismatched_path = tmp_path / "state_3.pt"
    _save_training_checkpoint(mismatched_path, policy=policy, algorithm=algorithm, config=config)
    payload = _load_checkpoint_payload(str(mismatched_path), device=torch.device(config.device))

    with pytest.raises(RuntimeError, match="filename/update_step mismatch"):
        _validate_resume_checkpoint(payload, str(mismatched_path))


def test_dppo_checkpoint_config_validation_rejects_task_mismatch(tmp_path) -> None:
    config = _make_config()
    bundle = _make_bundle()
    checkpoint_path = _make_checkpoint(tmp_path, config, bundle)
    payload = _load_checkpoint_payload(checkpoint_path, device=torch.device(config.device))
    mismatched = _make_config()
    mismatched.task = "square"
    mismatched.dataset.task = "square"

    with pytest.raises(RuntimeError, match="Checkpoint/config mismatch"):
        _assert_checkpoint_matches_config(mismatched, payload)
