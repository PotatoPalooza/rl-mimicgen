from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from rl_mimicgen.dppo.config.schema import DPPODiffusionConfig, DPPODatasetConfig, DPPOOnlineConfig, DPPORunConfig, DPPOTrainConfig
from rl_mimicgen.dppo.data.dataset import DPPODatasetBundle, DPPONormalizationStats
from rl_mimicgen.dppo.model import CriticObs, DiffusionModel
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
        num_envs=1,
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
        online=DPPOOnlineConfig(),
    )


def _make_actor(config: DPPORunConfig, bundle: DPPODatasetBundle) -> DiffusionModel:
    return DiffusionModel(
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


class _DummyModule(nn.Module):
    def forward(self, *args, **kwargs):
        raise NotImplementedError


def test_dppo_policy_loads_pretrain_checkpoint_without_derived_ddpm_buffers(tmp_path) -> None:
    config = _make_config()
    bundle = _make_bundle()
    actor = _make_actor(config, bundle)
    checkpoint_state = deepcopy(actor.state_dict())
    checkpoint_state.pop("ddpm_logvar_clipped")
    checkpoint_state.pop("ddpm_mu_coef1")
    checkpoint_state.pop("ddpm_mu_coef2")
    checkpoint_path = tmp_path / "pretrain.pt"
    torch.save({"model": checkpoint_state}, checkpoint_path)

    policy = DiffusionPolicyAdapter(config=config, bundle=bundle, checkpoint_path=str(checkpoint_path), deterministic=True)

    assert torch.allclose(next(policy.actor.parameters()), next(actor.parameters()))
    assert torch.allclose(next(policy.ema_actor.parameters()), next(actor.parameters()))


def test_dppo_policy_loads_local_finetune_checkpoint_including_critic(tmp_path) -> None:
    config = _make_config()
    bundle = _make_bundle()
    actor = _make_actor(config, bundle)
    ema_actor = _make_actor(config, bundle)
    critic = CriticObs(cond_dim=bundle.obs_dim * config.diffusion.cond_steps)
    with torch.no_grad():
        first_actor_param = next(actor.parameters())
        first_actor_param.fill_(0.25)
        first_ema_param = next(ema_actor.parameters())
        first_ema_param.fill_(0.5)
        first_critic_param = next(critic.parameters())
        first_critic_param.fill_(0.75)
    checkpoint_path = tmp_path / "finetune.pt"
    torch.save(
        {
            "actor": actor.state_dict(),
            "ema_actor": ema_actor.state_dict(),
            "critic": critic.state_dict(),
        },
        checkpoint_path,
    )

    policy = DiffusionPolicyAdapter(config=config, bundle=bundle, checkpoint_path=str(checkpoint_path), deterministic=False)

    assert torch.allclose(next(policy.actor.parameters()), next(actor.parameters()))
    assert torch.allclose(next(policy.ema_actor.parameters()), next(ema_actor.parameters()))
    assert torch.allclose(next(policy.value_net.parameters()), next(critic.parameters()))


def test_dppo_policy_act_reuses_queued_actions_and_marks_completed_envs(monkeypatch) -> None:
    adapter = DiffusionPolicyAdapter.__new__(DiffusionPolicyAdapter)
    nn.Module.__init__(adapter)
    adapter.actor = _DummyModule()
    adapter.ema_actor = _DummyModule()
    adapter.value_net = _DummyModule()
    adapter.device = torch.device("cpu")
    adapter.observation_horizon = 1
    adapter.prediction_horizon = 3
    adapter.act_steps = 2
    adapter._rollout_obs_history = None
    adapter._train_action_queues = None
    adapter._eval_action_queues = None

    obs_batch = torch.zeros((1, 3), dtype=torch.float32)
    monkeypatch.setattr(adapter, "prepare_observation_batch", lambda obs: obs_batch)
    monkeypatch.setattr(adapter, "_advance_history", lambda history, obs, starts: obs.unsqueeze(1))

    call_count = 0

    def fake_start_new_decisions(obs_history, replan_ids):
        del obs_history
        nonlocal call_count
        if not replan_ids:
            return None
        call_count += 1
        return {
            "env_indices": np.asarray(replan_ids, dtype=np.int64),
            "obs_history": {"state": torch.zeros((1, 1, 3), dtype=torch.float32)},
            "goals": None,
            "chain_samples": torch.zeros((1, 2, 3, 1), dtype=torch.float32),
            "chain_next_samples": torch.zeros((1, 2, 3, 1), dtype=torch.float32),
            "chain_timesteps": torch.zeros((1, 2), dtype=torch.long),
            "log_probs": torch.zeros((1, 2, 2, 1), dtype=torch.float32),
            "values": torch.zeros(1, dtype=torch.float32),
            "action_sequences": torch.tensor([[[1.0], [2.0], [99.0]]], dtype=torch.float32),
        }

    monkeypatch.setattr(adapter, "_start_new_decisions", fake_start_new_decisions)

    first_action, first_replan, first_completed = adapter.act(
        obs={"obs": np.zeros((1, 3), dtype=np.float32)},
        goal=None,
        episode_starts=np.asarray([True]),
        clip_actions=False,
    )
    second_action, second_replan, second_completed = adapter.act(
        obs={"obs": np.zeros((1, 3), dtype=np.float32)},
        goal=None,
        episode_starts=np.asarray([False]),
        clip_actions=False,
    )
    third_action, third_replan, third_completed = adapter.act(
        obs={"obs": np.zeros((1, 3), dtype=np.float32)},
        goal=None,
        episode_starts=np.asarray([False]),
        clip_actions=False,
    )

    assert first_action[0, 0] == 1.0
    assert first_replan is not None
    assert first_completed.size == 0

    assert second_action[0, 0] == 2.0
    assert second_replan is None
    assert second_completed.tolist() == [0]

    assert third_action[0, 0] == 1.0
    assert third_replan is not None
    assert third_completed.size == 0
    assert call_count == 2
