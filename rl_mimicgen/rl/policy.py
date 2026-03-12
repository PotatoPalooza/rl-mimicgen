from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.algo import RolloutPolicy
from robomimic.models.distributions import TanhWrappedDistribution
from robomimic.models.value_nets import ValueNetwork


@dataclass
class PolicyBundle:
    algo: Any
    rollout_policy: RolloutPolicy
    ckpt_dict: dict
    config: Any
    obs_normalization_stats: dict | None
    shape_meta: dict


def load_policy_bundle(checkpoint_path: str, device: torch.device) -> PolicyBundle:
    rollout_policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=checkpoint_path, device=device, verbose=False)
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict, verbose=False)
    return PolicyBundle(
        algo=rollout_policy.policy,
        rollout_policy=rollout_policy,
        ckpt_dict=ckpt_dict,
        config=config,
        obs_normalization_stats=rollout_policy.obs_normalization_stats,
        shape_meta=ckpt_dict["shape_metadata"],
    )


def _atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(min=-1.0 + eps, max=1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class OnlinePolicyAdapter(nn.Module):
    """Wrap a robomimic BC policy so it can be fine-tuned with policy gradients."""

    def __init__(
        self,
        bundle: PolicyBundle,
        device: torch.device,
        value_hidden_sizes: tuple[int, ...] = (256, 256),
        init_log_std: float = -0.5,
    ) -> None:
        super().__init__()
        self.bundle = bundle
        self.algo = bundle.algo
        self.rollout_policy = bundle.rollout_policy
        self.device = device
        self.actor = self.algo.nets["policy"]
        self.obs_shapes = deepcopy(self.algo.obs_shapes)
        self.goal_shapes = deepcopy(self.algo.goal_shapes)
        self.obs_keys = tuple(self.obs_shapes.keys())
        self.goal_keys = tuple(self.goal_shapes.keys())
        self.obs_normalization_stats = bundle.obs_normalization_stats
        self.is_recurrent = bool(_config_enabled(bundle.config.algo, "rnn"))
        self.uses_stochastic_head = bool(
            _config_enabled(bundle.config.algo, "gmm") or _config_enabled(bundle.config.algo, "gaussian")
        )

        self.value_net = ValueNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            mlp_layer_dims=list(value_hidden_sizes),
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.algo.obs_config.encoder),
        ).float().to(device)

        self.learned_log_std = None
        if not self.uses_stochastic_head:
            self.learned_log_std = nn.Parameter(torch.full((bundle.shape_meta["ac_dim"],), init_log_std, device=device))

        self._rollout_rnn_state = None

    def actor_parameters(self) -> list[nn.Parameter]:
        params = list(self.actor.parameters())
        if self.learned_log_std is not None:
            params.append(self.learned_log_std)
        return params

    def reset_rollout_state(self) -> None:
        self._rollout_rnn_state = None
        self.algo.reset()

    def prepare_observation(self, obs: dict | None) -> dict | None:
        if obs is None:
            return None
        filtered = {k: obs[k] for k in self.obs_keys if k in obs}
        prepared = TensorUtils.to_float(TensorUtils.to_device(TensorUtils.to_batch(TensorUtils.to_tensor(filtered)), self.device))
        if self.obs_normalization_stats is not None:
            stats = TensorUtils.to_float(
                TensorUtils.to_device(TensorUtils.to_tensor(self.obs_normalization_stats), self.device)
            )
            prepared = ObsUtils.normalize_obs(prepared, obs_normalization_stats=stats)
        return prepared

    def prepare_goal(self, goal: dict | None) -> dict | None:
        if goal is None or not self.goal_keys:
            return None
        filtered = {k: goal[k] for k in self.goal_keys if k in goal}
        if not filtered:
            return None
        return TensorUtils.to_float(TensorUtils.to_device(TensorUtils.to_batch(TensorUtils.to_tensor(filtered)), self.device))

    def prepare_observation_batch(self, observations: list[dict]) -> dict:
        batch = {k: np.stack([obs[k] for obs in observations], axis=0) for k in self.obs_keys}
        prepared = TensorUtils.to_float(TensorUtils.to_device(TensorUtils.to_tensor(batch), self.device))
        if self.obs_normalization_stats is not None:
            stats = TensorUtils.to_float(
                TensorUtils.to_device(TensorUtils.to_tensor(self.obs_normalization_stats), self.device)
            )
            prepared = ObsUtils.normalize_obs(prepared, obs_normalization_stats=stats)
        return prepared

    def prepare_goal_batch(self, goals: list[dict | None]) -> dict | None:
        if not self.goal_keys:
            return None
        valid_goals = [goal for goal in goals if goal is not None]
        if not valid_goals:
            return None
        batch = {k: np.stack([goal[k] for goal in valid_goals], axis=0) for k in self.goal_keys}
        return TensorUtils.to_float(TensorUtils.to_device(TensorUtils.to_tensor(batch), self.device))

    def _make_det_policy_dist(self, mean_action: torch.Tensor) -> TanhWrappedDistribution:
        assert self.learned_log_std is not None
        base_dist = D.Normal(loc=_atanh(mean_action), scale=self.learned_log_std.exp().expand_as(mean_action))
        return TanhWrappedDistribution(base_dist=D.Independent(base_dist, 1), scale=1.0)

    def _dist_from_step(self, obs: dict, goal: dict | None, rnn_state: Any = None) -> tuple[D.Distribution, Any]:
        if self.is_recurrent:
            if self.uses_stochastic_head:
                dist, next_state = self.actor.forward_train_step(obs_dict=obs, goal_dict=goal, rnn_state=rnn_state)
            else:
                mean_action, next_state = self.actor.forward_step(obs_dict=obs, goal_dict=goal, rnn_state=rnn_state)
                dist = self._make_det_policy_dist(mean_action)
            return dist, next_state

        if self.uses_stochastic_head:
            dist = self.actor.forward_train(obs_dict=obs, goal_dict=goal)
        else:
            mean_action = self.actor(obs_dict=obs, goal_dict=goal)
            dist = self._make_det_policy_dist(mean_action)
        return dist, None

    def act(self, obs: dict, goal: dict | None, deterministic: bool = False, clip_actions: bool = True) -> tuple[np.ndarray, float, float]:
        with torch.no_grad():
            if deterministic:
                self.actor.eval()
            else:
                self.actor.train()
            self.value_net.eval()

            obs_batch = self.prepare_observation(obs)
            goal_batch = self.prepare_goal(goal)

            dist, next_state = self._dist_from_step(obs_batch, goal_batch, self._rollout_rnn_state)
            if deterministic:
                action = self.rollout_policy(ob=obs, goal=goal)
                action_tensor = TensorUtils.to_float(
                    TensorUtils.to_device(TensorUtils.to_tensor(action[None, ...]), self.device)
                )
            else:
                action_tensor = dist.sample()
            if clip_actions:
                action_tensor = action_tensor.clamp(-1.0, 1.0)
            log_prob = dist.log_prob(action_tensor)
            value = self.value_net(obs_batch, goal_dict=goal_batch)

            self._rollout_rnn_state = next_state
            return (
                TensorUtils.to_numpy(action_tensor[0]),
                float(log_prob[0].detach().cpu()),
                float(value[0, 0].detach().cpu()),
            )

    def evaluate_actions_full_batch(
        self,
        observations: list[dict],
        goals: list[dict | None],
        actions: torch.Tensor,
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_recurrent:
            log_probs = []
            entropies = []
            rnn_state = None
            for idx, obs in enumerate(observations):
                if bool(episode_starts[idx].item()):
                    rnn_state = None
                obs_batch = self.prepare_observation(obs)
                goal_batch = self.prepare_goal(goals[idx])
                dist, rnn_state = self._dist_from_step(obs_batch, goal_batch, rnn_state)
                action = actions[idx : idx + 1]
                log_probs.append(dist.log_prob(action))
                entropies.append(_safe_entropy(dist))
            return torch.cat(log_probs, dim=0), torch.cat(entropies, dim=0)

        obs_batch = self.prepare_observation_batch(observations)
        goal_batch = self.prepare_goal_batch(goals)
        dist, _ = self._dist_from_step(obs_batch, goal_batch)
        return dist.log_prob(actions), _safe_entropy(dist)

    def value(self, observations: list[dict], goals: list[dict | None]) -> torch.Tensor:
        obs_batch = self.prepare_observation_batch(observations)
        goal_batch = self.prepare_goal_batch(goals)
        return self.value_net(obs_batch, goal_dict=goal_batch).squeeze(-1)

    def demo_loss(self, batch: dict) -> torch.Tensor:
        self.algo.set_train()
        processed = self.algo.process_batch_for_training(batch)
        processed = self.algo.postprocess_batch_for_training(processed, obs_normalization_stats=self.obs_normalization_stats)
        predictions = self.algo._forward_training(processed)  # type: ignore[attr-defined]
        losses = self.algo._compute_losses(predictions, processed)  # type: ignore[attr-defined]
        return losses["action_loss"]

    def predict_value(self, obs: dict, goal: dict | None) -> float:
        with torch.no_grad():
            obs_batch = self.prepare_observation(obs)
            goal_batch = self.prepare_goal(goal)
            value = self.value_net(obs_batch, goal_dict=goal_batch)
            return float(value[0, 0].detach().cpu())

    def export_robomimic_checkpoint(self) -> dict:
        ckpt = deepcopy(self.bundle.ckpt_dict)
        ckpt["model"] = self.algo.serialize()
        return ckpt


def _safe_entropy(dist: D.Distribution) -> torch.Tensor:
    try:
        ent = dist.entropy()
    except NotImplementedError:
        ent = torch.zeros(tuple(int(x) for x in dist.batch_shape), device=_distribution_device(dist))
    if ent.ndim == 0:
        ent = ent.unsqueeze(0)
    if ent.ndim > 1:
        ent = ent.sum(dim=-1)
    return ent


def _config_enabled(config_node: Any, attr_name: str) -> bool:
    node = getattr(config_node, attr_name, None)
    if node is None:
        return False
    return bool(getattr(node, "enabled", False))


def _distribution_device(dist: D.Distribution) -> torch.device:
    if hasattr(dist, "base_dist") and hasattr(dist.base_dist, "loc"):
        return dist.base_dist.loc.device
    if hasattr(dist, "component_distribution"):
        component = dist.component_distribution
        if hasattr(component, "base_dist") and hasattr(component.base_dist, "loc"):
            return component.base_dist.loc.device
    return torch.device("cpu")
