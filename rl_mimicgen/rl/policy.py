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
    """Wrap a robomimic BC policy so it can be fine-tuned with PPO updates."""

    def __init__(
        self,
        bundle: PolicyBundle,
        device: torch.device,
        value_hidden_sizes: tuple[int, ...] = (256, 256),
        init_log_std: float = -0.5,
        min_log_std: float = -2.0,
        residual_enabled: bool = False,
        residual_scale: float = 0.2,
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
        self.rnn_horizon = int(getattr(bundle.config.algo.rnn, "horizon", 0)) if self.is_recurrent else 0
        self.residual_enabled = bool(residual_enabled)
        self.residual_scale = float(residual_scale)

        self.value_net = ValueNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            mlp_layer_dims=list(value_hidden_sizes),
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.algo.obs_config.encoder),
        ).float().to(device)

        self.learned_log_std = None
        self.min_log_std = float(min_log_std)
        if not self.uses_stochastic_head:
            self.learned_log_std = nn.Parameter(torch.full((bundle.shape_meta["ac_dim"],), init_log_std, device=device))

        self.base_actor = None
        if self.residual_enabled:
            self.base_actor = deepcopy(self.actor).float().to(device)
            self.base_actor.eval()
            for param in self.base_actor.parameters():
                param.requires_grad_(False)

        self._rollout_rnn_state = None
        self._base_rollout_rnn_state = None
        self._rollout_step_count = 0
        self._base_rollout_step_count = 0

    def actor_parameters(self) -> list[nn.Parameter]:
        params = list(self.actor.parameters())
        if self.learned_log_std is not None:
            params.append(self.learned_log_std)
        return params

    def reset_rollout_state(self) -> None:
        self._rollout_rnn_state = None
        self._base_rollout_rnn_state = None
        self._rollout_step_count = 0
        self._base_rollout_step_count = 0
        self.algo.reset()

    def clone_rollout_state(self) -> Any:
        if self.residual_enabled:
            return (
                _clone_rnn_state(self._rollout_rnn_state),
                _clone_rnn_state(self._base_rollout_rnn_state),
                self._rollout_step_count,
                self._base_rollout_step_count,
            )
        return _clone_rnn_state(self._rollout_rnn_state)

    def restore_rollout_state(self, state: Any) -> None:
        if self.residual_enabled:
            residual_state, base_state, rollout_step_count, base_rollout_step_count = state
            self._rollout_rnn_state = _clone_rnn_state(residual_state)
            self._base_rollout_rnn_state = _clone_rnn_state(base_state)
            self._rollout_step_count = int(rollout_step_count)
            self._base_rollout_step_count = int(base_rollout_step_count)
            return
        self._rollout_rnn_state = _clone_rnn_state(state)

    def clone_training_rollout_state(self) -> Any:
        return {
            "hidden_state": _clone_rnn_state(self._rollout_rnn_state),
            "step_count": int(self._rollout_step_count),
        }

    def clone_hidden_state(self, state: Any) -> Any:
        return _clone_rnn_state(state)

    def _normalize_obs(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.obs_normalization_stats is None:
            return obs
        stats = TensorUtils.to_float(TensorUtils.to_device(TensorUtils.to_tensor(self.obs_normalization_stats), self.device))
        return ObsUtils.normalize_obs(obs, obs_normalization_stats=stats)

    def prepare_observation_batch(self, obs: dict[str, np.ndarray] | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        filtered = {k: obs[k] for k in self.obs_keys if k in obs}
        prepared = TensorUtils.to_float(TensorUtils.to_device(TensorUtils.to_tensor(filtered), self.device))
        return self._normalize_obs(prepared)

    def prepare_goal_batch(self, goal: dict[str, np.ndarray] | dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor] | None:
        if goal is None or not self.goal_keys:
            return None
        filtered = {k: goal[k] for k in self.goal_keys if k in goal}
        if not filtered:
            return None
        return TensorUtils.to_float(TensorUtils.to_device(TensorUtils.to_tensor(filtered), self.device))

    def _make_det_policy_dist(self, mean_action: torch.Tensor) -> TanhWrappedDistribution:
        assert self.learned_log_std is not None
        log_std = self.learned_log_std.clamp(min=self.min_log_std)
        base_dist = D.Normal(loc=_atanh(mean_action), scale=log_std.exp().expand_as(mean_action))
        return TanhWrappedDistribution(base_dist=D.Independent(base_dist, 1), scale=1.0)

    def current_log_std(self) -> torch.Tensor | None:
        if self.learned_log_std is None:
            return None
        return self.learned_log_std.detach().clamp(min=self.min_log_std)

    def _dist_from_step(self, obs: dict[str, torch.Tensor], goal: dict[str, torch.Tensor] | None, rnn_state: Any = None):
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

    def _dist_from_actor(
        self,
        actor_module: nn.Module,
        obs: dict[str, torch.Tensor],
        goal: dict[str, torch.Tensor] | None,
        rnn_state: Any = None,
    ):
        if self.is_recurrent:
            if self.uses_stochastic_head:
                dist, next_state = actor_module.forward_train_step(obs_dict=obs, goal_dict=goal, rnn_state=rnn_state)
            else:
                mean_action, next_state = actor_module.forward_step(obs_dict=obs, goal_dict=goal, rnn_state=rnn_state)
                dist = self._make_det_policy_dist(mean_action)
            return dist, next_state

        if self.uses_stochastic_head:
            dist = actor_module.forward_train(obs_dict=obs, goal_dict=goal)
        else:
            mean_action = actor_module(obs_dict=obs, goal_dict=goal)
            dist = self._make_det_policy_dist(mean_action)
        return dist, None

    def _init_rnn_state(self, batch_size: int) -> Any:
        return self.actor.get_rnn_init_state(batch_size=batch_size, device=self.device)

    def _reset_state_indices(self, rnn_state: Any, reset_mask: torch.Tensor) -> Any:
        if rnn_state is None or not bool(reset_mask.any().item()):
            return rnn_state
        if isinstance(rnn_state, tuple):
            return tuple(self._reset_state_indices(state, reset_mask) for state in rnn_state)
        state = rnn_state.clone()
        state[:, reset_mask] = 0
        return state

    def _base_action_step(
        self,
        obs: dict[str, torch.Tensor],
        goal: dict[str, torch.Tensor] | None,
        episode_starts: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        assert self.base_actor is not None
        batch_size = next(iter(obs.values())).shape[0]
        if self.is_recurrent:
            if self._base_rollout_rnn_state is None or (self.rnn_horizon > 0 and self._base_rollout_step_count % self.rnn_horizon == 0):
                self._base_rollout_rnn_state = self._init_rnn_state(batch_size=batch_size)
            reset_mask = torch.as_tensor(episode_starts, device=self.device, dtype=torch.bool)
            self._base_rollout_rnn_state = self._reset_state_indices(self._base_rollout_rnn_state, reset_mask)
        base_dist, next_state = self._dist_from_actor(self.base_actor, obs, goal, self._base_rollout_rnn_state)
        self._base_rollout_rnn_state = next_state
        self._base_rollout_step_count += 1
        return _distribution_mode(base_dist).detach()

    def _predict_actor_actions(
        self,
        actor_module: nn.Module,
        obs_batch: dict[str, torch.Tensor],
        goal_batch: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if self.is_recurrent:
            if self.uses_stochastic_head:
                dist = actor_module.forward_train(obs_dict=obs_batch, goal_dict=goal_batch)
                return _distribution_mode(dist)
            return actor_module.forward(obs_dict=obs_batch, goal_dict=goal_batch)
        if self.uses_stochastic_head:
            dist = actor_module.forward_train(obs_dict=obs_batch, goal_dict=goal_batch)
            return _distribution_mode(dist)
        return actor_module(obs_dict=obs_batch, goal_dict=goal_batch)

    def act(self, obs: dict[str, np.ndarray], goal: dict[str, np.ndarray] | None, episode_starts: np.ndarray, clip_actions: bool = True):
        with torch.no_grad():
            # PPO rollouts must come from the same train-time distribution family
            # that PPO later replays against; eval mode can enable low_noise_eval.
            self.actor.train()
            self.value_net.eval()

            obs_batch = self.prepare_observation_batch(obs)
            goal_batch = self.prepare_goal_batch(goal)
            batch_size = next(iter(obs_batch.values())).shape[0]

            if self.is_recurrent:
                if self._rollout_rnn_state is None or (self.rnn_horizon > 0 and self._rollout_step_count % self.rnn_horizon == 0):
                    self._rollout_rnn_state = self._init_rnn_state(batch_size=batch_size)
                reset_mask = torch.as_tensor(episode_starts, device=self.device, dtype=torch.bool)
                self._rollout_rnn_state = self._reset_state_indices(self._rollout_rnn_state, reset_mask)

            dist, next_state = self._dist_from_step(obs_batch, goal_batch, self._rollout_rnn_state)
            policy_action_tensor = dist.sample()
            action_tensor = policy_action_tensor
            if self.residual_enabled:
                action_tensor = self._base_action_step(obs_batch, goal_batch, episode_starts) + (self.residual_scale * policy_action_tensor)
            if clip_actions:
                action_tensor = action_tensor.clamp(-1.0, 1.0)
            log_prob = dist.log_prob(policy_action_tensor)
            value = self.value_net(obs_batch, goal_dict=goal_batch).squeeze(-1)

            self._rollout_rnn_state = next_state
            self._rollout_step_count += 1
            return (
                TensorUtils.to_numpy(action_tensor),
                TensorUtils.to_numpy(policy_action_tensor),
                TensorUtils.to_numpy(log_prob),
                TensorUtils.to_numpy(value),
            )

    def act_deterministic(
        self,
        obs: dict[str, np.ndarray],
        goal: dict[str, np.ndarray] | None,
        episode_starts: np.ndarray,
        clip_actions: bool = True,
    ) -> np.ndarray:
        with torch.no_grad():
            self.actor.eval()
            self.value_net.eval()

            obs_batch = self.prepare_observation_batch(obs)
            goal_batch = self.prepare_goal_batch(goal)
            batch_size = next(iter(obs_batch.values())).shape[0]

            if self.is_recurrent:
                if self._rollout_rnn_state is None or (self.rnn_horizon > 0 and self._rollout_step_count % self.rnn_horizon == 0):
                    self._rollout_rnn_state = self._init_rnn_state(batch_size=batch_size)
                reset_mask = torch.as_tensor(episode_starts, device=self.device, dtype=torch.bool)
                self._rollout_rnn_state = self._reset_state_indices(self._rollout_rnn_state, reset_mask)

            dist, next_state = self._dist_from_step(obs_batch, goal_batch, self._rollout_rnn_state)
            # For robomimic stochastic BC policies, eval mode already applies
            # low-noise action generation. Sampling here preserves that behavior.
            action_tensor = dist.sample() if self.uses_stochastic_head else _distribution_mode(dist)
            if self.residual_enabled:
                action_tensor = self._base_action_step(obs_batch, goal_batch, episode_starts) + (self.residual_scale * action_tensor)
            if clip_actions:
                action_tensor = action_tensor.clamp(-1.0, 1.0)

            self._rollout_rnn_state = next_state
            self._rollout_step_count += 1
            return TensorUtils.to_numpy(action_tensor)

    def evaluate_actions_rollout(
        self,
        observations: dict[str, torch.Tensor],
        goals: dict[str, torch.Tensor] | None,
        actions: torch.Tensor,
        episode_starts: torch.Tensor,
        initial_rnn_state: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rollout_steps, num_envs = next(iter(observations.values())).shape[:2]
        step_count = 0
        if isinstance(initial_rnn_state, dict):
            step_count = int(initial_rnn_state.get("step_count", 0))
            initial_rnn_state = initial_rnn_state.get("hidden_state")
        if self.is_recurrent:
            log_probs = []
            entropies = []
            rnn_state = self.clone_hidden_state(initial_rnn_state)
            for step in range(rollout_steps):
                if rnn_state is None or (self.rnn_horizon > 0 and step_count % self.rnn_horizon == 0):
                    rnn_state = self._init_rnn_state(batch_size=num_envs)
                reset_mask = episode_starts[step].to(dtype=torch.bool)
                rnn_state = self._reset_state_indices(rnn_state, reset_mask)
                obs_batch = {key: value[step] for key, value in observations.items()}
                goal_batch = None if goals is None else {key: value[step] for key, value in goals.items()}
                dist, rnn_state = self._dist_from_step(obs_batch, goal_batch, rnn_state)
                log_probs.append(dist.log_prob(actions[step]))
                entropies.append(_safe_entropy(dist))
                step_count += 1
            return torch.stack(log_probs, dim=0).reshape(-1), torch.stack(entropies, dim=0).reshape(-1)

        flat_obs = {key: value.reshape(-1, *value.shape[2:]) for key, value in observations.items()}
        flat_goals = None if goals is None else {key: value.reshape(-1, *value.shape[2:]) for key, value in goals.items()}
        dist, _ = self._dist_from_step(flat_obs, flat_goals)
        flat_actions = actions.reshape(-1, actions.shape[-1])
        return dist.log_prob(flat_actions), _safe_entropy(dist)

    def value(self, observations: dict[str, torch.Tensor], goals: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        flat_obs = {key: value.reshape(-1, *value.shape[2:]) for key, value in observations.items()}
        flat_goals = None if goals is None else {key: value.reshape(-1, *value.shape[2:]) for key, value in goals.items()}
        return self.value_net(flat_obs, goal_dict=flat_goals).squeeze(-1)

    def predict_value(self, obs: dict[str, np.ndarray], goal: dict[str, np.ndarray] | None) -> np.ndarray:
        with torch.no_grad():
            obs_batch = self.prepare_observation_batch(obs)
            goal_batch = self.prepare_goal_batch(goal)
            value = self.value_net(obs_batch, goal_dict=goal_batch).squeeze(-1)
            return TensorUtils.to_numpy(value)

    def demo_loss(self, batch: dict) -> torch.Tensor:
        if self.residual_enabled:
            return self._residual_demo_loss(batch)
        return self._standard_demo_loss(batch)

    def export_robomimic_checkpoint(self) -> dict:
        if self.residual_enabled:
            raise RuntimeError("Residual policies are not exportable as plain robomimic checkpoints.")
        ckpt = deepcopy(self.bundle.ckpt_dict)
        ckpt["model"] = self.algo.serialize()
        return ckpt

    def export_policy_artifact(self, checkpoint_path: str) -> dict:
        if not self.residual_enabled:
            return self.export_robomimic_checkpoint()
        return {
            "mode": "residual",
            "base_checkpoint_path": checkpoint_path,
            "base_checkpoint": deepcopy(self.bundle.ckpt_dict),
            "residual_actor_state": self.actor.state_dict(),
            "residual_log_std": None if self.learned_log_std is None else self.learned_log_std.detach().cpu(),
            "residual_scale": self.residual_scale,
        }

    def policy_artifact_extension(self) -> str:
        return ".pt" if self.residual_enabled else ".pth"

    def _residual_demo_loss(self, batch: dict) -> torch.Tensor:
        processed = self.algo.process_batch_for_training(batch)
        processed = self.algo.postprocess_batch_for_training(processed, obs_normalization_stats=self.obs_normalization_stats)
        obs_batch = processed["obs"]
        goal_batch = processed.get("goal_obs", None)
        target_actions = processed["actions"]
        assert self.base_actor is not None
        residual_actions = self._predict_actor_actions(self.actor, obs_batch, goal_batch)
        with torch.no_grad():
            base_actions = self._predict_actor_actions(self.base_actor, obs_batch, goal_batch)
        predicted_actions = torch.clamp(base_actions + (self.residual_scale * residual_actions), -1.0, 1.0)
        return torch.mean((predicted_actions - target_actions) ** 2)

    def _standard_demo_loss(self, batch: dict) -> torch.Tensor:
        processed = self.algo.process_batch_for_training(batch)
        processed = self.algo.postprocess_batch_for_training(processed, obs_normalization_stats=self.obs_normalization_stats)
        obs_batch = processed["obs"]
        goal_batch = processed.get("goal_obs", None)
        target_actions = processed["actions"]

        if self.uses_stochastic_head:
            dist = self.actor.forward_train(obs_dict=obs_batch, goal_dict=goal_batch)
            return -dist.log_prob(target_actions).mean()

        predicted_actions = self.actor.forward(obs_dict=obs_batch, goal_dict=goal_batch)
        return torch.mean((predicted_actions - target_actions) ** 2)


def _safe_entropy(dist: D.Distribution) -> torch.Tensor:
    try:
        ent = dist.entropy()
    except NotImplementedError:
        if hasattr(dist, "base_dist"):
            ent = dist.base_dist.entropy()
        elif hasattr(dist, "component_distribution") and hasattr(dist.component_distribution, "base_dist"):
            ent = dist.component_distribution.base_dist.entropy()
        else:
            ent = torch.zeros(tuple(int(x) for x in dist.batch_shape), device=_distribution_device(dist))
    if ent.ndim == 0:
        ent = ent.unsqueeze(0)
    if ent.ndim > 1:
        ent = ent.sum(dim=-1)
    return ent


def _distribution_mode(dist: D.Distribution) -> torch.Tensor:
    if isinstance(dist, TanhWrappedDistribution):
        return torch.tanh(dist.base_dist.mean) * dist.scale
    if hasattr(dist, "mean"):
        return dist.mean
    if hasattr(dist, "component_distribution") and hasattr(dist, "mixture_distribution"):
        logits = dist.mixture_distribution.logits
        mode_idx = torch.argmax(logits, dim=-1)
        means = dist.component_distribution.base_dist.loc
        gather_idx = mode_idx.view(-1, 1, 1).expand(-1, 1, means.shape[-1])
        return torch.gather(means, 1, gather_idx).squeeze(1)
    return dist.sample()


def _clone_rnn_state(state: Any) -> Any:
    if state is None:
        return None
    if isinstance(state, tuple):
        return tuple(_clone_rnn_state(value) for value in state)
    if torch.is_tensor(state):
        return state.detach().clone()
    return deepcopy(state)


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
