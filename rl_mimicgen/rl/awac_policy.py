from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.models.value_nets import ActionValueNetwork

from rl_mimicgen.rl.policy import OnlinePolicyAdapter


class AWACPolicyAdapter(OnlinePolicyAdapter):
    def __init__(
        self,
        *args,
        q_hidden_sizes: tuple[int, ...] = (256, 256),
        num_q_networks: int = 2,
        target_tau: float = 0.005,
        num_value_action_samples: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.residual_enabled:
            raise ValueError("Residual fine-tuning is not supported for replay-based AWAC.")

        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.algo.obs_config.encoder)
        self.q_networks = nn.ModuleList(
            [
                ActionValueNetwork(
                    obs_shapes=self.obs_shapes,
                    ac_dim=self.bundle.shape_meta["ac_dim"],
                    mlp_layer_dims=list(q_hidden_sizes),
                    goal_shapes=self.goal_shapes,
                    encoder_kwargs=encoder_kwargs,
                )
                for _ in range(max(1, int(num_q_networks)))
            ]
        ).float().to(self.device)
        self.target_q_networks = nn.ModuleList(
            [
                ActionValueNetwork(
                    obs_shapes=self.obs_shapes,
                    ac_dim=self.bundle.shape_meta["ac_dim"],
                    mlp_layer_dims=list(q_hidden_sizes),
                    goal_shapes=self.goal_shapes,
                    encoder_kwargs=encoder_kwargs,
                )
                for _ in range(max(1, int(num_q_networks)))
            ]
        ).float().to(self.device)
        self.target_tau = float(target_tau)
        self.num_value_action_samples = max(1, int(num_value_action_samples))

        with torch.no_grad():
            for q_net, target_q_net in zip(self.q_networks, self.target_q_networks):
                TorchUtils.hard_update(source=q_net, target=target_q_net)

    def q_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for q_net in self.q_networks:
            params.extend(list(q_net.parameters()))
        return params

    def q_values(
        self,
        observations: dict[str, torch.Tensor],
        actions: torch.Tensor,
        goals: dict[str, torch.Tensor] | None = None,
        use_target: bool = False,
    ) -> list[torch.Tensor]:
        q_networks = self.target_q_networks if use_target else self.q_networks
        return [q_net(obs_dict=observations, acts=actions, goal_dict=goals).squeeze(-1) for q_net in q_networks]

    def min_q_value(
        self,
        observations: dict[str, torch.Tensor],
        actions: torch.Tensor,
        goals: dict[str, torch.Tensor] | None = None,
        use_target: bool = False,
    ) -> torch.Tensor:
        q_values = self.q_values(observations=observations, actions=actions, goals=goals, use_target=use_target)
        return torch.stack(q_values, dim=0).min(dim=0)[0]

    def soft_update_targets(self) -> None:
        with torch.no_grad():
            for q_net, target_q_net in zip(self.q_networks, self.target_q_networks):
                TorchUtils.soft_update(source=q_net, target=target_q_net, tau=self.target_tau)

    def actor_log_prob_replay(
        self,
        observations: dict[str, torch.Tensor],
        goals: dict[str, torch.Tensor] | None,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_recurrent:
            seq_obs = {key: value.unsqueeze(0) for key, value in observations.items()}
            seq_goals = None if goals is None else {key: value.unsqueeze(0) for key, value in goals.items()}
            seq_actions = actions.unsqueeze(0)
            episode_starts = torch.ones((1, actions.shape[0]), dtype=torch.bool, device=actions.device)
            return self.evaluate_actions_rollout(
                observations=seq_obs,
                goals=seq_goals,
                actions=seq_actions,
                episode_starts=episode_starts,
                initial_rnn_state=None,
            )

        dist, _ = self._dist_from_step(observations, goals, None)
        return dist.log_prob(actions), _safe_entropy(dist)

    def actor_log_prob_replay_sequence(
        self,
        observations: dict[str, torch.Tensor],
        goals: dict[str, torch.Tensor] | None,
        actions: torch.Tensor,
        episode_starts: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_recurrent:
            flat_mask = mask.reshape(-1)
            flat_obs = {
                key: value.reshape(-1, *value.shape[2:])[flat_mask]
                for key, value in observations.items()
            }
            flat_goals = None
            if goals is not None:
                flat_goals = {
                    key: value.reshape(-1, *value.shape[2:])[flat_mask]
                    for key, value in goals.items()
                }
            flat_actions = actions.reshape(-1, actions.shape[-1])[flat_mask]
            return self.actor_log_prob_replay(flat_obs, flat_goals, flat_actions)

        log_probs, entropy = self.evaluate_actions_rollout(
            observations=observations,
            goals=goals,
            actions=actions,
            episode_starts=episode_starts,
            initial_rnn_state=None,
        )
        flat_mask = mask.reshape(-1)
        return log_probs[flat_mask], entropy[flat_mask]

    def behavior_kl_replay(
        self,
        observations: dict[str, torch.Tensor],
        goals: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_dist, _ = self._dist_from_step(observations, goals, None)
        with torch.no_grad():
            behavior_dist, _ = self._dist_from_actor(
                self.behavior_actor,
                observations,
                goals,
                None,
                log_std_override=self.behavior_log_std,
            )
            behavior_actions = _sample_from_distribution(behavior_dist)
            behavior_log_probs = behavior_dist.log_prob(behavior_actions)
        current_log_probs = current_dist.log_prob(behavior_actions)
        return behavior_log_probs - current_log_probs, _safe_entropy(current_dist)

    def behavior_kl_replay_sequence(
        self,
        observations: dict[str, torch.Tensor],
        goals: dict[str, torch.Tensor] | None,
        episode_starts: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rollout_steps, num_envs = next(iter(observations.values())).shape[:2]
        current_state = None
        behavior_state = None
        step_count = 0
        kl_values = []
        entropies = []
        for step in range(rollout_steps):
            if current_state is None or (self.rnn_horizon > 0 and step_count % self.rnn_horizon == 0):
                current_state = self._init_rnn_state(batch_size=num_envs)
                behavior_state = self._init_rnn_state(batch_size=num_envs)
            reset_mask = episode_starts[step].to(dtype=torch.bool)
            current_state = self._reset_state_indices(current_state, reset_mask)
            behavior_state = self._reset_state_indices(behavior_state, reset_mask)
            obs_batch = {key: value[step] for key, value in observations.items()}
            goal_batch = None if goals is None else {key: value[step] for key, value in goals.items()}
            current_dist, current_state = self._dist_from_step(obs_batch, goal_batch, current_state)
            with torch.no_grad():
                behavior_dist, behavior_state = self._dist_from_actor(
                    self.behavior_actor,
                    obs_batch,
                    goal_batch,
                    behavior_state,
                    log_std_override=self.behavior_log_std,
                )
                behavior_actions = _sample_from_distribution(behavior_dist)
                behavior_log_probs = behavior_dist.log_prob(behavior_actions)
            current_log_probs = current_dist.log_prob(behavior_actions)
            kl_values.append(behavior_log_probs - current_log_probs)
            entropies.append(_safe_entropy(current_dist))
            step_count += 1

        flat_mask = mask.reshape(-1)
        flat_kl = _flatten_metric(torch.stack(kl_values, dim=0), rollout_steps, num_envs)
        flat_entropy = _flatten_metric(torch.stack(entropies, dim=0), rollout_steps, num_envs)
        return flat_kl[flat_mask], flat_entropy[flat_mask]

    def sample_actions_for_value(
        self,
        observations: dict[str, torch.Tensor],
        goals: dict[str, torch.Tensor] | None,
        num_samples: int | None = None,
    ) -> torch.Tensor:
        sample_count = self.num_value_action_samples if num_samples is None else max(1, int(num_samples))
        batch_size = next(iter(observations.values())).shape[0]
        expanded_obs = {
            key: value.unsqueeze(1).expand(-1, sample_count, *value.shape[1:]).reshape(batch_size * sample_count, *value.shape[1:])
            for key, value in observations.items()
        }
        expanded_goals = None
        if goals is not None:
            expanded_goals = {
                key: value.unsqueeze(1).expand(-1, sample_count, *value.shape[1:]).reshape(batch_size * sample_count, *value.shape[1:])
                for key, value in goals.items()
            }

        if self.is_recurrent:
            rnn_state = self._init_rnn_state(batch_size=batch_size * sample_count)
            dist, _ = self._dist_from_step(expanded_obs, expanded_goals, rnn_state)
        else:
            dist, _ = self._dist_from_step(expanded_obs, expanded_goals, None)

        sampled_actions = dist.sample()
        return sampled_actions.view(batch_size, sample_count, -1)

    def approximate_v_value(
        self,
        observations: dict[str, torch.Tensor],
        goals: dict[str, torch.Tensor] | None = None,
        use_target: bool = False,
        num_action_samples: int | None = None,
    ) -> torch.Tensor:
        sample_count = self.num_value_action_samples if num_action_samples is None else max(1, int(num_action_samples))
        sampled_actions = self.sample_actions_for_value(observations=observations, goals=goals, num_samples=sample_count)
        batch_size = sampled_actions.shape[0]
        flat_actions = sampled_actions.reshape(batch_size * sample_count, -1)
        expanded_obs = {
            key: value.unsqueeze(1).expand(-1, sample_count, *value.shape[1:]).reshape(batch_size * sample_count, *value.shape[1:])
            for key, value in observations.items()
        }
        expanded_goals = None
        if goals is not None:
            expanded_goals = {
                key: value.unsqueeze(1).expand(-1, sample_count, *value.shape[1:]).reshape(batch_size * sample_count, *value.shape[1:])
                for key, value in goals.items()
            }
        q_values = self.min_q_value(
            observations=expanded_obs,
            actions=flat_actions,
            goals=expanded_goals,
            use_target=use_target,
        )
        return q_values.view(batch_size, sample_count).mean(dim=1)

    def export_policy_artifact(self, checkpoint_path: str) -> dict:
        artifact = super().export_policy_artifact(checkpoint_path)
        if isinstance(artifact, dict):
            artifact = deepcopy(artifact)
            artifact["awac_q_networks"] = [q_net.state_dict() for q_net in self.q_networks]
        return artifact


def _safe_entropy(dist) -> torch.Tensor:
    try:
        entropy = dist.entropy()
    except NotImplementedError:
        if hasattr(dist, "base_dist"):
            entropy = dist.base_dist.entropy()
        elif hasattr(dist, "component_distribution") and hasattr(dist.component_distribution, "base_dist"):
            entropy = dist.component_distribution.base_dist.entropy()
        else:
            entropy = torch.zeros(tuple(int(x) for x in dist.batch_shape), device=_distribution_device(dist))
    if entropy.ndim == 0:
        return entropy.unsqueeze(0)
    if entropy.ndim > 1:
        entropy = entropy.sum(dim=-1)
    return entropy


def _flatten_metric(metric: torch.Tensor, rollout_steps: int, num_envs: int) -> torch.Tensor:
    if metric.ndim == 0:
        return metric.unsqueeze(0)
    if metric.ndim == 1:
        return metric
    if metric.shape[0] != rollout_steps or metric.shape[1] != num_envs:
        metric = metric.reshape(rollout_steps, num_envs, *metric.shape[1:])
    while metric.ndim > 2:
        metric = metric.mean(dim=-1)
    return metric.reshape(-1)


def _sample_from_distribution(dist) -> torch.Tensor:
    if hasattr(dist, "rsample"):
        try:
            return dist.rsample()
        except NotImplementedError:
            pass
    return dist.sample()


def _distribution_device(dist) -> torch.device:
    if hasattr(dist, "base_dist") and hasattr(dist.base_dist, "loc"):
        return dist.base_dist.loc.device
    if hasattr(dist, "component_distribution"):
        component = dist.component_distribution
        if hasattr(component, "base_dist") and hasattr(component.base_dist, "loc"):
            return component.base_dist.loc.device
    return torch.device("cpu")
