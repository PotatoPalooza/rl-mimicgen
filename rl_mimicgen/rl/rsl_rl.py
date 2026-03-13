from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage

from rl_mimicgen.rl.policy import OnlinePolicyAdapter


def obs_to_tensordict(obs: dict[str, torch.Tensor] | dict[str, object], device: torch.device) -> TensorDict:
    tensor_obs = {
        key: value.to(device=device, dtype=torch.float32)
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in obs.items()
    }
    batch_size = [next(iter(tensor_obs.values())).shape[0]]
    return TensorDict(tensor_obs, batch_size=batch_size, device=device)


class RslRlRobomimicActor(nn.Module):
    is_recurrent: bool

    def __init__(self, adapter: OnlinePolicyAdapter) -> None:
        super().__init__()
        self.__dict__["adapter"] = adapter
        self.actor_net = adapter.actor
        if adapter.learned_log_std is not None:
            self.learned_log_std = adapter.learned_log_std
        self.is_recurrent = adapter.is_recurrent
        self._last_dist = None
        self._last_params: tuple[torch.Tensor, ...] | None = None

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        del masks
        prepared = self.adapter.prepare_observation_batch(dict(obs.items()))
        if self.adapter.is_recurrent:
            if hidden_state is not None:
                self.adapter._rollout_rnn_state = hidden_state
            elif self.adapter._rollout_rnn_state is None:
                batch_size = next(iter(prepared.values())).shape[0]
                self.adapter._rollout_rnn_state = self.adapter._init_rnn_state(batch_size)
        dist, next_state = self.adapter._dist_from_step(prepared, None, self.adapter._rollout_rnn_state)
        self.adapter._rollout_rnn_state = next_state
        self._last_dist = dist
        self._last_params = self._distribution_params(dist)
        if stochastic_output:
            return dist.sample()
        return self._deterministic_action(dist)

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        assert self._last_params is not None
        return self._last_params

    @property
    def output_entropy(self) -> torch.Tensor:
        assert self._last_dist is not None
        return self._safe_entropy(self._last_dist)

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        assert self._last_dist is not None
        log_prob = self._last_dist.log_prob(outputs)
        if log_prob.ndim > 1:
            log_prob = log_prob.sum(dim=-1)
        return log_prob

    def get_kl_divergence(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(old_params) == 2 and len(new_params) == 2:
            old_mean, old_std = old_params
            new_mean, new_std = new_params
            old_dist = torch.distributions.Normal(old_mean, old_std)
            new_dist = torch.distributions.Normal(new_mean, new_std)
            return torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1)
        return torch.zeros(old_params[0].shape[0], device=old_params[0].device)

    def update_normalization(self, obs: TensorDict) -> None:
        del obs

    def reset(self, dones: torch.Tensor | None = None) -> None:
        if not self.adapter.is_recurrent:
            return
        if dones is None:
            self.adapter.reset_rollout_state()
            return
        if self.adapter._rollout_rnn_state is None:
            return
        reset_mask = dones.to(device=self.adapter.device, dtype=torch.bool).view(-1)
        self.adapter._rollout_rnn_state = self.adapter._reset_state_indices(self.adapter._rollout_rnn_state, reset_mask)

    def get_hidden_state(self):
        return self.adapter._rollout_rnn_state

    def evaluate_actions_rollout(
        self,
        observations: TensorDict,
        actions: torch.Tensor,
        episode_starts: torch.Tensor,
        initial_rnn_state=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.adapter.evaluate_actions_rollout(
            observations={key: value for key, value in observations.items()},
            actions=actions,
            episode_starts=episode_starts,
            initial_rnn_state=initial_rnn_state,
        )

    @staticmethod
    def _safe_entropy(dist) -> torch.Tensor:
        try:
            ent = dist.entropy()
        except NotImplementedError:
            if hasattr(dist, "base_dist"):
                ent = dist.base_dist.entropy()
            elif hasattr(dist, "component_distribution") and hasattr(dist.component_distribution, "base_dist"):
                ent = dist.component_distribution.base_dist.entropy()
            else:
                ent = torch.zeros(
                    tuple(int(x) for x in dist.batch_shape),
                    device=RslRlRobomimicActor._distribution_device(dist),
                )
        if ent.ndim == 0:
            ent = ent.unsqueeze(0)
        if ent.ndim > 1:
            ent = ent.sum(dim=-1)
        return ent

    @staticmethod
    def _deterministic_action(dist):
        if hasattr(dist, "mean"):
            return dist.mean
        if hasattr(dist, "component_distribution") and hasattr(dist, "mixture_distribution"):
            logits = dist.mixture_distribution.logits
            mode_idx = torch.argmax(logits, dim=-1)
            means = dist.component_distribution.base_dist.loc
            gather_idx = mode_idx.view(-1, 1, 1).expand(-1, 1, means.shape[-1])
            return torch.gather(means, 1, gather_idx).squeeze(1)
        return dist.sample()

    @staticmethod
    def _distribution_params(dist) -> tuple[torch.Tensor, ...]:
        if hasattr(dist, "mean") and hasattr(dist, "stddev"):
            return dist.mean, dist.stddev
        if hasattr(dist, "mixture_distribution") and hasattr(dist, "component_distribution"):
            logits = dist.mixture_distribution.logits
            component = dist.component_distribution
            if hasattr(component, "base_dist") and hasattr(component.base_dist, "loc") and hasattr(component.base_dist, "scale"):
                return logits, component.base_dist.loc, component.base_dist.scale
        raise NotImplementedError(f"Unsupported distribution type for parameter export: {type(dist)}")

    @staticmethod
    def _distribution_device(dist) -> torch.device:
        if hasattr(dist, "base_dist") and hasattr(dist.base_dist, "loc"):
            return dist.base_dist.loc.device
        if hasattr(dist, "component_distribution"):
            component = dist.component_distribution
            if hasattr(component, "base_dist") and hasattr(component.base_dist, "loc"):
                return component.base_dist.loc.device
        return torch.device("cpu")


class RslRlValueCritic(nn.Module):
    is_recurrent: bool = False

    def __init__(self, adapter: OnlinePolicyAdapter) -> None:
        super().__init__()
        self.__dict__["adapter"] = adapter
        self.value_net = adapter.value_net

    def forward(self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state=None) -> torch.Tensor:
        del masks, hidden_state
        prepared = self.adapter.prepare_observation_batch(dict(obs.items()))
        return self.adapter.value_net(prepared, goal_dict=None)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def get_hidden_state(self):
        return None

    def update_normalization(self, obs: TensorDict) -> None:
        del obs

    def evaluate_rollout(self, observations: TensorDict) -> torch.Tensor:
        return self.adapter.value({key: value for key, value in observations.items()})


class DemoAugmentedPPO(PPO):
    def __init__(
        self,
        actor: RslRlRobomimicActor,
        critic: RslRlValueCritic,
        storage: RolloutStorage,
        demo_batch_iterator: Iterator[dict] | None = None,
        demo_loss_fn=None,
        demo_coef: float = 0.0,
        actor_learning_rate: float | None = None,
        critic_learning_rate: float | None = None,
        critic_warmup_updates: int = 0,
        actor_freeze_env_steps: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(actor=actor, critic=critic, storage=storage, **kwargs)
        self.demo_batch_iterator = demo_batch_iterator
        self.demo_loss_fn = demo_loss_fn
        self.demo_coef = demo_coef
        self.actor_learning_rate = actor_learning_rate if actor_learning_rate is not None else self.learning_rate
        self.critic_learning_rate = critic_learning_rate if critic_learning_rate is not None else self.learning_rate
        self.critic_warmup_updates = max(0, int(critic_warmup_updates))
        self.actor_freeze_env_steps = max(0, int(actor_freeze_env_steps))
        self.total_env_steps = 0
        self._update_step = 0

        optimizer_defaults = dict(self.optimizer.defaults)
        optimizer_defaults.pop("lr", None)
        optimizer_cls = type(self.optimizer)
        self.optimizer = optimizer_cls(
            [
                {"params": list(self.actor.parameters()), "lr": self.actor_learning_rate},
                {"params": list(self.critic.parameters()), "lr": self.critic_learning_rate},
            ],
            lr=self.actor_learning_rate,
            **optimizer_defaults,
        )

    def set_total_env_steps(self, total_env_steps: int) -> None:
        self.total_env_steps = max(0, int(total_env_steps))

    def _set_learning_rates(self) -> None:
        actor_lr = self.actor_learning_rate
        if self._update_step < self.critic_warmup_updates or self.total_env_steps <= self.actor_freeze_env_steps:
            actor_lr = 0.0
        self.optimizer.param_groups[0]["lr"] = actor_lr
        self.optimizer.param_groups[1]["lr"] = self.critic_learning_rate

    def update(self) -> dict[str, float]:
        self._set_learning_rates()
        rollout_steps = self.storage.num_transitions_per_env
        num_envs = self.storage.num_envs
        batch_size = rollout_steps * num_envs
        minibatch_size = max(1, batch_size // self.num_mini_batches)

        observations = self.storage.observations
        actions = self.storage.actions
        old_log_probs = self.storage.actions_log_prob.reshape(-1)
        returns = self.storage.returns.reshape(-1)
        advantages = self.storage.advantages.reshape(-1)
        if not self.normalize_advantage_per_mini_batch:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        episode_starts = torch.zeros(rollout_steps, num_envs, device=self.device, dtype=torch.bool)
        episode_starts[1:] = self.storage.dones[:-1, :, 0].bool()
        initial_rnn_state = None
        if self.actor.is_recurrent and self.storage.saved_hidden_state_a is not None:
            initial_rnn_state = [saved_state[0].clone() for saved_state in self.storage.saved_hidden_state_a]
            if len(initial_rnn_state) == 1:
                initial_rnn_state = initial_rnn_state[0]
            else:
                initial_rnn_state = tuple(initial_rnn_state)

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_demo_loss = 0.0

        for _ in range(self.num_learning_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                batch_idx = indices[start : start + minibatch_size]

                current_log_probs, entropy = self.actor.evaluate_actions_rollout(
                    observations,
                    actions,
                    episode_starts,
                    initial_rnn_state=initial_rnn_state,
                )
                current_values = self.critic.evaluate_rollout(observations)

                batch_log_probs = current_log_probs[batch_idx]
                batch_entropy = entropy[batch_idx]
                batch_values = current_values[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                ratio = torch.exp(batch_log_probs - batch_old_log_probs)
                surrogate = -batch_advantages * ratio
                surrogate_clipped = -batch_advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                value_loss = (batch_returns - batch_values).pow(2).mean()

                demo_loss = torch.zeros((), device=self.device)
                if self.demo_batch_iterator is not None and self.demo_loss_fn is not None and self.demo_coef > 0.0:
                    try:
                        demo_batch = next(self.demo_batch_iterator)
                    except StopIteration:
                        raise RuntimeError("Demo batch iterator was exhausted unexpectedly.")
                    demo_loss = self.demo_loss_fn(demo_batch)

                loss = (
                    surrogate_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * batch_entropy.mean()
                    + self.demo_coef * demo_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_entropy += batch_entropy.mean().item()
                mean_demo_loss += demo_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        self._update_step += 1

        return {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "demo": mean_demo_loss / num_updates,
        }
