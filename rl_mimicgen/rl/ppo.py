from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn

from rl_mimicgen.rl.policy import OnlinePolicyAdapter
from rl_mimicgen.rl.storage import RolloutBatch


class DemoAugmentedPPO:
    def __init__(
        self,
        policy: OnlinePolicyAdapter,
        demo_batch_iterator: Iterator[dict] | None = None,
        demo_loss_fn=None,
        demo_coef: float = 0.0,
        actor_learning_rate: float = 1e-4,
        critic_learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        critic_warmup_updates: int = 0,
        actor_freeze_env_steps: int = 0,
        num_learning_epochs: int = 4,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        target_kl: float | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.policy = policy
        self.demo_batch_iterator = demo_batch_iterator
        self.demo_loss_fn = demo_loss_fn
        self.demo_coef = float(demo_coef)
        self.actor_learning_rate = float(actor_learning_rate)
        self.critic_learning_rate = float(critic_learning_rate)
        self.critic_warmup_updates = max(0, int(critic_warmup_updates))
        self.actor_freeze_env_steps = max(0, int(actor_freeze_env_steps))
        self.num_learning_epochs = max(1, int(num_learning_epochs))
        self.num_mini_batches = max(1, int(num_mini_batches))
        self.clip_param = float(clip_param)
        self.value_loss_coef = float(value_loss_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.target_kl = None if target_kl is None else float(target_kl)
        self.device = torch.device(device)
        self.total_env_steps = 0
        self.update_step = 0

        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": list(self.policy.actor_parameters()),
                    "lr": self.actor_learning_rate,
                    "weight_decay": weight_decay,
                },
                {
                    "params": list(self.policy.value_net.parameters()),
                    "lr": self.critic_learning_rate,
                    "weight_decay": weight_decay,
                },
            ]
        )

    def train_mode(self) -> None:
        self.policy.train()
        self.policy.actor.train()
        self.policy.value_net.train()

    def set_total_env_steps(self, total_env_steps: int) -> None:
        self.total_env_steps = max(0, int(total_env_steps))

    def save(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer.state_dict(),
            "total_env_steps": self.total_env_steps,
            "update_step": self.update_step,
        }

    def load(self, state_dict: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.total_env_steps = int(state_dict.get("total_env_steps", 0))
        self.update_step = int(state_dict.get("update_step", 0))

    def _set_learning_rates(self) -> None:
        actor_lr = self.actor_learning_rate
        if self.update_step < self.critic_warmup_updates or self.total_env_steps <= self.actor_freeze_env_steps:
            actor_lr = 0.0
        self.optimizer.param_groups[0]["lr"] = actor_lr
        self.optimizer.param_groups[1]["lr"] = self.critic_learning_rate

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        batch = _detach_rollout_batch(batch)
        self.train_mode()
        self._set_learning_rates()

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_demo_loss = 0.0
        mean_approx_kl = 0.0
        num_updates = 0
        early_stop = False

        for _ in range(self.num_learning_epochs):
            env_indices = torch.randperm(batch.episode_starts.shape[1], device=self.device)
            for batch_env_ids in torch.tensor_split(env_indices, self.num_mini_batches):
                if batch_env_ids.numel() == 0:
                    continue
                minibatch = self._slice_batch(batch, batch_env_ids)

                current_log_probs, entropy = self.policy.evaluate_actions_rollout(
                    observations=minibatch["observations"],
                    goals=minibatch["goals"],
                    actions=minibatch["actions"],
                    episode_starts=minibatch["episode_starts"],
                    initial_rnn_state=minibatch["initial_rnn_state"],
                )
                current_values = self.policy.value(minibatch["observations"], goals=minibatch["goals"])

                old_log_probs = minibatch["log_probs"].reshape(-1)
                advantages = minibatch["advantages"].reshape(-1)
                returns = minibatch["returns"].reshape(-1)

                log_ratio = current_log_probs - old_log_probs
                ratio = log_ratio.exp()
                surrogate = -advantages * ratio
                surrogate_clipped = -advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
                value_loss = (returns - current_values).pow(2).mean()

                demo_loss = torch.zeros((), device=self.device)
                if self.demo_batch_iterator is not None and self.demo_loss_fn is not None and self.demo_coef > 0.0:
                    try:
                        demo_batch = next(self.demo_batch_iterator)
                    except StopIteration as exc:
                        raise RuntimeError("Demo batch iterator was exhausted unexpectedly.") from exc
                    demo_loss = self.demo_loss_fn(demo_batch)

                loss = (
                    surrogate_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy.mean()
                    + self.demo_coef * demo_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.actor_parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.policy.value_net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                approx_kl = (old_log_probs - current_log_probs).mean()
                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_entropy += entropy.mean().item()
                mean_demo_loss += demo_loss.item()
                mean_approx_kl += approx_kl.item()
                num_updates += 1

                if self.target_kl is not None and approx_kl.item() > 1.5 * self.target_kl:
                    early_stop = True
                    break
            if early_stop:
                break

        self.update_step += 1
        denom = max(1, num_updates)
        return {
            "value": mean_value_loss / denom,
            "surrogate": mean_surrogate_loss / denom,
            "entropy": mean_entropy / denom,
            "demo": mean_demo_loss / denom,
            "approx_kl": mean_approx_kl / denom,
            "early_stop": float(early_stop),
            "effective_num_minibatches": float(min(self.num_mini_batches, batch.episode_starts.shape[1])),
        }

    def _slice_batch(self, batch: RolloutBatch, env_indices: torch.Tensor) -> dict[str, Any]:
        env_indices_cpu = env_indices.to(device=batch.episode_starts.device)
        initial_rnn_state = _slice_rnn_state(batch.initial_rnn_state, env_indices_cpu)
        return {
            "observations": {key: value[:, env_indices_cpu] for key, value in batch.observations.items()},
            "goals": None if batch.goals is None else {key: value[:, env_indices_cpu] for key, value in batch.goals.items()},
            "actions": batch.actions[:, env_indices_cpu],
            "log_probs": batch.log_probs.view(batch.episode_starts.shape[0], batch.episode_starts.shape[1])[:, env_indices_cpu],
            "returns": batch.returns.view(batch.episode_starts.shape[0], batch.episode_starts.shape[1])[:, env_indices_cpu],
            "advantages": batch.advantages.view(batch.episode_starts.shape[0], batch.episode_starts.shape[1])[:, env_indices_cpu],
            "episode_starts": batch.episode_starts[:, env_indices_cpu].to(dtype=torch.bool),
            "initial_rnn_state": initial_rnn_state,
        }


def _slice_rnn_state(state: Any, env_indices: torch.Tensor) -> Any:
    if state is None:
        return None
    if isinstance(state, dict):
        sliced = {}
        for key, value in state.items():
            if key == "step_count":
                sliced[key] = int(value)
            else:
                sliced[key] = _slice_rnn_state(value, env_indices)
        return sliced
    if isinstance(state, tuple):
        return tuple(_slice_rnn_state(value, env_indices) for value in state)
    if torch.is_tensor(state):
        return state[:, env_indices].detach().clone()
    raise TypeError(f"Unsupported recurrent state type: {type(state)!r}")


def _detach_rollout_batch(batch: RolloutBatch) -> RolloutBatch:
    return RolloutBatch(
        observations={key: value.detach() for key, value in batch.observations.items()},
        goals=None if batch.goals is None else {key: value.detach() for key, value in batch.goals.items()},
        actions=batch.actions.detach(),
        log_probs=batch.log_probs.detach(),
        returns=batch.returns.detach(),
        advantages=batch.advantages.detach(),
        values=batch.values.detach(),
        rewards=batch.rewards.detach(),
        dones=batch.dones.detach(),
        episode_starts=batch.episode_starts.detach(),
        initial_rnn_state=_detach_nested_state(batch.initial_rnn_state),
    )


def _detach_nested_state(state: Any) -> Any:
    if state is None:
        return None
    if isinstance(state, dict):
        return {key: _detach_nested_state(value) for key, value in state.items()}
    if isinstance(state, tuple):
        return tuple(_detach_nested_state(value) for value in state)
    if torch.is_tensor(state):
        return state.detach()
    return state
