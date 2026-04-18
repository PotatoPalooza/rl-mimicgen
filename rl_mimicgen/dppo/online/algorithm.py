from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn

from rl_mimicgen.dppo.online.storage import DiffusionRolloutBatch
from rl_mimicgen.dppo.policy import DiffusionPolicyAdapter


class DiffusionPPO:
    def __init__(
        self,
        policy: DiffusionPolicyAdapter,
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
        gamma_denoising: float = 1.0,
        act_steps: int | None = None,
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
        self.gamma_denoising = float(gamma_denoising)
        self.act_steps = None if act_steps is None else int(act_steps)
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

    def update(self, batch: DiffusionRolloutBatch) -> dict[str, float]:
        batch = _detach_diffusion_batch(batch)
        self.train_mode()
        self._set_learning_rates()

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_demo_loss = 0.0
        mean_approx_kl = 0.0
        num_updates = 0
        early_stop = False
        num_decisions = batch.log_probs.shape[0]

        for _ in range(self.num_learning_epochs):
            decision_indices = torch.randperm(num_decisions, device=self.device)
            for batch_decision_ids in torch.tensor_split(decision_indices, self.num_mini_batches):
                if batch_decision_ids.numel() == 0:
                    continue
                minibatch = self._slice_batch(batch, batch_decision_ids)

                demo_loss = torch.zeros((), device=self.device)
                if self.demo_batch_iterator is not None and self.demo_loss_fn is not None and self.demo_coef > 0.0:
                    try:
                        demo_batch = next(self.demo_batch_iterator)
                    except StopIteration as exc:
                        raise RuntimeError("Demo batch iterator was exhausted unexpectedly.") from exc
                    demo_loss = self.demo_loss_fn(demo_batch)

                self.optimizer.zero_grad(set_to_none=True)
                current_log_probs = self.policy._chain_log_prob_subsample(
                    obs_history=minibatch["observations"],
                    goal_batch=minibatch["goals"],
                    chain_samples=minibatch["chain_prev"],
                    chain_next_samples=minibatch["chain_next"],
                    chain_timesteps=minibatch["timesteps"],
                )
                current_obs = {key: value[:, -1] for key, value in minibatch["observations"].items()}
                current_values = self.policy.value_net(current_obs, goal_dict=minibatch["goals"]).squeeze(-1)

                current_log_probs = self._reduce_denoising_log_probs(current_log_probs)
                old_log_probs = self._reduce_denoising_log_probs(minibatch["log_probs"])
                advantages = minibatch["advantages"]
                returns = minibatch["returns"]

                log_ratio = current_log_probs - old_log_probs
                ratio = log_ratio.exp()
                surrogate = -advantages * ratio
                surrogate_clipped = -advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
                value_loss = (returns - current_values).pow(2).mean()
                loss = surrogate_loss + self.value_loss_coef * value_loss
                loss.backward()

                if self.demo_coef > 0.0 and demo_loss.requires_grad:
                    (self.demo_coef * demo_loss).backward()

                nn.utils.clip_grad_norm_(self.policy.actor_parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.policy.value_net.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.policy.step_ema()

                approx_kl = (old_log_probs - current_log_probs).mean().item()
                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_demo_loss += demo_loss.item()
                mean_approx_kl += approx_kl
                num_updates += 1

                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    early_stop = True
                    break
            if early_stop:
                break

        self.update_step += 1
        denom = max(1, num_updates)
        return {
            "value": mean_value_loss / denom,
            "surrogate": mean_surrogate_loss / denom,
            "entropy": 0.0,
            "demo": mean_demo_loss / denom,
            "approx_kl": mean_approx_kl / denom,
            "early_stop": float(early_stop),
            "effective_num_minibatches": float(num_updates),
        }

    def _slice_batch(self, batch: DiffusionRolloutBatch, decision_indices: torch.Tensor) -> dict[str, Any]:
        return {
            "observations": {key: value[decision_indices] for key, value in batch.observations.items()},
            "goals": None if batch.goals is None else {key: value[decision_indices] for key, value in batch.goals.items()},
            "chain_prev": batch.chain_samples[decision_indices],
            "chain_next": batch.chain_next_samples[decision_indices],
            "timesteps": batch.chain_timesteps[decision_indices],
            "log_probs": batch.log_probs[decision_indices],
            "returns": batch.returns[decision_indices],
            "advantages": batch.advantages[decision_indices],
        }

    def _reduce_denoising_log_probs(self, log_probs: torch.Tensor) -> torch.Tensor:
        per_step_log_probs = log_probs.mean(dim=(-1, -2))
        if per_step_log_probs.ndim == 1:
            return per_step_log_probs
        chain_len = per_step_log_probs.shape[1]
        weights = torch.pow(
            torch.full((chain_len,), self.gamma_denoising, dtype=per_step_log_probs.dtype, device=per_step_log_probs.device),
            torch.arange(chain_len - 1, -1, -1, dtype=per_step_log_probs.dtype, device=per_step_log_probs.device),
        )
        weights = weights / weights.sum().clamp(min=torch.finfo(weights.dtype).eps)
        return (per_step_log_probs * weights.unsqueeze(0)).sum(dim=1)


def _detach_diffusion_batch(batch: DiffusionRolloutBatch) -> DiffusionRolloutBatch:
    return DiffusionRolloutBatch(
        observations={key: value.detach() for key, value in batch.observations.items()},
        goals=None if batch.goals is None else {key: value.detach() for key, value in batch.goals.items()},
        chain_samples=batch.chain_samples.detach(),
        chain_next_samples=batch.chain_next_samples.detach(),
        chain_timesteps=batch.chain_timesteps.detach(),
        log_probs=batch.log_probs.detach(),
        returns=batch.returns.detach(),
        advantages=batch.advantages.detach(),
        values=batch.values.detach(),
        rewards=batch.rewards.detach(),
        dones=batch.dones.detach(),
    )
