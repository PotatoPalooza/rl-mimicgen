from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn

from rl_mimicgen.rl.awac_policy import AWACPolicyAdapter
from rl_mimicgen.rl.replay_buffer import ReplayBuffer


class AWAC:
    def __init__(
        self,
        policy: AWACPolicyAdapter,
        batch_size: int,
        discount: float = 0.99,
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
        beta: float = 1.0,
        max_weight: float = 20.0,
        normalize_weights: bool = True,
        behavior_kl_coef: float = 0.0,
        critic_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        critic_huber_loss: bool = False,
        num_action_samples: int | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.policy = policy
        self.batch_size = max(1, int(batch_size))
        self.discount = float(discount)
        self.demo_batch_iterator = demo_batch_iterator
        self.demo_loss_fn = demo_loss_fn
        self.demo_coef = float(demo_coef)
        self.actor_learning_rate = float(actor_learning_rate)
        self.critic_learning_rate = float(critic_learning_rate)
        self.critic_warmup_updates = max(0, int(critic_warmup_updates))
        self.actor_freeze_env_steps = max(0, int(actor_freeze_env_steps))
        self.num_learning_epochs = max(1, int(num_learning_epochs))
        self.num_mini_batches = max(1, int(num_mini_batches))
        self.beta = float(beta)
        if self.beta <= 0.0:
            raise ValueError(f"AWAC beta must be positive, got {self.beta}.")
        self.max_weight = float(max_weight)
        self.normalize_weights = bool(normalize_weights)
        self.behavior_kl_coef = float(behavior_kl_coef)
        self.critic_loss_coef = float(critic_loss_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.device = torch.device(device)
        self.total_env_steps = 0
        self.update_step = 0
        self.num_action_samples = policy.num_value_action_samples if num_action_samples is None else max(1, int(num_action_samples))
        self.td_loss_fcn = nn.SmoothL1Loss() if critic_huber_loss else nn.MSELoss()

        self.actor_optimizer = torch.optim.Adam(
            self.policy.actor_parameters(),
            lr=self.actor_learning_rate,
            weight_decay=weight_decay,
        )
        self.critic_optimizers = [
            torch.optim.Adam(q_net.parameters(), lr=self.critic_learning_rate, weight_decay=weight_decay)
            for q_net in self.policy.q_networks
        ]

    def train_mode(self) -> None:
        self.policy.train()
        self.policy.actor.train()
        for q_net in self.policy.q_networks:
            q_net.train()
        for target_q_net in self.policy.target_q_networks:
            target_q_net.eval()

    def set_total_env_steps(self, total_env_steps: int) -> None:
        self.total_env_steps = max(0, int(total_env_steps))

    def save(self) -> dict[str, Any]:
        return {
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizers": [optimizer.state_dict() for optimizer in self.critic_optimizers],
            "total_env_steps": self.total_env_steps,
            "update_step": self.update_step,
        }

    def load(self, state_dict: dict[str, Any]) -> None:
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        for optimizer, optimizer_state in zip(self.critic_optimizers, state_dict.get("critic_optimizers", [])):
            optimizer.load_state_dict(optimizer_state)
        self.total_env_steps = int(state_dict.get("total_env_steps", 0))
        self.update_step = int(state_dict.get("update_step", 0))

    def _set_learning_rates(self) -> None:
        actor_lr = self.actor_learning_rate
        if self.update_step < self.critic_warmup_updates or self.total_env_steps <= self.actor_freeze_env_steps:
            actor_lr = 0.0
        self.actor_optimizer.param_groups[0]["lr"] = actor_lr
        for optimizer in self.critic_optimizers:
            optimizer.param_groups[0]["lr"] = self.critic_learning_rate

    def update(self, replay_buffer: ReplayBuffer) -> dict[str, float]:
        if len(replay_buffer) < self.batch_size:
            raise RuntimeError(
                f"Replay buffer needs at least {self.batch_size} transitions before AWAC updates, got {len(replay_buffer)}."
            )

        self.train_mode()
        self._set_learning_rates()

        mean_q_loss = 0.0
        mean_actor_loss = 0.0
        mean_entropy = 0.0
        mean_demo_loss = 0.0
        mean_q_pred = 0.0
        mean_target_q = 0.0
        mean_q_gap = 0.0
        mean_weight = 0.0
        mean_advantage = 0.0
        mean_advantage_std = 0.0
        mean_advantage_min = 0.0
        mean_advantage_max = 0.0
        mean_weight_std = 0.0
        mean_weight_min = 0.0
        mean_weight_max = 0.0
        mean_clipped_weight_frac = 0.0
        mean_log_prob = 0.0
        mean_log_prob_std = 0.0
        mean_log_prob_min = 0.0
        mean_log_prob_max = 0.0
        mean_behavior_kl = 0.0
        num_updates = 0

        for _ in range(self.num_learning_epochs):
            for _ in range(self.num_mini_batches):
                batch = replay_buffer.sample(batch_size=self.batch_size, device=self.device)
                critic_metrics = self._update_critics(batch)
                if self.policy.is_recurrent and replay_buffer.can_sample_sequences(self.policy.rnn_horizon):
                    actor_metrics = self._update_actor_sequence(replay_buffer)
                else:
                    actor_metrics = self._update_actor(batch)
                self.policy.soft_update_targets()

                mean_q_loss += critic_metrics["q_loss"]
                mean_q_pred += critic_metrics["q_pred_mean"]
                mean_target_q += critic_metrics["target_q_mean"]
                mean_q_gap += critic_metrics["q_gap_mean"]
                mean_actor_loss += actor_metrics["actor_loss"]
                mean_entropy += actor_metrics["entropy"]
                mean_demo_loss += actor_metrics["demo_loss"]
                mean_weight += actor_metrics["weight_mean"]
                mean_advantage += actor_metrics["advantage_mean"]
                mean_advantage_std += actor_metrics["advantage_std"]
                mean_advantage_min += actor_metrics["advantage_min"]
                mean_advantage_max += actor_metrics["advantage_max"]
                mean_weight_std += actor_metrics["weight_std"]
                mean_weight_min += actor_metrics["weight_min"]
                mean_weight_max += actor_metrics["weight_max"]
                mean_clipped_weight_frac += actor_metrics["weight_clipped_frac"]
                mean_log_prob += actor_metrics["log_prob_mean"]
                mean_log_prob_std += actor_metrics["log_prob_std"]
                mean_log_prob_min += actor_metrics["log_prob_min"]
                mean_log_prob_max += actor_metrics["log_prob_max"]
                mean_behavior_kl += actor_metrics["behavior_kl"]
                num_updates += 1

        self.update_step += 1
        denom = max(1, num_updates)
        return {
            "value": mean_q_loss / denom,
            "surrogate": mean_actor_loss / denom,
            "entropy": mean_entropy / denom,
            "demo": mean_demo_loss / denom,
            "approx_kl": 0.0,
            "weight_mean": mean_weight / denom,
            "advantage_mean": mean_advantage / denom,
            "advantage_std": mean_advantage_std / denom,
            "advantage_min": mean_advantage_min / denom,
            "advantage_max": mean_advantage_max / denom,
            "weight_std": mean_weight_std / denom,
            "weight_min": mean_weight_min / denom,
            "weight_max": mean_weight_max / denom,
            "weight_clipped_frac": mean_clipped_weight_frac / denom,
            "log_prob_mean": mean_log_prob / denom,
            "log_prob_std": mean_log_prob_std / denom,
            "log_prob_min": mean_log_prob_min / denom,
            "log_prob_max": mean_log_prob_max / denom,
            "behavior_kl": mean_behavior_kl / denom,
            "q_pred_mean": mean_q_pred / denom,
            "target_q_mean": mean_target_q / denom,
            "q_gap_mean": mean_q_gap / denom,
            "early_stop": 0.0,
            "effective_num_minibatches": float(num_updates),
        }

    def _update_critics(self, batch) -> dict[str, float]:
        with torch.no_grad():
            next_v = self.policy.approximate_v_value(
                observations=batch.next_observations,
                goals=batch.next_goals,
                use_target=True,
                num_action_samples=self.num_action_samples,
            )
            target_q = batch.rewards + ((1.0 - batch.dones) * self.discount * next_v)

        pred_qs = self.policy.q_values(
            observations=batch.observations,
            actions=batch.actions,
            goals=batch.goals,
            use_target=False,
        )

        q_losses = [self.td_loss_fcn(pred_q, target_q) for pred_q in pred_qs]
        for optimizer, q_loss, q_net in zip(self.critic_optimizers, q_losses, self.policy.q_networks):
            optimizer.zero_grad(set_to_none=True)
            (self.critic_loss_coef * q_loss).backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), self.max_grad_norm)
            optimizer.step()

        stacked_qs = torch.stack([pred_q.detach() for pred_q in pred_qs], dim=0)
        return {
            "q_loss": float(torch.stack([q_loss.detach() for q_loss in q_losses]).mean().item()),
            "q_pred_mean": float(stacked_qs.mean().item()),
            "target_q_mean": float(target_q.mean().item()),
            "q_gap_mean": float((stacked_qs.max(dim=0)[0] - stacked_qs.min(dim=0)[0]).mean().item()),
        }

    def _update_actor(self, batch) -> dict[str, float]:
        with torch.no_grad():
            q_dataset = self.policy.min_q_value(
                observations=batch.observations,
                actions=batch.actions,
                goals=batch.goals,
                use_target=False,
            )
            v_values = self.policy.approximate_v_value(
                observations=batch.observations,
                goals=batch.goals,
                use_target=False,
                num_action_samples=self.num_action_samples,
            )
            advantages = q_dataset - v_values
            unclipped_weights = torch.exp(advantages / self.beta)
            clipped_weight_frac = (unclipped_weights > self.max_weight).to(dtype=torch.float32).mean()
            weights = torch.clamp(unclipped_weights, max=self.max_weight)
            if self.normalize_weights:
                weights = weights / weights.mean().clamp(min=1e-8)

        current_log_probs, entropy = self.policy.actor_log_prob_replay(
            observations=batch.observations,
            goals=batch.goals,
            actions=batch.actions,
        )
        actor_loss = -(weights.detach() * current_log_probs).mean()
        behavior_kl = torch.zeros((), device=self.device)
        if self.behavior_kl_coef > 0.0:
            behavior_kl, _ = self.policy.behavior_kl_replay(
                observations=batch.observations,
                goals=batch.goals,
            )
            behavior_kl = behavior_kl.mean()

        demo_loss = torch.zeros((), device=self.device)
        if self.demo_batch_iterator is not None and self.demo_loss_fn is not None and self.demo_coef > 0.0:
            try:
                demo_batch = next(self.demo_batch_iterator)
            except StopIteration as exc:
                raise RuntimeError("Demo batch iterator was exhausted unexpectedly.") from exc
            demo_loss = self.demo_loss_fn(demo_batch)

        loss = actor_loss - (self.entropy_coef * entropy.mean()) + (self.demo_coef * demo_loss)
        loss = loss + (self.behavior_kl_coef * behavior_kl)
        self.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.actor_parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        return {
            "actor_loss": float(actor_loss.item()),
            "entropy": float(entropy.mean().item()),
            "demo_loss": float(demo_loss.item()),
            "log_prob_mean": float(current_log_probs.mean().item()),
            "log_prob_std": float(current_log_probs.std(unbiased=False).item()),
            "log_prob_min": float(current_log_probs.min().item()),
            "log_prob_max": float(current_log_probs.max().item()),
            "behavior_kl": float(behavior_kl.item()),
            "weight_mean": float(weights.mean().item()),
            "weight_std": float(weights.std(unbiased=False).item()),
            "weight_min": float(weights.min().item()),
            "weight_max": float(weights.max().item()),
            "weight_clipped_frac": float(clipped_weight_frac.item()),
            "advantage_mean": float(advantages.mean().item()),
            "advantage_std": float(advantages.std(unbiased=False).item()),
            "advantage_min": float(advantages.min().item()),
            "advantage_max": float(advantages.max().item()),
        }

    def _update_actor_sequence(self, replay_buffer: ReplayBuffer) -> dict[str, float]:
        if not self.policy.is_recurrent:
            raise RuntimeError("Sequence actor updates are only valid for recurrent policies.")

        sequence_length = max(1, int(self.policy.rnn_horizon))
        num_sequences = max(1, self.batch_size // sequence_length)
        sequence_batch = replay_buffer.sample_sequences(
            sequence_length=sequence_length,
            batch_size=num_sequences,
            device=self.device,
        )
        flat_mask = sequence_batch.mask.reshape(-1)
        observations = {
            key: value.reshape(-1, *value.shape[2:])[flat_mask]
            for key, value in sequence_batch.observations.items()
        }
        goals = None
        if sequence_batch.goals is not None:
            goals = {
                key: value.reshape(-1, *value.shape[2:])[flat_mask]
                for key, value in sequence_batch.goals.items()
            }
        actions = sequence_batch.actions.reshape(-1, sequence_batch.actions.shape[-1])[flat_mask]

        with torch.no_grad():
            q_dataset = self.policy.min_q_value(
                observations=observations,
                actions=actions,
                goals=goals,
                use_target=False,
            )
            v_values = self.policy.approximate_v_value(
                observations=observations,
                goals=goals,
                use_target=False,
                num_action_samples=self.num_action_samples,
            )
            advantages = q_dataset - v_values
            unclipped_weights = torch.exp(advantages / self.beta)
            clipped_weight_frac = (unclipped_weights > self.max_weight).to(dtype=torch.float32).mean()
            weights = torch.clamp(unclipped_weights, max=self.max_weight)
            if self.normalize_weights:
                weights = weights / weights.mean().clamp(min=1e-8)

        current_log_probs, entropy = self.policy.actor_log_prob_replay_sequence(
            observations=sequence_batch.observations,
            goals=sequence_batch.goals,
            actions=sequence_batch.actions,
            episode_starts=sequence_batch.episode_starts,
            mask=sequence_batch.mask,
        )
        actor_loss = -(weights.detach() * current_log_probs).mean()
        behavior_kl = torch.zeros((), device=self.device)
        if self.behavior_kl_coef > 0.0:
            behavior_kl, _ = self.policy.behavior_kl_replay_sequence(
                observations=sequence_batch.observations,
                goals=sequence_batch.goals,
                episode_starts=sequence_batch.episode_starts,
                mask=sequence_batch.mask,
            )
            behavior_kl = behavior_kl.mean()

        demo_loss = torch.zeros((), device=self.device)
        if self.demo_batch_iterator is not None and self.demo_loss_fn is not None and self.demo_coef > 0.0:
            try:
                demo_batch = next(self.demo_batch_iterator)
            except StopIteration as exc:
                raise RuntimeError("Demo batch iterator was exhausted unexpectedly.") from exc
            demo_loss = self.demo_loss_fn(demo_batch)

        loss = actor_loss - (self.entropy_coef * entropy.mean()) + (self.demo_coef * demo_loss)
        loss = loss + (self.behavior_kl_coef * behavior_kl)
        self.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.actor_parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        return {
            "actor_loss": float(actor_loss.item()),
            "entropy": float(entropy.mean().item()),
            "demo_loss": float(demo_loss.item()),
            "log_prob_mean": float(current_log_probs.mean().item()),
            "log_prob_std": float(current_log_probs.std(unbiased=False).item()),
            "log_prob_min": float(current_log_probs.min().item()),
            "log_prob_max": float(current_log_probs.max().item()),
            "behavior_kl": float(behavior_kl.item()),
            "weight_mean": float(weights.mean().item()),
            "weight_std": float(weights.std(unbiased=False).item()),
            "weight_min": float(weights.min().item()),
            "weight_max": float(weights.max().item()),
            "weight_clipped_frac": float(clipped_weight_frac.item()),
            "advantage_mean": float(advantages.mean().item()),
            "advantage_std": float(advantages.std(unbiased=False).item()),
            "advantage_min": float(advantages.min().item()),
            "advantage_max": float(advantages.max().item()),
        }
