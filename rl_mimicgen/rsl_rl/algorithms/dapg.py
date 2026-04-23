"""Demo-Augmented Policy Gradient (DAPG) on top of RSL-RL's PPO.

Reference: Rajeswaran et al., "Learning Complex Dexterous Manipulation with Deep
Reinforcement Learning and Demonstrations" (RSS 2018) -- https://arxiv.org/pdf/1709.10087

Augments each PPO mini-batch with a demonstration log-likelihood term:

    L_DAPG = L_PPO  -  w_k * E_(s,a)~demo [ log pi(a | s) ]
    w_k    = lambda0 * lambda1 ** update_count * max(|A_PPO|)

where the ``max`` is taken over the current PPO mini-batch's advantages. The
weight decays geometrically with training iterations, matching the paper.

Demo forward pass
-----------------
The actor is an LSTM-backed ``RNNModel``. We reuse the same recurrent forward
the BC-RNN was trained under: fixed-length windows, zero initial hidden state,
no padding. The demo batch is a ``(T, B, obs_dim)`` tensor; we normalize via
the actor's ``obs_normalizer``, run the raw ``nn.LSTM`` with ``hidden_state=None``
(zero-init), then apply the MLP head and score the demo actions. RSL-RL's
``masks``/``unpad_trajectories`` pipeline is bypassed for demos because it
only round-trips correctly when every chunk in the batch is full-length -- a
guarantee ``DemoStorage`` already makes.
"""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import PPO

from rl_mimicgen.rsl_rl.modules import DemoStorage


class DAPG(PPO):
    """PPO + demonstration-augmented policy gradient.

    The subclass only adds DAPG-specific state in ``__init__`` and overrides
    ``update()`` to mix in the demo log-likelihood term inside the existing
    mini-batch loop. Everything else (rollout, GAE, value loss, adaptive LR,
    clipping, logging) is inherited from PPO untouched.
    """

    def __init__(
        self,
        *args: Any,
        demo_dataset_path: str,
        demo_obs_keys: list[str],
        demo_seq_length: int = 10,
        demo_stride: int | None = None,
        demo_filter_key: str | None = None,
        dapg_lambda0: float = 0.1,
        dapg_lambda1: float = 0.995,
        dapg_weight_floor: float = 0.0,
        dapg_batch_size: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        if not self.actor.is_recurrent:
            raise ValueError(
                "DAPG currently assumes a recurrent actor (BC-RNN warm-start). "
                "Got a non-recurrent actor -- extend DAPG before using."
            )

        self.demo_storage = DemoStorage(
            dataset_path=demo_dataset_path,
            obs_keys=list(demo_obs_keys),
            seq_length=int(demo_seq_length),
            stride=demo_stride,
            filter_key=demo_filter_key,
            device=self.device,
        )
        self.dapg_lambda0 = float(dapg_lambda0)
        self.dapg_lambda1 = float(dapg_lambda1)
        self.dapg_weight_floor = float(dapg_weight_floor)
        self.dapg_batch_size = int(dapg_batch_size)
        # Incremented once per call to ``update()``; exponent for w_k decay.
        self._update_count: int = 0

        print(
            f"[DAPG] demo bank: {self.demo_storage.num_chunks} chunks "
            f"(seq_length={self.demo_storage.seq_length}, "
            f"obs_dim={self.demo_storage.obs_dim}, "
            f"action_dim={self.demo_storage.action_dim}, "
            f"{self.demo_storage.memory_mb():.1f} MB)"
        )

    def _demo_log_prob(self, demo_obs: TensorDict, demo_actions: torch.Tensor) -> torch.Tensor:
        """Compute per-timestep log pi(a|s) over a padded demo batch.

        Args:
            demo_obs: TensorDict with key ``"policy"`` of shape ``(T, B, obs_dim)``.
            demo_actions: tensor of shape ``(T, B, action_dim)``.

        Returns:
            Tensor of shape ``(T, B)`` holding ``log pi`` summed over action dims.
        """
        actor = self.actor

        # Match MLPModel.get_latent: concatenate selected obs groups, normalize.
        obs_cat = torch.cat([demo_obs[g] for g in actor.obs_groups], dim=-1)
        obs_norm = actor.obs_normalizer(obs_cat)

        # Zero initial hidden state matches BC-RNN training; chunks are
        # full-length so RSL-RL's masks/unpad wrapper would misalign shapes.
        rnn_out, _ = actor.rnn.rnn(obs_norm, None)  # (T, B, hidden_dim)

        mlp_out = actor.mlp(rnn_out)  # (T, B, distribution.input_dim)
        actor.distribution.update(mlp_out)
        return actor.distribution.log_prob(demo_actions)  # (T, B)

    def update(self) -> dict[str, float]:
        """PPO update with DAPG demo-augmented gradient mixed into each mini-batch.

        This is a near-verbatim copy of :meth:`PPO.update` (preserving RND and
        symmetry plumbing); the only additions are the demo sampling + loss term
        inserted before the backward pass, and a final ``_update_count`` bump.
        """
        import torch.nn as nn  # local import to match upstream style

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_rnd_loss = 0.0 if self.rnd else None
        mean_symmetry_loss = 0.0 if self.symmetry else None
        mean_demo_loss = 0.0
        mean_demo_weight = 0.0

        # Off-policy drift diagnostics (exposed via self._train_metrics for the
        # runner-level hook in train_rl.py to log under Train/*). Track tails,
        # not means — adaptive LR already constrains mean KL.
        kl_max_all = 0.0
        kl_mean_sum = 0.0
        clip_frac_sum = 0.0
        ratio_max_all = 0.0
        metric_count = 0

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        w_decay = max(
            self.dapg_weight_floor,
            self.dapg_lambda0 * (self.dapg_lambda1 ** self._update_count),
        )

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (
                        batch.advantages - batch.advantages.mean()
                    ) / (batch.advantages.std() + 1e-8)

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                batch.observations, batch.actions = data_augmentation_func(
                    env=self.symmetry["_env"],
                    obs=batch.observations,
                    actions=batch.actions,
                )
                num_aug = int(batch.observations.batch_size[0] / original_batch_size)
                batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
                batch.values = batch.values.repeat(num_aug, 1)
                batch.advantages = batch.advantages.repeat(num_aug, 1)
                batch.returns = batch.returns.repeat(num_aug, 1)

            # PPO forward on the RL minibatch
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(
                batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1]
            )
            distribution_params = tuple(
                p[:original_batch_size] for p in self.actor.output_distribution_params
            )
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            with torch.no_grad():
                kl_mb = self.actor.get_kl_divergence(
                    batch.old_distribution_params, distribution_params
                )
                kl_max_all = max(kl_max_all, float(kl_mb.max().item()))
                kl_mean_sum += float(kl_mb.mean().item())
                ratio_dev = (ratio - 1.0).abs()
                clip_frac_sum += float((ratio_dev > self.clip_param).float().mean().item())
                ratio_max_all = max(ratio_max_all, float(ratio.max().item()))
                metric_count += 1

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            # Weight by max |adv| so the demo gradient tracks the policy gradient scale.
            demo_obs, demo_actions = self.demo_storage.sample(self.dapg_batch_size)
            demo_log_prob = self._demo_log_prob(demo_obs, demo_actions)  # (T, B)
            adv_scale = float(batch.advantages.detach().abs().max().item())
            demo_weight = w_decay * adv_scale
            demo_loss = -demo_log_prob.mean()
            loss = loss + demo_weight * demo_loss
            mean_demo_loss += float(demo_loss.item())
            mean_demo_weight += demo_weight
            # -------------------------------------------------------------------

            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    batch.observations, _ = data_augmentation_func(
                        obs=batch.observations, actions=None, env=self.symmetry["_env"]
                    )
                mean_actions = self.actor(batch.observations.detach().clone())
                action_mean_orig = mean_actions[:original_batch_size]
                _, actions_mean_symm = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions[original_batch_size:], actions_mean_symm.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            if self.rnd:
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_demo_loss /= num_updates
        mean_demo_weight /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        self.storage.clear()
        self._update_count += 1

        denom = max(metric_count, 1)
        self._train_metrics = {
            "kl_max": kl_max_all,
            "kl_mean": kl_mean_sum / denom,
            "clip_frac": clip_frac_sum / denom,
            "ratio_max": ratio_max_all,
        }

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "dapg_demo_nll": mean_demo_loss,
            "dapg_weight": mean_demo_weight,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict
