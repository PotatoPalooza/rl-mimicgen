from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.models.value_nets import ValueNetwork

from rl_mimicgen.rl.policy import PolicyBundle


class DiffusionOnlinePolicyAdapter(nn.Module):
    """Diffusion-native policy adapter for DPPO-style online fine-tuning."""

    def __init__(
        self,
        bundle: PolicyBundle,
        device: torch.device,
        value_hidden_sizes: tuple[int, ...] = (256, 256),
        num_inference_timesteps: int | None = None,
        use_ema: bool = False,
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
        self.residual_enabled = False
        self.learned_log_std = None
        self.is_diffusion_policy = True

        if not bool(getattr(bundle.config.algo.ddpm, "enabled", False)):
            raise NotImplementedError("The diffusion online RL path currently supports DDPM schedulers only.")

        self.observation_horizon = int(bundle.config.algo.horizon.observation_horizon)
        self.action_horizon = int(bundle.config.algo.horizon.action_horizon)
        self.prediction_horizon = int(bundle.config.algo.horizon.prediction_horizon)
        self.action_dim = int(bundle.shape_meta["ac_dim"])
        self.num_inference_timesteps = int(
            num_inference_timesteps or getattr(bundle.config.algo.ddpm, "num_inference_timesteps")
        )
        # PPO rollouts must come from the same live policy weights that are
        # later replayed inside the PPO objective. Using EMA samples for data
        # collection makes the batch off-policy immediately, so the EMA path is
        # reserved for deterministic evaluation only.
        self.use_ema_for_evaluation = bool(use_ema)

        self.value_net = ValueNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            mlp_layer_dims=list(value_hidden_sizes),
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.algo.obs_config.encoder),
        ).float().to(device)

        self._rollout_obs_history: dict[str, torch.Tensor] | None = None
        self._train_action_queues: list[deque[torch.Tensor]] | None = None
        self._eval_action_queues: list[deque[torch.Tensor]] | None = None

    def actor_parameters(self) -> list[nn.Parameter]:
        return list(self.actor.parameters())

    def current_log_std(self) -> torch.Tensor | None:
        return None

    def step_ema(self) -> None:
        if self.algo.ema is not None:
            self.algo.ema.step(self.algo.nets)

    def reset_rollout_state(self) -> None:
        self._rollout_obs_history = None
        self._train_action_queues = None
        self._eval_action_queues = None
        self.algo.reset()

    def clone_rollout_state(self) -> Any:
        return {
            "obs_history": _clone_history_state(self._rollout_obs_history),
            "train_action_queues": _clone_action_queues(self._train_action_queues),
            "eval_action_queues": _clone_action_queues(self._eval_action_queues),
        }

    def restore_rollout_state(self, state: Any) -> None:
        if isinstance(state, dict) and "obs_history" in state:
            self._rollout_obs_history = _clone_history_state(state.get("obs_history"))
            self._train_action_queues = _clone_action_queues(state.get("train_action_queues"))
            self._eval_action_queues = _clone_action_queues(state.get("eval_action_queues"))
            return
        self._rollout_obs_history = _clone_history_state(state)
        self._train_action_queues = None
        self._eval_action_queues = None

    def clone_training_rollout_state(self) -> Any:
        return {
            "obs_history": _clone_history_state(self._rollout_obs_history),
            "train_action_queues": _clone_action_queues(self._train_action_queues),
        }

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

    def act(
        self,
        obs: dict[str, np.ndarray],
        goal: dict[str, np.ndarray] | None,
        episode_starts: np.ndarray,
        clip_actions: bool = True,
    ):
        with torch.no_grad():
            self.actor.train()
            self.value_net.eval()

            obs_batch = self.prepare_observation_batch(obs)
            goal_batch = self.prepare_goal_batch(goal)
            obs_history = self._advance_history(self._rollout_obs_history, obs_batch, episode_starts)
            batch_size = next(iter(obs_history.values())).shape[0]
            self._ensure_train_action_queues(batch_size)
            reset_mask = torch.as_tensor(episode_starts, device=self.device, dtype=torch.bool)
            if reset_mask.any():
                assert self._train_action_queues is not None
                for env_idx in torch.nonzero(reset_mask, as_tuple=False).flatten().tolist():
                    self._train_action_queues[env_idx].clear()

            assert self._train_action_queues is not None
            replan_ids = [env_idx for env_idx, queue in enumerate(self._train_action_queues) if len(queue) == 0]
            replan_data = None
            if replan_ids:
                replan_index = torch.as_tensor(replan_ids, device=self.device, dtype=torch.long)
                replan_obs_history = {key: value.index_select(0, replan_index) for key, value in obs_history.items()}
                replan_goal_batch = None
                if goal_batch is not None:
                    replan_goal_batch = {key: value.index_select(0, replan_index) for key, value in goal_batch.items()}
                action_sequences, log_probs, chain_samples, chain_next_samples, chain_timesteps = self._sample_action_sequence_with_chain(
                    obs_history=replan_obs_history,
                    goal_batch=replan_goal_batch,
                    use_ema=False,
                )
                values = self.value_net({key: value[:, -1] for key, value in replan_obs_history.items()}, goal_dict=replan_goal_batch).squeeze(-1)
                for local_idx, env_idx in enumerate(replan_ids):
                    self._train_action_queues[env_idx].extend(action_sequences[local_idx].unbind(dim=0))
                replan_data = {
                    "env_indices": np.asarray(replan_ids, dtype=np.int64),
                    "obs_history": replan_obs_history,
                    "goals": replan_goal_batch,
                    "chain_samples": chain_samples,
                    "chain_next_samples": chain_next_samples,
                    "chain_timesteps": chain_timesteps,
                    "log_probs": log_probs,
                    "values": values,
                }

            env_action = torch.stack([queue.popleft() for queue in self._train_action_queues], dim=0)
            completed_envs = np.asarray([env_idx for env_idx, queue in enumerate(self._train_action_queues) if len(queue) == 0], dtype=np.int64)
            if clip_actions:
                env_action = env_action.clamp(-1.0, 1.0)
            self._rollout_obs_history = _clone_history_state(obs_history)
            return (
                TensorUtils.to_numpy(env_action),
                replan_data,
                completed_envs,
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
            obs_history = self._advance_history(self._rollout_obs_history, obs_batch, episode_starts)
            batch_size = next(iter(obs_history.values())).shape[0]
            self._ensure_eval_action_queues(batch_size)
            reset_mask = torch.as_tensor(episode_starts, device=self.device, dtype=torch.bool)
            if reset_mask.any():
                assert self._eval_action_queues is not None
                for env_idx in torch.nonzero(reset_mask, as_tuple=False).flatten().tolist():
                    self._eval_action_queues[env_idx].clear()

            assert self._eval_action_queues is not None
            replan_ids = [env_idx for env_idx, queue in enumerate(self._eval_action_queues) if len(queue) == 0]
            if replan_ids:
                replan_index = torch.as_tensor(replan_ids, device=self.device, dtype=torch.long)
                replan_obs_history = {key: value.index_select(0, replan_index) for key, value in obs_history.items()}
                replan_goal_batch = None
                if goal_batch is not None:
                    replan_goal_batch = {key: value.index_select(0, replan_index) for key, value in goal_batch.items()}
                action_sequences = self._sample_action_sequence(
                    obs_history=replan_obs_history,
                    goal_batch=replan_goal_batch,
                    use_ema=self.use_ema_for_evaluation,
                )
                for local_idx, env_idx in enumerate(replan_ids):
                    self._eval_action_queues[env_idx].extend(action_sequences[local_idx].unbind(dim=0))

            env_action = torch.stack([queue.popleft() for queue in self._eval_action_queues], dim=0)
            if clip_actions:
                env_action = env_action.clamp(-1.0, 1.0)
            self._rollout_obs_history = _clone_history_state(obs_history)
            return TensorUtils.to_numpy(env_action)

    def evaluate_diffusion_rollout(
        self,
        observations: dict[str, torch.Tensor],
        goals: dict[str, torch.Tensor] | None,
        chain_samples: torch.Tensor,
        chain_next_samples: torch.Tensor,
        chain_timesteps: torch.Tensor,
        episode_starts: torch.Tensor,
        initial_history_state: Any = None,
    ) -> torch.Tensor:
        rollout_steps = next(iter(observations.values())).shape[0]
        obs_history = _clone_history_state(initial_history_state)
        log_probs = []
        for step in range(rollout_steps):
            obs_batch = {key: value[step] for key, value in observations.items()}
            goal_batch = None if goals is None else {key: value[step] for key, value in goals.items()}
            obs_history = self._advance_history(obs_history, obs_batch, episode_starts[step])
            log_probs.append(
                self._chain_log_prob(
                    obs_history=obs_history,
                    goal_batch=goal_batch,
                    chain_samples=chain_samples[step],
                    chain_next_samples=chain_next_samples[step],
                    chain_timesteps=chain_timesteps[step],
                )
            )
        return torch.stack(log_probs, dim=0).reshape(-1)

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
        processed = self.algo.process_batch_for_training(batch)
        actions = processed["actions"]
        obs_history = processed["obs"]
        goal_batch = processed.get("goal_obs")
        obs_cond = self._encode_obs_history(obs_history=obs_history, goal_batch=goal_batch, use_ema=False)
        batch_size = actions.shape[0]
        noise = torch.randn(actions.shape, device=self.device)
        timesteps = torch.randint(
            0,
            self.algo.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
        ).long()
        noisy_actions = self.algo.noise_scheduler.add_noise(actions, noise, timesteps)
        noise_pred = self._policy_nets(use_ema=False)["noise_pred_net"](
            noisy_actions,
            timesteps,
            global_cond=obs_cond,
        )
        return F.mse_loss(noise_pred, noise)

    def export_robomimic_checkpoint(self) -> dict:
        ckpt = deepcopy(self.bundle.ckpt_dict)
        ckpt["model"] = self.algo.serialize()
        return ckpt

    def export_policy_artifact(self, checkpoint_path: str) -> dict:
        del checkpoint_path
        return self.export_robomimic_checkpoint()

    def policy_artifact_extension(self) -> str:
        return ".pth"

    def _advance_history(
        self,
        history: dict[str, torch.Tensor] | None,
        obs_batch: dict[str, torch.Tensor],
        episode_starts: np.ndarray | torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = next(iter(obs_batch.values())).shape[0]
        reset_mask = torch.as_tensor(episode_starts, device=self.device, dtype=torch.bool)
        next_history: dict[str, torch.Tensor] = {}
        for key, obs_value in obs_batch.items():
            if history is None or key not in history or history[key].shape[0] != batch_size:
                repeated = obs_value.unsqueeze(1).repeat(1, self.observation_horizon, *([1] * (obs_value.ndim - 1)))
                next_history[key] = repeated
                continue
            updated = torch.cat([history[key][:, 1:], obs_value.unsqueeze(1)], dim=1)
            if reset_mask.any():
                updated[reset_mask] = obs_value[reset_mask].unsqueeze(1).repeat(
                    1, self.observation_horizon, *([1] * (obs_value.ndim - 1))
                )
            next_history[key] = updated
        return next_history

    def _policy_nets(self, use_ema: bool) -> nn.ModuleDict:
        if use_ema and self.algo.ema is not None:
            return self.algo.ema.averaged_model["policy"]
        return self.algo.nets["policy"]

    def _encode_obs_history(
        self,
        obs_history: dict[str, torch.Tensor],
        goal_batch: dict[str, torch.Tensor] | None,
        use_ema: bool,
    ) -> torch.Tensor:
        nets = self._policy_nets(use_ema=use_ema)
        inputs = {
            "obs": obs_history,
            "goal": goal_batch,
        }
        obs_features = TensorUtils.time_distributed(inputs, nets["obs_encoder"], inputs_as_kwargs=True)
        return obs_features.flatten(start_dim=1)

    def _sample_action_chain(
        self,
        obs_history: dict[str, torch.Tensor],
        goal_batch: dict[str, torch.Tensor] | None,
        deterministic: bool,
        use_ema: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        final_trajectory, chain_log_prob, chain_samples, chain_next_samples, chain_timesteps, _ = self._run_diffusion_chain(
            obs_history=obs_history,
            goal_batch=goal_batch,
            deterministic=deterministic,
            use_ema=use_ema,
        )
        action_start = self.observation_horizon - 1
        env_action = final_trajectory[:, action_start]
        return (
            env_action,
            chain_log_prob,
            chain_samples,
            chain_next_samples,
            chain_timesteps,
        )

    def _sample_action_sequence(
        self,
        obs_history: dict[str, torch.Tensor],
        goal_batch: dict[str, torch.Tensor] | None,
        use_ema: bool,
    ) -> torch.Tensor:
        final_trajectory, _, _, _, _, _ = self._run_diffusion_chain(
            obs_history=obs_history,
            goal_batch=goal_batch,
            deterministic=True,
            use_ema=use_ema,
        )
        action_start = self.observation_horizon - 1
        action_end = action_start + self.action_horizon
        return final_trajectory[:, action_start:action_end]

    def _sample_action_sequence_with_chain(
        self,
        obs_history: dict[str, torch.Tensor],
        goal_batch: dict[str, torch.Tensor] | None,
        use_ema: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        final_trajectory, _, chain_samples, chain_next_samples, chain_timesteps, chain_log_probs = self._run_diffusion_chain(
            obs_history=obs_history,
            goal_batch=goal_batch,
            deterministic=False,
            use_ema=use_ema,
            record_step_log_probs=True,
        )
        action_start = self.observation_horizon - 1
        action_end = action_start + self.action_horizon
        assert chain_log_probs is not None
        return (
            final_trajectory[:, action_start:action_end],
            chain_log_probs[:, :, action_start:action_end],
            chain_samples,
            chain_next_samples,
            chain_timesteps,
        )

    def _run_diffusion_chain(
        self,
        obs_history: dict[str, torch.Tensor],
        goal_batch: dict[str, torch.Tensor] | None,
        deterministic: bool,
        use_ema: bool,
        record_step_log_probs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        scheduler = self.algo.noise_scheduler
        scheduler.set_timesteps(self.num_inference_timesteps)
        obs_cond = self._encode_obs_history(obs_history=obs_history, goal_batch=goal_batch, use_ema=use_ema)
        batch_size = next(iter(obs_history.values())).shape[0]
        current = torch.randn((batch_size, self.prediction_horizon, self.action_dim), device=self.device)
        chain_samples = []
        chain_next_samples = []
        chain_timesteps = []
        chain_step_log_probs = [] if record_step_log_probs else None
        chain_log_prob = torch.zeros(batch_size, device=self.device)
        noise_pred_net = self._policy_nets(use_ema=use_ema)["noise_pred_net"]

        for timestep in scheduler.timesteps:
            timesteps = torch.full((batch_size,), int(timestep), device=self.device, dtype=torch.long)
            noise_pred = noise_pred_net(sample=current, timestep=timesteps, global_cond=obs_cond)
            mean, variance = _ddpm_reverse_stats(scheduler, noise_pred, timesteps, current)
            if deterministic or int(timestep) == 0:
                next_sample = mean
                step_log_prob = torch.zeros_like(next_sample)
            else:
                std = variance.sqrt()
                next_sample = mean + (std * torch.randn_like(mean))
                step_log_prob = _gaussian_log_prob_per_dim(next_sample, mean, variance)
                chain_log_prob += _gaussian_log_prob(next_sample, mean, variance)
            chain_samples.append(current)
            chain_next_samples.append(next_sample)
            chain_timesteps.append(timesteps)
            if chain_step_log_probs is not None:
                chain_step_log_probs.append(step_log_prob)
            current = next_sample

        return (
            current,
            chain_log_prob,
            torch.stack(chain_samples, dim=1),
            torch.stack(chain_next_samples, dim=1),
            torch.stack(chain_timesteps, dim=1),
            None if chain_step_log_probs is None else torch.stack(chain_step_log_probs, dim=1),
        )

    def _ensure_eval_action_queues(self, batch_size: int) -> None:
        if self._eval_action_queues is None or len(self._eval_action_queues) != batch_size:
            self._eval_action_queues = [deque() for _ in range(batch_size)]

    def _ensure_train_action_queues(self, batch_size: int) -> None:
        if self._train_action_queues is None or len(self._train_action_queues) != batch_size:
            self._train_action_queues = [deque() for _ in range(batch_size)]

    def _chain_log_prob(
        self,
        obs_history: dict[str, torch.Tensor],
        goal_batch: dict[str, torch.Tensor] | None,
        chain_samples: torch.Tensor,
        chain_next_samples: torch.Tensor,
        chain_timesteps: torch.Tensor,
    ) -> torch.Tensor:
        obs_cond = self._encode_obs_history(obs_history=obs_history, goal_batch=goal_batch, use_ema=False)
        noise_pred_net = self._policy_nets(use_ema=False)["noise_pred_net"]
        total_log_prob = torch.zeros(chain_samples.shape[0], device=self.device)
        for chain_idx in range(chain_samples.shape[1]):
            current = chain_samples[:, chain_idx]
            next_sample = chain_next_samples[:, chain_idx]
            timesteps = chain_timesteps[:, chain_idx]
            noise_pred = noise_pred_net(sample=current, timestep=timesteps, global_cond=obs_cond)
            mean, variance = _ddpm_reverse_stats(self.algo.noise_scheduler, noise_pred, timesteps, current)
            stochastic_mask = timesteps > 0
            if stochastic_mask.any():
                log_prob = _gaussian_log_prob(next_sample, mean, variance)
                total_log_prob = total_log_prob + torch.where(
                    stochastic_mask,
                    log_prob,
                    torch.zeros_like(log_prob),
                )
        return total_log_prob

    def _chain_log_prob_subsample(
        self,
        obs_history: dict[str, torch.Tensor],
        goal_batch: dict[str, torch.Tensor] | None,
        chain_samples: torch.Tensor,
        chain_next_samples: torch.Tensor,
        chain_timesteps: torch.Tensor,
    ) -> torch.Tensor:
        obs_cond = self._encode_obs_history(obs_history=obs_history, goal_batch=goal_batch, use_ema=False)
        noise_pred_net = self._policy_nets(use_ema=False)["noise_pred_net"]
        noise_pred = noise_pred_net(sample=chain_samples, timestep=chain_timesteps, global_cond=obs_cond)
        mean, variance = _ddpm_reverse_stats(self.algo.noise_scheduler, noise_pred, chain_timesteps, chain_samples)
        return _gaussian_log_prob_per_dim(chain_next_samples, mean, variance)

    def _chain_reverse_stats_tensor(
        self,
        obs_history: dict[str, torch.Tensor],
        goal_batch: dict[str, torch.Tensor] | None,
        chain_samples: torch.Tensor,
        chain_timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, chain_len = chain_timesteps.shape
        obs_cond = self._encode_obs_history(obs_history=obs_history, goal_batch=goal_batch, use_ema=False)
        obs_cond = obs_cond.unsqueeze(1).expand(-1, chain_len, -1).reshape(batch_size * chain_len, -1)
        flat_samples = chain_samples.reshape(batch_size * chain_len, chain_samples.shape[-2], chain_samples.shape[-1])
        flat_timesteps = chain_timesteps.reshape(-1)
        noise_pred_net = self._policy_nets(use_ema=False)["noise_pred_net"]
        noise_pred = noise_pred_net(sample=flat_samples, timestep=flat_timesteps, global_cond=obs_cond)
        mean, variance = _ddpm_reverse_stats(self.algo.noise_scheduler, noise_pred, flat_timesteps, flat_samples)
        mean = mean.reshape(batch_size, chain_len, *chain_samples.shape[-2:])
        variance = variance.reshape(batch_size, chain_len, 1, 1)
        return mean, variance


def _clone_history_state(state: Any) -> Any:
    if state is None:
        return None
    return {key: value.detach().clone() for key, value in state.items()}


def _clone_action_queues(queues: list[deque[torch.Tensor]] | None) -> list[deque[torch.Tensor]] | None:
    if queues is None:
        return None
    return [deque([action.detach().clone() for action in queue], maxlen=queue.maxlen) for queue in queues]


def _gaussian_log_prob(value: torch.Tensor, mean: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    variance = variance.clamp(min=1e-12)
    log_variance = variance.log()
    quadratic = ((value - mean) ** 2) / variance
    log_prob = -0.5 * (quadratic + log_variance + np.log(2.0 * np.pi))
    return log_prob.reshape(log_prob.shape[0], -1).sum(dim=-1)


def _gaussian_log_prob_per_dim(value: torch.Tensor, mean: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    variance = variance.clamp(min=1e-12)
    log_variance = variance.log()
    quadratic = ((value - mean) ** 2) / variance
    return -0.5 * (quadratic + log_variance + np.log(2.0 * np.pi))


def _ddpm_reverse_stats(
    scheduler,
    model_output: torch.Tensor,
    timesteps: torch.Tensor,
    sample: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    timesteps = timesteps.to(device=sample.device, dtype=torch.long)
    num_train_timesteps = int(scheduler.config.num_train_timesteps)
    if torch.any(timesteps < 0) or torch.any(timesteps >= num_train_timesteps):
        invalid_min = int(timesteps.min().item())
        invalid_max = int(timesteps.max().item())
        raise ValueError(
            f"Diffusion timesteps out of range: min={invalid_min}, max={invalid_max}, "
            f"expected in [0, {num_train_timesteps - 1}]"
        )

    alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
    betas = scheduler.betas.to(device=sample.device, dtype=sample.dtype)
    alphas = scheduler.alphas.to(device=sample.device, dtype=sample.dtype)

    alpha_prod_t = alphas_cumprod[timesteps].view(-1, 1, 1)
    alpha_prod_t_prev = torch.where(
        timesteps.view(-1, 1, 1) > 0,
        alphas_cumprod[(timesteps - 1).clamp(min=0)].view(-1, 1, 1),
        torch.ones_like(alpha_prod_t),
    )
    betas_t = betas[timesteps].view(-1, 1, 1)
    alphas_t = alphas[timesteps].view(-1, 1, 1)
    beta_prod_t = 1.0 - alpha_prod_t
    beta_prod_t_prev = 1.0 - alpha_prod_t_prev

    prediction_type = scheduler.config.prediction_type
    if prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
    elif prediction_type == "sample":
        pred_original_sample = model_output
    elif prediction_type == "v_prediction":
        pred_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
    else:
        raise ValueError(f"Unsupported DDPM prediction type: {prediction_type}")

    if getattr(scheduler.config, "clip_sample", False):
        pred_original_sample = pred_original_sample.clamp(-1.0, 1.0)

    pred_original_coeff = (alpha_prod_t_prev.sqrt() * betas_t) / beta_prod_t
    current_sample_coeff = (alphas_t.sqrt() * beta_prod_t_prev) / beta_prod_t
    mean = pred_original_coeff * pred_original_sample + current_sample_coeff * sample
    variance = ((1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t) * betas_t).clamp(min=1e-20)
    return mean, variance
