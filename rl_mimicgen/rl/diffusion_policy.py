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
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
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
        ft_denoising_steps: int | None = None,
        use_ddim: bool = False,
        ddim_steps: int | None = None,
        act_steps: int | None = None,
        min_sampling_denoising_std: float | None = None,
        min_logprob_denoising_std: float | None = None,
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
        self.use_ddim = bool(use_ddim)

        self.observation_horizon = int(bundle.config.algo.horizon.observation_horizon)
        self.action_horizon = int(bundle.config.algo.horizon.action_horizon)
        self.prediction_horizon = int(bundle.config.algo.horizon.prediction_horizon)
        self.act_steps = self._resolve_act_steps(bundle=bundle, act_steps=act_steps)
        self.action_dim = int(bundle.shape_meta["ac_dim"])
        self.num_inference_timesteps = int(
            self._resolve_num_inference_timesteps(
                bundle=bundle,
                num_inference_timesteps=num_inference_timesteps,
                ddim_steps=ddim_steps,
            )
        )
        train_config = _safe_getattr(bundle.config, "train", None)
        configured_ft_steps = _safe_getattr(train_config, "ft_denoising_steps", None)
        if configured_ft_steps is None:
            configured_ft_steps = _safe_getattr(bundle.config.algo, "ft_denoising_steps", None)
        if ft_denoising_steps is not None:
            configured_ft_steps = ft_denoising_steps
        self.ft_denoising_steps = None if configured_ft_steps is None else int(configured_ft_steps)
        self.min_sampling_denoising_std = (
            None if min_sampling_denoising_std is None else float(min_sampling_denoising_std)
        )
        self.min_logprob_denoising_std = (
            None if min_logprob_denoising_std is None else float(min_logprob_denoising_std)
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
        self.sampling_scheduler = self._build_sampling_scheduler(bundle=bundle)

        self._rollout_obs_history: dict[str, torch.Tensor] | None = None
        self._train_action_queues: list[deque[torch.Tensor]] | None = None
        self._eval_action_queues: list[deque[torch.Tensor]] | None = None

    def _resolve_num_inference_timesteps(
        self,
        bundle: PolicyBundle,
        num_inference_timesteps: int | None,
        ddim_steps: int | None,
    ) -> int:
        if self.use_ddim:
            configured_ddim_steps = ddim_steps
            if configured_ddim_steps is None:
                configured_ddim_steps = _safe_getattr(bundle.config.algo.ddim, "num_inference_timesteps", None)
            if num_inference_timesteps is not None:
                configured_ddim_steps = num_inference_timesteps
            if configured_ddim_steps is None:
                raise ValueError("DDIM fine-tuning requires ddim_steps or num_inference_timesteps to be configured.")
            return int(configured_ddim_steps)

        configured_ddpm_steps = num_inference_timesteps
        if configured_ddpm_steps is None:
            configured_ddpm_steps = _safe_getattr(bundle.config.algo.ddpm, "num_inference_timesteps", None)
        if configured_ddpm_steps is None:
            raise ValueError("DDPM fine-tuning requires num_inference_timesteps to be configured.")
        return int(configured_ddpm_steps)

    def _resolve_act_steps(self, bundle: PolicyBundle, act_steps: int | None) -> int:
        configured_act_steps = act_steps
        if configured_act_steps is None:
            configured_act_steps = _safe_getattr(bundle.config.algo.horizon, "act_steps", None)
        if configured_act_steps is None:
            configured_act_steps = self.action_horizon
        configured_act_steps = int(configured_act_steps)
        if configured_act_steps <= 0:
            raise ValueError(f"act_steps must be positive, got {configured_act_steps}.")
        if configured_act_steps > self.prediction_horizon:
            raise ValueError(
                f"act_steps={configured_act_steps} exceeds prediction_horizon={self.prediction_horizon}."
            )
        return configured_act_steps

    def _build_sampling_scheduler(self, bundle: PolicyBundle):
        if self.use_ddim:
            ddim_cfg = bundle.config.algo.ddim
            return DDIMScheduler(
                num_train_timesteps=int(_safe_getattr(ddim_cfg, "num_train_timesteps", 100)),
                beta_schedule=str(_safe_getattr(ddim_cfg, "beta_schedule", "squaredcos_cap_v2")),
                clip_sample=bool(_safe_getattr(ddim_cfg, "clip_sample", True)),
                set_alpha_to_one=bool(_safe_getattr(ddim_cfg, "set_alpha_to_one", True)),
                steps_offset=int(_safe_getattr(ddim_cfg, "steps_offset", 0)),
                prediction_type=str(_safe_getattr(ddim_cfg, "prediction_type", "epsilon")),
            )

        ddpm_cfg = bundle.config.algo.ddpm
        if not bool(_safe_getattr(ddpm_cfg, "enabled", False)):
            raise NotImplementedError("The diffusion online RL path requires a DDPM checkpoint config when DDIM is not enabled.")
        return DDPMScheduler(
            num_train_timesteps=int(_safe_getattr(ddpm_cfg, "num_train_timesteps", 100)),
            beta_schedule=str(_safe_getattr(ddpm_cfg, "beta_schedule", "squaredcos_cap_v2")),
            clip_sample=bool(_safe_getattr(ddpm_cfg, "clip_sample", True)),
            prediction_type=str(_safe_getattr(ddpm_cfg, "prediction_type", "epsilon")),
        )

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
        action_end = action_start + self.act_steps
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
        action_end = action_start + self.act_steps
        assert chain_log_probs is not None
        chain_samples, chain_next_samples, chain_timesteps, chain_log_probs = self._truncate_finetuning_chain(
            chain_samples=chain_samples,
            chain_next_samples=chain_next_samples,
            chain_timesteps=chain_timesteps,
            chain_log_probs=chain_log_probs,
        )
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
        scheduler = self.sampling_scheduler
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
            mean, variance = _scheduler_reverse_stats(scheduler, noise_pred, timesteps, current)
            if deterministic or int(timestep) == 0:
                next_sample = mean
                step_log_prob = torch.zeros_like(next_sample)
            else:
                sampling_variance = _apply_min_std_floor(variance, self.min_sampling_denoising_std)
                std = sampling_variance.sqrt()
                next_sample = mean + (std * torch.randn_like(mean))
                logprob_variance = _apply_min_std_floor(variance, self.min_logprob_denoising_std)
                step_log_prob = _gaussian_log_prob_per_dim(next_sample, mean, logprob_variance)
                chain_log_prob += _gaussian_log_prob(next_sample, mean, logprob_variance)
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

    def _truncate_finetuning_chain(
        self,
        chain_samples: torch.Tensor,
        chain_next_samples: torch.Tensor,
        chain_timesteps: torch.Tensor,
        chain_log_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.ft_denoising_steps is None:
            return chain_samples, chain_next_samples, chain_timesteps, chain_log_probs
        retained_steps = max(1, min(int(self.ft_denoising_steps), chain_timesteps.shape[1]))
        return (
            chain_samples[:, -retained_steps:],
            chain_next_samples[:, -retained_steps:],
            chain_timesteps[:, -retained_steps:],
            chain_log_probs[:, -retained_steps:],
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
            mean, variance = _scheduler_reverse_stats(self.sampling_scheduler, noise_pred, timesteps, current)
            variance = _apply_min_std_floor(variance, self.min_logprob_denoising_std)
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
        if chain_timesteps.ndim == 1:
            noise_pred = noise_pred_net(sample=chain_samples, timestep=chain_timesteps, global_cond=obs_cond)
            mean, variance = _scheduler_reverse_stats(self.sampling_scheduler, noise_pred, chain_timesteps, chain_samples)
            variance = _apply_min_std_floor(variance, self.min_logprob_denoising_std)
            return _gaussian_log_prob_per_dim(chain_next_samples, mean, variance)

        batch_size, chain_len = chain_timesteps.shape
        obs_cond = obs_cond.unsqueeze(1).expand(-1, chain_len, -1).reshape(batch_size * chain_len, -1)
        flat_samples = chain_samples.reshape(batch_size * chain_len, chain_samples.shape[-2], chain_samples.shape[-1])
        flat_next_samples = chain_next_samples.reshape(
            batch_size * chain_len, chain_next_samples.shape[-2], chain_next_samples.shape[-1]
        )
        flat_timesteps = chain_timesteps.reshape(-1)
        noise_pred = noise_pred_net(sample=flat_samples, timestep=flat_timesteps, global_cond=obs_cond)
        mean, variance = _scheduler_reverse_stats(self.sampling_scheduler, noise_pred, flat_timesteps, flat_samples)
        variance = _apply_min_std_floor(variance, self.min_logprob_denoising_std)
        log_probs = _gaussian_log_prob_per_dim(flat_next_samples, mean, variance)
        return log_probs.reshape(batch_size, chain_len, *chain_next_samples.shape[-2:])

def _clone_history_state(state: Any) -> Any:
    if state is None:
        return None
    return {key: value.detach().clone() for key, value in state.items()}


def _clone_action_queues(queues: list[deque[torch.Tensor]] | None) -> list[deque[torch.Tensor]] | None:
    if queues is None:
        return None
    return [deque([action.detach().clone() for action in queue], maxlen=queue.maxlen) for queue in queues]


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    try:
        return getattr(obj, name)
    except (AttributeError, RuntimeError):
        return default


def _apply_min_std_floor(variance: torch.Tensor, min_std: float | None) -> torch.Tensor:
    variance = variance.clamp(min=1e-12)
    if min_std is None:
        return variance
    min_variance = max(float(min_std), 0.0) ** 2
    if min_variance == 0.0:
        return variance
    return variance.clamp(min=min_variance)


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


def _scheduler_reverse_stats(
    scheduler,
    model_output: torch.Tensor,
    timesteps: torch.Tensor,
    sample: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(scheduler, DDIMScheduler):
        return _ddim_reverse_stats(scheduler, model_output, timesteps, sample)
    if isinstance(scheduler, DDPMScheduler):
        return _ddpm_reverse_stats(scheduler, model_output, timesteps, sample)
    raise TypeError(f"Unsupported diffusion scheduler type: {type(scheduler)!r}")


def _ddpm_reverse_stats(
    scheduler: DDPMScheduler,
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


def _ddim_reverse_stats(
    scheduler: DDIMScheduler,
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

    if scheduler.num_inference_steps is None:
        raise ValueError("DDIM scheduler timesteps have not been initialized; call set_timesteps first.")

    step_stride = max(1, num_train_timesteps // int(scheduler.num_inference_steps))
    prev_timesteps = timesteps - step_stride

    flat_timesteps = timesteps.reshape(-1)
    flat_prev_timesteps = prev_timesteps.reshape(-1)
    flat_model_output = model_output.reshape(-1, *model_output.shape[1:])
    flat_sample = sample.reshape(-1, *sample.shape[1:])

    prev_samples = []
    variances = []
    for idx in range(flat_timesteps.shape[0]):
        timestep = int(flat_timesteps[idx].item())
        prev_timestep = int(flat_prev_timesteps[idx].item())
        step_out = scheduler.step(
            model_output=flat_model_output[idx],
            timestep=timestep,
            sample=flat_sample[idx],
            eta=0.0,
            return_dict=True,
        )
        prev_samples.append(step_out.prev_sample)
        variances.append(float(scheduler._get_variance(timestep, prev_timestep)))

    mean = torch.stack(prev_samples, dim=0).reshape_as(sample)
    variance = torch.tensor(variances, dtype=sample.dtype, device=sample.device).view(-1, 1, 1)
    variance = variance.reshape(*timesteps.shape, 1, 1)
    return mean, variance
