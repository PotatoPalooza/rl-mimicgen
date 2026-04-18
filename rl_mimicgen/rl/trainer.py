from __future__ import annotations

import json
import random
from collections import OrderedDict
from pathlib import Path
from time import perf_counter

import h5py
import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader

from robomimic.utils.train_utils import dataset_factory

from rl_mimicgen.rl.config import OnlineRLConfig
from rl_mimicgen.rl.awac import AWAC
from rl_mimicgen.rl.awac_policy import AWACPolicyAdapter
from rl_mimicgen.rl.diffusion_policy import DiffusionOnlinePolicyAdapter
from rl_mimicgen.rl.diffusion_storage import DiffusionRolloutStorage
from rl_mimicgen.rl.dppo import DiffusionPPO
from rl_mimicgen.rl.policy import OnlinePolicyAdapter, load_policy_bundle
from rl_mimicgen.rl.ppo import DemoAugmentedPPO
from rl_mimicgen.rl.replay_buffer import ReplayBuffer
from rl_mimicgen.rl.robomimic_env import ParallelRobomimicVectorEnv, SerialRobomimicVectorEnv, make_robosuite_env_from_checkpoint
from rl_mimicgen.rl.storage import RolloutStorage


class OnlineRLTrainer:
    def __init__(self, config: OnlineRLConfig):
        self.config = config
        self.output_dir = Path(config.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"

        self.device = self._resolve_device(config.device)
        self._seed_everything(config.seed)

        self.bundle = load_policy_bundle(config.checkpoint_path, self.device)
        self.is_diffusion_policy = getattr(self.bundle.config, "algo_name", "") == "diffusion_policy"
        self.algorithm_name = str(config.algorithm).lower()
        if self.is_diffusion_policy:
            if self.algorithm_name != "dppo":
                raise ValueError("Diffusion policies currently support only algorithm='dppo'.")
            if config.residual.enabled:
                raise ValueError("Residual fine-tuning is not supported for diffusion policies.")
            self.policy = DiffusionOnlinePolicyAdapter(
                bundle=self.bundle,
                device=self.device,
                num_inference_timesteps=config.diffusion.num_inference_timesteps,
                ft_denoising_steps=config.diffusion.ft_denoising_steps,
                use_ddim=config.diffusion.use_ddim,
                ddim_steps=config.diffusion.ddim_steps,
                act_steps=config.diffusion.act_steps,
                min_sampling_denoising_std=config.diffusion.min_sampling_denoising_std,
                min_logprob_denoising_std=config.diffusion.min_logprob_denoising_std,
                use_ema=config.diffusion.use_ema,
            )
        else:
            if self.algorithm_name == "awac":
                self.policy = AWACPolicyAdapter(
                    bundle=self.bundle,
                    device=self.device,
                    init_log_std=config.ppo.init_log_std,
                    min_log_std=config.ppo.min_log_std,
                    residual_enabled=config.residual.enabled,
                    residual_scale=config.residual.scale,
                    q_hidden_sizes=tuple(int(dim) for dim in config.awac.q_hidden_sizes),
                    num_q_networks=config.awac.num_q_networks,
                    target_tau=config.awac.target_tau,
                    num_value_action_samples=config.awac.num_action_samples,
                )
            else:
                self.policy = OnlinePolicyAdapter(
                    bundle=self.bundle,
                    device=self.device,
                    init_log_std=config.ppo.init_log_std,
                    min_log_std=config.ppo.min_log_std,
                    residual_enabled=config.residual.enabled,
                    residual_scale=config.residual.scale,
                )

        self.env = self._build_vector_env()
        self.eval_env = self._build_eval_env(
            render=config.evaluation.render,
            render_offscreen=bool(config.evaluation.video_path),
        )
        self._emit(
            "[env] "
            f"train_backend={self._env_backend_name(self.env)} "
            f"train_num_envs={getattr(self.env, 'num_envs', 1)} "
            f"eval_backend={self._env_backend_name(self.eval_env)} "
            f"eval_num_envs={getattr(self.eval_env, 'num_envs', 1)} "
            f"device={self.device}"
        )
        action_dim = int(np.prod(self.env.single_action_space.shape))

        self.demo_loader = None
        self.demo_iter = None
        if config.demo.enabled:
            self.demo_loader = self._build_demo_loader()
            self.demo_iter = iter(self.demo_loader)

        self.storage = None
        self.replay_buffer = None
        self._awac_episode_observations: list[dict[str, list[np.ndarray]]] | None = None
        self._awac_episode_goals: list[dict[str, list[np.ndarray]]] | None = None
        self._awac_episode_actions: list[list[np.ndarray]] | None = None

        if self.is_diffusion_policy:
            self.storage = DiffusionRolloutStorage(
                rollout_steps=self.config.rollout_steps,
                num_envs=self.config.num_envs,
                obs_shapes=self.policy.obs_shapes,
                goal_shapes=self.policy.goal_shapes,
                action_dim=action_dim,
                prediction_horizon=self.policy.prediction_horizon,
                chain_length=self.policy.num_inference_timesteps,
                device=self.device,
            )
            self.algorithm = DiffusionPPO(
                policy=self.policy,
                demo_batch_iterator=self._demo_batch_generator() if self.config.demo.enabled else None,
                demo_loss_fn=self.policy.demo_loss if self.config.demo.enabled else None,
                demo_coef=self.config.demo.coef if self.config.demo.enabled else 0.0,
                actor_learning_rate=self.config.optimizer.actor_lr,
                critic_learning_rate=self.config.optimizer.value_lr,
                weight_decay=self.config.optimizer.weight_decay,
                critic_warmup_updates=self.config.ppo.critic_warmup_updates,
                actor_freeze_env_steps=self.config.ppo.actor_freeze_env_steps,
                num_learning_epochs=self.config.ppo.update_epochs,
                num_mini_batches=self.config.ppo.num_minibatches,
                clip_param=self.config.ppo.clip_ratio,
                value_loss_coef=self.config.ppo.value_coef,
                gamma_denoising=self.config.diffusion.gamma_denoising,
                act_steps=self.policy.act_steps,
                target_kl=self.config.ppo.target_kl,
                max_grad_norm=self.config.optimizer.max_grad_norm,
                device=self.device,
            )
        else:
            demo_batch_iterator = self._demo_batch_generator() if self.config.demo.enabled else None
            demo_loss_fn = self.policy.demo_loss if self.config.demo.enabled else None
            if self.algorithm_name == "awac":
                self.replay_buffer = ReplayBuffer(
                    capacity=self.config.awac.replay_capacity,
                    obs_shapes=self.policy.obs_shapes,
                    goal_shapes=self.policy.goal_shapes,
                    action_dim=action_dim,
                )
                self._init_awac_episode_storage()
                self._load_awac_offline_data()
                self.algorithm = AWAC(
                    policy=self.policy,
                    batch_size=self.config.awac.actor_batch_size,
                    discount=self.config.awac.discount,
                    demo_batch_iterator=demo_batch_iterator,
                    demo_loss_fn=demo_loss_fn,
                    demo_coef=self.config.demo.coef if self.config.demo.enabled else 0.0,
                    actor_learning_rate=self.config.optimizer.actor_lr,
                    critic_learning_rate=self.config.optimizer.value_lr,
                    weight_decay=self.config.optimizer.weight_decay,
                    critic_warmup_updates=self.config.awac.critic_warmup_updates,
                    actor_freeze_env_steps=self.config.awac.actor_freeze_env_steps,
                    num_learning_epochs=self.config.awac.update_epochs,
                    num_mini_batches=self.config.awac.num_minibatches,
                    beta=self.config.awac.beta,
                    max_weight=self.config.awac.max_weight,
                    normalize_weights=self.config.awac.normalize_weights,
                    critic_loss_coef=self.config.awac.value_coef,
                    entropy_coef=self.config.awac.entropy_coef,
                    max_grad_norm=self.config.optimizer.max_grad_norm,
                    critic_huber_loss=self.config.awac.critic_huber_loss,
                    num_action_samples=self.config.awac.num_action_samples,
                    device=self.device,
                )
            elif self.algorithm_name == "ppo":
                self.storage = RolloutStorage(
                    rollout_steps=self.config.rollout_steps,
                    num_envs=self.config.num_envs,
                    obs_shapes=self.policy.obs_shapes,
                    goal_shapes=self.policy.goal_shapes,
                    action_dim=action_dim,
                    device=self.device,
                )
                self.algorithm = DemoAugmentedPPO(
                    policy=self.policy,
                    demo_batch_iterator=demo_batch_iterator,
                    demo_loss_fn=demo_loss_fn,
                    demo_coef=self.config.demo.coef if self.config.demo.enabled else 0.0,
                    actor_learning_rate=self.config.optimizer.actor_lr,
                    critic_learning_rate=self.config.optimizer.value_lr,
                    weight_decay=self.config.optimizer.weight_decay,
                    critic_warmup_updates=self.config.ppo.critic_warmup_updates,
                    actor_freeze_env_steps=self.config.ppo.actor_freeze_env_steps,
                    num_learning_epochs=self.config.ppo.update_epochs,
                    num_mini_batches=self.config.ppo.num_minibatches,
                    clip_param=self.config.ppo.clip_ratio,
                    value_loss_coef=self.config.ppo.value_coef,
                    entropy_coef=self.config.ppo.entropy_coef,
                    target_kl=self.config.ppo.target_kl,
                    max_grad_norm=self.config.optimizer.max_grad_norm,
                    device=self.device,
                )
            else:
                raise ValueError(f"Unsupported non-diffusion algorithm '{self.algorithm_name}'.")

        self.best_success = float("-inf")
        self.last_eval_metrics: dict[str, float] | None = None
        self.best_policy_path: Path | None = None
        self.best_trainer_path: Path | None = None
        self.current_demo_coef = config.demo.coef

        self.config.dump_json(self.output_dir / "config.json")

    def train(self) -> None:
        if self.algorithm_name == "awac":
            self._train_awac()
            return

        obs, _ = self.env.reset(seed=self.config.seed)
        goal = self._current_goal(self.env)
        running_returns = np.zeros(self.config.num_envs, dtype=np.float32)
        running_lengths = np.zeros(self.config.num_envs, dtype=np.float32)
        episode_horizon = self._env_horizon()
        total_env_steps = 0
        episode_starts = np.ones(self.config.num_envs, dtype=bool)

        for update in range(self.config.total_updates):
            completed_returns = []
            completed_lengths = []
            completed_success = []

            self.algorithm.train_mode()
            self.algorithm.demo_coef = self.current_demo_coef
            self.storage.reset()
            if not self.is_diffusion_policy:
                self.storage.set_initial_rnn_state(self.policy.clone_training_rollout_state())

            rollout_step_count = 0
            rollout_start_time = perf_counter()
            while rollout_step_count < self.config.rollout_steps:
                if self.is_diffusion_policy:
                    env_actions, replan_data, completed_envs = self.policy.act(
                        obs=obs,
                        goal=goal,
                        episode_starts=episode_starts,
                        clip_actions=self.config.ppo.clip_actions,
                    )
                    if replan_data is not None:
                        self.storage.start_decisions(
                            env_indices=replan_data["env_indices"],
                            obs_history=replan_data["obs_history"],
                            goals=replan_data["goals"],
                            chain_samples=replan_data["chain_samples"],
                            chain_next_samples=replan_data["chain_next_samples"],
                            chain_timesteps=replan_data["chain_timesteps"],
                            log_probs=replan_data["log_probs"],
                            values=replan_data["values"],
                        )
                else:
                    env_actions, policy_actions, log_probs, values = self.policy.act(
                        obs=obs,
                        goal=goal,
                        episode_starts=episode_starts,
                        clip_actions=self.config.ppo.clip_actions,
                    )
                next_obs, rewards, terminated, truncated, infos = self.env.step(env_actions)
                done = terminated | truncated

                if self.is_diffusion_policy:
                    self.storage.accumulate_step(
                        rewards=rewards,
                        dones=done.astype(np.float32),
                    )
                    finalize_envs = np.union1d(completed_envs, np.nonzero(done)[0].astype(np.int64))
                    if finalize_envs.size:
                        self.storage.finalize_decisions(finalize_envs)
                else:
                    self.storage.add(
                        obs=obs,
                        actions=policy_actions,
                        goals=goal,
                        log_probs=log_probs,
                        rewards=rewards,
                        dones=done.astype(np.float32),
                        episode_starts=episode_starts.astype(np.float32),
                        values=values,
                    )

                running_returns += rewards
                running_lengths += 1

                final_info = infos.get("final_info", [None] * self.config.num_envs)
                for idx, info in enumerate(final_info):
                    if info is None:
                        continue
                    completed_returns.append(float(running_returns[idx]))
                    completed_lengths.append(float(running_lengths[idx]))
                    completed_success.append(float(info.get("success", False)))
                    running_returns[idx] = 0.0
                    running_lengths[idx] = 0.0

                obs = next_obs
                goal = self._current_goal(self.env)
                episode_starts = done
                rollout_step_count += 1

            rollout_duration_sec = perf_counter() - rollout_start_time
            rollout_env_steps = rollout_step_count * self.config.num_envs
            total_env_steps += rollout_env_steps
            self.algorithm.set_total_env_steps(total_env_steps)
            last_values = self.policy.predict_value(obs=obs, goal=goal)
            self.storage.compute_returns_and_advantages(
                last_value=last_values,
                gamma=self.config.ppo.gamma,
                gae_lambda=self.config.ppo.gae_lambda,
            )

            update_start_time = perf_counter()
            update_metrics = self.algorithm.update(self.storage.as_batch())
            update_duration_sec = perf_counter() - update_start_time
            train_metrics = {
                "policy_loss": float(update_metrics["surrogate"]),
                "value_loss": float(update_metrics["value"]),
                "entropy": float(update_metrics["entropy"]),
                "demo_loss": float(update_metrics["demo"]),
                "demo_weight": float(self.current_demo_coef),
                "approx_kl": float(update_metrics["approx_kl"]),
                "ppo_early_stop": float(update_metrics["early_stop"]),
                "effective_num_minibatches": float(update_metrics["effective_num_minibatches"]),
            }
            train_metrics["update"] = update
            train_metrics["episodes_completed"] = float(len(completed_returns))
            train_metrics["episode_return_mean"] = float(np.mean(completed_returns)) if completed_returns else 0.0
            train_metrics["episode_length_mean"] = float(np.mean(completed_lengths)) if completed_lengths else 0.0
            train_metrics["success_rate_mean"] = float(np.mean(completed_success)) if completed_success else 0.0
            train_metrics["rollout_horizon"] = float(episode_horizon)
            train_metrics["env_steps"] = float(total_env_steps)
            train_metrics["algorithm"] = 0.0 if self.algorithm_name == "ppo" else 1.0
            train_metrics["timing/rollout_sec"] = rollout_duration_sec
            train_metrics["timing/update_sec"] = update_duration_sec
            train_metrics["timing/rollout_sps"] = (
                float(rollout_env_steps) / rollout_duration_sec if rollout_duration_sec > 0.0 else 0.0
            )
            for metric_name in (
                "weight_mean",
                "weight_std",
                "weight_min",
                "weight_max",
                "weight_clipped_frac",
                "advantage_mean",
                "advantage_std",
                "advantage_min",
                "advantage_max",
            ):
                if metric_name in update_metrics:
                    train_metrics[f"awac_{metric_name}"] = float(update_metrics[metric_name])
            current_log_std = self.policy.current_log_std()
            if current_log_std is not None:
                train_metrics["log_std_mean"] = float(current_log_std.mean().item())
                train_metrics["log_std_min"] = float(current_log_std.min().item())
                train_metrics["log_std_max"] = float(current_log_std.max().item())

            if self.config.evaluation.enabled and ((update + 1) % self.config.evaluation.every_n_updates == 0):
                eval_start_time = perf_counter()
                eval_metrics = self.evaluate(update)
                eval_duration_sec = perf_counter() - eval_start_time
                self.last_eval_metrics = eval_metrics
                train_metrics.update({f"eval/{k}": v for k, v in eval_metrics.items()})
                train_metrics["eval/was_run"] = 1.0
                train_metrics["timing/eval_sec"] = eval_duration_sec
                if eval_metrics["success_rate"] > self.best_success:
                    self.best_success = eval_metrics["success_rate"]
                    self.save_best_checkpoint(update + 1, eval_metrics["success_rate"])
            else:
                train_metrics["eval/was_run"] = 0.0
                train_metrics["timing/eval_sec"] = 0.0
                if self.last_eval_metrics is not None:
                    train_metrics["eval/last_success_rate"] = self.last_eval_metrics["success_rate"]
                    train_metrics["eval/last_return_mean"] = self.last_eval_metrics["return_mean"]
                    train_metrics["eval/last_length_mean"] = self.last_eval_metrics["length_mean"]
                    train_metrics["eval/last_evaluated_episodes"] = self.last_eval_metrics["evaluated_episodes"]
            train_metrics["eval/best_success_rate"] = self.best_success if self.best_success != float("-inf") else 0.0

            self._log_metrics(train_metrics)

            if (update + 1) % self.config.save_every_n_updates == 0:
                self.save_checkpoint(tag=f"update_{update + 1:04d}")

            self.current_demo_coef *= self.config.demo.decay

        self.save_checkpoint(tag="latest")
        self._close_env(self.env)
        self._close_env(self.eval_env)

    def _train_awac(self) -> None:
        if self.replay_buffer is None:
            raise RuntimeError("AWAC replay buffer was not initialized.")

        obs, _ = self.env.reset(seed=self.config.seed)
        goal = self._current_goal(self.env)
        running_returns = np.zeros(self.config.num_envs, dtype=np.float32)
        running_lengths = np.zeros(self.config.num_envs, dtype=np.float32)
        episode_horizon = self._env_horizon()
        total_env_steps = 0
        episode_starts = np.ones(self.config.num_envs, dtype=bool)

        for update in range(self.config.total_updates):
            completed_returns = []
            completed_lengths = []
            completed_success = []

            self.algorithm.train_mode()
            self.algorithm.demo_coef = self.current_demo_coef

            rollout_step_count = 0
            rollout_start_time = perf_counter()
            while rollout_step_count < self.config.rollout_steps:
                env_actions, policy_actions, _, _ = self.policy.act(
                    obs=obs,
                    goal=goal,
                    episode_starts=episode_starts,
                    clip_actions=self.config.ppo.clip_actions,
                )
                next_obs, rewards, terminated, truncated, infos = self.env.step(env_actions)
                done = terminated | truncated
                next_goal = self._current_goal(self.env)

                self.replay_buffer.add_batch(
                    obs=obs,
                    actions=policy_actions,
                    rewards=rewards,
                    next_obs=next_obs,
                    dones=done.astype(np.float32),
                    goals=goal,
                    next_goals=next_goal,
                )
                self._record_awac_online_sequences(
                    obs=obs,
                    actions=policy_actions,
                    goals=goal,
                    done=done,
                )

                running_returns += rewards
                running_lengths += 1

                final_info = infos.get("final_info", [None] * self.config.num_envs)
                for idx, info in enumerate(final_info):
                    if info is None:
                        continue
                    completed_returns.append(float(running_returns[idx]))
                    completed_lengths.append(float(running_lengths[idx]))
                    completed_success.append(float(info.get("success", False)))
                    running_returns[idx] = 0.0
                    running_lengths[idx] = 0.0

                obs = next_obs
                goal = next_goal
                episode_starts = done
                rollout_step_count += 1

            rollout_duration_sec = perf_counter() - rollout_start_time
            rollout_env_steps = rollout_step_count * self.config.num_envs
            total_env_steps += rollout_env_steps
            self.algorithm.set_total_env_steps(total_env_steps)

            update_start_time = perf_counter()
            update_metrics = self.algorithm.update(self.replay_buffer)
            update_duration_sec = perf_counter() - update_start_time
            train_metrics = {
                "policy_loss": float(update_metrics["surrogate"]),
                "value_loss": float(update_metrics["value"]),
                "entropy": float(update_metrics["entropy"]),
                "demo_loss": float(update_metrics["demo"]),
                "demo_weight": float(self.current_demo_coef),
                "approx_kl": float(update_metrics["approx_kl"]),
                "ppo_early_stop": float(update_metrics["early_stop"]),
                "effective_num_minibatches": float(update_metrics["effective_num_minibatches"]),
                "replay_size": float(len(self.replay_buffer)),
                "update": update,
                "episodes_completed": float(len(completed_returns)),
                "episode_return_mean": float(np.mean(completed_returns)) if completed_returns else 0.0,
                "episode_length_mean": float(np.mean(completed_lengths)) if completed_lengths else 0.0,
                "success_rate_mean": float(np.mean(completed_success)) if completed_success else 0.0,
                "rollout_horizon": float(episode_horizon),
                "env_steps": float(total_env_steps),
                "algorithm": 1.0,
                "timing/rollout_sec": rollout_duration_sec,
                "timing/update_sec": update_duration_sec,
                "timing/rollout_sps": float(rollout_env_steps) / rollout_duration_sec if rollout_duration_sec > 0.0 else 0.0,
            }
            for metric_name in (
                "weight_mean",
                "weight_std",
                "weight_min",
                "weight_max",
                "weight_clipped_frac",
                "advantage_mean",
                "advantage_std",
                "advantage_min",
                "advantage_max",
                "log_prob_mean",
                "log_prob_std",
                "log_prob_min",
                "log_prob_max",
                "q_pred_mean",
                "target_q_mean",
                "q_gap_mean",
            ):
                if metric_name in update_metrics:
                    train_metrics[f"awac_{metric_name}"] = float(update_metrics[metric_name])
            current_log_std = self.policy.current_log_std()
            if current_log_std is not None:
                train_metrics["log_std_mean"] = float(current_log_std.mean().item())
                train_metrics["log_std_min"] = float(current_log_std.min().item())
                train_metrics["log_std_max"] = float(current_log_std.max().item())

            if self.config.evaluation.enabled and ((update + 1) % self.config.evaluation.every_n_updates == 0):
                eval_start_time = perf_counter()
                eval_metrics = self.evaluate(update)
                eval_duration_sec = perf_counter() - eval_start_time
                self.last_eval_metrics = eval_metrics
                train_metrics.update({f"eval/{k}": v for k, v in eval_metrics.items()})
                train_metrics["eval/was_run"] = 1.0
                train_metrics["timing/eval_sec"] = eval_duration_sec
                if eval_metrics["success_rate"] > self.best_success:
                    self.best_success = eval_metrics["success_rate"]
                    self.save_best_checkpoint(update + 1, eval_metrics["success_rate"])
            else:
                train_metrics["eval/was_run"] = 0.0
                train_metrics["timing/eval_sec"] = 0.0
                if self.last_eval_metrics is not None:
                    train_metrics["eval/last_success_rate"] = self.last_eval_metrics["success_rate"]
                    train_metrics["eval/last_return_mean"] = self.last_eval_metrics["return_mean"]
                    train_metrics["eval/last_length_mean"] = self.last_eval_metrics["length_mean"]
                    train_metrics["eval/last_evaluated_episodes"] = self.last_eval_metrics["evaluated_episodes"]
            train_metrics["eval/best_success_rate"] = self.best_success if self.best_success != float("-inf") else 0.0

            self._log_metrics(train_metrics)

            if (update + 1) % self.config.save_every_n_updates == 0:
                self.save_checkpoint(tag=f"update_{update + 1:04d}")

            self.current_demo_coef *= self.config.demo.decay

        self.save_checkpoint(tag="latest")
        self._close_env(self.env)
        self._close_env(self.eval_env)

    def evaluate(self, update: int) -> dict[str, float]:
        saved_rollout_state = self.policy.clone_rollout_state()
        try:
            if hasattr(self.eval_env, "num_envs"):
                return self._evaluate_vectorized(update)
            return self._evaluate_single(update)
        finally:
            self.policy.restore_rollout_state(saved_rollout_state)

    def _evaluate_single(self, update: int) -> dict[str, float]:
        video_writer = None
        if self.config.evaluation.video_path:
            video_relpath = self.config.evaluation.video_path.format(update=update + 1)
            video_path = self.output_dir / video_relpath
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_writer = imageio.get_writer(video_path, fps=20)

        returns = []
        lengths = []
        successes = []
        self.policy.reset_rollout_state()
        for episode_idx in range(self.config.evaluation.episodes):
            obs = self.eval_env.reset()
            total_reward = 0.0
            success = False
            horizon = self._env_horizon()
            episode_length = horizon
            episode_starts = np.ones(1, dtype=bool)

            for step_idx in range(horizon):
                goal = self._current_goal(self.eval_env)
                action = self.policy.act_deterministic(
                    obs={key: value[None, ...] for key, value in obs.items()},
                    goal=None if goal is None else {key: value[None, ...] for key, value in goal.items()},
                    episode_starts=episode_starts,
                )[0]
                obs, reward, done, _ = self.eval_env.step(action)
                total_reward += reward
                success = success or bool(self.eval_env.is_success()["task"])
                episode_starts[0] = bool(done)

                if video_writer is not None:
                    frame = self.eval_env.render(
                        mode="rgb_array",
                        camera_name=self.config.robosuite.camera_name,
                        height=512,
                        width=512,
                    )
                    video_writer.append_data(frame)

                if done or (self.config.robosuite.terminate_on_success and success):
                    episode_length = step_idx + 1
                    break
            returns.append(total_reward)
            lengths.append(episode_length)
            successes.append(float(success))

            if self.config.evaluation.render:
                self._emit(f"[eval] episode={episode_idx} return={total_reward:.4f} success={float(success):.1f}")

        if video_writer is not None:
            video_writer.close()

        return {
            "return_mean": float(np.mean(returns)),
            "length_mean": float(np.mean(lengths)) if lengths else 0.0,
            "success_rate": float(np.mean(successes)),
            "evaluated_episodes": float(len(returns)),
        }

    def _evaluate_vectorized(self, update: int) -> dict[str, float]:
        del update
        returns = []
        lengths = []
        successes = []
        num_envs = self.eval_env.num_envs
        horizon = self._env_horizon()
        episode_target = self.config.evaluation.episodes
        episodes_recorded = 0
        wave_idx = 0

        while episodes_recorded < episode_target:
            active_envs = min(num_envs, episode_target - episodes_recorded)
            active_mask = np.zeros(num_envs, dtype=bool)
            active_mask[:active_envs] = True
            completed_mask = np.zeros(num_envs, dtype=bool)
            running_returns = np.zeros(num_envs, dtype=np.float32)
            running_lengths = np.zeros(num_envs, dtype=np.float32)
            episode_starts = np.ones(num_envs, dtype=bool)

            obs, _ = self.eval_env.reset(seed=self.config.seed + wave_idx * num_envs)
            self.policy.reset_rollout_state()

            while not np.all(completed_mask[active_mask]):
                goal = self._current_goal(self.eval_env)
                actions = self.policy.act_deterministic(obs=obs, goal=goal, episode_starts=episode_starts)
                next_obs, rewards, terminated, truncated, infos = self.eval_env.step(actions)
                done = terminated | truncated

                running_returns[active_mask] += rewards[active_mask]
                running_lengths[active_mask] += 1

                final_info = infos.get("final_info", [None] * num_envs)
                for idx, info in enumerate(final_info):
                    if not active_mask[idx] or completed_mask[idx] or info is None:
                        continue
                    returns.append(float(running_returns[idx]))
                    lengths.append(float(min(running_lengths[idx], horizon)))
                    successes.append(float(info.get("success", False)))
                    completed_mask[idx] = True
                    episodes_recorded += 1

                obs = next_obs
                episode_starts = done
            wave_idx += 1

        return {
            "return_mean": float(np.mean(returns)),
            "length_mean": float(np.mean(lengths)) if lengths else 0.0,
            "success_rate": float(np.mean(successes)),
            "evaluated_episodes": float(len(returns)),
        }

    def save_checkpoint(self, tag: str) -> None:
        policy_path = self.output_dir / f"policy_{tag}{self.policy.policy_artifact_extension()}"
        torch.save(self.policy.export_policy_artifact(self.config.checkpoint_path), policy_path)

        trainer_state = {
            "algorithm_state": self.algorithm.save(),
            "actor_state": self.policy.actor.state_dict(),
            "value_net": self.policy.value_net.state_dict() if hasattr(self.policy, "value_net") else None,
            "q_networks": [q_net.state_dict() for q_net in getattr(self.policy, "q_networks", [])],
            "target_q_networks": [q_net.state_dict() for q_net in getattr(self.policy, "target_q_networks", [])],
            "learned_log_std": None
            if self.policy.learned_log_std is None
            else self.policy.learned_log_std.detach().cpu(),
            "demo_coef": self.current_demo_coef,
            "config": self.config.to_dict(),
            "residual_enabled": self.policy.residual_enabled,
        }
        torch.save(trainer_state, self.output_dir / f"trainer_{tag}.pt")

    def save_best_checkpoint(self, update: int, success_rate: float) -> None:
        success_str = f"{success_rate:.4f}"
        policy_path = self.output_dir / f"policy_best_update_{update:04d}_success_{success_str}{self.policy.policy_artifact_extension()}"
        trainer_path = self.output_dir / f"trainer_best_update_{update:04d}_success_{success_str}.pt"

        torch.save(self.policy.export_policy_artifact(self.config.checkpoint_path), policy_path)
        trainer_state = {
            "algorithm_state": self.algorithm.save(),
            "actor_state": self.policy.actor.state_dict(),
            "value_net": self.policy.value_net.state_dict() if hasattr(self.policy, "value_net") else None,
            "q_networks": [q_net.state_dict() for q_net in getattr(self.policy, "q_networks", [])],
            "target_q_networks": [q_net.state_dict() for q_net in getattr(self.policy, "target_q_networks", [])],
            "learned_log_std": None
            if self.policy.learned_log_std is None
            else self.policy.learned_log_std.detach().cpu(),
            "demo_coef": self.current_demo_coef,
            "config": self.config.to_dict(),
            "residual_enabled": self.policy.residual_enabled,
            "best_success_rate": success_rate,
            "best_update": update,
        }
        torch.save(trainer_state, trainer_path)

        self.best_policy_path = policy_path
        self.best_trainer_path = trainer_path
        self._emit(f"[checkpoint] saved best checkpoint update={update} success_rate={success_rate:.4f}")

    def _build_demo_loader(self) -> DataLoader:
        dataset_path = self.config.demo.dataset_path or self.bundle.config.train.data
        if dataset_path is None:
            raise ValueError("Demo regularization was enabled, but no dataset path was found in the checkpoint or config.")
        self._ensure_demo_dataset_config_defaults()
        dataset = dataset_factory(
            config=self.bundle.config,
            obs_keys=self.bundle.config.all_obs_keys,
            dataset_path=dataset_path,
        )
        return DataLoader(
            dataset,
            batch_size=self.config.demo.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

    def _load_awac_offline_data(self) -> None:
        if self.replay_buffer is None:
            raise RuntimeError("Replay buffer must exist before loading offline AWAC data.")
        dataset_path = self._resolve_awac_dataset_path()
        if dataset_path is None:
            raise ValueError("AWAC requires an offline dataset path to seed replay.")
        dataset_path = Path(dataset_path).expanduser().resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"AWAC dataset path does not exist: {dataset_path}")

        transitions_loaded = 0
        with h5py.File(dataset_path, "r") as dataset_file:
            data_group = dataset_file["data"]
            for demo_name in data_group.keys():
                demo_group = data_group[demo_name]
                actions = np.asarray(demo_group["actions"], dtype=np.float32)
                rewards = np.asarray(demo_group["rewards"], dtype=np.float32)
                dones = np.asarray(demo_group["dones"], dtype=np.float32)
                obs_group = demo_group["obs"]
                observations = {
                    key: np.asarray(obs_group[key], dtype=np.float32)
                    for key in self.policy.obs_shapes
                    if key in obs_group
                }
                missing_obs = [key for key in self.policy.obs_shapes if key not in observations]
                if missing_obs:
                    raise KeyError(f"Offline dataset is missing observation keys required by policy: {missing_obs}")

                next_observations = {}
                for key, value in observations.items():
                    shifted = np.zeros_like(value)
                    if value.shape[0] > 1:
                        shifted[:-1] = value[1:]
                    shifted[-1] = value[-1]
                    next_observations[key] = shifted

                self.replay_buffer.add_batch(
                    obs=observations,
                    actions=actions,
                    rewards=rewards,
                    next_obs=next_observations,
                    dones=dones,
                    goals=None,
                    next_goals=None,
                )
                self.replay_buffer.add_trajectory(
                    obs=observations,
                    actions=actions,
                    goals=None,
                )
                transitions_loaded += actions.shape[0]

        self._emit(f"[awac] loaded_offline_transitions={transitions_loaded} replay_size={len(self.replay_buffer)}")

    def _init_awac_episode_storage(self) -> None:
        if self.algorithm_name != "awac":
            return
        self._awac_episode_observations = [
            {key: [] for key in self.policy.obs_shapes}
            for _ in range(self.config.num_envs)
        ]
        self._awac_episode_goals = [
            {key: [] for key in self.policy.goal_shapes}
            for _ in range(self.config.num_envs)
        ]
        self._awac_episode_actions = [[] for _ in range(self.config.num_envs)]

    def _record_awac_online_sequences(
        self,
        obs: dict[str, np.ndarray],
        actions: np.ndarray,
        goals: dict[str, np.ndarray] | None,
        done: np.ndarray,
    ) -> None:
        if self.replay_buffer is None or self._awac_episode_observations is None or self._awac_episode_actions is None:
            return

        for env_idx in range(self.config.num_envs):
            obs_store = self._awac_episode_observations[env_idx]
            for key in obs_store:
                obs_store[key].append(np.asarray(obs[key][env_idx], dtype=np.float32))

            if self._awac_episode_goals is not None and self.policy.goal_shapes:
                goal_store = self._awac_episode_goals[env_idx]
                if goals is not None:
                    for key in goal_store:
                        goal_store[key].append(np.asarray(goals[key][env_idx], dtype=np.float32))

            self._awac_episode_actions[env_idx].append(np.asarray(actions[env_idx], dtype=np.float32))

            if not bool(done[env_idx]):
                continue

            trajectory_obs = {
                key: np.stack(values, axis=0)
                for key, values in obs_store.items()
                if values
            }
            trajectory_goals = None
            if self.policy.goal_shapes and self._awac_episode_goals is not None:
                goal_store = self._awac_episode_goals[env_idx]
                if any(goal_store.values()):
                    trajectory_goals = {
                        key: np.stack(values, axis=0)
                        for key, values in goal_store.items()
                        if values
                    }
            trajectory_actions = np.stack(self._awac_episode_actions[env_idx], axis=0)
            self.replay_buffer.add_trajectory(
                obs=trajectory_obs,
                actions=trajectory_actions,
                goals=trajectory_goals,
            )

            self._awac_episode_observations[env_idx] = {key: [] for key in self.policy.obs_shapes}
            if self._awac_episode_goals is not None:
                self._awac_episode_goals[env_idx] = {key: [] for key in self.policy.goal_shapes}
            self._awac_episode_actions[env_idx] = []

    def _resolve_awac_dataset_path(self) -> str | None:
        if self.config.awac.dataset_path is not None:
            return self.config.awac.dataset_path
        if self.config.demo.dataset_path is not None:
            return self.config.demo.dataset_path
        data_value = getattr(self.bundle.config.train, "data", None)
        if isinstance(data_value, str):
            return data_value
        if isinstance(data_value, (list, tuple)) and data_value:
            first_entry = data_value[0]
            if isinstance(first_entry, dict):
                return first_entry.get("path")
            return str(first_entry)
        return None

    def _ensure_demo_dataset_config_defaults(self) -> None:
        train_cfg = self.bundle.config.train
        with self.bundle.config.unlocked():
            data_value = getattr(train_cfg, "data", None)
            if isinstance(data_value, str):
                train_cfg.data = [{"path": data_value, "weight": 1.0}]
            elif isinstance(data_value, tuple):
                train_cfg.data = list(data_value)
            if "dataset_keys" not in train_cfg:
                train_cfg.dataset_keys = ("actions", "rewards", "dones")
            if "action_keys" not in train_cfg:
                train_cfg.action_keys = ["actions"]
            if "action_config" not in train_cfg:
                train_cfg.action_config = {
                    "actions": {
                        "normalization": None,
                        "rot_conversion": None,
                    }
                }
            if "normalize_weights_by_ds_size" not in train_cfg:
                train_cfg.normalize_weights_by_ds_size = False

    def _next_demo_batch(self) -> dict:
        assert self.demo_iter is not None
        try:
            return next(self.demo_iter)
        except StopIteration:
            assert self.demo_loader is not None
            self.demo_iter = iter(self.demo_loader)
            return next(self.demo_iter)

    def _demo_batch_generator(self):
        while True:
            if self.demo_iter is None:
                return
            yield self._next_demo_batch()

    def _build_vector_env(self):
        factory, _, _ = make_robosuite_env_from_checkpoint(
            checkpoint_path=self.config.checkpoint_path,
            task_override=self.config.robosuite.env_name,
            render=self.config.robosuite.render_train,
            render_offscreen=False,
            reward_shaping_override=self.config.robosuite.reward_shaping,
            horizon_override=self._env_horizon(),
        )
        env_fns = [factory for _ in range(self.config.num_envs)]
        if self.config.robosuite.parallel_envs and self.config.num_envs > 1:
            try:
                return ParallelRobomimicVectorEnv(env_fns, start_method=self.config.robosuite.start_method)
            except (ConnectionResetError, BrokenPipeError, EOFError) as exc:
                self._emit(
                    "parallel training env startup failed; falling back to serial envs "
                    f"(start_method={self.config.robosuite.start_method}, error={exc})"
                )
        return SerialRobomimicVectorEnv(env_fns)

    def _build_eval_env(self, render: bool, render_offscreen: bool):
        factory, _, _ = make_robosuite_env_from_checkpoint(
            checkpoint_path=self.config.checkpoint_path,
            task_override=self.config.robosuite.env_name,
            render=render,
            render_offscreen=render_offscreen,
            reward_shaping_override=self.config.robosuite.reward_shaping,
            horizon_override=self._env_horizon(),
        )
        if render or render_offscreen or self.config.evaluation.num_envs <= 1:
            return factory()

        env_fns = [factory for _ in range(self.config.evaluation.num_envs)]
        if self.config.robosuite.parallel_envs and self.config.evaluation.num_envs > 1:
            try:
                return ParallelRobomimicVectorEnv(env_fns, start_method=self.config.robosuite.start_method)
            except (ConnectionResetError, BrokenPipeError, EOFError) as exc:
                self._emit(
                    "parallel eval env startup failed; falling back to serial envs "
                    f"(start_method={self.config.robosuite.start_method}, error={exc})"
                )
        return SerialRobomimicVectorEnv(env_fns)

    def _env_horizon(self) -> int:
        if self.config.robosuite.horizon is not None:
            return self.config.robosuite.horizon
        return int(self.bundle.config.experiment.rollout.horizon)

    @staticmethod
    def _current_goal(env) -> dict[str, np.ndarray] | None:
        get_goal = getattr(env, "get_goal", None)
        if not callable(get_goal):
            return None
        try:
            return get_goal()
        except (AttributeError, NotImplementedError):
            return None

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

        if metrics["update"] % self.config.log_every_n_updates == 0:
            printable = OrderedDict(sorted(metrics.items()))
            self._emit(" ".join(f"{key}={value}" for key, value in printable.items()))

    @staticmethod
    def _emit(message: str) -> None:
        print(message, flush=True)

    @staticmethod
    def _close_env(env) -> None:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device)

    @staticmethod
    def _env_backend_name(env) -> str:
        if isinstance(env, ParallelRobomimicVectorEnv):
            return "parallel"
        if isinstance(env, SerialRobomimicVectorEnv):
            return "serial"
        return type(env).__name__

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
