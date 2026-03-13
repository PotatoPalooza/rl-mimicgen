from __future__ import annotations

import json
import random
from collections import OrderedDict
from pathlib import Path

import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader
from rsl_rl.storage import RolloutStorage

from robomimic.utils.train_utils import dataset_factory

from rl_mimicgen.rl.config import OnlineRLConfig
from rl_mimicgen.rl.policy import OnlinePolicyAdapter, load_policy_bundle
from rl_mimicgen.rl.robomimic_env import ParallelRobomimicVectorEnv, SerialRobomimicVectorEnv, make_robosuite_env_from_checkpoint
from rl_mimicgen.rl.rsl_rl import DemoAugmentedPPO, RslRlRobomimicActor, RslRlValueCritic, obs_to_tensordict


class OnlineRLTrainer:
    def __init__(self, config: OnlineRLConfig):
        self.config = config
        self.output_dir = Path(config.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"

        self.device = self._resolve_device(config.device)
        self._seed_everything(config.seed)

        self.bundle = load_policy_bundle(config.checkpoint_path, self.device)
        self.policy = OnlinePolicyAdapter(
            bundle=self.bundle,
            device=self.device,
            init_log_std=config.ppo.init_log_std,
            min_log_std=config.ppo.min_log_std,
        )
        self.actor = RslRlRobomimicActor(self.policy)
        self.critic = RslRlValueCritic(self.policy)

        self.env = self._build_vector_env()
        self.eval_env = self._build_eval_env(
            render=config.evaluation.render,
            render_offscreen=bool(config.evaluation.video_path),
        )

        self.demo_loader = None
        self.demo_iter = None
        if config.demo.enabled:
            self.demo_loader = self._build_demo_loader()
            self.demo_iter = iter(self.demo_loader)

        reset_obs, _ = self.env.reset(seed=self.config.seed)
        storage = RolloutStorage(
            "rl",
            self.config.num_envs,
            self.config.rollout_steps,
            obs_to_tensordict(reset_obs, self.device),
            [int(np.prod(self.env.single_action_space.shape))],
            device=str(self.device),
        )
        self.algorithm = DemoAugmentedPPO(
            actor=self.actor,
            critic=self.critic,
            storage=storage,
            demo_batch_iterator=self._demo_batch_generator() if self.config.demo.enabled else None,
            demo_loss_fn=self.policy.demo_loss if self.config.demo.enabled else None,
            demo_coef=self.config.demo.coef if self.config.demo.enabled else 0.0,
            actor_learning_rate=self.config.optimizer.actor_lr,
            critic_learning_rate=self.config.optimizer.value_lr,
            critic_warmup_updates=self.config.ppo.critic_warmup_updates,
            actor_freeze_env_steps=self.config.ppo.actor_freeze_env_steps,
            num_learning_epochs=self.config.ppo.update_epochs,
            num_mini_batches=self.config.ppo.num_minibatches,
            clip_param=self.config.ppo.clip_ratio,
            gamma=self.config.ppo.gamma,
            lam=self.config.ppo.gae_lambda,
            value_loss_coef=self.config.ppo.value_coef,
            entropy_coef=self.config.ppo.entropy_coef,
            learning_rate=self.config.optimizer.actor_lr,
            max_grad_norm=self.config.optimizer.max_grad_norm,
            device=str(self.device),
        )

        self.best_success = float("-inf")
        self.last_eval_metrics: dict[str, float] | None = None
        self.best_policy_path: Path | None = None
        self.best_trainer_path: Path | None = None
        self.current_demo_coef = config.demo.coef

        self.config.dump_json(self.output_dir / "config.json")

    def train(self) -> None:
        obs, _ = self.env.reset(seed=self.config.seed)
        obs_td = obs_to_tensordict(obs, self.device)
        running_returns = np.zeros(self.config.num_envs, dtype=np.float32)
        running_lengths = np.zeros(self.config.num_envs, dtype=np.float32)
        episode_horizon = self._env_horizon()
        total_env_steps = 0

        for update in range(self.config.total_updates):
            completed_returns = []
            completed_lengths = []
            completed_success = []

            self.algorithm.train_mode()
            self.algorithm.demo_coef = self.current_demo_coef

            for _ in range(self.config.rollout_steps):
                actions = self.algorithm.act(obs_td)
                next_obs, rewards, terminated, truncated, infos = self.env.step(actions.detach().cpu().numpy())
                done = terminated | truncated

                next_obs_td = obs_to_tensordict(next_obs, self.device)
                self.algorithm.process_env_step(
                    next_obs_td,
                    torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
                    torch.as_tensor(done, dtype=torch.bool, device=self.device),
                    {
                        "time_outs": torch.as_tensor(truncated, dtype=torch.float32, device=self.device),
                    },
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
                obs_td = next_obs_td

            total_env_steps += self.config.rollout_steps * self.config.num_envs
            self.algorithm.set_total_env_steps(total_env_steps)
            self.algorithm.compute_returns(obs_td)

            update_metrics = self.algorithm.update()
            train_metrics = {
                "policy_loss": float(update_metrics["surrogate"]),
                "value_loss": float(update_metrics["value"]),
                "entropy": float(update_metrics["entropy"]),
                "demo_loss": float(update_metrics["demo"]),
                "demo_weight": float(self.current_demo_coef),
            }
            train_metrics["update"] = update
            train_metrics["episodes_completed"] = float(len(completed_returns))
            train_metrics["episode_return_mean"] = float(np.mean(completed_returns)) if completed_returns else 0.0
            train_metrics["episode_length_mean"] = float(np.mean(completed_lengths)) if completed_lengths else 0.0
            train_metrics["success_rate_mean"] = float(np.mean(completed_success)) if completed_success else 0.0
            train_metrics["rollout_horizon"] = float(episode_horizon)
            train_metrics["env_steps"] = float(total_env_steps)
            current_log_std = self.policy.current_log_std()
            if current_log_std is not None:
                train_metrics["log_std_mean"] = float(current_log_std.mean().item())
                train_metrics["log_std_min"] = float(current_log_std.min().item())
                train_metrics["log_std_max"] = float(current_log_std.max().item())

            if self.config.evaluation.enabled and ((update + 1) % self.config.evaluation.every_n_updates == 0):
                eval_metrics = self.evaluate(update)
                self.last_eval_metrics = eval_metrics
                train_metrics.update({f"eval/{k}": v for k, v in eval_metrics.items()})
                train_metrics["eval/was_run"] = 1.0
                if eval_metrics["success_rate"] > self.best_success:
                    self.best_success = eval_metrics["success_rate"]
                    self.save_best_checkpoint(update + 1, eval_metrics["success_rate"])
            else:
                train_metrics["eval/was_run"] = 0.0
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
                action = self.policy.act_deterministic(obs={key: value[None, ...] for key, value in obs.items()}, goal=None, episode_starts=episode_starts)[0]
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
                actions = self.policy.act_deterministic(obs=obs, goal=None, episode_starts=episode_starts)
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
        policy_path = self.output_dir / f"policy_{tag}.pth"
        torch.save(self.policy.export_robomimic_checkpoint(), policy_path)

        trainer_state = {
            "ppo_state": self.algorithm.save(),
            "value_net": self.policy.value_net.state_dict(),
            "learned_log_std": None
            if self.policy.learned_log_std is None
            else self.policy.learned_log_std.detach().cpu(),
            "demo_coef": self.current_demo_coef,
            "config": self.config.to_dict(),
        }
        torch.save(trainer_state, self.output_dir / f"trainer_{tag}.pt")

    def save_best_checkpoint(self, update: int, success_rate: float) -> None:
        success_str = f"{success_rate:.4f}"
        policy_path = self.output_dir / f"policy_best_update_{update:04d}_success_{success_str}.pth"
        trainer_path = self.output_dir / f"trainer_best_update_{update:04d}_success_{success_str}.pt"

        torch.save(self.policy.export_robomimic_checkpoint(), policy_path)
        trainer_state = {
            "ppo_state": self.algorithm.save(),
            "value_net": self.policy.value_net.state_dict(),
            "learned_log_std": None
            if self.policy.learned_log_std is None
            else self.policy.learned_log_std.detach().cpu(),
            "demo_coef": self.current_demo_coef,
            "config": self.config.to_dict(),
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
            horizon_override=self.config.robosuite.horizon,
        )
        env_fns = [factory for _ in range(self.config.num_envs)]
        if self.config.robosuite.parallel_envs and self.config.num_envs > 1:
            return ParallelRobomimicVectorEnv(env_fns, start_method=self.config.robosuite.start_method)
        return SerialRobomimicVectorEnv(env_fns)

    def _build_eval_env(self, render: bool, render_offscreen: bool):
        factory, _, _ = make_robosuite_env_from_checkpoint(
            checkpoint_path=self.config.checkpoint_path,
            task_override=self.config.robosuite.env_name,
            render=render,
            render_offscreen=render_offscreen,
            reward_shaping_override=self.config.robosuite.reward_shaping,
            horizon_override=self.config.robosuite.horizon,
        )
        if render or render_offscreen or self.config.evaluation.num_envs <= 1:
            return factory()

        env_fns = [factory for _ in range(self.config.evaluation.num_envs)]
        if self.config.robosuite.parallel_envs and self.config.evaluation.num_envs > 1:
            return ParallelRobomimicVectorEnv(env_fns, start_method=self.config.robosuite.start_method)
        return SerialRobomimicVectorEnv(env_fns)

    def _env_horizon(self) -> int:
        if self.config.robosuite.horizon is not None:
            return self.config.robosuite.horizon
        return int(self.bundle.config.experiment.rollout.horizon)

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
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
