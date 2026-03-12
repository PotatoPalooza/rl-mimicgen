from __future__ import annotations

import json
import random
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader

import mimicgen  # noqa: F401
import robomimic.utils.env_utils as EnvUtils
from robomimic.utils.train_utils import dataset_factory

from rl_mimicgen.rl.config import OnlineRLConfig
from rl_mimicgen.rl.policy import OnlinePolicyAdapter, load_policy_bundle
from rl_mimicgen.rl.storage import RolloutBatch


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
        )

        self.env = self._build_env(
            render=config.robosuite.render_train,
            render_offscreen=bool(config.evaluation.video_path),
        )
        self.eval_env = self._build_env(
            render=config.evaluation.render,
            render_offscreen=bool(config.evaluation.video_path),
        )

        actor_params = self.policy.actor_parameters()
        self.actor_optimizer = torch.optim.Adam(
            actor_params,
            lr=config.optimizer.actor_lr,
            weight_decay=config.optimizer.weight_decay,
        )
        self.value_optimizer = torch.optim.Adam(
            self.policy.value_net.parameters(),
            lr=config.optimizer.value_lr,
            weight_decay=config.optimizer.weight_decay,
        )

        self.demo_loader = None
        self.demo_iter = None
        if config.demo.enabled:
            self.demo_loader = self._build_demo_loader()
            self.demo_iter = iter(self.demo_loader)

        self.best_success = float("-inf")
        self.best_policy_path: Path | None = None
        self.best_trainer_path: Path | None = None
        self.current_demo_coef = config.demo.coef

        self.config.dump_json(self.output_dir / "config.json")

    def train(self) -> None:
        obs = self.env.reset()
        goal = self._get_goal(self.env)
        episode_start = True
        episode_horizon = self._env_horizon()

        for update in range(self.config.total_updates):
            rollout = RolloutBatch()
            episode_return = 0.0
            episode_length = 0
            completed_returns = []
            completed_lengths = []
            completed_success = []

            self.policy.reset_rollout_state()
            self.policy.actor.train()
            self.policy.value_net.train()

            for _ in range(self.config.rollout_steps):
                action, log_prob, value = self.policy.act(
                    obs=obs,
                    goal=goal,
                    deterministic=False,
                    clip_actions=self.config.ppo.clip_actions,
                )

                next_obs, reward, done, _ = self.env.step(action)
                if self.config.robosuite.render_train:
                    self.env.render(mode="human", camera_name=self.config.robosuite.camera_name)
                success = bool(self.env.is_success()["task"])
                reward_terminal = reward > 0.0
                horizon_terminal = episode_length >= episode_horizon
                terminal = done or horizon_terminal or (
                    self.config.robosuite.terminate_on_success and (success or reward_terminal)
                )
                rollout.add(
                    observation=deepcopy(obs),
                    goal=deepcopy(goal),
                    action=action.astype(np.float32),
                    reward=reward,
                    done=terminal,
                    episode_start=episode_start,
                    value=value,
                    log_prob=log_prob,
                )

                episode_return += reward
                episode_length += 1

                obs = next_obs
                goal = self._get_goal(self.env)
                episode_start = False

                if terminal:
                    completed_returns.append(episode_return)
                    completed_lengths.append(episode_length)
                    completed_success.append(float(success or reward_terminal))
                    if self.config.robosuite.render_train:
                        self._emit(
                            f"[train] reset return={episode_return:.4f} length={episode_length} "
                            f"success={int(success)} reward_terminal={int(reward_terminal)} "
                            f"horizon_terminal={int(horizon_terminal)}"
                        )
                    obs = self.env.reset()
                    goal = self._get_goal(self.env)
                    self.policy.reset_rollout_state()
                    episode_return = 0.0
                    episode_length = 0
                    episode_start = True

            last_value = 0.0
            if not episode_start:
                last_value = self.policy.predict_value(obs=obs, goal=goal)
            rollout.finish(
                last_value=last_value,
                gamma=self.config.ppo.gamma,
                gae_lambda=self.config.ppo.gae_lambda,
            )

            train_metrics = self._update_from_rollout(rollout)
            train_metrics["update"] = update
            train_metrics["episodes_completed"] = float(len(completed_returns))
            train_metrics["episode_return_mean"] = float(np.mean(completed_returns)) if completed_returns else 0.0
            train_metrics["episode_length_mean"] = float(np.mean(completed_lengths)) if completed_lengths else 0.0
            train_metrics["success_rate_mean"] = float(np.mean(completed_success)) if completed_success else 0.0

            if self.config.evaluation.enabled and ((update + 1) % self.config.evaluation.every_n_updates == 0):
                eval_metrics = self.evaluate(update)
                train_metrics.update({f"eval/{k}": v for k, v in eval_metrics.items()})
                if eval_metrics["success_rate"] > self.best_success:
                    self.best_success = eval_metrics["success_rate"]
                    self.save_best_checkpoint(update + 1, eval_metrics["success_rate"])
            else:
                train_metrics["eval/success_rate"] = self.best_success if self.best_success != float("-inf") else 0.0

            self._log_metrics(train_metrics)

            if (update + 1) % self.config.save_every_n_updates == 0:
                self.save_checkpoint(tag=f"update_{update + 1:04d}")

            self.current_demo_coef *= self.config.demo.decay

        self.save_checkpoint(tag="latest")
        self._close_env(self.env)
        self._close_env(self.eval_env)

    def evaluate(self, update: int) -> dict[str, float]:
        video_writer = None
        if self.config.evaluation.video_path:
            video_relpath = self.config.evaluation.video_path.format(update=update + 1)
            video_path = self.output_dir / video_relpath
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_writer = imageio.get_writer(video_path, fps=20)

        returns = []
        lengths = []
        successes = []
        for episode_idx in range(self.config.evaluation.episodes):
            obs = self.eval_env.reset()
            goal = self._get_goal(self.eval_env)
            self.policy.rollout_policy.start_episode()
            total_reward = 0.0
            success = False
            horizon = self._env_horizon()
            episode_length = horizon

            for step_idx in range(horizon):
                action = self.policy.rollout_policy(ob=obs, goal=goal)
                obs, reward, done, _ = self.eval_env.step(action)
                total_reward += reward
                success = success or bool(self.eval_env.is_success()["task"])
                goal = self._get_goal(self.eval_env)

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

    def save_checkpoint(self, tag: str) -> None:
        policy_path = self.output_dir / f"policy_{tag}.pth"
        torch.save(self.policy.export_robomimic_checkpoint(), policy_path)

        trainer_state = {
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
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

        if self.best_policy_path is not None and self.best_policy_path.exists():
            self.best_policy_path.unlink()
        if self.best_trainer_path is not None and self.best_trainer_path.exists():
            self.best_trainer_path.unlink()

        torch.save(self.policy.export_robomimic_checkpoint(), policy_path)
        trainer_state = {
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
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

    def _update_from_rollout(self, rollout: RolloutBatch) -> dict[str, float]:
        tensors = rollout.as_numpy()
        actions = torch.as_tensor(tensors["actions"], device=self.device, dtype=torch.float32)
        old_log_probs = torch.as_tensor(tensors["log_probs"], device=self.device, dtype=torch.float32)
        advantages = torch.as_tensor(tensors["advantages"], device=self.device, dtype=torch.float32)
        returns = torch.as_tensor(tensors["returns"], device=self.device, dtype=torch.float32)
        episode_starts = torch.as_tensor(tensors["episode_starts"], device=self.device, dtype=torch.float32)

        metrics = {}
        for _ in range(self.config.ppo.update_epochs):
            new_log_probs, entropy = self.policy.evaluate_actions_full_batch(
                observations=rollout.observations,
                goals=rollout.goals,
                actions=actions,
                episode_starts=episode_starts,
            )
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1.0 - self.config.ppo.clip_ratio, 1.0 + self.config.ppo.clip_ratio)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

            values = self.policy.value(rollout.observations, rollout.goals)
            value_loss = torch.nn.functional.mse_loss(values, returns)
            entropy_bonus = entropy.mean()

            demo_loss = torch.tensor(0.0, device=self.device)
            if self.demo_iter is not None:
                demo_batch = self._next_demo_batch()
                demo_loss = self.policy.demo_loss(demo_batch)

            actor_loss = policy_loss + (self.current_demo_coef * demo_loss) - self.config.ppo.entropy_coef * entropy_bonus
            total_value_loss = self.config.ppo.value_coef * value_loss

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.actor_parameters(), self.config.optimizer.max_grad_norm)
            self.actor_optimizer.step()

            self.value_optimizer.zero_grad(set_to_none=True)
            total_value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.value_net.parameters(), self.config.optimizer.max_grad_norm)
            self.value_optimizer.step()

            approx_kl = float((old_log_probs - new_log_probs).mean().detach().cpu())
            metrics = {
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "entropy": float(entropy_bonus.detach().cpu()),
                "demo_loss": float(demo_loss.detach().cpu()),
                "demo_weight": float(self.current_demo_coef),
                "approx_kl": approx_kl,
            }
            if approx_kl > self.config.ppo.target_kl:
                break

        return metrics

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

    def _build_env(self, render: bool, render_offscreen: bool):
        env_meta = deepcopy(self.bundle.ckpt_dict["env_metadata"])
        if self.config.robosuite.env_name is not None:
            env_meta["env_name"] = self.config.robosuite.env_name
        if self.config.robosuite.reward_shaping is not None:
            env_meta["env_kwargs"]["reward_shaping"] = self.config.robosuite.reward_shaping

        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            env_name=env_meta["env_name"],
            render=render,
            render_offscreen=render_offscreen,
            use_image_obs=self.bundle.shape_meta.get("use_images", False),
            use_depth_obs=self.bundle.shape_meta.get("use_depths", False),
        )
        env = EnvUtils.wrap_env_from_config(env, config=self.bundle.config)
        return env

    def _env_horizon(self) -> int:
        if self.config.robosuite.horizon is not None:
            return self.config.robosuite.horizon
        return int(self.bundle.config.experiment.rollout.horizon)

    def _get_goal(self, env) -> dict | None:
        if not self.bundle.config.use_goals:
            return None
        return env.get_goal()

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
