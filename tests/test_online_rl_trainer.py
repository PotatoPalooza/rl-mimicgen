from __future__ import annotations

import numpy as np

from rl_mimicgen.rl.config import OnlineRLConfig
from rl_mimicgen.rl.trainer import OnlineRLTrainer


class _DummyAlgorithm:
    def __init__(self) -> None:
        self.total_env_steps_history: list[int] = []
        self.demo_coef = 0.0

    def train_mode(self) -> None:
        pass

    def set_total_env_steps(self, total_env_steps: int) -> None:
        self.total_env_steps_history.append(total_env_steps)

    def update(self, batch) -> dict[str, float]:
        del batch
        return {
            "surrogate": 0.0,
            "value": 0.0,
            "entropy": 0.0,
            "demo": 0.0,
            "approx_kl": 0.0,
            "early_stop": 0.0,
            "effective_num_minibatches": 1.0,
        }


class _DummyStorage:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.add_calls = 0
        self.compute_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def set_initial_rnn_state(self, state) -> None:
        del state

    def add(self, **kwargs) -> None:
        del kwargs
        self.add_calls += 1

    def compute_returns_and_advantages(self, last_value, gamma: float, gae_lambda: float) -> None:
        del last_value, gamma, gae_lambda
        self.compute_calls += 1

    def as_batch(self):
        return object()


class _DummyPolicy:
    def clone_training_rollout_state(self):
        return None

    def act(self, obs, goal, episode_starts, clip_actions):
        del goal, episode_starts, clip_actions
        num_envs = obs["obs"].shape[0]
        env_actions = np.zeros((num_envs, 2), dtype=np.float32)
        policy_actions = np.zeros((num_envs, 2), dtype=np.float32)
        log_probs = np.zeros(num_envs, dtype=np.float32)
        values = np.zeros(num_envs, dtype=np.float32)
        return env_actions, policy_actions, log_probs, values

    def predict_value(self, obs, goal):
        del goal
        return np.zeros(obs["obs"].shape[0], dtype=np.float32)

    def current_log_std(self):
        return None


class _DummyEnv:
    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs

    def reset(self, seed=None):
        del seed
        obs = {"obs": np.zeros((self.num_envs, 3), dtype=np.float32)}
        return obs, {}

    def step(self, actions):
        del actions
        obs = {"obs": np.zeros((self.num_envs, 3), dtype=np.float32)}
        rewards = np.ones(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos = {"final_info": [None] * self.num_envs}
        return obs, rewards, terminated, truncated, infos


def test_trainer_counts_rollout_steps_per_time_step_and_env_steps_separately(tmp_path) -> None:
    trainer = OnlineRLTrainer.__new__(OnlineRLTrainer)
    trainer.config = OnlineRLConfig(
        output_dir=str(tmp_path),
        total_updates=1,
        num_envs=16,
        rollout_steps=192,
    )
    trainer.is_diffusion_policy = False
    trainer.algorithm_name = "ppo"
    trainer.env = _DummyEnv(num_envs=trainer.config.num_envs)
    trainer.policy = _DummyPolicy()
    trainer.storage = _DummyStorage()
    trainer.replay_buffer = None
    trainer.algorithm = _DummyAlgorithm()
    trainer.current_demo_coef = 0.0
    trainer.best_success = float("-inf")
    trainer.last_eval_metrics = None
    trainer.metrics_path = tmp_path / "metrics.jsonl"
    trainer.output_dir = tmp_path
    trainer.best_policy_path = None
    trainer.best_trainer_path = None
    trainer.eval_env = None

    trainer._current_goal = lambda env: None
    trainer._env_horizon = lambda: 0
    trainer._log_metrics = lambda metrics: None
    trainer.evaluate = lambda update: (_ for _ in ()).throw(AssertionError(f"unexpected eval {update}"))
    trainer.save_checkpoint = lambda update: None
    trainer.save_best_checkpoint = lambda update, success: None
    trainer.close = lambda: None

    trainer.train()

    assert trainer.storage.reset_calls == 1
    assert trainer.storage.add_calls == trainer.config.rollout_steps
    assert trainer.storage.compute_calls == 1
    assert trainer.algorithm.total_env_steps_history == [trainer.config.rollout_steps * trainer.config.num_envs]
