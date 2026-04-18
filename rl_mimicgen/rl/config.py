from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rl_mimicgen.diffusion_runtime import apply_runtime_profile_to_online_rl_config

@dataclass
class OptimizerConfig:
    actor_lr: float = 1e-4
    value_lr: float = 1e-3
    weight_decay: float = 0.0
    max_grad_norm: float = 0.5


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.006
    update_epochs: int = 4
    num_minibatches: int = 4
    target_kl: float = 0.03
    init_log_std: float = -0.5
    min_log_std: float = -2.0
    clip_actions: bool = True
    critic_warmup_updates: int = 0
    actor_freeze_env_steps: int = 8192


@dataclass
class AWACConfig:
    dataset_path: str | None = None
    offline_pretrain_updates: int = 0
    discount: float = 0.99
    beta: float = 1.0
    max_weight: float = 20.0
    normalize_weights: bool = True
    native_actor: bool = False
    actor_hidden_sizes: tuple[int, ...] | None = None
    actor_fixed_std: bool = False
    actor_init_std: float = 0.3
    actor_use_tanh: bool = False
    behavior_kl_coef: float = 0.0
    actor_batch_size: int = 256
    replay_capacity: int = 500000
    q_hidden_sizes: tuple[int, ...] = (256, 256)
    num_q_networks: int = 2
    target_tau: float = 0.005
    num_action_samples: int = 4
    critic_huber_loss: bool = False
    value_coef: float = 1.0
    entropy_coef: float = 0.0
    update_epochs: int = 4
    num_minibatches: int = 4
    critic_warmup_updates: int = 0
    actor_freeze_env_steps: int = 8192


@dataclass
class DemoConfig:
    enabled: bool = True
    batch_size: int = 16
    coef: float = 0.1
    decay: float = 0.99
    dataset_path: str | None = None


@dataclass
class ResidualConfig:
    enabled: bool = False
    scale: float = 0.2


@dataclass
class EvalConfig:
    enabled: bool = True
    episodes: int = 5
    num_envs: int = 8
    every_n_updates: int = 10
    render: bool = False
    video_path: str | None = None


@dataclass
class RobosuiteConfig:
    env_name: str | None = None
    horizon: int | None = None
    reward_shaping: bool | None = None
    terminate_on_success: bool = True
    parallel_envs: bool = True
    start_method: str = "spawn"
    envs_per_worker: int = 1
    render_train: bool = False
    camera_name: str = "agentview"


@dataclass
class DiffusionConfig:
    enabled: bool = True
    runtime_profile: str | None = None
    num_inference_timesteps: int | None = None
    use_ema: bool = False
    ft_denoising_steps: int | None = None
    use_ddim: bool = False
    ddim_steps: int | None = None
    gamma_denoising: float = 1.0
    min_sampling_denoising_std: float | None = None
    min_logprob_denoising_std: float | None = None
    act_steps: int | None = None


@dataclass
class OnlineRLConfig:
    checkpoint_path: str = ""
    output_dir: str = "logs/online_rl"
    device: str = "cuda"
    seed: int = 0
    algorithm: str = "ppo"
    total_updates: int = 50
    num_envs: int = 8
    rollout_steps: int = 128
    log_every_n_updates: int = 1
    save_every_n_updates: int = 10
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    awac: AWACConfig = field(default_factory=AWACConfig)
    demo: DemoConfig = field(default_factory=DemoConfig)
    residual: ResidualConfig = field(default_factory=ResidualConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    robosuite: RobosuiteConfig = field(default_factory=RobosuiteConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "OnlineRLConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        config = cls(
            **{
                **data,
                "optimizer": OptimizerConfig(**data.get("optimizer", {})),
                "ppo": PPOConfig(**data.get("ppo", {})),
                "awac": AWACConfig(**data.get("awac", {})),
                "demo": DemoConfig(**data.get("demo", {})),
                "residual": ResidualConfig(**data.get("residual", {})),
                "evaluation": EvalConfig(**data.get("evaluation", {})),
                "robosuite": RobosuiteConfig(**data.get("robosuite", {})),
                "diffusion": DiffusionConfig(**data.get("diffusion", {})),
            }
        )
        config.algorithm = str(config.algorithm).lower()
        if config.algorithm not in {"ppo", "awac", "dppo"}:
            raise ValueError(f"Unsupported online RL algorithm '{config.algorithm}'.")
        apply_runtime_profile_to_online_rl_config(config.diffusion)
        return config

    def to_dict(self) -> dict:
        return asdict(self)

    def dump_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
