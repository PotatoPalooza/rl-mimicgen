from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class OptimizerConfig:
    actor_lr: float = 3e-4
    value_lr: float = 3e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 0.5


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    update_epochs: int = 4
    target_kl: float = 0.03
    init_log_std: float = -0.5
    clip_actions: bool = True


@dataclass
class DemoConfig:
    enabled: bool = True
    batch_size: int = 16
    coef: float = 0.1
    decay: float = 0.99
    dataset_path: str | None = None


@dataclass
class EvalConfig:
    enabled: bool = True
    episodes: int = 5
    every_n_updates: int = 10
    render: bool = False
    video_path: str | None = None


@dataclass
class RobosuiteConfig:
    env_name: str | None = None
    horizon: int | None = None
    reward_shaping: bool | None = None
    terminate_on_success: bool = True
    render_train: bool = False
    camera_name: str = "agentview"


@dataclass
class OnlineRLConfig:
    checkpoint_path: str = ""
    output_dir: str = "logs/online_rl"
    device: str = "cuda"
    seed: int = 0
    total_updates: int = 50
    rollout_steps: int = 400
    log_every_n_updates: int = 1
    save_every_n_updates: int = 10
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    demo: DemoConfig = field(default_factory=DemoConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    robosuite: RobosuiteConfig = field(default_factory=RobosuiteConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "OnlineRLConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            **{
                **data,
                "optimizer": OptimizerConfig(**data.get("optimizer", {})),
                "ppo": PPOConfig(**data.get("ppo", {})),
                "demo": DemoConfig(**data.get("demo", {})),
                "evaluation": EvalConfig(**data.get("evaluation", {})),
                "robosuite": RobosuiteConfig(**data.get("robosuite", {})),
            }
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def dump_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
