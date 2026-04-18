from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class DPPODatasetConfig:
    source_hdf5: str = ""
    bundle_dir: str = "runs/dppo_dataset"
    task: str = ""
    variant: str = ""
    obs_keys: tuple[str, ...] = ()
    max_demos: int | None = None


@dataclass
class DPPODiffusionConfig:
    horizon_steps: int = 4
    act_steps: int = 4
    cond_steps: int = 1
    denoising_steps: int = 20
    predict_epsilon: bool = True
    denoised_clip_value: float | None = 1.0
    time_dim: int = 16
    mlp_dims: tuple[int, ...] = (512, 512, 512)
    residual_style: bool = True


@dataclass
class DPPOTrainConfig:
    epochs: int = 3000
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    lr_warmup_steps: int = 100
    lr_min: float = 1e-5
    save_every_n_epochs: int = 500
    log_every_n_epochs: int = 1
    update_ema_every_n_steps: int = 10
    ema_decay: float = 0.995
    ema_start_epoch: int = 1
    normalize_dataset: bool = True
    num_workers: int = 0


@dataclass
class DPPOOnlineConfig:
    rollout_steps: int = 16
    update_epochs: int = 2
    num_minibatches: int = 1
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    gamma_denoising: float = 1.0
    max_grad_norm: float = 0.5
    target_kl: float | None = None


@dataclass
class DPPORunConfig:
    task: str = ""
    variant: str = ""
    seed: int = 0
    device: str = "cuda"
    dataset: DPPODatasetConfig = field(default_factory=DPPODatasetConfig)
    diffusion: DPPODiffusionConfig = field(default_factory=DPPODiffusionConfig)
    training: DPPOTrainConfig = field(default_factory=DPPOTrainConfig)
    online: DPPOOnlineConfig = field(default_factory=DPPOOnlineConfig)
    output_dir: str = "logs/dppo"
    train_steps: int = 0
    num_envs: int = 1
    checkpoint_path: str | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "DPPORunConfig":
        with open(path, "r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
        return cls(
            **{
                **data,
                "dataset": DPPODatasetConfig(**data.get("dataset", {})),
                "diffusion": DPPODiffusionConfig(**data.get("diffusion", {})),
                "training": DPPOTrainConfig(**data.get("training", {})),
                "online": DPPOOnlineConfig(**data.get("online", {})),
            }
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def dump_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as file_obj:
            json.dump(self.to_dict(), file_obj, indent=2)
