from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from rl_mimicgen.dppo.config.schema import DPPORunConfig
from rl_mimicgen.dppo.data import DPPODatasetBundle, DPPODiffusionDataset
from rl_mimicgen.dppo.model import DiffusionModel


def _move_to_device(batch, device: torch.device):
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: _move_to_device(value, device) for key, value in batch.items()}
    if hasattr(batch, "_fields"):
        return type(batch)(*(_move_to_device(getattr(batch, field), device) for field in batch._fields))
    raise TypeError(f"Unsupported batch type: {type(batch)}")


class ExponentialMovingAverage:
    def __init__(self, decay: float) -> None:
        self.decay = decay

    def update(self, ema_model: torch.nn.Module, model: torch.nn.Module) -> None:
        with torch.no_grad():
            for ema_param, param in zip(ema_model.parameters(), model.parameters(), strict=True):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)


class WarmupCosineScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, total_epochs: int, warmup_steps: int, min_lr: float) -> None:
        self.optimizer = optimizer
        self.total_epochs = max(total_epochs, 1)
        self.warmup_steps = max(warmup_steps, 0)
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.epoch = 0

    def step(self) -> None:
        self.epoch += 1
        if self.warmup_steps > 0 and self.epoch <= self.warmup_steps:
            scale = self.epoch / self.warmup_steps
            lrs = [max(self.min_lr, base_lr * scale) for base_lr in self.base_lrs]
        else:
            if self.total_epochs <= self.warmup_steps:
                progress = 1.0
            else:
                progress = (self.epoch - self.warmup_steps) / max(self.total_epochs - self.warmup_steps, 1)
            cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()
            lrs = [self.min_lr + (base_lr - self.min_lr) * cosine for base_lr in self.base_lrs]
        for group, lr in zip(self.optimizer.param_groups, lrs, strict=True):
            group["lr"] = lr

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline diffusion pretraining entry point for the DPPO stack.")
    parser.add_argument("--config", required=True, help="Path to the DPPO JSON config.")
    parser.add_argument("--dry_run", action="store_true", help="Load config and exit without training.")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    config = DPPORunConfig.from_json(args.config)
    dataset = DPPODatasetBundle.load(config.dataset.bundle_dir)
    device = torch.device(config.device)
    if args.dry_run:
        print(
            f"Loaded pretrain config for task={config.task} variant={config.variant} "
            f"bundle={config.dataset.bundle_dir}"
        )
        print(dataset.summary())
        return

    train_dataset = DPPODiffusionDataset(
        bundle=dataset,
        horizon_steps=config.diffusion.horizon_steps,
        cond_steps=config.diffusion.cond_steps,
        normalize=config.training.normalize_dataset,
        device="cpu",
        max_demos=config.dataset.max_demos,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = DiffusionModel(
        obs_dim=dataset.obs_dim,
        action_dim=dataset.action_dim,
        horizon_steps=config.diffusion.horizon_steps,
        cond_steps=config.diffusion.cond_steps,
        denoising_steps=config.diffusion.denoising_steps,
        predict_epsilon=config.diffusion.predict_epsilon,
        denoised_clip_value=config.diffusion.denoised_clip_value,
        time_dim=config.diffusion.time_dim,
        mlp_dims=config.diffusion.mlp_dims,
        residual_style=config.diffusion.residual_style,
        device=config.device,
    ).to(device)
    ema_model = copy.deepcopy(model)
    ema = ExponentialMovingAverage(config.training.ema_decay)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        total_epochs=config.training.epochs,
        warmup_steps=config.training.lr_warmup_steps,
        min_lr=config.training.lr_min,
    )

    output_dir = Path(config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w", encoding="utf-8") as file_obj:
        json.dump(config.to_dict(), file_obj, indent=2)

    global_step = 0
    for epoch in range(1, config.training.epochs + 1):
        epoch_losses: list[float] = []
        model.train()
        for batch in train_loader:
            batch = _move_to_device(batch, device)
            loss = model.loss(batch.actions, batch.conditions)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            epoch_losses.append(float(loss.item()))
            if global_step % config.training.update_ema_every_n_steps == 0:
                if epoch < config.training.ema_start_epoch:
                    ema_model.load_state_dict(model.state_dict())
                else:
                    ema.update(ema_model, model)
            global_step += 1

        scheduler.step()
        mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        if epoch % config.training.log_every_n_epochs == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"epoch={epoch} loss={mean_loss:.6f} lr={current_lr:.6e}")
        if epoch % config.training.save_every_n_epochs == 0 or epoch == config.training.epochs:
            checkpoint_path = checkpoint_dir / f"state_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "ema": ema_model.state_dict(),
                    "config": config.to_dict(),
                    "dataset_metadata": dataset.metadata,
                },
                checkpoint_path,
            )
            print(f"saved_checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
