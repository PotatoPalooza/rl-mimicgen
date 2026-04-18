from __future__ import annotations

import math
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from rl_mimicgen.dppo.model.mlp import MLP, ResidualMLP


DPPOSample = namedtuple("DPPOSample", "trajectories chains")


def cosine_beta_schedule(timesteps: int, s: float = 0.008, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.tensor(np.clip(betas, a_min=0.0, a_max=0.999), dtype=dtype)


def extract(values: torch.Tensor, timesteps: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    batch_size, *_ = timesteps.shape
    out = values.gather(-1, timesteps)
    return out.reshape(batch_size, *((1,) * (len(target_shape) - 1)))


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        exponent = math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(torch.arange(half_dim, device=device) * -exponent)
        embeddings = x[:, None] * frequencies[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class DiffusionMLP(nn.Module):
    def __init__(
        self,
        action_dim: int,
        horizon_steps: int,
        cond_dim: int,
        time_dim: int = 16,
        mlp_dims: tuple[int, ...] = (512, 512, 512),
        residual_style: bool = True,
    ) -> None:
        super().__init__()
        output_dim = action_dim * horizon_steps
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )
        model_cls = ResidualMLP if residual_style else MLP
        input_dim = time_dim + action_dim * horizon_steps + cond_dim
        self.mlp = model_cls([input_dim, *mlp_dims, output_dim], activation_type="mish", out_activation_type="identity")
        self.time_dim = time_dim

    def forward(self, x: torch.Tensor, time: torch.Tensor, cond: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size, horizon_steps, action_dim = x.shape
        x = x.reshape(batch_size, -1)
        state = cond["state"].reshape(batch_size, -1)
        time_embedding = self.time_embedding(time.view(batch_size, 1)).reshape(batch_size, self.time_dim)
        output = self.mlp(torch.cat([x, time_embedding, state], dim=-1))
        return output.reshape(batch_size, horizon_steps, action_dim)


class DiffusionModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        horizon_steps: int,
        cond_steps: int,
        denoising_steps: int = 20,
        predict_epsilon: bool = True,
        denoised_clip_value: float | None = 1.0,
        time_dim: int = 16,
        mlp_dims: tuple[int, ...] = (512, 512, 512),
        residual_style: bool = True,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device_name = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps
        self.denoising_steps = int(denoising_steps)
        self.predict_epsilon = predict_epsilon
        self.denoised_clip_value = denoised_clip_value
        self.network = DiffusionMLP(
            action_dim=action_dim,
            horizon_steps=horizon_steps,
            cond_dim=obs_dim * cond_steps,
            time_dim=time_dim,
            mlp_dims=mlp_dims,
            residual_style=residual_style,
        )

        betas = cosine_beta_schedule(self.denoising_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape) * noise
        )

    def p_losses(self, x_start: torch.Tensor, cond: dict[str, torch.Tensor], timesteps: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x_start)
        noisy_actions = self.q_sample(x_start=x_start, timesteps=timesteps, noise=noise)
        prediction = self.network(noisy_actions, timesteps, cond=cond)
        if self.predict_epsilon:
            return F.mse_loss(prediction, noise, reduction="mean")
        return F.mse_loss(prediction, x_start, reduction="mean")

    def loss(self, actions: torch.Tensor, conditions: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = actions.shape[0]
        timesteps = torch.randint(0, self.denoising_steps, (batch_size,), device=actions.device, dtype=torch.long)
        return self.p_losses(actions, conditions, timesteps)
