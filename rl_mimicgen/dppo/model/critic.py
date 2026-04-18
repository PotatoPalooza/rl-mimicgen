from __future__ import annotations

import torch
from torch import nn

from rl_mimicgen.dppo.model.mlp import MLP, ResidualMLP


class CriticObs(nn.Module):
    """State-only critic for low-dim DPPO."""

    def __init__(
        self,
        cond_dim: int,
        mlp_dims: tuple[int, ...] = (256, 256),
        residual_style: bool = False,
    ) -> None:
        super().__init__()
        model_cls = ResidualMLP if residual_style else MLP
        self.value_head = model_cls([cond_dim, *mlp_dims, 1], activation_type="mish", out_activation_type="identity")

    def forward(
        self,
        cond: dict[str, torch.Tensor] | torch.Tensor,
        goal_dict: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        del goal_dict
        if isinstance(cond, dict):
            state = cond["state"]
        else:
            state = cond
        if state.ndim > 2:
            state = state.reshape(state.shape[0], -1)
        return self.value_head(state)
