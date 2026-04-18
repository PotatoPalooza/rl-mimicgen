from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


ACTIVATIONS = nn.ModuleDict(
    {
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "mish": nn.Mish(),
        "identity": nn.Identity(),
        "softplus": nn.Softplus(),
    }
)


class MLP(nn.Module):
    def __init__(
        self,
        dim_list: list[int],
        activation_type: str = "mish",
        out_activation_type: str = "identity",
        use_layernorm: bool = False,
    ) -> None:
        super().__init__()
        modules = []
        num_layers = len(dim_list) - 1
        for index in range(num_layers):
            in_dim = dim_list[index]
            out_dim = dim_list[index + 1]
            layers = [("linear", nn.Linear(in_dim, out_dim))]
            if use_layernorm and index < num_layers - 1:
                layers.append(("norm", nn.LayerNorm(out_dim)))
            activation_key = activation_type if index < num_layers - 1 else out_activation_type
            layers.append(("act", ACTIVATIONS[activation_key]))
            modules.append(nn.Sequential(OrderedDict(layers)))
        self.layers = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, activation_type: str = "mish", use_layernorm: bool = False) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = ACTIVATIONS[activation_type]
        self.norm1 = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
        self.norm2 = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(self.activation(self.norm1(x)))
        x = self.fc2(self.activation(self.norm2(x)))
        return x + residual


class ResidualMLP(nn.Module):
    def __init__(
        self,
        dim_list: list[int],
        activation_type: str = "mish",
        out_activation_type: str = "identity",
        use_layernorm: bool = False,
    ) -> None:
        super().__init__()
        hidden_dim = dim_list[1]
        num_hidden_layers = len(dim_list) - 3
        if num_hidden_layers % 2 != 0:
            raise ValueError("ResidualMLP expects an even number of hidden layers after the input layer.")
        layers: list[nn.Module] = [nn.Linear(dim_list[0], hidden_dim)]
        layers.extend(
            ResidualBlock(
                hidden_dim=hidden_dim,
                activation_type=activation_type,
                use_layernorm=use_layernorm,
            )
            for _ in range(1, num_hidden_layers, 2)
        )
        layers.append(nn.Linear(hidden_dim, dim_list[-1]))
        layers.append(ACTIVATIONS[out_activation_type])
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
