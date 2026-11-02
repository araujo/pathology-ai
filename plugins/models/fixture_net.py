"""The `torch-fixture` network (imports torch; loaded lazily by `fixture`)."""

from __future__ import annotations

import torch
from torch import nn


def build_network(embedding_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=8, stride=8),
        nn.ReLU(),
        nn.Conv2d(8, 16, kernel_size=4, stride=4),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, embedding_dim),
        nn.Tanh(),
    )


def forward(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out: torch.Tensor = module(x)
    return out
