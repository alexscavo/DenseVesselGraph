"""Legacy-style image/instance domain discriminators with gradient reversal."""

from __future__ import annotations

import torch
from torch import nn


class _ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, alpha):
        ctx.alpha = float(alpha)
        return features.view_as(features)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.alpha * gradient, None


class DomainDiscriminator(nn.Module):
    """Match Image-to-Graph-Vessel's four-layer 2048-wide DANN classifier."""

    def __init__(self, input_size: int, hidden_size: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 2),
            nn.LogSoftmax(dim=1),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, features: torch.Tensor, alpha: float):
        reversed_prediction = self.net(_ReverseGradient.apply(features, alpha))
        plain_prediction = self.net(_ReverseGradient.apply(features, -alpha))
        return reversed_prediction, plain_prediction


__all__ = ["DomainDiscriminator"]
