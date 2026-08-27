"""Simple TCN baseline."""

from __future__ import annotations

import torch
from torch import nn


class SimpleTCN(nn.Module):
    """A small dilated convolution classifier."""

    def __init__(self, in_channels: int, num_classes: int, hidden: int = 32):
        super().__init__()
        layers = []
        channels = in_channels
        for dilation in [1, 2, 4]:
            layers.extend(
                [
                    nn.Conv1d(channels, hidden, kernel_size=5, padding=2 * dilation, dilation=dilation),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(),
                ]
            )
            channels = hidden
        self.encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x).squeeze(-1))
