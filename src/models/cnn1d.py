"""Small 1-D CNN baselines."""

from __future__ import annotations

import torch
from torch import nn


class _SmallCNN(nn.Module):
    """Compact channel-first 1-D CNN classifier."""

    def __init__(self, in_channels: int, num_classes: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=9, padding=4),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(hidden, hidden * 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.net(x).squeeze(-1)
        return self.classifier(features)


class VibrationOnlyCNN(_SmallCNN):
    """CNN for vibration channels."""


class CurrentOnlyCNN(_SmallCNN):
    """CNN for current channels."""


class VibCurrentCNN(_SmallCNN):
    """CNN for concatenated vibration and current channels."""
