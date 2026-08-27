"""Small Transformer encoder baseline for 1-D signals."""

from __future__ import annotations

import torch
from torch import nn


class Transformer1DClassifier(nn.Module):
    """Patch-free Transformer over downsampled temporal tokens."""

    def __init__(self, in_channels: int, num_classes: int, d_model: int = 64, nhead: int = 4):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, d_model, kernel_size=9, stride=8, padding=4)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(x).transpose(1, 2)
        encoded = self.encoder(tokens)
        return self.classifier(encoded.mean(dim=1))
