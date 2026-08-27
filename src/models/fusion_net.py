"""Compact multisource fusion network."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class _Branch(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(max(1, in_channels), hidden, kernel_size=9, padding=4),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(hidden, hidden * 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MultisourceFusionNet(nn.Module):
    """Small vibration/current/scalar fusion classifier.

    Any order-related scalar features are prepared upstream. The network does
    not itself perform angular resampling or order normalization.
    """

    def __init__(
        self,
        vibration_channels: int,
        current_channels: int,
        scalar_dim: int,
        num_classes: int,
        hidden: int = 32,
        dropout: float = 0.2,
        health_head: bool = False,
    ):
        super().__init__()
        self.vibration_branch = _Branch(vibration_channels, hidden)
        self.current_branch = _Branch(current_channels, hidden)
        self.scalar_branch = nn.Sequential(
            nn.Linear(max(1, scalar_dim), hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        fusion_dim = hidden * 2 + hidden * 2 + hidden
        self.gate = nn.Sequential(nn.Linear(fusion_dim, fusion_dim), nn.Sigmoid())
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, num_classes),
        )
        self.health_head = nn.Sequential(nn.Linear(fusion_dim, 1), nn.Sigmoid()) if health_head else None

    def forward(
        self,
        vibration_seq: torch.Tensor,
        current_seq: torch.Tensor,
        scalar_features: torch.Tensor,
    ):
        vib = self.vibration_branch(vibration_seq)
        cur = self.current_branch(current_seq)
        scal = self.scalar_branch(scalar_features)
        fused = torch.cat([vib, cur, scal], dim=1)
        fused = fused * self.gate(fused)
        logits = self.classifier(fused)
        if self.health_head is None:
            return logits
        return logits, self.health_head(fused)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model_summary(path: str | Path = "results/logs/model_summary.txt") -> None:
    """Run a forward smoke test and save a model summary."""
    model = MultisourceFusionNet(3, 3, 16, 4)
    vib = torch.zeros(2, 3, 8192)
    cur = torch.zeros(2, 3, 8192)
    scal = torch.zeros(2, 16)
    logits = model(vib, cur, scal)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "\n".join(
            [
                "MultisourceFusionNet",
                f"parameters: {count_parameters(model)}",
                f"forward_output_shape: {tuple(logits.shape)}",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    save_model_summary()


# Backward-compatible alias for archived run scripts and checkpoint metadata.
OrderNormalizedMultisourceFusionNet = MultisourceFusionNet
