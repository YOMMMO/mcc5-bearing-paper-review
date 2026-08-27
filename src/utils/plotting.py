"""Matplotlib-only plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from src.utils.io import ensure_dir

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - depends on the runtime environment
    plt = None


def _skip_plot(out_path: str | Path, reason: str) -> None:
    path = Path(out_path)
    ensure_dir(path.parent)
    note_path = path.with_suffix(path.suffix + ".txt")
    note_path.write_text(f"Plot skipped: {reason}\nRequested output: {path}\n", encoding="utf-8")


def _finish(out_path: str | Path, dpi: int = 300) -> None:
    if plt is None:
        _skip_plot(out_path, "matplotlib is not installed")
        return
    path = Path(out_path)
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    if path.suffix.lower() == ".png":
        plt.savefig(path.with_suffix(".pdf"))
    plt.close()


def save_line_plot(
    x: Sequence[float],
    y: Sequence[float],
    out_path: str | Path,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
) -> None:
    """Save a simple line plot."""
    if plt is None:
        _skip_plot(out_path, "matplotlib is not installed")
        return
    plt.figure(figsize=(7, 4))
    plt.plot(x, y, linewidth=1.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    _finish(out_path)


def save_bar_plot(
    labels: Sequence[str],
    values: Sequence[float],
    out_path: str | Path,
    title: str = "",
    ylabel: str = "",
) -> None:
    """Save a simple bar plot."""
    if plt is None:
        _skip_plot(out_path, "matplotlib is not installed")
        return
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values)
    plt.xticks(rotation=30, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    _finish(out_path)


def save_training_curve(history: dict[str, Sequence[float]], out_path: str | Path) -> None:
    """Save training curves from a metric history dictionary."""
    if plt is None:
        _skip_plot(out_path, "matplotlib is not installed")
        return
    plt.figure(figsize=(7, 4))
    for key, values in history.items():
        plt.plot(list(range(1, len(values) + 1)), values, label=key)
    plt.xlabel("Epoch")
    plt.legend()
    _finish(out_path)


def save_confusion_matrix_plot(
    matrix: np.ndarray,
    labels: Sequence[str],
    out_path: str | Path,
    title: str = "Confusion Matrix",
) -> None:
    """Save a confusion matrix heatmap using matplotlib."""
    if plt is None:
        _skip_plot(out_path, "matplotlib is not installed")
        return
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8)
    _finish(out_path)
