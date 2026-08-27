"""Basic signal preprocessing helpers."""

from __future__ import annotations

import numpy as np


def clean_signal(x) -> np.ndarray:
    """Return a finite 1-D float signal with NaNs replaced by robust values."""
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr)
    fill = float(np.nanmedian(arr[finite]))
    arr = np.where(finite, arr, fill)
    return arr


def resample_or_pad(x, length: int) -> np.ndarray:
    """Pad/truncate a 1-D or 2-D channel-first array to a fixed length."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] == length:
        return arr
    if arr.shape[1] > length:
        return arr[:, :length]
    pad = np.zeros((arr.shape[0], length - arr.shape[1]), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=1)


def zscore_train_only(train: np.ndarray, *others: np.ndarray, eps: float = 1e-8):
    """Fit z-score parameters on train and transform train plus optional arrays."""
    mean = np.nanmean(train, axis=0)
    std = np.nanstd(train, axis=0)
    std = np.where(std < eps, 1.0, std)
    transformed = [(train - mean) / std]
    transformed.extend((arr - mean) / std for arr in others)
    return transformed, mean, std
