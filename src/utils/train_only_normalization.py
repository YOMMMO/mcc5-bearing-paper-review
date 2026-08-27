"""Train-only normalization for sequence and scalar model inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SequenceNormalizationStats:
    """Per-channel statistics fitted on training sequences only."""

    mean: np.ndarray
    scale: np.ndarray
    raw_std: np.ndarray
    constant_mask: np.ndarray
    value_count_per_channel: int


@dataclass(frozen=True)
class ScalarNormalizationStats:
    """Per-feature imputation and normalization statistics."""

    columns: tuple[str, ...]
    median: pd.Series
    mean: np.ndarray
    scale: np.ndarray
    raw_std: np.ndarray
    constant_mask: np.ndarray


def fit_sequence_stats(
    train: np.ndarray,
    *,
    eps: float = 1e-8,
    chunk_size: int = 256,
) -> SequenceNormalizationStats:
    """Fit per-channel mean and standard deviation without a float64 full copy."""

    array = np.asarray(train)
    if array.ndim != 3 or array.shape[0] == 0 or array.shape[2] == 0:
        raise ValueError("Training sequences must have non-empty [N, C, L] shape.")

    channel_count = int(array.shape[1])
    total = int(array.shape[0] * array.shape[2])
    sums = np.zeros(channel_count, dtype=np.float64)
    squared_sums = np.zeros(channel_count, dtype=np.float64)

    for start in range(0, array.shape[0], chunk_size):
        block = np.asarray(array[start : start + chunk_size], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("Sequence inputs contain NaN or infinite values.")
        sums += block.sum(axis=(0, 2), dtype=np.float64)
        squared_sums += np.square(block).sum(axis=(0, 2), dtype=np.float64)

    mean_1d = sums / total
    variance = np.maximum(squared_sums / total - np.square(mean_1d), 0.0)
    raw_std_1d = np.sqrt(variance)
    constant_mask_1d = raw_std_1d < eps
    scale_1d = np.where(constant_mask_1d, 1.0, raw_std_1d)
    shape = (1, channel_count, 1)
    return SequenceNormalizationStats(
        mean=mean_1d.reshape(shape),
        scale=scale_1d.reshape(shape),
        raw_std=raw_std_1d.reshape(shape),
        constant_mask=constant_mask_1d.reshape(shape),
        value_count_per_channel=total,
    )


def transform_sequences(
    array: np.ndarray,
    stats: SequenceNormalizationStats,
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    """Apply saved training statistics to a sequence array."""

    source = np.asarray(array)
    if source.ndim != 3 or source.shape[1] != stats.mean.shape[1]:
        raise ValueError("Sequence shape is incompatible with fitted channel statistics.")
    output = np.empty(source.shape, dtype=np.float32)
    for start in range(0, source.shape[0], chunk_size):
        block = np.asarray(source[start : start + chunk_size], dtype=np.float64)
        transformed = (block - stats.mean) / stats.scale
        output[start : start + chunk_size] = transformed.astype(np.float32)
    return output


def standardize_sequence_splits(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    *,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SequenceNormalizationStats]:
    """Fit on train and transform train, validation, and test sequences."""

    stats = fit_sequence_stats(train, eps=eps)
    return (
        transform_sequences(train, stats),
        transform_sequences(val, stats),
        transform_sequences(test, stats),
        stats,
    )


def standardize_scalar_frames(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: list[str],
    *,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ScalarNormalizationStats]:
    """Median-impute and z-score scalar features using training rows only."""

    if not columns:
        raise ValueError("Scalar columns must be non-empty.")

    train = train_df.reindex(columns=columns).replace([np.inf, -np.inf], np.nan)
    val = val_df.reindex(columns=columns).replace([np.inf, -np.inf], np.nan)
    test = test_df.reindex(columns=columns).replace([np.inf, -np.inf], np.nan)

    median = train.median(numeric_only=True).reindex(columns).fillna(0.0)
    x_train = train.fillna(median).to_numpy(dtype=np.float64)
    x_val = val.fillna(median).to_numpy(dtype=np.float64)
    x_test = test.fillna(median).to_numpy(dtype=np.float64)
    if not all(np.isfinite(array).all() for array in (x_train, x_val, x_test)):
        raise ValueError("Scalar inputs remain non-finite after train-only imputation.")

    mean = x_train.mean(axis=0, keepdims=True, dtype=np.float64)
    raw_std = x_train.std(axis=0, keepdims=True, dtype=np.float64)
    constant_mask = raw_std < eps
    scale = np.where(constant_mask, 1.0, raw_std)
    stats = ScalarNormalizationStats(
        columns=tuple(columns),
        median=median,
        mean=mean,
        scale=scale,
        raw_std=raw_std,
        constant_mask=constant_mask,
    )
    return (
        ((x_train - mean) / scale).astype(np.float32),
        ((x_val - mean) / scale).astype(np.float32),
        ((x_test - mean) / scale).astype(np.float32),
        stats,
    )


def sequence_training_audit(array: np.ndarray, stats: SequenceNormalizationStats) -> pd.DataFrame:
    """Return transformed training-channel means and standard deviations."""

    means = np.asarray(array, dtype=np.float64).mean(axis=(0, 2))
    stds = np.asarray(array, dtype=np.float64).std(axis=(0, 2))
    return pd.DataFrame(
        {
            "channel": np.arange(array.shape[1], dtype=int),
            "transformed_mean": means,
            "transformed_std": stds,
            "raw_std": stats.raw_std.reshape(-1),
            "scale": stats.scale.reshape(-1),
            "constant": stats.constant_mask.reshape(-1),
        }
    )


def scalar_training_audit(array: np.ndarray, stats: ScalarNormalizationStats) -> pd.DataFrame:
    """Return transformed training-feature means and standard deviations."""

    values = np.asarray(array, dtype=np.float64)
    return pd.DataFrame(
        {
            "feature": list(stats.columns),
            "transformed_mean": values.mean(axis=0),
            "transformed_std": values.std(axis=0),
            "raw_std": stats.raw_std.reshape(-1),
            "scale": stats.scale.reshape(-1),
            "constant": stats.constant_mask.reshape(-1),
        }
    )
