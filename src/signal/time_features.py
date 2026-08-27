"""Time-domain feature extraction."""

from __future__ import annotations

import numpy as np

from src.signal.preprocessing import clean_signal


def _nan_features(prefix: str) -> dict[str, float]:
    keys = [
        "mean",
        "std",
        "rms",
        "peak",
        "peak_to_peak",
        "skewness",
        "kurtosis",
        "crest_factor",
        "impulse_factor",
        "shape_factor",
        "clearance_factor",
        "energy",
    ]
    return {f"{prefix}_{k}": np.nan for k in keys}


def extract_time_features(x, prefix: str) -> dict[str, float]:
    """Extract robust time-domain features for one signal."""
    arr = clean_signal(x)
    if arr.size == 0:
        return _nan_features(prefix)
    abs_arr = np.abs(arr)
    mean_abs = float(np.mean(abs_arr))
    rms = float(np.sqrt(np.mean(arr**2)))
    peak = float(np.max(abs_arr))
    sqrt_mean = float(np.mean(np.sqrt(abs_arr))) if arr.size else 0.0
    eps = 1e-12
    centered = arr - np.mean(arr)
    std = float(np.std(arr))
    skewness = float(np.mean(centered**3) / (std**3 + eps)) if arr.size > 2 else 0.0
    kurt = float(np.mean(centered**4) / (std**4 + eps) - 3.0) if arr.size > 3 else 0.0
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": std,
        f"{prefix}_rms": rms,
        f"{prefix}_peak": peak,
        f"{prefix}_peak_to_peak": float(np.ptp(arr)),
        f"{prefix}_skewness": skewness if std > eps else 0.0,
        f"{prefix}_kurtosis": kurt if std > eps else 0.0,
        f"{prefix}_crest_factor": peak / (rms + eps),
        f"{prefix}_impulse_factor": peak / (mean_abs + eps),
        f"{prefix}_shape_factor": rms / (mean_abs + eps),
        f"{prefix}_clearance_factor": peak / (sqrt_mean**2 + eps),
        f"{prefix}_energy": float(np.sum(arr**2)),
    }
