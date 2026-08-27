"""Frequency-domain feature extraction."""

from __future__ import annotations

import numpy as np

from src.signal.preprocessing import clean_signal


def _band_energy(freqs, power, low, high) -> float:
    mask = (freqs >= low) & (freqs < high)
    return float(np.sum(power[mask])) if np.any(mask) else 0.0


def extract_frequency_features(x, fs: float, prefix: str) -> dict[str, float]:
    """Extract FFT-based features for one signal."""
    arr = clean_signal(x)
    if arr.size < 4 or fs <= 0:
        keys = [
            "dominant_frequency",
            "spectral_centroid",
            "spectral_entropy",
            "spectral_kurtosis",
            "band_energy_0_500",
            "band_energy_500_2000",
            "band_energy_2000_5000",
            "high_frequency_ratio",
        ]
        return {f"{prefix}_{k}": np.nan for k in keys}
    arr = arr - np.mean(arr)
    spectrum = np.fft.rfft(arr)
    freqs = np.fft.rfftfreq(arr.size, d=1.0 / fs)
    power = np.abs(spectrum) ** 2
    total = float(np.sum(power)) + 1e-12
    prob = power / total
    dominant_idx = int(np.argmax(power[1:]) + 1) if power.size > 1 else 0
    pstd = float(np.std(power))
    spectral_kurtosis = (
        float(np.mean((power - np.mean(power)) ** 4) / (pstd**4 + 1e-12) - 3.0)
        if power.size > 3 and pstd > 1e-12
        else 0.0
    )
    return {
        f"{prefix}_dominant_frequency": float(freqs[dominant_idx]),
        f"{prefix}_spectral_centroid": float(np.sum(freqs * power) / total),
        f"{prefix}_spectral_entropy": float(-np.sum(prob * np.log(prob + 1e-12))),
        f"{prefix}_spectral_kurtosis": spectral_kurtosis,
        f"{prefix}_band_energy_0_500": _band_energy(freqs, power, 0, 500),
        f"{prefix}_band_energy_500_2000": _band_energy(freqs, power, 500, 2000),
        f"{prefix}_band_energy_2000_5000": _band_energy(freqs, power, 2000, 5000),
        f"{prefix}_high_frequency_ratio": _band_energy(freqs, power, 5000, fs / 2 + 1) / total,
    }
