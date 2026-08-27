"""Envelope feature extraction."""

from __future__ import annotations

import numpy as np

from src.signal.frequency_features import _band_energy
from src.signal.preprocessing import clean_signal


def hilbert_envelope(x) -> np.ndarray:
    """Compute Hilbert envelope."""
    arr = clean_signal(x)
    if arr.size < 4:
        return np.full_like(arr, np.nan, dtype=float)
    centered = arr - np.mean(arr)
    spectrum = np.fft.fft(centered)
    h = np.zeros(arr.size)
    if arr.size % 2 == 0:
        h[0] = 1
        h[arr.size // 2] = 1
        h[1 : arr.size // 2] = 2
    else:
        h[0] = 1
        h[1 : (arr.size + 1) // 2] = 2
    analytic = np.fft.ifft(spectrum * h)
    return np.abs(analytic)


def envelope_spectrum(x, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Return envelope spectrum frequencies and amplitudes."""
    env = hilbert_envelope(x)
    if env.size < 4 or not np.isfinite(env).any():
        return np.array([]), np.array([])
    env = env - np.nanmean(env)
    amp = np.abs(np.fft.rfft(env))
    freqs = np.fft.rfftfreq(env.size, d=1.0 / fs)
    return freqs, amp


def extract_envelope_features(x, fs: float, prefix: str) -> dict[str, float]:
    """Extract envelope statistics and spectrum features."""
    env = hilbert_envelope(x)
    if env.size < 4 or not np.isfinite(env).any():
        return {
            f"{prefix}_envelope_rms": np.nan,
            f"{prefix}_envelope_kurtosis": np.nan,
            f"{prefix}_envelope_spectrum_peak_freq": np.nan,
            f"{prefix}_envelope_spectrum_peak_amp": np.nan,
            f"{prefix}_envelope_band_energy": np.nan,
        }
    freqs, amp = envelope_spectrum(x, fs)
    idx = int(np.argmax(amp[1:]) + 1) if amp.size > 1 else 0
    std = float(np.std(env))
    env_kurtosis = (
        float(np.mean((env - np.mean(env)) ** 4) / (std**4 + 1e-12) - 3.0)
        if env.size > 3 and std > 1e-12
        else 0.0
    )
    return {
        f"{prefix}_envelope_rms": float(np.sqrt(np.mean(env**2))),
        f"{prefix}_envelope_kurtosis": env_kurtosis,
        f"{prefix}_envelope_spectrum_peak_freq": float(freqs[idx]) if freqs.size else np.nan,
        f"{prefix}_envelope_spectrum_peak_amp": float(amp[idx]) if amp.size else np.nan,
        f"{prefix}_envelope_band_energy": _band_energy(freqs, amp**2, 0, min(1000, fs / 2)),
    }
