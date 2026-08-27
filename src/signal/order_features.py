"""Order-domain feature extraction."""

from __future__ import annotations

import numpy as np

from src.signal.preprocessing import clean_signal

ORDER_BANDS = [(0.5, 1.5), (1.5, 2.5), (2.5, 5), (5, 10), (10, 20), (20, 50)]


def _nan_order(prefix: str) -> dict[str, float]:
    out = {f"{prefix}_order_energy_{lo:g}_{hi:g}": np.nan for lo, hi in ORDER_BANDS}
    out.update(
        {
            f"{prefix}_peak_order": np.nan,
            f"{prefix}_peak_order_amp": np.nan,
            f"{prefix}_order_entropy": np.nan,
        }
    )
    return out


def _order_spectrum_features(orders: np.ndarray, amp: np.ndarray, prefix: str) -> dict[str, float]:
    """Summarize an already computed order spectrum."""
    if orders.size == 0 or amp.size == 0:
        return _nan_order(prefix)
    power = amp**2
    total = float(np.sum(power)) + 1e-12
    out = {}
    for lo, hi in ORDER_BANDS:
        mask = (orders >= lo) & (orders < hi)
        out[f"{prefix}_order_energy_{lo:g}_{hi:g}"] = float(np.sum(power[mask]))
    idx = int(np.argmax(power[1:]) + 1) if power.size > 1 else 0
    prob = power / total
    out[f"{prefix}_peak_order"] = float(orders[idx])
    out[f"{prefix}_peak_order_amp"] = float(amp[idx])
    out[f"{prefix}_order_entropy"] = float(-np.sum(prob * np.log(prob + 1e-12)))
    return out


def frequency_to_order_features(x, fs: float, rpm: float | None, prefix: str) -> dict[str, float]:
    """Convert FFT frequencies to orders and extract band energies."""
    arr = clean_signal(x)
    if arr.size < 4 or fs <= 0 or rpm is None or not np.isfinite(rpm) or rpm <= 0:
        return _nan_order(prefix)
    arr = arr - np.mean(arr)
    amp = np.abs(np.fft.rfft(arr))
    freqs = np.fft.rfftfreq(arr.size, d=1.0 / fs)
    orders = freqs / (rpm / 60.0)
    return _order_spectrum_features(orders, amp, prefix)


def _nan_keyphase_order(prefix: str) -> dict[str, float]:
    """NaN order features plus key-phase diagnostic fields."""
    out = _nan_order(prefix)
    out.update(
        {
            f"{prefix}_pulse_count": np.nan,
            f"{prefix}_estimated_rpm": np.nan,
            f"{prefix}_rpm_std": np.nan,
            f"{prefix}_angle_revolutions": np.nan,
        }
    )
    return out


def _dedupe_indices(indices: np.ndarray, min_dist: int) -> np.ndarray:
    """Keep candidate pulse indices separated by at least min_dist samples."""
    kept: list[int] = []
    last = -min_dist
    for idx in np.asarray(indices, dtype=int):
        if idx - last >= min_dist:
            kept.append(int(idx))
            last = int(idx)
    return np.asarray(kept, dtype=int)


def _detect_key_phase_pulses(key: np.ndarray, fs: float) -> np.ndarray:
    """Detect rising-edge or peak-like key-phase pulses without scipy."""
    spread = float(np.nanmax(key) - np.nanmin(key))
    if not np.isfinite(spread) or spread <= 1e-12:
        spread = float(np.nanpercentile(key, 95) - np.nanpercentile(key, 5))
    if not np.isfinite(spread) or spread <= 1e-12:
        return np.asarray([], dtype=int)
    threshold = float(np.nanmin(key) + 0.5 * spread)
    above = key > threshold
    rising_edges = np.where((~above[:-1]) & above[1:])[0] + 1
    min_dist = max(1, int(fs * 0.002))
    if rising_edges.size >= 2:
        return _dedupe_indices(rising_edges, min_dist)

    peak_threshold = float(np.nanmean(key) + 0.5 * np.nanstd(key))
    candidates = np.where(
        (key[1:-1] > key[:-2]) & (key[1:-1] >= key[2:]) & (key[1:-1] > peak_threshold)
    )[0] + 1
    return _dedupe_indices(candidates, min_dist)


def _angle_resample(x: np.ndarray, pulses: np.ndarray, samples_per_rev: int) -> np.ndarray:
    """Resample each detected revolution to a fixed number of angle samples."""
    revs = []
    phase_new = np.linspace(0.0, 1.0, samples_per_rev, endpoint=False)
    for start, stop in zip(pulses[:-1], pulses[1:]):
        if stop - start < 4:
            continue
        segment = x[start:stop]
        phase_old = np.linspace(0.0, 1.0, segment.size, endpoint=False)
        revs.append(np.interp(phase_new, phase_old, segment))
    if not revs:
        return np.asarray([], dtype=float)
    return np.concatenate(revs).astype(float)


def key_phase_order_features(x, key_phase, fs: float, prefix: str) -> dict[str, float]:
    """Compute order features from key-phase pulses using angle resampling.

    Pulses are detected from rising edges or local peaks. When at least two
    full revolutions are available, each revolution is resampled to a fixed
    number of angle samples before the order spectrum is computed. If pulse
    detection is unreliable, NaN order features are returned.
    """
    key = clean_signal(key_phase)
    arr = clean_signal(x)
    n = min(arr.size, key.size)
    if n < 8 or fs <= 0:
        return _nan_keyphase_order(prefix)
    arr = arr[:n]
    key = key[:n]
    pulses = _detect_key_phase_pulses(key, fs)
    if pulses.size < 2:
        return _nan_keyphase_order(prefix)
    periods = np.diff(pulses) / fs
    periods = periods[periods > 0]
    if periods.size == 0:
        return _nan_keyphase_order(prefix)

    rpm_inst = 60.0 / periods
    rpm = float(np.nanmedian(rpm_inst))
    diagnostics = {
        f"{prefix}_pulse_count": float(pulses.size),
        f"{prefix}_estimated_rpm": rpm,
        f"{prefix}_rpm_std": float(np.nanstd(rpm_inst)),
        f"{prefix}_angle_revolutions": float(max(0, pulses.size - 1)),
    }
    if not np.isfinite(rpm) or rpm <= 0:
        out = _nan_order(prefix)
        out.update(diagnostics)
        return out

    samples_per_rev = 256
    angle_signal = _angle_resample(arr, pulses, samples_per_rev=samples_per_rev)
    if angle_signal.size >= samples_per_rev:
        angle_signal = angle_signal - np.mean(angle_signal)
        amp = np.abs(np.fft.rfft(angle_signal))
        orders = np.fft.rfftfreq(angle_signal.size, d=1.0 / samples_per_rev)
        out = _order_spectrum_features(orders, amp, prefix)
    else:
        out = frequency_to_order_features(arr, fs, rpm, prefix)
    out.update(diagnostics)
    return out
