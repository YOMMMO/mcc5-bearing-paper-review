"""Infer signal columns from heterogeneous CSV files."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import re

import pandas as pd


@dataclass
class SignalColumns:
    time_col: str | None
    vibration_cols: list[str]
    current_cols: list[str]
    torque_col: str | None
    key_phase_col: str | None
    rpm_col: str | None
    other_cols: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _norm(col: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(col).lower())


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def infer_signal_columns(df: pd.DataFrame, dataset_name: str = "") -> SignalColumns:
    """Infer time, vibration, current, torque, key-phase, and rpm columns."""
    cols = list(df.columns)
    numeric_cols = _numeric_columns(df)
    norm = {c: _norm(c) for c in cols}

    time_col = None
    torque_col = None
    key_phase_col = None
    rpm_col = None
    vibration_cols: list[str] = []
    current_cols: list[str] = []

    for c in cols:
        n = norm[c]
        if time_col is None and n in {"time", "timestamp", "t", "sec", "seconds"}:
            time_col = c
        elif rpm_col is None and any(k in n for k in ["rpm", "speed"]):
            rpm_col = c
        elif torque_col is None and "torque" in n:
            torque_col = c
        elif key_phase_col is None and any(k in n for k in ["keyphase", "key", "tach", "encoder"]):
            key_phase_col = c

    for c in numeric_cols:
        n = norm[c]
        if c in {time_col, torque_col, key_phase_col, rpm_col}:
            continue
        if any(k in n for k in ["vibration", "vib", "accel", "acc", "acceleration"]):
            vibration_cols.append(c)
        elif any(k in n for k in ["current", "phase", "ia", "ib", "ic"]) or n in {"u", "v", "w"}:
            current_cols.append(c)

    # Positional fallback for unnamed or non-descriptive numeric data.
    if not vibration_cols and not current_cols:
        nc = len(numeric_cols)
        if nc >= 9:
            time_col = time_col or numeric_cols[0]
            key_phase_col = key_phase_col or numeric_cols[1]
            torque_col = torque_col or numeric_cols[2]
            vibration_cols = numeric_cols[3:6]
            current_cols = numeric_cols[6:9]
        elif nc == 8:
            key_phase_col = key_phase_col or numeric_cols[0]
            torque_col = torque_col or numeric_cols[1]
            vibration_cols = numeric_cols[2:5]
            current_cols = numeric_cols[5:8]
        elif nc == 7:
            time_col = time_col or numeric_cols[0]
            vibration_cols = numeric_cols[1:4]
            current_cols = numeric_cols[4:7]
        elif nc >= 3:
            vibration_cols = numeric_cols[: min(3, nc)]
            current_cols = numeric_cols[3: min(6, nc)]
        elif nc:
            vibration_cols = numeric_cols[:1]

    used = set([c for c in [time_col, torque_col, key_phase_col, rpm_col] if c is not None])
    used.update(vibration_cols)
    used.update(current_cols)
    other_cols = [c for c in cols if c not in used]
    return SignalColumns(
        time_col=time_col,
        vibration_cols=vibration_cols,
        current_cols=current_cols,
        torque_col=torque_col,
        key_phase_col=key_phase_col,
        rpm_col=rpm_col,
        other_cols=other_cols,
    )
