"""Table read/write helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import ensure_dir


def read_table(path: str | Path) -> pd.DataFrame:
    """Read parquet or CSV based on suffix."""
    p = Path(path)
    if not p.exists() and p.suffix.lower() == ".parquet":
        csv_fallback = p.with_suffix(".csv")
        if csv_fallback.exists():
            p = csv_fallback
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame, falling back to CSV if parquet support is missing."""
    p = Path(path)
    ensure_dir(p.parent)
    if p.suffix.lower() == ".parquet":
        try:
            df.to_parquet(p, index=False)
            fallback = p.with_suffix(".csv")
            if fallback.exists():
                df.to_csv(fallback, index=False)
            return p
        except Exception:
            fallback = p.with_suffix(".csv")
            df.to_csv(fallback, index=False)
            return fallback
    df.to_csv(p, index=False)
    return p
