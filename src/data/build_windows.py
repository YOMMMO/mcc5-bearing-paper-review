"""Build leakage-traceable sliding windows from CSV metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.column_inference import infer_signal_columns
from src.utils.io import ensure_dir, safe_read_csv
from src.utils.logger import get_logger
from src.utils.tables import write_table


def _read_signal_csv(path: str | Path) -> pd.DataFrame:
    """Read a signal CSV, detecting no-header numeric files."""
    path = Path(path)
    try:
        df = pd.read_csv(path, header=None, low_memory=False)
        numeric = df.apply(pd.to_numeric, errors="coerce")
        numeric_ratio = numeric.notna().mean().mean()
        if numeric_ratio > 0.95:
            df = numeric
            df.columns = [f"col_{i}" for i in range(df.shape[1])]
            return df
    except Exception:
        pass
    return safe_read_csv(path)


def _array(df: pd.DataFrame, cols: list[str] | str | None, start: int, end: int) -> np.ndarray:
    if cols is None or cols == []:
        return np.empty((0, end - start), dtype=np.float32)
    if isinstance(cols, str):
        cols = [cols]
    data = df.loc[start : end - 1, cols].apply(pd.to_numeric, errors="coerce").to_numpy().T
    return np.nan_to_num(data, nan=0.0).astype(np.float32)


def build_windows(
    metadata: pd.DataFrame,
    out: Path,
    segment_dir: Path,
    dataset: str,
    window_sec: float,
    stride_sec: float,
    sampling_rate: float,
    max_files: int | None = None,
    max_windows_per_file: int | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Create compressed NPZ windows and return window metadata."""
    logger = get_logger("build_windows")
    ensure_dir(segment_dir)
    rows = []
    file_rows = metadata.head(max_files) if max_files else metadata
    win = int(round(window_sec * sampling_rate))
    stride = int(round(stride_sec * sampling_rate))
    if win <= 0 or stride <= 0:
        raise ValueError("Window and stride must be positive.")

    for _, row in file_rows.iterrows():
        path = str(row.get("extracted_path") or "").strip()
        if not path or path.lower() == "nan":
            path = str(row.get("file_path") or "").strip()
        if not path or path.lower() == "nan" or not Path(path).exists():
            logger.warning("Skipping missing CSV path for sample %s", row.get("sample_id"))
            continue
        try:
            df = _read_signal_csv(path)
            cols = infer_signal_columns(df, dataset)
        except Exception as exc:
            logger.warning("Skipping unreadable CSV %s: %s", path, exc)
            continue
        if len(df) < win:
            logger.warning("Skipping short file %s with %d rows", path, len(df))
            continue
        n_windows = 0
        for start in range(0, len(df) - win + 1, stride):
            if max_windows_per_file is not None and n_windows >= max_windows_per_file:
                break
            end = start + win
            window_id = f"{row.get('sample_id')}_w{n_windows:05d}"
            npz_path = segment_dir / f"{window_id}.npz"
            vibration = _array(df, cols.vibration_cols, start, end)
            current = _array(df, cols.current_cols, start, end)
            torque = _array(df, cols.torque_col, start, end)
            key_phase = _array(df, cols.key_phase_col, start, end)
            if not dry_run:
                np.savez_compressed(
                    npz_path,
                    vibration=vibration,
                    current=current,
                    torque=torque,
                    key_phase=key_phase,
                )
            rows.append(
                {
                    "window_id": window_id,
                    "sample_id": row.get("sample_id"),
                    "source_file": row.get("source_file"),
                    "label_group": row.get("label_group"),
                    "label_raw": row.get("label_raw"),
                    "condition_type": row.get("condition_type"),
                    "rpm_nominal": row.get("rpm_nominal"),
                    "load_nm": row.get("load_nm"),
                    "start_index": start,
                    "end_index": end,
                    "start_time": start / sampling_rate,
                    "end_time": end / sampling_rate,
                    "npz_path": str(npz_path),
                    "has_vibration": vibration.size > 0,
                    "has_current": current.size > 0,
                    "has_torque": torque.size > 0,
                    "has_key_phase": key_phase.size > 0,
                }
            )
            n_windows += 1

    windows = pd.DataFrame(rows)
    if not dry_run:
        actual = write_table(windows, out)
        logger.info("Saved window metadata: %s (%d rows)", actual, len(windows))
    if not windows.empty:
        print("Window class distribution:")
        print(windows["label_group"].value_counts(dropna=False).to_string())
        print("Window condition distribution:")
        print(windows["condition_type"].value_counts(dropna=False).to_string())
    else:
        logger.warning("No windows were generated.")
    return windows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--segment_dir", required=True)
    parser.add_argument("--window_sec", type=float, default=1.0)
    parser.add_argument("--stride_sec", type=float, default=0.5)
    parser.add_argument("--sampling_rate", type=float, default=12800)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_windows", type=int, default=None, help="Alias for max total windows; unused for compatibility.")
    parser.add_argument("--max_windows_per_file", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)

    metadata = pd.read_csv(args.metadata) if Path(args.metadata).exists() else pd.DataFrame()
    if metadata.empty:
        get_logger("build_windows").warning("Metadata is empty or missing: %s", args.metadata)
        write_table(pd.DataFrame(), args.out)
        return 0
    windows = build_windows(
        metadata=metadata,
        out=Path(args.out),
        segment_dir=Path(args.segment_dir),
        dataset=args.dataset,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        sampling_rate=args.sampling_rate,
        max_files=args.max_files,
        max_windows_per_file=args.max_windows_per_file,
        dry_run=args.dry_run,
    )
    if args.max_windows and len(windows) > args.max_windows:
        windows = windows.head(args.max_windows)
        if not args.dry_run:
            write_table(windows, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
