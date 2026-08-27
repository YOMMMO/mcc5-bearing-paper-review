"""Extract physical signal features from saved NPZ windows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.signal.envelope_features import extract_envelope_features
from src.signal.frequency_features import extract_frequency_features
from src.signal.order_features import frequency_to_order_features, key_phase_order_features
from src.signal.time_features import extract_time_features
from src.utils.logger import get_logger
from src.utils.tables import read_table, write_table


def _channel_features(arr: np.ndarray, fs: float, rpm, prefix: str, include_envelope: bool) -> dict:
    feats = {}
    feats.update(extract_time_features(arr, prefix))
    feats.update(extract_frequency_features(arr, fs, prefix))
    if include_envelope:
        feats.update(extract_envelope_features(arr, fs, prefix))
    feats.update(frequency_to_order_features(arr, fs, rpm, prefix))
    return feats


def extract_features(windows: pd.DataFrame, fs: float, max_windows: int | None = None) -> pd.DataFrame:
    """Extract one feature row per window."""
    logger = get_logger("extract_features")
    rows = []
    work = windows.head(max_windows) if max_windows else windows
    for _, meta in work.iterrows():
        npz_path = Path(str(meta.get("npz_path", "")))
        if not npz_path.exists():
            logger.warning("Missing NPZ: %s", npz_path)
            continue
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                vibration = data.get("vibration", np.empty((0, 0)))
                current = data.get("current", np.empty((0, 0)))
                torque = data.get("torque", np.empty((0, 0)))
                key_phase = data.get("key_phase", np.empty((0, 0)))
        except Exception as exc:
            logger.warning("Skipping corrupt NPZ %s: %s", npz_path, exc)
            continue

        rpm = pd.to_numeric(pd.Series([meta.get("rpm_nominal")]), errors="coerce").iloc[0]
        row = {
            "window_id": meta.get("window_id"),
            "source_file": meta.get("source_file"),
            "sample_id": meta.get("sample_id"),
            "label_group": meta.get("label_group"),
            "label_raw": meta.get("label_raw"),
            "condition_type": meta.get("condition_type"),
            "rpm_nominal": rpm,
            "load_nm": meta.get("load_nm"),
        }
        for i in range(vibration.shape[0]):
            row.update(_channel_features(vibration[i], fs, rpm, f"vib_{i}", include_envelope=True))
            if key_phase.size:
                row.update(key_phase_order_features(vibration[i], key_phase[0], fs, f"vib_{i}_keyphase"))
        for i in range(current.shape[0]):
            row.update(_channel_features(current[i], fs, rpm, f"cur_{i}", include_envelope=False))
        for i in range(torque.shape[0]):
            row.update(extract_time_features(torque[i], f"torque_{i}"))
        if key_phase.size:
            row.update(extract_time_features(key_phase[0], "key_phase_0"))
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fs", type=float, default=12800)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)

    logger = get_logger("extract_features")
    if not Path(args.windows).exists() and not Path(args.windows).with_suffix(".csv").exists():
        logger.warning("Window metadata missing: %s", args.windows)
        write_table(pd.DataFrame(), args.out)
        return 0
    windows = read_table(args.windows)
    if args.max_files and "source_file" in windows:
        keep = windows["source_file"].drop_duplicates().head(args.max_files)
        windows = windows[windows["source_file"].isin(keep)]
    features = extract_features(windows, args.fs, args.max_windows)
    if args.dry_run:
        logger.info("Dry run: extracted %d feature rows", len(features))
        return 0
    actual = write_table(features, args.out)
    preview = Path(args.out).with_suffix(".preview.csv")
    features.head(50).to_csv(preview, index=False)
    logger.info("Saved features: %s (%d rows)", actual, len(features))
    if not features.empty:
        missing = features.isna().mean().sort_values(ascending=False).head(20)
        print("Top missing feature rates:")
        print(missing.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
