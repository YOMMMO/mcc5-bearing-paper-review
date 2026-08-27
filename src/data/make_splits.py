"""Create leakage-safe source-file and condition-level split files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.io import ensure_dir
from src.utils.logger import get_logger
from src.utils.tables import read_table


def _save(df: pd.DataFrame, out_dir: Path, dataset: str, name: str) -> Path:
    path = out_dir / f"{dataset}_{name}.csv"
    df.to_csv(path, index=False)
    return path


def _source_file_split(features: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    groups = features[["source_file", "label_group"]].drop_duplicates("source_file")
    if len(groups) < 3:
        split_map = {sf: "train" for sf in groups["source_file"]}
    else:
        rng = np.random.default_rng(seed)
        train_parts = []
        val_parts = []
        test_parts = []
        for _, group in groups.groupby("label_group", dropna=False):
            shuffled = group.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000)))
            n = len(shuffled)
            n_train = max(1, int(round(n * 0.70)))
            n_val = max(0, int(round(n * 0.15)))
            if n_train + n_val >= n and n >= 3:
                n_train = n - 2
                n_val = 1
            train_parts.append(shuffled.iloc[:n_train])
            val_parts.append(shuffled.iloc[n_train : n_train + n_val])
            test_parts.append(shuffled.iloc[n_train + n_val :])
        train_g = pd.concat(train_parts, ignore_index=True)
        val_g = pd.concat(val_parts, ignore_index=True) if val_parts else groups.iloc[0:0]
        test_g = pd.concat(test_parts, ignore_index=True) if test_parts else groups.iloc[0:0]
        split_map = {sf: "train" for sf in train_g["source_file"]}
        split_map.update({sf: "val" for sf in val_g["source_file"]})
        split_map.update({sf: "test" for sf in test_g["source_file"]})
        counts = pd.Series(split_map).value_counts()
        for target in ["val", "test"]:
            if counts.get(target, 0) > 0:
                continue
            donor = next((name for name in ["train", "test", "val"] if counts.get(name, 0) > 1), None)
            if donor is None:
                continue
            candidates = sorted(sf for sf, split in split_map.items() if split == donor)
            chosen = candidates[seed % len(candidates)]
            split_map[chosen] = target
            counts = pd.Series(split_map).value_counts()
    return pd.DataFrame(
        {
            "window_id": features["window_id"],
            "split": features["source_file"].map(split_map).fillna("train"),
            "split_type": "source_file_split",
            "reason": "grouped_by_source_file",
        }
    )


def _cross_condition(features: pd.DataFrame) -> pd.DataFrame | None:
    conds = set(features["condition_type"].dropna().astype(str))
    if {"speed_circulation", "torque_circulation"}.issubset(conds):
        split = np.where(features["condition_type"].astype(str) == "speed_circulation", "train", "test")
    elif len(conds) >= 2:
        first = sorted(conds)[0]
        split = np.where(features["condition_type"].astype(str) == first, "train", "test")
    else:
        return None
    return pd.DataFrame(
        {
            "window_id": features["window_id"],
            "split": split,
            "split_type": "cross_condition_split",
            "reason": "condition_holdout",
        }
    )


def _numeric_holdout(features: pd.DataFrame, col: str, split_type: str) -> pd.DataFrame | None:
    values = pd.to_numeric(features[col], errors="coerce")
    if values.notna().sum() < 2 or values.nunique(dropna=True) < 2:
        return None
    median = values.median()
    split = np.where(values <= median, "train", "test")
    return pd.DataFrame(
        {
            "window_id": features["window_id"],
            "split": split,
            "split_type": split_type,
            "reason": f"median_holdout_{col}",
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)

    logger = get_logger("make_splits")
    out_dir = ensure_dir(args.out_dir)
    if not Path(args.features).exists() and not Path(args.features).with_suffix(".csv").exists():
        logger.warning("Features file missing: %s", args.features)
        return 0
    features = read_table(args.features)
    if args.max_windows:
        features = features.head(args.max_windows)
    if features.empty or "window_id" not in features:
        logger.warning("Features file has no rows/window_id: %s", args.features)
        return 0
    if args.dry_run:
        logger.info("Dry run: would create split files from %d rows", len(features))
        return 0

    paths = [_save(_source_file_split(features), out_dir, args.dataset, "source_file_split")]
    cond = _cross_condition(features)
    if cond is not None:
        paths.append(_save(cond, out_dir, args.dataset, "cross_condition_split"))
    rpm = _numeric_holdout(features, "rpm_nominal", "cross_rpm_split") if "rpm_nominal" in features else None
    if rpm is not None:
        paths.append(_save(rpm, out_dir, args.dataset, "cross_rpm_split"))
    load = _numeric_holdout(features, "load_nm", "cross_load_split") if "load_nm" in features else None
    if load is not None:
        paths.append(_save(load, out_dir, args.dataset, "cross_load_split"))
    for path in paths:
        logger.info("Saved split: %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
