"""Audit split files for source-file leakage and distribution drift."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.io import ensure_dir, save_json
from src.utils.logger import get_logger
from src.utils.tables import read_table


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.fillna("<missing>").astype(str).value_counts().items()}


def _split_counts(df: pd.DataFrame, column: str) -> dict[str, dict[str, int]]:
    if column not in df:
        return {}
    out: dict[str, dict[str, int]] = {}
    for split, sub in df.groupby("split", dropna=False):
        out[str(split)] = _counts(sub[column])
    return out


def _range_by_split(df: pd.DataFrame, column: str) -> dict[str, dict[str, float]]:
    if column not in df:
        return {}
    values = pd.to_numeric(df[column], errors="coerce")
    out: dict[str, dict[str, float]] = {}
    for split, idx in df.groupby("split", dropna=False).groups.items():
        sub = values.loc[idx].dropna()
        if sub.empty:
            continue
        out[str(split)] = {
            "min": float(sub.min()),
            "median": float(sub.median()),
            "max": float(sub.max()),
        }
    return out


def _audit_one(features: pd.DataFrame, split_path: Path) -> dict[str, Any]:
    split = pd.read_csv(split_path)
    merged = features.merge(split, on="window_id", how="inner", suffixes=("", "_split"))
    missing_from_features = int(len(split) - len(merged))
    feature_missing_split = int(features["window_id"].nunique() - merged["window_id"].nunique())

    source_leakage_count = 0
    source_leakage_examples: list[dict[str, Any]] = []
    if "source_file" in merged:
        source_split_counts = merged.groupby("source_file")["split"].nunique()
        leaky_sources = source_split_counts[source_split_counts > 1]
        source_leakage_count = int(len(leaky_sources))
        for source in leaky_sources.index[:10]:
            sub = merged[merged["source_file"] == source]
            source_leakage_examples.append(
                {
                    "source_file": str(source),
                    "splits": sorted(sub["split"].dropna().astype(str).unique().tolist()),
                    "window_count": int(len(sub)),
                }
            )

    split_counts = _counts(merged["split"]) if "split" in merged else {}
    expected = {"train", "test"}
    if Path(split_path).stem.endswith("source_file_split"):
        expected.add("val")
    missing_expected = sorted(expected.difference(split_counts))
    status = "ok"
    issues = []
    if source_leakage_count > 0:
        issues.append("source_file_leakage")
    if missing_from_features > 0:
        issues.append("split_rows_missing_features")
    if feature_missing_split > 0:
        issues.append("feature_rows_missing_split")
    if missing_expected:
        issues.append("missing_expected_splits:" + ",".join(missing_expected))
    if issues:
        status = "warning" if source_leakage_count == 0 else "failed"

    return {
        "split_file": str(split_path),
        "status": status,
        "issues": ";".join(issues),
        "rows_in_split": int(len(split)),
        "matched_rows": int(len(merged)),
        "feature_rows": int(len(features)),
        "missing_from_features": missing_from_features,
        "feature_rows_missing_split": feature_missing_split,
        "split_counts": split_counts,
        "unique_source_files": int(merged["source_file"].nunique()) if "source_file" in merged else 0,
        "source_files_by_split": (
            {str(k): int(v) for k, v in merged.groupby("split")["source_file"].nunique().items()}
            if "source_file" in merged and "split" in merged
            else {}
        ),
        "source_file_leakage_count": source_leakage_count,
        "source_file_leakage_examples": source_leakage_examples,
        "label_counts_by_split": _split_counts(merged, "label_group"),
        "condition_counts_by_split": _split_counts(merged, "condition_type"),
        "rpm_range_by_split": _range_by_split(merged, "rpm_nominal"),
        "load_range_by_split": _range_by_split(merged, "load_nm"),
    }


def audit_splits(
    features_path: Path,
    split_root: Path,
    split_glob: str,
    max_windows: int | None = None,
) -> list[dict[str, Any]]:
    """Return audit records for all matching split files."""
    features = read_table(features_path)
    if max_windows:
        features = features.head(max_windows)
    if "window_id" not in features:
        raise ValueError(f"Features table lacks window_id: {features_path}")
    split_paths = sorted(split_root.glob(split_glob)) if split_root.is_dir() else [split_root]
    return [_audit_one(features, path) for path in split_paths if path.exists()]


def _flatten_for_csv(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rec in records:
        row = {}
        for key, value in rec.items():
            if isinstance(value, (dict, list)):
                row[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
            else:
                row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--split", required=True, help="Split file or directory.")
    parser.add_argument("--split_glob", default="*_split.csv")
    parser.add_argument("--out", default="results/tables/split_audit.csv")
    parser.add_argument("--log_out", default="results/logs/split_audit.json")
    parser.add_argument("--max_files", type=int, default=None, help="Accepted for common data-processing CLI compatibility.")
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)

    logger = get_logger("audit_splits")
    if args.dry_run:
        logger.info("Dry run: would audit splits in %s using features %s.", args.split, args.features)
        return 0

    records = audit_splits(Path(args.features), Path(args.split), args.split_glob, args.max_windows)
    ensure_dir(Path(args.out).parent)
    ensure_dir(Path(args.log_out).parent)
    _flatten_for_csv(records).to_csv(args.out, index=False)
    save_json({"records": records}, args.log_out)
    failed = [rec for rec in records if rec.get("source_file_leakage_count", 0) > 0]
    logger.info("Saved split audit: %s and %s (%d records)", args.out, args.log_out, len(records))
    if failed:
        logger.error("Detected source-file leakage in %d split files.", len(failed))
        return 2 if args.strict else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
