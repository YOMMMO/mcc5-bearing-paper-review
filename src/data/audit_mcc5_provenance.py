"""Audit MCC5 raw-file identity and acquisition-session metadata.

The public review repository does not redistribute the MCC5 recordings. This
script lets a reviewer who downloaded the official dataset recreate compact
file-integrity and timestamp-collision evidence without publishing raw signals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


TIMESTAMP_RE = re.compile(r"(?P<timestamp>\d{12})(?:[A-Za-z])?$")
ROLE_COLUMNS = [
    "source_file_role",
    "cross_condition_role",
    "cross_load_role",
    "cross_rpm_role",
]


def _timestamp_from_name(name: str) -> str:
    match = TIMESTAMP_RE.search(name)
    if not match:
        raise ValueError(f"No YYMMDDhhmmss timestamp in recording name: {name}")
    return match.group("timestamp")


def _file_sha_shape(path: Path) -> tuple[str, int, int, int]:
    digest = hashlib.sha256()
    newline_count = 0
    last_byte = b""
    with path.open("rb") as handle:
        first_line = handle.readline()
        digest.update(first_line)
        newline_count += first_line.count(b"\n")
        if first_line:
            last_byte = first_line[-1:]
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            newline_count += chunk.count(b"\n")
            last_byte = chunk[-1:]
    physical_lines = newline_count + (1 if last_byte and last_byte != b"\n" else 0)
    # MCC5 raw CSV files contain numeric rows without a header.
    row_count = physical_lines
    decoded = first_line.decode("utf-8-sig").rstrip("\r\n")
    column_count = len(next(csv.reader([decoded]))) if decoded else 0
    return digest.hexdigest(), path.stat().st_size, row_count, column_count


def _build_file_index(raw_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in raw_root.rglob("*.csv"):
        key = path.name.casefold()
        if key in index:
            duplicates.add(key)
        else:
            index[key] = path
    if duplicates:
        names = ", ".join(sorted(duplicates)[:10])
        raise RuntimeError(f"Raw root contains duplicate CSV basenames: {names}")
    return index


def _cramers_v(table: pd.DataFrame) -> float:
    observed = table.to_numpy(dtype=float)
    total = observed.sum()
    if total <= 0:
        return float("nan")
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / total
    valid = expected > 0
    chi2 = float(np.sum(((observed - expected) ** 2)[valid] / expected[valid]))
    denominator = total * max(1, min(observed.shape[0] - 1, observed.shape[1] - 1))
    return math.sqrt(chi2 / denominator)


def _normalized_mutual_information(table: pd.DataFrame) -> float:
    observed = table.to_numpy(dtype=float)
    total = observed.sum()
    joint = observed / total
    p_rows = joint.sum(axis=1)
    p_cols = joint.sum(axis=0)
    mutual_information = 0.0
    for row in range(joint.shape[0]):
        for column in range(joint.shape[1]):
            probability = joint[row, column]
            if probability > 0:
                mutual_information += probability * math.log(
                    probability / (p_rows[row] * p_cols[column])
                )
    h_rows = -sum(value * math.log(value) for value in p_rows if value > 0)
    h_cols = -sum(value * math.log(value) for value in p_cols if value > 0)
    denominator = math.sqrt(h_rows * h_cols)
    return mutual_information / denominator if denominator else 0.0


def _write_session_audit(catalog: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    session = catalog.copy()
    session["acquisition_timestamp"] = session["original_recording_name"].map(_timestamp_from_name)
    session["acquisition_date"] = session["acquisition_timestamp"].str[:6]
    session.to_csv(out_dir / "session_metadata_audit.csv", index=False)

    class_counts = (
        session.groupby(["acquisition_date", "publication_label"], as_index=False)
        .size()
        .rename(columns={"size": "recording_count"})
    )
    class_counts.to_csv(out_dir / "acquisition_date_class_counts.csv", index=False)

    role_rows: list[dict[str, object]] = []
    for role_column in ROLE_COLUMNS:
        protocol = role_column.removesuffix("_role")
        grouped = session.groupby(
            ["acquisition_date", role_column, "publication_label"], as_index=False
        ).size()
        for row in grouped.itertuples(index=False):
            role_rows.append(
                {
                    "protocol": protocol,
                    "acquisition_date": row.acquisition_date,
                    "role": getattr(row, role_column),
                    "publication_label": row.publication_label,
                    "recording_count": int(row.size),
                }
            )
    pd.DataFrame(role_rows).to_csv(
        out_dir / "acquisition_date_protocol_role_counts.csv", index=False
    )

    contingency = pd.crosstab(session["acquisition_date"], session["publication_label"])
    majority_correct = int(contingency.max(axis=1).sum())
    summary: dict[str, object] = {
        "recording_count": int(len(session)),
        "acquisition_date_count": int(session["acquisition_date"].nunique()),
        "date_majority_correct_in_sample": majority_correct,
        "date_majority_accuracy_in_sample": majority_correct / len(session),
        "cramers_v_date_vs_class": _cramers_v(contingency),
        "normalized_mutual_information_date_vs_class": _normalized_mutual_information(contingency),
        "interpretation": (
            "Descriptive association only. The in-sample date-majority score is not "
            "an out-of-sample diagnostic result and indicates possible session confounding."
        ),
    }
    (out_dir / "session_confounding_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def _stream_pair_comparison(path_a: Path, path_b: Path, chunksize: int) -> dict[str, object]:
    readers = (
        pd.read_csv(path_a, header=None, chunksize=chunksize),
        pd.read_csv(path_b, header=None, chunksize=chunksize),
    )
    n = None
    sum_x = sum_y = sum_x2 = sum_y2 = sum_xy = None
    rows = 0
    all_values_equal = True
    time_axis_equal = True
    signal_values_equal = True
    max_abs_signal_difference = 0.0
    column_names: list[str] | None = None

    while True:
        try:
            chunk_a = next(readers[0])
        except StopIteration:
            chunk_a = None
        try:
            chunk_b = next(readers[1])
        except StopIteration:
            chunk_b = None
        if chunk_a is None or chunk_b is None:
            if chunk_a is not None or chunk_b is not None:
                all_values_equal = time_axis_equal = signal_values_equal = False
            break
        if list(chunk_a.columns) != list(chunk_b.columns) or len(chunk_a) != len(chunk_b):
            all_values_equal = time_axis_equal = signal_values_equal = False
            break
        if column_names is None:
            if len(chunk_a.columns) == 9:
                column_names = [
                    "time",
                    "key_phase",
                    "torque",
                    "vibration_1",
                    "vibration_2",
                    "vibration_3",
                    "current_1",
                    "current_2",
                    "current_3",
                ]
            else:
                column_names = [f"column_{column}" for column in chunk_a.columns]
            width = max(0, len(column_names) - 1)
            n = np.zeros(width, dtype=np.int64)
            sum_x = np.zeros(width, dtype=np.float64)
            sum_y = np.zeros(width, dtype=np.float64)
            sum_x2 = np.zeros(width, dtype=np.float64)
            sum_y2 = np.zeros(width, dtype=np.float64)
            sum_xy = np.zeros(width, dtype=np.float64)
        array_a = chunk_a.to_numpy(dtype=np.float64)
        array_b = chunk_b.to_numpy(dtype=np.float64)
        equal = np.array_equal(array_a, array_b, equal_nan=True)
        all_values_equal = all_values_equal and equal
        time_axis_equal = time_axis_equal and np.array_equal(
            array_a[:, 0], array_b[:, 0], equal_nan=True
        )
        signals_a = array_a[:, 1:]
        signals_b = array_b[:, 1:]
        signal_values_equal = signal_values_equal and np.array_equal(
            signals_a, signals_b, equal_nan=True
        )
        finite = np.isfinite(signals_a) & np.isfinite(signals_b)
        safe_a = np.where(finite, signals_a, 0.0)
        safe_b = np.where(finite, signals_b, 0.0)
        n += finite.sum(axis=0)
        sum_x += safe_a.sum(axis=0)
        sum_y += safe_b.sum(axis=0)
        sum_x2 += (safe_a * safe_a).sum(axis=0)
        sum_y2 += (safe_b * safe_b).sum(axis=0)
        sum_xy += (safe_a * safe_b).sum(axis=0)
        if finite.any():
            difference = np.where(finite, np.abs(signals_a - signals_b), 0.0)
            max_abs_signal_difference = max(
                max_abs_signal_difference, float(np.max(difference))
            )
        rows += len(chunk_a)

    correlations: dict[str, float | None] = {}
    if column_names and n is not None:
        numerator = n * sum_xy - sum_x * sum_y
        denominator = np.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))
        for index, name in enumerate(column_names[1:]):
            correlations[name] = (
                float(numerator[index] / denominator[index])
                if denominator[index] > 0
                else None
            )
    return {
        "rows_compared": rows,
        "column_count": len(column_names or []),
        "all_values_equal": all_values_equal,
        "time_axis_equal": time_axis_equal,
        "signal_values_equal": signal_values_equal,
        "max_abs_signal_difference": max_abs_signal_difference,
        "signal_correlations_json": json.dumps(correlations, sort_keys=True),
    }


def _write_raw_audit(
    catalog: pd.DataFrame,
    raw_root: Path,
    out_dir: Path,
    deep_compare: bool,
    chunksize: int,
) -> dict[str, object]:
    file_index = _build_file_index(raw_root)
    rows: list[dict[str, object]] = []
    resolved: dict[str, Path] = {}
    for row in catalog.itertuples(index=False):
        filename = f"{row.original_recording_name}.csv"
        path = file_index.get(filename.casefold())
        if path is None:
            raise FileNotFoundError(f"Raw recording not found below {raw_root}: {filename}")
        sha256, size, row_count, column_count = _file_sha_shape(path)
        timestamp = _timestamp_from_name(row.original_recording_name)
        resolved[row.recording_id] = path
        rows.append(
            {
                "recording_id": row.recording_id,
                "original_recording_name": row.original_recording_name,
                "publication_label": row.publication_label,
                "condition_type": row.condition_type,
                "rpm_nominal": row.rpm_nominal,
                "load_nm": row.load_nm,
                "acquisition_timestamp": timestamp,
                "relative_raw_file": path.relative_to(raw_root).as_posix(),
                "file_size_bytes": size,
                "data_row_count": row_count,
                "column_count": column_count,
                "sha256": sha256,
            }
        )
    integrity = pd.DataFrame(rows)
    sha_counts = Counter(integrity["sha256"])
    timestamp_counts = Counter(integrity["acquisition_timestamp"])
    integrity["sha256_group_size"] = integrity["sha256"].map(sha_counts)
    integrity["timestamp_group_size"] = integrity["acquisition_timestamp"].map(timestamp_counts)
    integrity["exact_duplicate_content"] = integrity["sha256_group_size"] > 1
    integrity.to_csv(out_dir / "raw_file_integrity_audit.csv", index=False)

    collision_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    for timestamp, group in integrity.groupby("acquisition_timestamp", sort=True):
        if len(group) < 2:
            continue
        collision_rows.append(
            {
                "acquisition_timestamp": timestamp,
                "recording_count": len(group),
                "unique_sha256_count": int(group["sha256"].nunique()),
                "all_file_hashes_unique": bool(group["sha256"].nunique() == len(group)),
                "recording_ids_json": json.dumps(sorted(group["recording_id"])),
            }
        )
        if deep_compare:
            records = group.to_dict(orient="records")
            for record_a, record_b in combinations(records, 2):
                comparison = _stream_pair_comparison(
                    resolved[record_a["recording_id"]],
                    resolved[record_b["recording_id"]],
                    chunksize,
                )
                pair_rows.append(
                    {
                        "acquisition_timestamp": timestamp,
                        "recording_a": record_a["recording_id"],
                        "recording_b": record_b["recording_id"],
                        "sha256_equal": record_a["sha256"] == record_b["sha256"],
                        **comparison,
                    }
                )
    pd.DataFrame(collision_rows).to_csv(out_dir / "timestamp_collision_audit.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(out_dir / "same_timestamp_pair_comparison.csv", index=False)

    summary: dict[str, object] = {
        "catalog_recording_count": int(len(catalog)),
        "resolved_raw_file_count": int(len(integrity)),
        "unique_sha256_count": int(integrity["sha256"].nunique()),
        "exact_duplicate_sha256_group_count": int(
            integrity.loc[integrity["sha256_group_size"] > 1, "sha256"].nunique()
        ),
        "repeated_timestamp_group_count": len(collision_rows),
        "deep_timestamp_pair_comparison_performed": deep_compare,
        "raw_dataset_redistributed": False,
    }
    (out_dir / "raw_integrity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/recording_catalog_and_splits.csv")
    parser.add_argument("--raw-root")
    parser.add_argument("--out", default="evidence")
    parser.add_argument("--deep-compare-timestamp-groups", action="store_true")
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args(argv)

    catalog = pd.read_csv(args.catalog)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    session_summary = _write_session_audit(catalog, out_dir)
    print(json.dumps({"session": session_summary}, indent=2))
    if args.raw_root:
        raw_summary = _write_raw_audit(
            catalog,
            Path(args.raw_root),
            out_dir,
            args.deep_compare_timestamp_groups,
            args.chunksize,
        )
        print(json.dumps({"raw_integrity": raw_summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
