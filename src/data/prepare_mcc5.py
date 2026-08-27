"""Prepare MCC5 metadata from extracted CSV files or zip archives."""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.parsing import BEARING_CLASSES, parse_mcc5_filename
from src.utils.io import ensure_dir
from src.utils.logger import get_logger


def str2bool(value) -> bool:
    """Parse common CLI boolean strings."""
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def _iter_csv_candidates(root: Path):
    for path in sorted(root.rglob("*.csv")):
        yield {"file_path": path, "zip_path": None, "member_path": None}
    for zip_path in sorted(root.rglob("*.zip")):
        try:
            with zipfile.ZipFile(zip_path) as zf:
                for member in sorted(zf.namelist()):
                    if member.lower().endswith(".csv") and not member.endswith("/"):
                        yield {"file_path": None, "zip_path": zip_path, "member_path": member}
        except zipfile.BadZipFile:
            continue


def _balanced_limit(rows: list[dict], max_files: int | None) -> list[dict]:
    if max_files is None or max_files <= 0 or len(rows) <= max_files:
        return rows
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("condition_type", "unknown")), str(row.get("label_group", "unknown")))].append(row)
    ordered_keys = sorted(groups)
    for offset, key in enumerate(ordered_keys):
        group_rows = sorted(
            groups[key],
            key=lambda row: (
                float(row.get("rpm_nominal") or 0),
                float(row.get("load_nm") or 0),
                str(row.get("source_file", "")),
            ),
        )
        if group_rows:
            rotation = offset % len(group_rows)
            groups[key] = group_rows[rotation:] + group_rows[:rotation]
    selected: list[dict] = []
    while len(selected) < max_files and any(groups.values()):
        for key in ordered_keys:
            if groups[key] and len(selected) < max_files:
                selected.append(groups[key].pop(0))
    return selected


def build_metadata(
    root: Path,
    extract_to: Path | None = None,
    max_files: int | None = None,
    bearing_only: bool = True,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Build MCC5 metadata and optionally copy/extract selected CSVs."""
    rows = []
    for idx, item in enumerate(_iter_csv_candidates(root), start=1):
        source = item["member_path"] or str(item["file_path"])
        parsed = parse_mcc5_filename(source)
        if bearing_only and parsed["label_group"] not in BEARING_CLASSES:
            continue
        sample_id = f"mcc5_{idx:06d}_{Path(source).stem}"
        rows.append(
            {
                "dataset": "mcc5",
                "sample_id": sample_id,
                "zip_path": str(item["zip_path"] or ""),
                "member_path": str(item["member_path"] or ""),
                "extracted_path": "",
                "file_path": str(item["file_path"] or ""),
                **parsed,
                "source_file": source.replace("\\", "/"),
            }
        )

    rows = _balanced_limit(rows, max_files)

    if extract_to is not None and not dry_run:
        ensure_dir(extract_to)
        for row in rows:
            target = extract_to / f"{row['sample_id']}.csv"
            if row["file_path"]:
                if not target.exists():
                    shutil.copy2(row["file_path"], target)
            elif row["zip_path"] and row["member_path"]:
                if not target.exists():
                    with zipfile.ZipFile(row["zip_path"]) as zf, zf.open(row["member_path"]) as src:
                        with target.open("wb") as dst:
                            shutil.copyfileobj(src, dst)
            row["extracted_path"] = str(target)

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--extract_to", default=None)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_windows", type=int, default=None, help="Accepted for pipeline CLI compatibility; metadata preparation does not window signals.")
    parser.add_argument("--bearing_only", type=str2bool, default=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)

    logger = get_logger("prepare_mcc5")
    root = Path(args.root)
    if not root.exists():
        logger.warning("MCC5 root does not exist: %s", root)
        df = pd.DataFrame()
    else:
        df = build_metadata(
            root=root,
            extract_to=Path(args.extract_to) if args.extract_to else None,
            max_files=args.max_files,
            bearing_only=args.bearing_only,
            dry_run=args.dry_run,
        )

    out = Path(args.out)
    ensure_dir(out.parent)
    if not args.dry_run:
        df.to_csv(out, index=False)
        logger.info("Saved metadata: %s (%d rows)", out, len(df))
    else:
        logger.info("Dry run: would save %d rows to %s", len(df), out)

    if not df.empty:
        print("Class distribution:")
        print(df["label_group"].value_counts(dropna=False).to_string())
        print("Condition distribution:")
        print(df["condition_type"].value_counts(dropna=False).to_string())
    else:
        logger.warning("No MCC5 CSV metadata rows found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
