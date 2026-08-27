"""Prepare VAT 2023 metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.parsing import parse_vat_filename
from src.utils.io import ensure_dir
from src.utils.logger import get_logger


def _csv_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.csv")) if root.exists() else []


def _select_root(root: Path, fallback_root: Path) -> Path:
    if _csv_files(root):
        return root
    if _csv_files(fallback_root):
        return fallback_root
    return root if root.exists() else fallback_root


def _measurement_key(path: Path, root: Path) -> tuple[str, str]:
    parsed = parse_vat_filename(path.name)
    stem = path.stem.lower()
    signal = parsed["signal_type"]
    if signal != "unknown" and stem.startswith(f"{signal}_"):
        measurement = stem[len(signal) + 1 :]
    else:
        measurement = stem
    parent = str(path.parent.relative_to(root)) if path.parent != root else "."
    return parent, measurement


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/vat_2023")
    parser.add_argument("--fallback_root", default="data/raw/vat_speed")
    parser.add_argument("--out", default="data/processed/metadata/vat2023_metadata.csv")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)
    root = _select_root(Path(args.root), Path(args.fallback_root))
    groups: dict[tuple[str, str], dict] = {}
    files = _csv_files(root)
    for i, path in enumerate(files):
        parsed = parse_vat_filename(path.name)
        parent, measurement = _measurement_key(path, root)
        key = (parent, measurement)
        row = groups.setdefault(
            key,
            {
                "dataset": "vat2023",
                "sample_id": "",
                "subset": parent,
                "measurement_id": measurement,
                "label_group": parsed["label_group"],
                "condition_type": "constant_speed" if "constant" in measurement else "varying_speed",
                "vibration_path": "",
                "current_path": "",
                "rpm_path": "",
                "file_path": "",
                "available_signal_types": "",
                "source_file": f"{parent}/{measurement}",
            },
        )
        signal = parsed["signal_type"]
        if signal in {"vibration", "current", "rpm"}:
            row[f"{signal}_path"] = str(path)
        if not row["file_path"] or signal == "vibration":
            row["file_path"] = str(path)
        if row["label_group"] == "unknown" and parsed["label_group"] != "unknown":
            row["label_group"] = parsed["label_group"]
    rows = []
    for i, row in enumerate(groups.values()):
        if args.max_files and i >= args.max_files:
            break
        available = [name for name in ["vibration", "current", "rpm"] if row.get(f"{name}_path")]
        row["sample_id"] = f"vat_{i:05d}"
        row["available_signal_types"] = ",".join(available)
        rows.append(row)
    df = pd.DataFrame(rows)
    if not args.dry_run:
        out = Path(args.out)
        ensure_dir(out.parent)
        df.to_csv(out, index=False)
        get_logger("prepare_vat").info("Saved %s (%d rows)", out, len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
