"""Prepare Lenze-MB metadata when files are available."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.parsing import parse_lenze_metadata_row
from src.utils.io import ensure_dir, safe_read_excel
from src.utils.logger import get_logger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/lenze_mb")
    parser.add_argument("--out", default="data/processed/metadata/lenze_metadata.csv")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)
    logger = get_logger("prepare_lenze")
    root = Path(args.root)
    rows = []
    mat_files = {path.stem: path for path in sorted(root.rglob("*.mat"))} if root.exists() else {}
    meta = root / "Meta_Data.xlsx"
    if meta.exists():
        try:
            df = safe_read_excel(meta)
            if args.max_files:
                df = df.head(args.max_files)
            for i, row in df.iterrows():
                parsed = parse_lenze_metadata_row(row)
                record = row.to_dict()
                sample_key = str(record.get("ID", f"lenze_{i:05d}"))
                rows.append(
                    {
                        "dataset": "lenze_mb",
                        "sample_id": f"lenze_{i:05d}",
                        "source_id": sample_key,
                        "file_path": str(mat_files.get(sample_key, "")),
                        "rpm_nominal": record.get("Motor Speed (RPM)"),
                        "belt_tension_nm": record.get("Belt_Tension (Nm)"),
                        "counter_momentum_nm": record.get("Counter_Momentum (Nm)"),
                        "condition_type": f"rpm_{record.get('Motor Speed (RPM)')}_belt_{record.get('Belt_Tension (Nm)')}_counter_{record.get('Counter_Momentum (Nm)')}",
                        **parsed,
                        **record,
                    }
                )
        except Exception as exc:
            logger.warning("Could not read %s (%s); falling back to .mat scan.", meta, exc)
    else:
        logger.warning("Lenze metadata workbook not found: %s", meta)
    if not rows:
        for i, path in enumerate(sorted(root.rglob("*.mat")) if root.exists() else []):
            if args.max_files and i >= args.max_files:
                break
            rows.append(
                {
                    "dataset": "lenze_mb",
                    "sample_id": f"lenze_{i:05d}",
                    "source_id": path.stem,
                    "file_path": str(path),
                    "label_group": "unknown",
                    "label_raw": path.stem,
                    "fault_family": "unknown",
                    "severity": None,
                    "condition_type": "unknown",
                }
            )
    out_df = pd.DataFrame(rows)
    if not args.dry_run:
        out = Path(args.out)
        ensure_dir(out.parent)
        out_df.to_csv(out, index=False)
        logger.info("Saved %s (%d rows)", out, len(out_df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
