"""Prepare metadata for Gearbox variable conditions 2025."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.parsing import parse_gearbox_filename
from src.utils.io import ensure_dir
from src.utils.logger import get_logger


def _file_sort_key(path: Path) -> tuple[int, str]:
    parsed = parse_gearbox_filename(str(path))
    test_id = parsed.get("test_id")
    return (int(test_id) if test_id is not None else 10_000, path.name)


def _parse_header(path: Path) -> dict:
    header: dict[str, object] = {
        "title": "",
        "frequency_limit_hz": None,
        "total_data_rows": None,
        "sampling_rate_hz": None,
        "channel_names": "",
        "volts_per_unit": "",
    }
    try:
        lines = []
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                stripped = line.rstrip("\n\r")
                lines.append(stripped)
                if stripped.startswith("Time (seconds) and Data Channels"):
                    break
                if len(lines) > 80:
                    break
        for line in lines:
            parts = line.split("\t")
            if line.startswith("Title:"):
                header["title"] = parts[-1] if len(parts) > 1 else line.replace("Title:", "").strip()
            elif line.startswith("Frequency Limit") and len(parts) > 1:
                header["frequency_limit_hz"] = float(parts[1])
            elif line.startswith("Total Data Rows") and len(parts) > 1:
                header["total_data_rows"] = int(float(parts[1]))
            elif line.startswith("Legend"):
                header["channel_names"] = "|".join(parts[1:])
            elif line.startswith("Volts/Unit"):
                header["volts_per_unit"] = "|".join(parts[1:])
        if header["frequency_limit_hz"]:
            header["sampling_rate_hz"] = float(header["frequency_limit_hz"]) * 2.56
    except Exception:
        pass
    return header


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/gearbox_2025")
    parser.add_argument("--out", default="data/processed/metadata/gearbox_2025_metadata.csv")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)
    rows = []
    root = Path(args.root)
    files = sorted((p for p in root.rglob("*") if p.is_file()), key=_file_sort_key) if root.exists() else []
    for i, path in enumerate(files):
        if args.max_files and i >= args.max_files:
            break
        parsed = parse_gearbox_filename(str(path))
        rows.append(
            {
                "dataset": "gearbox_2025",
                "sample_id": f"gearbox_{i:05d}",
                "file_path": str(path),
                **parsed,
                **_parse_header(path),
            }
        )
    df = pd.DataFrame(rows)
    if not args.dry_run:
        out = Path(args.out)
        ensure_dir(out.parent)
        df.to_csv(out, index=False)
        get_logger("prepare_gearbox").info("Saved %s (%d rows)", out, len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
