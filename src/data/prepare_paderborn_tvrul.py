"""Prepare Paderborn TV-RUL metadata for selected bearings."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.io import ensure_dir
from src.utils.config import config_default, load_yaml_config
from src.utils.logger import get_logger


def _infer_bearing(path: Path, bearings: set[str]) -> str | None:
    """Infer a TV-RUL bearing id from a path or filename."""
    text = str(path).replace("\\", "/")
    match = re.search(r"\b(B\d{2})\b", text, flags=re.IGNORECASE)
    if match:
        bearing = match.group(1).upper()
        return bearing if bearing in bearings else None
    match = re.search(r"data_(B\d{2})_M\d+", path.name, flags=re.IGNORECASE)
    if match:
        bearing = match.group(1).upper()
        return bearing if bearing in bearings else None
    return None


def _infer_cycle_index(path: Path, fallback: int) -> int:
    """Infer the measurement cycle index from a TV-RUL filename."""
    match = re.search(r"_M(\d+)", path.stem, flags=re.IGNORECASE)
    if match:
        return max(0, int(match.group(1)) - 1)
    return fallback


def _discover_files(root: Path, bearings: list[str]) -> list[tuple[str, Path]]:
    """Find TV-RUL .mat files in organized or mixed raw folders."""
    bearing_set = {b.upper() for b in bearings}
    files_by_bearing: dict[str, list[Path]] = {b: [] for b in bearing_set}
    for path in sorted(root.rglob("*.mat")):
        bearing = _infer_bearing(path, bearing_set)
        if bearing is None:
            continue
        if not re.search(r"data_B\d{2}_M\d+", path.stem, flags=re.IGNORECASE):
            continue
        files_by_bearing[bearing].append(path)
    discovered: list[tuple[str, Path]] = []
    for bearing in bearings:
        for path in sorted(files_by_bearing.get(bearing.upper(), [])):
            discovered.append((bearing.upper(), path))
    return discovered


def main(argv: list[str] | None = None) -> int:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="configs/paderborn_tvrul.yaml")
    config_args, _ = config_parser.parse_known_args(argv)
    cfg = load_yaml_config(config_args.config)

    parser = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    parser.add_argument("--root", default=config_default(cfg, "root", "data/raw/paderborn_tvrul_2024"))
    parser.add_argument("--out", default=config_default(cfg, "metadata", "data/processed/metadata/paderborn_tvrul_metadata.csv"))
    parser.add_argument("--bearings", nargs="*", default=config_default(cfg, "bearings", ["B01", "B02", "B03", "B05"]))
    parser.add_argument("--max_files", type=int, default=config_default(cfg, "max_files", None))
    parser.add_argument("--max_windows", type=int, default=config_default(cfg, "max_windows", None))
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)
    rows = []
    root = Path(args.root)
    discovered = _discover_files(root, args.bearings)
    for bearing in args.bearings:
        files = [path for b, path in discovered if b == bearing.upper()]
        if args.max_files:
            files = files[: args.max_files]
        for i, path in enumerate(files):
            rows.append(
                {
                    "dataset": "paderborn_tvrul_2024",
                    "sample_id": f"{bearing.upper()}_{i:05d}",
                    "bearing": bearing.upper(),
                    "cycle_index": _infer_cycle_index(path, i),
                    "file_path": str(path),
                    "source_file": str(path.relative_to(root)) if path.is_relative_to(root) else path.name,
                }
            )
    df = pd.DataFrame(rows)
    if not args.dry_run:
        out = Path(args.out)
        ensure_dir(out.parent)
        df.to_csv(out, index=False)
        get_logger("prepare_paderborn").info("Saved %s (%d rows)", out, len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
