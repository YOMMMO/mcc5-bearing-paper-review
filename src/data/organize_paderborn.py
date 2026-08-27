"""Extract and organize mixed Paderborn raw files.

This helper is intentionally conservative: it never overwrites existing files
unless ``--overwrite`` is given, and ``--dry_run`` prints planned actions
without touching the filesystem.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.io import ensure_dir
from src.utils.logger import get_logger

ARCHIVE_SUFFIXES = (
    ".zip",
    ".7z",
    ".rar",
    ".tar",
    ".tgz",
    ".tar.gz",
    ".tar.bz2",
    ".tar.xz",
    ".gz",
    ".bz2",
    ".xz",
)

TVRUL_BEARINGS = {"B01", "B02", "B03", "B05"}
CLASSIC_RE = re.compile(r"(?<![A-Z0-9])((?:K\d{3})|(?:KA\d{2})|(?:KB\d{2})|(?:KI\d{2}))(?![A-Z0-9])", re.I)
TVRUL_DATA_RE = re.compile(r"data_(B\d{2})_M\d+\.mat$", re.I)
TVRUL_SIDE_RE = re.compile(r"(B\d{2})_(?:operatingConditions|meanTemperatures|log)\.(?:csv|pdf)$", re.I)


@dataclass(frozen=True)
class PlanItem:
    action: str
    source: Path
    target: Path


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _archive_stem(path: Path) -> str:
    name = path.name
    lower = name.lower()
    for suffix in (".tar.gz", ".tar.bz2", ".tar.xz"):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _is_archive(path: Path) -> bool:
    lower = path.name.lower()
    return any(lower.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def _safe_target(base: Path, member_name: str) -> Path:
    target = (base / member_name).resolve()
    base_resolved = base.resolve()
    try:
        target.relative_to(base_resolved)
    except ValueError:
        raise ValueError(f"Unsafe archive member path: {member_name}")
    return target


def _extract_zip(path: Path, out_dir: Path, dry_run: bool) -> int:
    count = 0
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            target = _safe_target(out_dir, info.filename)
            if info.is_dir():
                if not dry_run:
                    ensure_dir(target)
                continue
            count += 1
            if dry_run:
                continue
            ensure_dir(target.parent)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    return count


def _extract_tar(path: Path, out_dir: Path, dry_run: bool) -> int:
    count = 0
    with tarfile.open(path) as tf:
        for member in tf.getmembers():
            _safe_target(out_dir, member.name)
            if member.isfile():
                count += 1
        if not dry_run:
            tf.extractall(out_dir)
    return count


def _extract_external(path: Path, out_dir: Path, dry_run: bool) -> int:
    tool = shutil.which("7z") or shutil.which("7za")
    if tool is None:
        raise RuntimeError(f"Need 7z/7za to extract {path.name}")
    if dry_run:
        return 0
    ensure_dir(out_dir)
    subprocess.run([tool, "x", str(path), f"-o{out_dir}", "-y"], check=True)
    return 0


def extract_archive(path: Path, out_dir: Path, dry_run: bool) -> int:
    lower = path.name.lower()
    if lower.endswith(".zip"):
        return _extract_zip(path, out_dir, dry_run)
    if lower.endswith((".tar", ".tgz", ".tar.gz", ".tar.bz2", ".tar.xz")):
        return _extract_tar(path, out_dir, dry_run)
    if lower.endswith((".7z", ".rar")):
        return _extract_external(path, out_dir, dry_run)
    raise RuntimeError(f"Unsupported archive format: {path}")


def classify_target(path: Path, tvrul_root: Path, classic_root: Path) -> Path | None:
    name = path.name
    match = TVRUL_DATA_RE.search(name)
    if match:
        bearing = match.group(1).upper()
        if bearing in TVRUL_BEARINGS:
            return tvrul_root / bearing / "vibrationData" / name

    match = TVRUL_SIDE_RE.search(name)
    if match:
        bearing = match.group(1).upper()
        if bearing in TVRUL_BEARINGS:
            return tvrul_root / bearing / name

    match = CLASSIC_RE.search(path.stem.upper())
    if match:
        bearing = match.group(1).upper()
        return classic_root / bearing / name
    return None


def build_archive_plan(source: Path, recursive: bool) -> list[PlanItem]:
    pattern = "**/*" if recursive else "*"
    items = []
    for path in sorted(source.glob(pattern)):
        if path.is_file() and _is_archive(path):
            items.append(PlanItem("extract", path, path.parent / _archive_stem(path)))
    return items


def build_organize_plan(source: Path, tvrul_root: Path, classic_root: Path, recursive: bool) -> list[PlanItem]:
    pattern = "**/*" if recursive else "*"
    items = []
    for path in sorted(source.glob(pattern)):
        if not path.is_file() or _is_archive(path):
            continue
        if _is_within(path, tvrul_root) or _is_within(path, classic_root):
            continue
        target = classify_target(path, tvrul_root, classic_root)
        if target is not None and path.resolve() != target.resolve():
            items.append(PlanItem("organize", path, target))
    return items


def _copy_or_move(path: Path, target: Path, move: bool, overwrite: bool, dry_run: bool) -> str:
    if target.exists():
        same_size = path.stat().st_size == target.stat().st_size
        if same_size and not overwrite:
            return "skip_exists_same_size"
        if not overwrite:
            return "skip_exists_different_size"
    if dry_run:
        return "planned_move" if move else "planned_copy"
    ensure_dir(target.parent)
    if move:
        shutil.move(str(path), str(target))
    else:
        shutil.copy2(path, target)
    return "moved" if move else "copied"


def _print_preview(items: list[PlanItem], limit: int) -> None:
    for item in items[:limit]:
        print(f"{item.action}: {item.source} -> {item.target}")
    if len(items) > limit:
        print(f"... {len(items) - limit} more planned actions")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/raw/paderborn", help="Mixed folder containing archives or raw files.")
    parser.add_argument("--tvrul_root", default="data/raw/paderborn_tvrul_2024")
    parser.add_argument("--classic_root", default="data/raw/paderborn_classic")
    parser.add_argument("--recursive", action="store_true", help="Scan source recursively.")
    parser.add_argument("--skip_extract", action="store_true")
    parser.add_argument("--skip_organize", action="store_true")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying during organization.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing targets.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--preview", type=int, default=30, help="Number of planned actions to print.")
    args = parser.parse_args(argv)

    logger = get_logger("organize_paderborn")
    source = Path(args.source)
    tvrul_root = Path(args.tvrul_root)
    classic_root = Path(args.classic_root)
    if not source.exists():
        logger.warning("Source folder does not exist: %s", source)
        return 0

    archive_plan = [] if args.skip_extract else build_archive_plan(source, args.recursive)
    organize_plan = [] if args.skip_organize else build_organize_plan(source, tvrul_root, classic_root, args.recursive)

    print(f"archives_to_extract={len(archive_plan)}")
    print(f"files_to_organize={len(organize_plan)}")
    _print_preview(archive_plan + organize_plan, args.preview)

    extracted_members = 0
    for item in archive_plan:
        logger.info("Extracting %s -> %s", item.source, item.target)
        if not args.dry_run:
            ensure_dir(item.target)
        extracted_members += extract_archive(item.source, item.target, args.dry_run)

    status_counts: dict[str, int] = {}
    for item in organize_plan:
        status = _copy_or_move(item.source, item.target, args.move, args.overwrite, args.dry_run)
        status_counts[status] = status_counts.get(status, 0) + 1

    print(f"extracted_members={extracted_members}")
    for status, count in sorted(status_counts.items()):
        print(f"{status}={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
