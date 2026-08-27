"""File I/O helpers with conservative fallbacks."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create a directory and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: object, path: str | Path) -> None:
    """Write JSON with UTF-8 encoding."""
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> object:
    """Read a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def list_files_recursive(
    root: str | Path, extensions: Iterable[str] | None = None
) -> list[Path]:
    """List files recursively, optionally filtered by lowercase suffix."""
    root_path = Path(root)
    if not root_path.exists():
        return []
    suffixes = None
    if extensions is not None:
        suffixes = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    files = []
    for path in root_path.rglob("*"):
        if path.is_file() and (suffixes is None or path.suffix.lower() in suffixes):
            files.append(path)
    return sorted(files)


def safe_read_csv(path_or_buffer, **kwargs) -> pd.DataFrame:
    """Read CSV with a small set of encoding fallbacks."""
    kwargs.setdefault("low_memory", False)
    encodings = [kwargs.pop("encoding", None), "utf-8", "utf-8-sig", "gbk", "latin1"]
    errors = []
    for enc in encodings:
        try:
            if enc is None:
                return pd.read_csv(path_or_buffer, **kwargs)
            return pd.read_csv(path_or_buffer, encoding=enc, **kwargs)
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError(f"Could not read CSV after fallbacks: {errors[-1]}")


def safe_read_excel(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read an Excel file using pandas."""
    return pd.read_excel(path, **kwargs)


def safe_extract_zip_member(
    zip_path: str | Path, member: str, out_dir: str | Path
) -> Path:
    """Extract a single zip member while preventing path traversal."""
    zip_path = Path(zip_path)
    out_dir = ensure_dir(out_dir).resolve()
    with zipfile.ZipFile(zip_path) as zf:
        info = zf.getinfo(member)
        target = (out_dir / Path(info.filename).name).resolve()
        if not str(target).lower().startswith(str(out_dir).lower()):
            raise ValueError(f"Unsafe zip member path: {member}")
        with zf.open(info) as src, target.open("wb") as dst:
            dst.write(src.read())
    return target


def compute_file_md5(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute an MD5 hash for a file."""
    h = hashlib.md5()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
