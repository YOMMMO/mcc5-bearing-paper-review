"""Inspect raw dataset structure and produce summary tables."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.parsing import parse_gearbox_filename, parse_mcc5_filename, parse_vat_filename
from src.utils.io import ensure_dir
from src.utils.tables import write_table


def _parse(dataset: str, path: Path) -> dict:
    if dataset == "mcc5":
        parsed = parse_mcc5_filename(path.name)
        return {
            "inferred_label": parsed["label_group"],
            "inferred_condition": parsed["condition_type"],
            "inferred_rpm": parsed["rpm_nominal"],
            "inferred_load_nm": parsed["load_nm"],
        }
    if dataset == "vat":
        parsed = parse_vat_filename(path.name)
        return {"inferred_label": parsed["label_group"], "inferred_condition": parsed.get("signal_type")}
    if dataset == "gearbox":
        parsed = parse_gearbox_filename(str(path))
        return {"inferred_label": parsed["label_group"], "inferred_condition": parsed["condition_type"]}
    return {"inferred_label": "unknown", "inferred_condition": "unknown"}


def inspect(
    root: Path,
    dataset: str,
    out: Path,
    max_zip_members: int = 200,
    max_files: int | None = None,
) -> pd.DataFrame:
    """Scan files recursively and write CSV plus markdown structure notes."""
    rows = []
    scanned_files = 0
    for path in sorted(root.rglob("*")) if root.exists() else []:
        if not path.is_file():
            continue
        if max_files is not None and scanned_files >= max_files:
            break
        scanned_files += 1
        rel = path.relative_to(root)
        base = {
            "relative_path": str(rel).replace("\\", "/"),
            "extension": path.suffix.lower(),
            "size_mb": round(path.stat().st_size / (1024 * 1024), 4),
            "parent_folder": path.parent.name,
            "inferred_dataset": dataset,
            "rows": None,
            "columns": None,
            "column_names": "",
            "zip_member": "",
        }
        base.update(_parse(dataset, path))
        if path.suffix.lower() == ".csv" and path.stat().st_size < 50 * 1024 * 1024:
            try:
                df = pd.read_csv(path, nrows=20, header=None)
                base["rows"] = "sampled_20"
                base["columns"] = df.shape[1]
                base["column_names"] = ",".join(str(c) for c in df.columns)
            except Exception:
                pass
        rows.append(base)
        if path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(path) as zf:
                    for member in zf.namelist()[:max_zip_members]:
                        if member.endswith("/"):
                            continue
                        member_row = base.copy()
                        member_row["relative_path"] = str(rel).replace("\\", "/")
                        member_row["zip_member"] = member
                        member_row["extension"] = Path(member).suffix.lower()
                        member_row.update(_parse(dataset, Path(member)))
                        rows.append(member_row)
            except zipfile.BadZipFile:
                pass
    df = pd.DataFrame(rows)
    actual = write_table(df, out)
    ensure_dir("docs")
    md = Path(f"docs/dataset_structure_{dataset}.md")
    lines = [f"# Dataset Structure: {dataset}", "", f"Root: `{root}`", "", f"Rows: {len(df)}", ""]
    if not df.empty:
        lines.append("## Extensions")
        lines.append("")
        lines.extend(
            f"- `{idx}`: {count}"
            for idx, count in df["extension"].value_counts(dropna=False).items()
        )
        lines.append("")
        lines.append("## Label Counts")
        lines.append("")
        lines.extend(
            f"- `{idx}`: {count}"
            for idx, count in df["inferred_label"].value_counts(dropna=False).items()
        )
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {actual} and {md}")
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(f"Would inspect {args.root}")
        return 0
    df = inspect(Path(args.root), args.dataset, Path(args.out), max_files=args.max_files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
