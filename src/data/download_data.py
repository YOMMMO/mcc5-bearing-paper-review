"""Dataset download helper with conservative manual-instruction fallback."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.io import ensure_dir
from src.utils.logger import get_logger

DOC_PATH = Path("docs/dataset_download_plan.md")

MENDELEY_MANUAL_NOTES = {
    "gearbox_2025": """
## Gearbox 2025 Mendeley manual download note

Automatic Gearbox 2025 download is intentionally not attempted because the
dataset is hosted on Mendeley and scraping the web page is unreliable.

Manual setup:

1. Open `https://data.mendeley.com/datasets/whj3wxhw8j/1`.
2. Download the available archive or individual `Nextmon_GPS_*` text exports.
3. Place the extracted files under `data/raw/gearbox_2025/`.
4. Verify with:

```bash
python src/data/prepare_gearbox2025.py --root data/raw/gearbox_2025 --out data/processed/metadata/gearbox_2025_metadata.csv --dry_run
python src/experiments/exp08_gearbox_support.py --max_files 4 --max_samples_per_file 2048 --dry_run
```
""",
    "vat_2023": """
## VAT 2023 Mendeley manual download note

Automatic VAT 2023 download is intentionally not attempted because the three
subsets are hosted on Mendeley and should be downloaded manually.

Manual setup:

1. Open the subset pages:
   - `https://data.mendeley.com/datasets/vxkj334rzv/7`
   - `https://data.mendeley.com/datasets/x3vhp8t6hg/7`
   - `https://data.mendeley.com/datasets/j8d8pfkvj2/7`
2. Download the available vibration/current/RPM CSV files.
3. Place them under `data/raw/vat_2023/`, or keep the current fallback layout under `data/raw/vat_speed/subset*/`.
4. Verify with:

```bash
python src/data/prepare_vat2023.py --root data/raw/vat_2023 --fallback_root data/raw/vat_speed --out data/processed/metadata/vat2023_metadata.csv --dry_run
python src/experiments/exp09_vat2023_validation.py --max_files 4 --max_samples_per_signal 2048 --dry_run
```
""",
}


def _append_plan(text: str) -> None:
    ensure_dir(DOC_PATH.parent)
    with DOC_PATH.open("a", encoding="utf-8") as f:
        f.write("\n\n" + text.strip() + "\n")


def _append_plan_once(marker: str, text: str) -> None:
    """Append a plan section only if its marker is not already present."""
    ensure_dir(DOC_PATH.parent)
    current = DOC_PATH.read_text(encoding="utf-8") if DOC_PATH.exists() else ""
    if marker in current:
        return
    _append_plan(text)


def _download_file(url: str, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _zenodo_files(record: str) -> list[dict]:
    url = f"https://zenodo.org/api/records/{record}"
    data = requests.get(url, timeout=30).json()
    return data.get("files", [])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["mcc5", "lenze", "paderborn_tvrul", "gearbox_2025", "gearbox", "vat_2023", "vat"],
        required=True,
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--download_big_data", action="store_true")
    parser.add_argument("--include", nargs="*", default=None)
    parser.add_argument("--max_files", type=int, default=None, help="Limit downloaded/listed files for smoke checks.")
    parser.add_argument("--max_windows", type=int, default=None, help="Accepted for common pipeline CLI compatibility.")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)

    logger = get_logger("download_data")
    out = ensure_dir(args.out)
    dataset = {"gearbox": "gearbox_2025", "vat": "vat_2023"}.get(args.dataset, args.dataset)

    if dataset == "mcc5":
        if args.dry_run:
            logger.info("Dry run: would handle MCC5 download into %s; no files downloaded.", out)
            return 0
        if not args.download_big_data:
            _append_plan(
                """
## MCC5 command note

Automatic MCC5 download was skipped because `--download_big_data` was not set.
Use one of:

```bash
python src/data/download_data.py --dataset mcc5 --out data/raw/mcc5_thu_motor --download_big_data
huggingface-cli download Samlzy/MCC5-THU-Motor --repo-type dataset --local-dir data/raw/mcc5_thu_motor
```
"""
            )
            logger.info("MCC5 big download skipped; instructions appended.")
            return 0
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="Samlzy/MCC5-THU-Motor",
            repo_type="dataset",
            local_dir=out,
            local_dir_use_symlinks=False,
        )
        return 0

    if dataset == "lenze":
        files = _zenodo_files("14762423")
        if args.max_files:
            files = files[: args.max_files]
        if args.dry_run:
            logger.info("Dry run: would handle %d Lenze files into %s.", len(files), out)
            for item in files:
                logger.info("  %s", item.get("key"))
            return 0
        if not args.download_big_data:
            logger.info("Lenze files available from Zenodo:")
            for item in files:
                logger.info("  %s", item.get("key"))
            _append_plan("## Lenze command note\n\nRun with `--download_big_data` to download files from Zenodo record `14762423`.")
            return 0
        for item in files:
            key = item.get("key")
            link = item.get("links", {}).get("self")
            if key and link:
                logger.info("Downloading %s", key)
                _download_file(link, out / key)
        return 0

    if dataset in {"gearbox_2025", "vat_2023"}:
        marker = f"## {'Gearbox 2025' if dataset == 'gearbox_2025' else 'VAT 2023'} Mendeley manual download note"
        _append_plan_once(marker, MENDELEY_MANUAL_NOTES[dataset])
        logger.info("%s is hosted on Mendeley; manual instructions are recorded in %s.", dataset, DOC_PATH)
        if args.download_big_data:
            logger.info("Ignoring --download_big_data for %s because Mendeley scraping is intentionally disabled.", dataset)
        if args.dry_run:
            logger.info("Dry run: would not download %s; manual setup remains required.", dataset)
        return 0

    files = _zenodo_files("10868257")
    include = set(args.include or ["B01.zip", "B02.zip", "B03.zip", "B05.zip"])
    files = [item for item in files if item.get("key") in include]
    if args.max_files:
        files = files[: args.max_files]
    if args.dry_run:
        logger.info("Dry run: would handle %d Paderborn TV-RUL files into %s.", len(files), out)
        for item in files:
            logger.info("  %s", item.get("key"))
        return 0
    if not args.download_big_data:
        logger.info("Paderborn TV-RUL files available from Zenodo:")
        for item in files:
            logger.info("  %s", item.get("key"))
        _append_plan("## Paderborn TV-RUL command note\n\nRun with `--download_big_data --include B01.zip B02.zip B03.zip B05.zip`.")
        return 0
    for item in files:
        key = item.get("key")
        link = item.get("links", {}).get("self")
        if key and link:
            logger.info("Downloading %s", key)
            _download_file(link, out / key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
