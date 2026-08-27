"""Run the MCC5 formal evidence pipeline with an isolated run directory.

This runner intentionally keeps formal artifacts separate from the bounded
pilot outputs. It does not pass pilot-limiting flags to data, feature, or model
commands. Use ``--stop_after`` to advance the formal run in auditable stages.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except Exception:  # pragma: no cover - depends on local environment
    plt = None
    MATPLOTLIB_AVAILABLE = False

from src.utils.config import config_default, load_yaml_config


STEPS = ["preflight", "metadata", "windows", "features", "splits", "ml", "dl", "fusion", "ablation"]
MAIN_LABELS = {"healthy", "bearing_inner", "bearing_outer", "bearing_ball"}
ROOT = Path(__file__).resolve().parent


def _now_run_id() -> str:
    return datetime.now().strftime("formal_%Y%m%d_%H%M%S")


def _tail(text: str, limit: int = 4000) -> str:
    text = text or ""
    return text[-limit:]


def _ensure_run_dirs(run_id: str) -> dict[str, Path]:
    run_root = ROOT / "results" / "formal_runs" / run_id
    dirs = {
        "root": run_root,
        "tables": run_root / "tables",
        "figures": run_root / "figures",
        "logs": run_root / "logs",
        "checkpoints": run_root / "checkpoints",
        "manuscript": run_root / "manuscript",
        "docs": ROOT / "docs" / "formal_runs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "formal_runs" / "latest_run_id.txt").write_text(run_id + "\n", encoding="utf-8")
    (dirs["docs"] / "latest_run_id.txt").write_text(run_id + "\n", encoding="utf-8")
    return dirs


def _copy_if_exists(src: str | Path, dst: str | Path) -> None:
    src_path = ROOT / src
    dst_path = Path(dst)
    if src_path.exists():
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)


def _archive_pilot(run_id: str, dirs: dict[str, Path]) -> None:
    log_dir = dirs["logs"]
    _copy_if_exists("docs/preliminary_run_summary.md", log_dir / "preliminary_run_summary_before_formal.md")
    _copy_if_exists("docs/preliminary_completion_check.md", log_dir / "preliminary_completion_check_before_formal.md")
    _copy_if_exists("results/tables/artifact_audit.csv", log_dir / "pilot_artifact_audit_before_formal.csv")
    _copy_if_exists("results/logs/full_mcc5_command_log.json", log_dir / "pilot_full_mcc5_command_log_before_formal.json")
    (log_dir / "archive_note.md").write_text(
        "\n".join(
            [
                f"# Pilot Archive For {run_id}",
                "",
                "Bounded pilot artifacts were copied before formal processing.",
                "Original pilot files were not deleted.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _table_shape(path: str | Path) -> dict[str, Any]:
    p = ROOT / path
    if not p.exists() and not (p.suffix == ".parquet" and p.with_suffix(".csv").exists()):
        return {"path": str(path), "missing": True}
    actual = p if p.exists() else p.with_suffix(".csv")
    try:
        df = pd.read_csv(actual) if actual.suffix.lower() == ".csv" else pd.read_parquet(actual)
        return {"path": str(path), "shape": [int(df.shape[0]), int(df.shape[1])]}
    except Exception as exc:
        return {"path": str(path), "error": repr(exc)}


def _raw_mcc5_stats() -> dict[str, Any]:
    root = ROOT / "data" / "raw" / "mcc5_thu_motor"
    files = list(root.rglob("*")) if root.exists() else []
    file_paths = [p for p in files if p.is_file()]
    csv_files = [p for p in file_paths if p.suffix.lower() == ".csv"]
    zip_files = [p for p in file_paths if p.suffix.lower() == ".zip"]
    total_size = sum(p.stat().st_size for p in file_paths)
    return {
        "root": str(root.relative_to(ROOT)),
        "exists": root.exists(),
        "files": len(file_paths),
        "csv_files": len(csv_files),
        "zip_files": len(zip_files),
        "size_gb": round(total_size / 1024**3, 3),
        "largest_csv": str(max(csv_files, key=lambda p: p.stat().st_size).relative_to(ROOT)) if csv_files else "",
    }


def _torch_info() -> dict[str, Any]:
    info: dict[str, Any] = {"torch_available": False, "cuda_available": False, "gpu_name": ""}
    try:
        import torch

        info["torch_available"] = True
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    except Exception as exc:
        info["torch_error"] = repr(exc)
    return info


def _preflight(run_id: str, dirs: dict[str, Path], feature_config: Path, fusion_config: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(str(ROOT.anchor or ROOT))
    raw = _raw_mcc5_stats()
    torch = _torch_info()
    run_full_text = (ROOT / "run_full_mcc5_pipeline.py").read_text(encoding="utf-8")
    run_commands_text = (ROOT / "docs" / "run_commands.md").read_text(encoding="utf-8")
    formal_section = run_commands_text.split("## Formal MCC5 Run", 1)[-1]
    pilot_flags = ["--max_files", "--max_windows_per_file", "--max_windows", "--max_train_samples"]
    has_formal_no_limit_command = (
        "run_full_mcc5_pipeline.py" in formal_section
        and not any(flag in formal_section.split("## External", 1)[0] for flag in pilot_flags)
    )
    recommendation = "continue"
    limitations: list[str] = []
    if not raw["exists"] or (raw["csv_files"] == 0 and raw["zip_files"] == 0):
        recommendation = "stop"
        limitations.append("MCC5 raw CSV/ZIP files are missing or unreadable.")
    if disk.free < 80 * 1024**3:
        limitations.append("Free disk space is below 80 GB; full sequence DL should be skipped.")
    if not torch.get("cuda_available"):
        limitations.append("CUDA GPU is unavailable; DL/fusion should be skipped unless CPU training is explicitly allowed.")

    processed_shapes = {
        "metadata": _table_shape("data/processed/metadata/mcc5_metadata.csv"),
        "windows": _table_shape("data/processed/windows/mcc5_windows.parquet"),
        "features": _table_shape("data/processed/features/mcc5_features.parquet"),
    }
    payload = {
        "run_id": run_id,
        "created_local": datetime.now().isoformat(timespec="seconds"),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version,
        "torch": torch,
        "free_disk_gb": round(disk.free / 1024**3, 3),
        "total_disk_gb": round(disk.total / 1024**3, 3),
        "raw_mcc5": raw,
        "processed_shapes": processed_shapes,
        "feature_config": str(feature_config),
        "fusion_config": str(fusion_config),
        "pilot_flags_present_in_run_full_mcc5_pipeline": [flag for flag in pilot_flags if flag in run_full_text],
        "docs_run_commands_has_formal_no_limit_command": has_formal_no_limit_command,
        "recommendation": recommendation,
        "limitations": limitations,
    }
    lines = [
        f"# Formal Preflight Audit: {run_id}",
        "",
        f"- Python: `{sys.version}`",
        f"- Torch available: `{torch.get('torch_available')}`",
        f"- CUDA available: `{torch.get('cuda_available')}`",
        f"- GPU name: `{torch.get('gpu_name') or 'none'}`",
        f"- Free disk on {ROOT.anchor or ROOT}: `{payload['free_disk_gb']}` GB",
        f"- Raw MCC5 files: `{raw['files']}` total, `{raw['csv_files']}` CSV, `{raw['zip_files']}` ZIP",
        f"- Raw MCC5 size: `{raw['size_gb']}` GB",
        f"- Existing processed metadata shape: `{processed_shapes['metadata']}`",
        f"- Existing processed windows shape: `{processed_shapes['windows']}`",
        f"- Existing processed features shape: `{processed_shapes['features']}`",
        f"- Pilot flags present in bounded runner: `{payload['pilot_flags_present_in_run_full_mcc5_pipeline']}`",
        f"- Formal no-limit command present in docs/run_commands.md: `{has_formal_no_limit_command}`",
        f"- Recommendation: `{recommendation}`",
        "",
        "## Limitations",
        "",
    ]
    lines.extend([f"- {item}" for item in limitations] or ["- None."])
    lines.extend(["", "## JSON", "", "```json", json.dumps(payload, indent=2, ensure_ascii=False), "```", ""])
    (dirs["docs"] / f"{run_id}_preflight_audit.md").write_text("\n".join(lines), encoding="utf-8")
    (dirs["logs"] / "preflight_audit.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _run_command(cmd: list[str], history: list[dict[str, Any]]) -> int:
    print("\n$ " + " ".join(cmd), flush=True)
    started = datetime.now(timezone.utc)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, check=False)
    finished = datetime.now(timezone.utc)
    if proc.stdout:
        print(_tail(proc.stdout), flush=True)
    if proc.stderr:
        print(_tail(proc.stderr), file=sys.stderr, flush=True)
    history.append(
        {
            "command": cmd,
            "command_string": " ".join(cmd),
            "returncode": int(proc.returncode),
            "started_utc": started.isoformat(timespec="seconds"),
            "finished_utc": finished.isoformat(timespec="seconds"),
            "duration_sec": round(time.time() - t0, 3),
            "stdout_tail": _tail(proc.stdout),
            "stderr_tail": _tail(proc.stderr),
        }
    )
    return int(proc.returncode)


def _write_command_log(run_id: str, dirs: dict[str, Path], history: list[dict[str, Any]], status: str, notes: list[str]) -> Path:
    path = dirs["logs"] / "formal_mcc5_command_log.json"
    payload = {
        "pipeline": "formal_mcc5_pipeline",
        "run_id": run_id,
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commands": history,
        "notes": notes,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_failure_report(dirs: dict[str, Path], phase: str, command: list[str] | None, history: list[dict[str, Any]], suggestion: str) -> None:
    failed = history[-1] if history else {}
    lines = [
        "# Formal Failure Report",
        "",
        f"- Phase: `{phase}`",
        f"- Failed command: `{' '.join(command or [])}`",
        f"- Return code: `{failed.get('returncode', 'n/a')}`",
        "",
        "## Stdout Tail",
        "",
        "```text",
        str(failed.get("stdout_tail", "")),
        "```",
        "",
        "## Stderr Tail",
        "",
        "```text",
        str(failed.get("stderr_tail", "")),
        "```",
        "",
        "## Suggested Fix",
        "",
        suggestion,
        "",
    ]
    (dirs["logs"] / "formal_failure_report.md").write_text("\n".join(lines), encoding="utf-8")


def _read_table(path: str | Path) -> pd.DataFrame:
    p = ROOT / path
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _save_bar(counts: pd.Series, out: Path, title: str, ylabel: str = "Count") -> None:
    if not MATPLOTLIB_AVAILABLE:
        out.with_suffix(out.suffix + ".txt").write_text("Plot skipped: matplotlib unavailable.\n", encoding="utf-8")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    counts.plot(kind="bar")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    if out.suffix.lower() == ".png":
        plt.savefig(out.with_suffix(".pdf"))
    plt.close()


def _metadata_reports(run_id: str, dirs: dict[str, Path]) -> None:
    all_path = ROOT / "data" / "processed" / "metadata" / "mcc5_metadata_all_labels_formal.csv"
    main_path = ROOT / "data" / "processed" / "metadata" / "mcc5_metadata_formal.csv"
    all_df = pd.read_csv(all_path) if all_path.exists() else pd.DataFrame()
    main_df = pd.read_csv(main_path) if main_path.exists() else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    if not main_df.empty:
        for name, series in {
            "label_group": main_df["label_group"],
            "condition_type": main_df["condition_type"],
            "rpm_nominal": main_df["rpm_nominal"],
            "load_nm": main_df["load_nm"],
        }.items():
            for value, count in series.value_counts(dropna=False).sort_index().items():
                rows.append({"distribution": name, "value": value, "count": int(count)})
    dist = pd.DataFrame(rows)
    dist.to_csv(dirs["tables"] / "mcc5_formal_metadata_distribution.csv", index=False)
    shutil.copy2(dirs["tables"] / "mcc5_formal_metadata_distribution.csv", ROOT / "results" / "tables" / "mcc5_formal_metadata_distribution.csv")
    if not main_df.empty:
        _save_bar(main_df["label_group"].value_counts(), dirs["figures"] / "mcc5_formal_label_distribution.png", "Formal MCC5 Label Distribution")
        _save_bar(main_df["condition_type"].value_counts(), dirs["figures"] / "mcc5_formal_condition_distribution.png", "Formal MCC5 Condition Distribution")
    excluded = pd.DataFrame()
    if not all_df.empty and not main_df.empty:
        main_sources = set(main_df["source_file"].astype(str))
        excluded = all_df[~all_df["source_file"].astype(str).isin(main_sources)].copy()
        excluded["exclusion_reason"] = "label_not_in_main_bearing_classes"
        excluded.to_csv(dirs["tables"] / "mcc5_formal_excluded_labels.csv", index=False)
    lines = [
        f"# Formal MCC5 Dataset Audit: {run_id}",
        "",
        f"- All-label source files: `{len(all_df)}`",
        f"- Main bearing-only source files: `{len(main_df)}`",
        f"- Main labels: `{sorted(main_df['label_group'].dropna().unique().tolist()) if not main_df.empty else []}`",
        "",
        "## Per-Class Counts",
        "",
        "```text",
        main_df["label_group"].value_counts(dropna=False).to_string() if not main_df.empty else "No main metadata rows.",
        "```",
        "",
        "## Condition Counts",
        "",
        "```text",
        main_df["condition_type"].value_counts(dropna=False).to_string() if not main_df.empty else "No main metadata rows.",
        "```",
        "",
        "## Excluded Labels",
        "",
        "```text",
        excluded["label_group"].value_counts(dropna=False).to_string() if not excluded.empty else "No excluded rows.",
        "```",
        "",
    ]
    (dirs["docs"] / f"{run_id}_dataset_audit.md").write_text("\n".join(lines), encoding="utf-8")


def _window_reports(dirs: dict[str, Path]) -> None:
    path = ROOT / "data" / "processed" / "windows" / "mcc5_windows_formal.parquet"
    windows = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    rows = []
    if not windows.empty:
        rows.append({"metric": "rows", "value": len(windows)})
        rows.append({"metric": "unique_source_files", "value": windows["source_file"].nunique()})
        rows.append({"metric": "unique_window_ids", "value": windows["window_id"].nunique()})
        for value, count in windows["label_group"].value_counts(dropna=False).items():
            rows.append({"metric": "label_count", "value": value, "count": int(count)})
        for value, count in windows["condition_type"].value_counts(dropna=False).items():
            rows.append({"metric": "condition_count", "value": value, "count": int(count)})
    pd.DataFrame(rows).to_csv(dirs["tables"] / "mcc5_formal_window_summary.csv", index=False)
    shutil.copy2(dirs["tables"] / "mcc5_formal_window_summary.csv", ROOT / "results" / "tables" / "mcc5_formal_window_summary.csv")
    if not windows.empty:
        _save_bar(windows["label_group"].value_counts(), dirs["figures"] / "mcc5_formal_windows_per_class.png", "Formal Windows Per Class")
        _save_bar(windows["condition_type"].value_counts(), dirs["figures"] / "mcc5_formal_windows_per_condition.png", "Formal Windows Per Condition")


def _feature_reports(dirs: dict[str, Path]) -> None:
    path = ROOT / "data" / "processed" / "features" / "mcc5_features_formal.parquet"
    features = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    if features.empty:
        pd.DataFrame().to_csv(dirs["tables"] / "mcc5_formal_feature_missingness.csv", index=False)
        pd.DataFrame().to_csv(dirs["tables"] / "mcc5_formal_feature_summary.csv", index=False)
        return
    preview = ROOT / "data" / "processed" / "features" / "mcc5_features_formal_preview.csv"
    features.head(50).to_csv(preview, index=False)
    csv_out = ROOT / "data" / "processed" / "features" / "mcc5_features_formal.csv"
    features.to_csv(csv_out, index=False)
    missing = features.isna().mean().rename("missing_rate").reset_index().rename(columns={"index": "feature"})
    missing.sort_values("missing_rate", ascending=False).to_csv(dirs["tables"] / "mcc5_formal_feature_missingness.csv", index=False)
    numeric = features.select_dtypes(include=["number", "bool"])
    summary = numeric.describe().T.reset_index().rename(columns={"index": "feature"})
    summary.to_csv(dirs["tables"] / "mcc5_formal_feature_summary.csv", index=False)
    for file_name in ["mcc5_formal_feature_missingness.csv", "mcc5_formal_feature_summary.csv"]:
        shutil.copy2(dirs["tables"] / file_name, ROOT / "results" / "tables" / file_name)


def _split_reports(dirs: dict[str, Path]) -> bool:
    audit_path = ROOT / "results" / "tables" / "mcc5_formal_split_audit.csv"
    if audit_path.exists():
        shutil.copy2(audit_path, dirs["tables"] / "mcc5_formal_split_audit.csv")
    features_path = ROOT / "data" / "processed" / "features" / "mcc5_features_formal.parquet"
    features = pd.read_parquet(features_path) if features_path.exists() else pd.DataFrame()
    rows = []
    for split_path in sorted((ROOT / "data" / "processed" / "splits").glob("mcc5_formal_*_split.csv")):
        split = pd.read_csv(split_path)
        merged = features[["window_id", "label_group"]].merge(split[["window_id", "split"]], on="window_id", how="inner")
        counts = merged.groupby(["split", "label_group"]).size().reset_index(name="count")
        counts.insert(0, "split_file", split_path.name)
        rows.append(counts)
    class_counts = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    class_counts.to_csv(dirs["tables"] / "mcc5_formal_split_class_counts.csv", index=False)
    shutil.copy2(dirs["tables"] / "mcc5_formal_split_class_counts.csv", ROOT / "results" / "tables" / "mcc5_formal_split_class_counts.csv")
    if not audit_path.exists():
        return False
    audit = pd.read_csv(audit_path)
    return bool(not audit.empty and (audit["status"].astype(str) == "ok").all())


def _sync_formal_artifacts(dirs: dict[str, Path]) -> None:
    for pattern in ["mcc5_formal*.csv", "mcc5_formal*.xlsx"]:
        for path in (ROOT / "results" / "tables").glob(pattern):
            shutil.copy2(path, dirs["tables"] / path.name)
    for pattern in ["mcc5_formal*.png", "mcc5_formal*.pdf", "dl_*mcc5_formal*.png", "dl_*mcc5_formal*.pdf"]:
        for path in (ROOT / "results" / "figures").glob(pattern):
            shutil.copy2(path, dirs["figures"] / path.name)
    for pattern in ["mcc5_formal*.joblib", "dl_*mcc5_formal*.pt", "mcc5_fusion_mcc5_formal*.pt"]:
        for path in (ROOT / "results" / "checkpoints").glob(pattern):
            shutil.copy2(path, dirs["checkpoints"] / path.name)


def _write_skip_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _concat_formal_per_class(dirs: dict[str, Path]) -> None:
    rows = []
    for path in sorted((ROOT / "results" / "tables").glob("mcc5_formal_*_split_*_per_class_metrics.csv")):
        df = pd.read_csv(path)
        df.insert(0, "source_table", path.name)
        rows.append(df)
    out = ROOT / "results" / "tables" / "mcc5_formal_ml_per_class_metrics.csv"
    pd.concat(rows, ignore_index=True).to_csv(out, index=False) if rows else pd.DataFrame().to_csv(out, index=False)
    shutil.copy2(out, dirs["tables"] / out.name)


def _step_reached(stop_after: str, step: str) -> bool:
    return STEPS.index(step) <= STEPS.index(stop_after)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--feature_config", default="configs/mcc5_feature.yaml")
    parser.add_argument("--fusion_config", default="configs/mcc5_fusion.yaml")
    parser.add_argument("--stop_after", default="ablation", choices=STEPS)
    parser.add_argument("--dl_jobs", nargs="*", default=["cnn:vib_current", "tcn:vib_current", "transformer:vib_current"])
    parser.add_argument("--allow_cpu_dl", action="store_true")
    parser.add_argument("--resume_existing", action="store_true", help="Reuse completed formal artifacts for the same run_id.")
    parser.add_argument("--skip_dl", action="store_true")
    parser.add_argument("--skip_fusion", action="store_true")
    args = parser.parse_args(argv)

    run_id = args.run_id or _now_run_id()
    dirs = _ensure_run_dirs(run_id)
    _archive_pilot(run_id, dirs)
    history: list[dict[str, Any]] = []
    notes: list[str] = [f"run_id={run_id}", "scope=formal_mcc5", "pilot_artifacts_archived=True"]
    existing_log = dirs["logs"] / "formal_mcc5_command_log.json"
    if args.resume_existing and existing_log.exists():
        try:
            old_log = json.loads(existing_log.read_text(encoding="utf-8"))
            history = list(old_log.get("commands", []))
            notes.extend(str(note) for note in old_log.get("notes", []) if str(note) not in notes)
            notes.append("existing_command_log_loaded=True")
        except Exception as exc:
            notes.append(f"existing_command_log_load_error={exc!r}")

    feature_cfg = load_yaml_config(args.feature_config)
    fusion_cfg = load_yaml_config(args.fusion_config)
    fs = float(config_default(feature_cfg, "fs", config_default(fusion_cfg, "fs", 12800)))
    window_sec = float(config_default(feature_cfg, "window_sec", config_default(fusion_cfg, "window_sec", 1.0)))
    stride_sec = float(config_default(feature_cfg, "stride_sec", config_default(fusion_cfg, "stride_sec", 0.5)))
    fixed_length = int(config_default(fusion_cfg, "fixed_length", 8192))
    epochs = int(config_default(fusion_cfg, "epochs", 30))
    patience = int(config_default(fusion_cfg, "early_stopping_patience", 8))
    batch_size = int(config_default(fusion_cfg, "batch_size", 64))
    notes.extend(
        [
            f"feature_config={args.feature_config}",
            f"fusion_config={args.fusion_config}",
            f"fs={fs}",
            f"window_sec={window_sec}",
            f"stride_sec={stride_sec}",
            f"fixed_length={fixed_length}",
            f"epochs={epochs}",
            f"early_stopping_patience={patience}",
            f"batch_size={batch_size}",
        ]
    )

    preflight = _preflight(run_id, dirs, Path(args.feature_config), Path(args.fusion_config))
    cuda_available = bool(preflight["torch"].get("cuda_available"))
    if preflight["recommendation"] == "stop":
        notes.append("preflight_recommendation=stop")
        _write_command_log(run_id, dirs, history, "stopped_preflight", notes)
        return 1
    if args.stop_after == "preflight":
        _write_command_log(run_id, dirs, history, "preflight_only", notes)
        print(f"Formal run ID: {run_id}")
        print(f"Preflight audit: {dirs['docs'] / f'{run_id}_preflight_audit.md'}")
        return 0

    py = sys.executable
    raw_stats = preflight["raw_mcc5"]
    use_extract_to = raw_stats["csv_files"] == 0 and raw_stats["zip_files"] > 0
    all_meta = "data/processed/metadata/mcc5_metadata_all_labels_formal.csv"
    main_meta = "data/processed/metadata/mcc5_metadata_formal.csv"
    windows_path = "data/processed/windows/mcc5_windows_formal.parquet"
    segments_dir = "data/processed/windows/mcc5_segments_formal"
    features_path = "data/processed/features/mcc5_features_formal.parquet"
    split_dir = "data/processed/splits"
    split_glob = "mcc5_formal_*_split.csv"

    if _step_reached(args.stop_after, "metadata"):
        metadata_ready = (ROOT / all_meta).exists() and (ROOT / main_meta).exists()
        if args.resume_existing and metadata_ready:
            print("Reusing existing formal metadata.")
            notes.append("metadata_reused=True")
        else:
            for cmd in [
                [py, "src/data/prepare_mcc5.py", "--root", "data/raw/mcc5_thu_motor", "--out", all_meta, "--bearing_only", "false"],
                [py, "src/data/prepare_mcc5.py", "--root", "data/raw/mcc5_thu_motor", "--out", main_meta, "--bearing_only", "true"],
            ]:
                if use_extract_to and cmd[-1] == "true":
                    cmd.extend(["--extract_to", "data/interim/mcc5_thu_motor/formal_selected_csv"])
                code = _run_command(cmd, history)
                if code != 0:
                    _write_failure_report(dirs, "metadata", cmd, history, "Check MCC5 raw paths and filename parsing.")
                    _write_command_log(run_id, dirs, history, "failed_metadata", notes)
                    return code
        _metadata_reports(run_id, dirs)
        _sync_formal_artifacts(dirs)
        if args.stop_after == "metadata":
            _write_command_log(run_id, dirs, history, "completed_until_metadata", notes)
            print(f"Formal run ID: {run_id}")
            return 0

    if _step_reached(args.stop_after, "windows"):
        windows_ready = (ROOT / windows_path).exists() and any((ROOT / segments_dir).glob("*.npz"))
        if args.resume_existing and windows_ready:
            print("Reusing existing formal windows and NPZ segments.")
            notes.append("windows_reused=True")
        else:
            cmd = [
                py,
                "src/data/build_windows.py",
                "--dataset",
                "mcc5",
                "--metadata",
                main_meta,
                "--out",
                windows_path,
                "--segment_dir",
                segments_dir,
                "--window_sec",
                str(window_sec),
                "--stride_sec",
                str(stride_sec),
                "--sampling_rate",
                str(fs),
            ]
            code = _run_command(cmd, history)
            if code != 0:
                _write_failure_report(dirs, "windows", cmd, history, "Inspect unreadable MCC5 CSV files and available disk space.")
                _write_command_log(run_id, dirs, history, "failed_windows", notes)
                return code
        _window_reports(dirs)
        _sync_formal_artifacts(dirs)
        if args.stop_after == "windows":
            _write_command_log(run_id, dirs, history, "completed_until_windows", notes)
            print(f"Formal run ID: {run_id}")
            return 0

    if _step_reached(args.stop_after, "features"):
        features_ready = (ROOT / features_path).exists()
        if args.resume_existing and features_ready:
            print("Reusing existing formal features.")
            notes.append("features_reused=True")
        else:
            cmd = [py, "src/experiments/exp01_extract_features.py", "--windows", windows_path, "--out", features_path, "--fs", str(fs)]
            code = _run_command(cmd, history)
            if code != 0:
                _write_failure_report(dirs, "features", cmd, history, "Check formal NPZ windows and feature extraction dependencies.")
                _write_command_log(run_id, dirs, history, "failed_features", notes)
                return code
        _feature_reports(dirs)
        _sync_formal_artifacts(dirs)
        if args.stop_after == "features":
            _write_command_log(run_id, dirs, history, "completed_until_features", notes)
            print(f"Formal run ID: {run_id}")
            return 0

    if _step_reached(args.stop_after, "splits"):
        commands = [
            [py, "src/data/make_splits.py", "--features", features_path, "--dataset", "mcc5_formal", "--out_dir", split_dir],
            [
                py,
                "src/data/audit_splits.py",
                "--features",
                features_path,
                "--split",
                split_dir,
                "--split_glob",
                split_glob,
                "--out",
                "results/tables/mcc5_formal_split_audit.csv",
                "--log_out",
                str(dirs["logs"] / "mcc5_formal_split_audit.json"),
                "--strict",
            ],
        ]
        for cmd in commands:
            code = _run_command(cmd, history)
            if code != 0:
                _write_failure_report(dirs, "splits", cmd, history, "Fix split generation or leakage audit issues before training.")
                _write_command_log(run_id, dirs, history, "failed_splits", notes)
                return code
        if not _split_reports(dirs):
            _write_failure_report(dirs, "split_audit", commands[-1], history, "Split audit did not prove all formal splits are ok.")
            _write_command_log(run_id, dirs, history, "failed_split_audit", notes)
            return 1
        _sync_formal_artifacts(dirs)
        if args.stop_after == "splits":
            _write_command_log(run_id, dirs, history, "completed_until_splits", notes)
            print(f"Formal run ID: {run_id}")
            return 0

    if _step_reached(args.stop_after, "ml"):
        cmd = [
            py,
            "src/experiments/exp02_ml_baselines.py",
            "--features",
            features_path,
            "--split",
            split_dir,
            "--split_glob",
            split_glob,
            "--out",
            "results/tables/mcc5_formal_ml_baselines_all_splits.csv",
            "--seed",
            "42",
        ]
        code = _run_command(cmd, history)
        if code != 0:
            _write_failure_report(dirs, "ml", cmd, history, "Inspect formal features and split class coverage.")
            _write_command_log(run_id, dirs, history, "failed_ml", notes)
            return code
        _concat_formal_per_class(dirs)
        _sync_formal_artifacts(dirs)
        if args.stop_after == "ml":
            _write_command_log(run_id, dirs, history, "completed_until_ml", notes)
            print(f"Formal run ID: {run_id}")
            return 0

    dl_allowed = not args.skip_dl and (cuda_available or args.allow_cpu_dl)
    if _step_reached(args.stop_after, "dl"):
        if not dl_allowed:
            reason = "CUDA unavailable and --allow_cpu_dl was not set." if not args.skip_dl else "Skipped by --skip_dl."
            notes.append(f"formal_dl_skipped={reason}")
            rows = []
            for job in args.dl_jobs:
                model, _, mode = job.partition(":")
                rows.append({"status": "skipped", "reason": reason, "model": model, "input_mode": mode or "vib_current"})
            out = ROOT / "results" / "tables" / "mcc5_formal_dl_results.csv"
            _write_skip_table(out, rows)
            shutil.copy2(out, dirs["tables"] / out.name)
        else:
            for split_path in sorted((ROOT / split_dir).glob(split_glob)):
                for job in args.dl_jobs:
                    model, _, mode = job.partition(":")
                    cmd = [
                        py,
                        "src/experiments/exp03_dl_baselines.py",
                        "--windows",
                        windows_path,
                        "--split",
                        str(split_path.relative_to(ROOT)),
                        "--input_mode",
                        mode or "vib_current",
                        "--model",
                        model,
                        "--epochs",
                        str(epochs),
                        "--early_stopping_patience",
                        str(patience),
                        "--batch_size",
                        str(batch_size),
                        "--fixed_length",
                        str(fixed_length),
                    ]
                    code = _run_command(cmd, history)
                    if code != 0:
                        _write_failure_report(dirs, "dl", cmd, history, "Inspect DL memory/runtime and consider skipping DL with documented limitation.")
                        _write_command_log(run_id, dirs, history, "failed_dl", notes)
                        return code
            # Preserve whatever formal split-named DL outputs were produced.
            _sync_formal_artifacts(dirs)
        if args.stop_after == "dl":
            _write_command_log(run_id, dirs, history, "completed_until_dl", notes)
            print(f"Formal run ID: {run_id}")
            return 0

    fusion_allowed = not args.skip_fusion and (cuda_available or args.allow_cpu_dl)
    if _step_reached(args.stop_after, "fusion"):
        if not fusion_allowed:
            reason = "CUDA unavailable and --allow_cpu_dl was not set." if not args.skip_fusion else "Skipped by --skip_fusion."
            notes.append(f"formal_fusion_skipped={reason}")
            out = ROOT / "results" / "tables" / "mcc5_formal_fusion_results.csv"
            _write_skip_table(out, [{"status": "skipped", "reason": reason, "model": "fusion"}])
            shutil.copy2(out, dirs["tables"] / out.name)
        else:
            # Clear old non-formal fusion table before collecting formal split rows.
            formal_fusion_out = ROOT / "results" / "tables" / "mcc5_formal_fusion_results.csv"
            if formal_fusion_out.exists():
                formal_fusion_out.unlink()
            for split_path in sorted((ROOT / split_dir).glob(split_glob)):
                cmd = [
                    py,
                    "src/experiments/exp04_train_fusion.py",
                    "--windows",
                    windows_path,
                    "--features",
                    features_path,
                    "--split",
                    str(split_path.relative_to(ROOT)),
                    "--config",
                    args.fusion_config,
                    "--epochs",
                    str(epochs),
                    "--early_stopping_patience",
                    str(patience),
                ]
                code = _run_command(cmd, history)
                if code != 0:
                    _write_failure_report(dirs, "fusion", cmd, history, "Inspect fusion training runtime and dependencies.")
                    _write_command_log(run_id, dirs, history, "failed_fusion", notes)
                    return code
            nonformal = ROOT / "results" / "tables" / "mcc5_fusion_results.csv"
            if nonformal.exists():
                df = pd.read_csv(nonformal)
                df = df[df.get("split_file", pd.Series(dtype=str)).astype(str).str.startswith("mcc5_formal_")]
                df.to_csv(formal_fusion_out, index=False)
                shutil.copy2(formal_fusion_out, dirs["tables"] / formal_fusion_out.name)
            _sync_formal_artifacts(dirs)
        if args.stop_after == "fusion":
            _write_command_log(run_id, dirs, history, "completed_until_fusion", notes)
            print(f"Formal run ID: {run_id}")
            return 0

    if _step_reached(args.stop_after, "ablation"):
        commands = [
            [
                py,
                "src/experiments/exp05_ablation.py",
                "--features",
                features_path,
                "--split",
                "data/processed/splits/mcc5_formal_source_file_split.csv",
                "--out",
                "results/tables/mcc5_formal_ablation.csv",
                "--seed",
                "42",
            ],
            [
                py,
                "src/experiments/exp06_representation_comparison.py",
                "--features",
                features_path,
                "--split",
                "data/processed/splits/mcc5_formal_source_file_split.csv",
                "--out",
                "results/tables/mcc5_formal_representation_comparison.csv",
                "--dl_results",
                "results/tables/mcc5_formal_dl_results.csv",
                "--fusion_results",
                "results/tables/mcc5_formal_fusion_results.csv",
            ],
        ]
        for cmd in commands:
            code = _run_command(cmd, history)
            if code != 0:
                _write_failure_report(dirs, "ablation", cmd, history, "Inspect formal ML/feature outputs and representation inputs.")
                _write_command_log(run_id, dirs, history, "failed_ablation", notes)
                return code
        _sync_formal_artifacts(dirs)
        _write_command_log(run_id, dirs, history, "completed_until_ablation", notes)
        print(f"Formal run ID: {run_id}")
        return 0

    _write_command_log(run_id, dirs, history, f"completed_until_{args.stop_after}", notes)
    print(f"Formal run ID: {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
