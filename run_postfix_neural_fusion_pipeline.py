"""Corrected MCC5 neural/fusion rerun with train-only z-score and source-level audit."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_formal_dl_fusion_gpu import (
    FEATURES_PATH,
    SPLIT_PATHS,
    WINDOWS_PATH,
    RunPaths,
    load_formal_tables,
    prepare_fusion_arrays,
    prepare_raw_arrays,
    require_cuda,
    summarize_results,
    train_fusion_one,
    train_raw_one,
)
from src.utils.io import ensure_dir
from src.utils.metrics import classification_metrics


DEFAULT_MODELS = ["cnn", "tcn", "transformer"]
DEFAULT_SETTINGS = [
    "vibration_current",
    "vibration_current_rpm_load_only",
    "vibration_current_auxiliary_only",
    "vibration_current_rpm_load",
    "full_multisource_fusion",
    "full_engineered_features_254",
]
SETTING_LABELS = {
    "vibration_current": "no_scalars",
    "vibration_current_rpm_load_only": "context_2",
    "vibration_current_auxiliary_only": "auxiliary_26",
    "vibration_current_rpm_load": "auxiliary_context_28",
    "full_multisource_fusion": "full_fusion_legacy_262",
    "full_engineered_features_254": "full_engineered_254",
}


def now_run_id() -> str:
    return datetime.now().strftime("postfix_neural_fusion_%Y%m%d_%H%M%S")


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def dataframe_markdown(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""
    if frame.empty:
        return "_No rows._"

    def clean(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(clean(value) for value in row) + " |")
    return "\n".join(lines)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_paths(run_id: str) -> RunPaths:
    root = ensure_dir(Path("results/formal_runs") / run_id)
    paths = RunPaths(
        run_id=run_id,
        root=root,
        tables=ensure_dir(root / "tables"),
        figures=ensure_dir(root / "figures"),
        logs=ensure_dir(root / "logs"),
        checkpoints=ensure_dir(root / "checkpoints"),
        docs=ensure_dir("docs/postfix_runs"),
    )
    ensure_dir(root / "predictions")
    ensure_dir(root / "normalization")
    ensure_dir(root / "splits")
    Path("results/formal_runs/latest_postfix_run_id.txt").write_text(run_id + "\n", encoding="utf-8")
    return paths


def write_preflight(paths: RunPaths, env: dict[str, object]) -> None:
    split_rows = []
    for name, path in SPLIT_PATHS.items():
        split_rows.append({"split": name, "path": str(path), "sha256": file_sha256(path)})
    pd.DataFrame(split_rows).to_csv(paths.tables / "formal_input_hashes.csv", index=False)
    lines = [
        "# Post-fix Neural/Fusion Preflight",
        "",
        f"- Run ID: `{paths.run_id}`",
        f"- Created: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Python: `{sys.executable}`",
        f"- Python version: `{sys.version.split()[0]}`",
        f"- Torch: `{torch.__version__}`",
        f"- Torch CUDA: `{torch.version.cuda}`",
        f"- CUDA available: `{torch.cuda.is_available()}`",
        f"- GPU: `{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}`",
        f"- Windows table: `{WINDOWS_PATH}`",
        f"- Feature table: `{FEATURES_PATH}`",
        "",
        "The historical neural/fusion results are retained but excluded from final claims because their preprocessing centered inputs without dividing by the fitted training standard deviation.",
        "",
        "## Environment JSON",
        "",
        "```json",
        json.dumps(env, indent=2, default=str),
        "```",
    ]
    (paths.docs / f"{paths.run_id}_preflight.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _select_validation_sources(train_sources: pd.DataFrame, seed: int, fraction: float) -> set[str]:
    rng = np.random.default_rng(seed)
    selected: set[str] = set()
    for _label, group in train_sources.groupby("label_group", sort=True):
        sources = group["source_file"].astype(str).to_numpy()
        if len(sources) < 2:
            raise RuntimeError(f"Cannot create grouped validation for class {_label!r}: fewer than two train sources.")
        count = min(len(sources) - 1, max(1, int(round(fraction * len(sources)))))
        selected.update(str(value) for value in rng.choice(sources, size=count, replace=False))
    return selected


def freeze_grouped_splits(
    paths: RunPaths,
    windows: pd.DataFrame,
    original_splits: dict[str, pd.DataFrame],
    *,
    validation_seed: int,
    validation_fraction: float,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    frozen: dict[str, pd.DataFrame] = {}
    audit_rows: list[dict[str, object]] = []
    split_dir = ensure_dir(Path("data/processed/splits/postfix") / paths.run_id)

    base_columns = [
        column
        for column in ["window_id", "source_file", "label_group", "condition_type", "rpm_nominal", "load_nm"]
        if column in windows.columns
    ]
    for split_name, split in original_splits.items():
        merged = windows[base_columns].merge(split[["window_id", "split"]], on="window_id", how="inner")
        merged["role"] = merged["split"].astype(str)
        original_has_validation = bool((merged["role"] == "val").any())
        if not original_has_validation:
            train_sources = (
                merged[merged["role"] == "train"]
                .groupby("source_file", as_index=False)
                .agg(label_group=("label_group", "first"))
            )
            validation_sources = _select_validation_sources(train_sources, validation_seed, validation_fraction)
            merged.loc[
                (merged["role"] == "train") & merged["source_file"].astype(str).isin(validation_sources),
                "role",
            ] = "val"

        source_role_counts = merged.groupby("source_file")["role"].nunique()
        if int(source_role_counts.max()) != 1:
            raise RuntimeError(f"A source recording crosses roles in frozen split {split_name}.")
        role_sources = {
            role: set(group["source_file"].astype(str))
            for role, group in merged.groupby("role", sort=False)
        }
        role_windows = {role: set(group["window_id"]) for role, group in merged.groupby("role", sort=False)}
        overlaps = {}
        for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
            overlaps[f"{left}_{right}_source_overlap"] = len(role_sources.get(left, set()) & role_sources.get(right, set()))
            overlaps[f"{left}_{right}_window_overlap"] = len(role_windows.get(left, set()) & role_windows.get(right, set()))
        if any(value != 0 for value in overlaps.values()):
            raise RuntimeError(f"Frozen split overlap detected for {split_name}: {overlaps}")

        frozen_split = merged[["window_id", "role"]].rename(columns={"role": "split"})
        frozen_path = split_dir / f"mcc5_postfix_{split_name}_split.csv"
        frozen_split.to_csv(frozen_path, index=False)
        frozen[split_name] = frozen_split

        source_counts = (
            merged.groupby(["role", "label_group"])["source_file"]
            .nunique()
            .unstack(fill_value=0)
            .to_dict(orient="index")
        )
        window_counts = merged.groupby(["role", "label_group"]).size().unstack(fill_value=0).to_dict(orient="index")
        audit_rows.append(
            {
                "split_name": split_name,
                "frozen_split_path": str(frozen_path),
                "validation_seed": validation_seed,
                "validation_fraction": validation_fraction,
                "original_validation_preserved": original_has_validation,
                "fallback_used": False,
                "train_source_count": len(role_sources.get("train", set())),
                "validation_source_count": len(role_sources.get("val", set())),
                "test_source_count": len(role_sources.get("test", set())),
                "train_window_count": len(role_windows.get("train", set())),
                "validation_window_count": len(role_windows.get("val", set())),
                "test_window_count": len(role_windows.get("test", set())),
                "source_counts_by_role_and_label": json.dumps(source_counts, sort_keys=True),
                "window_counts_by_role_and_label": json.dumps(window_counts, sort_keys=True),
                **overlaps,
                "sha256": file_sha256(frozen_path),
            }
        )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(paths.tables / "internal_validation_source_audit.csv", index=False)
    (paths.docs / f"{paths.run_id}_internal_validation_audit.md").write_text(
        "# Frozen Grouped Internal Validation Audit\n\n"
        + dataframe_markdown(audit)
        + "\n\nAll final neural and fusion runs use these fixed source-recording assignments for every optimization seed. No window-level fallback was used.\n",
        encoding="utf-8",
    )
    return frozen, audit


def _normalization_rows(arrays: dict[str, object], split_name: str, setting: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add_sequence(prefix: str, array_key: str, scale_key: str) -> None:
        values = np.asarray(arrays[array_key], dtype=np.float64)
        means = values.mean(axis=(0, 2))
        stds = values.std(axis=(0, 2))
        scales = np.asarray(arrays[scale_key], dtype=np.float64).reshape(-1)
        for index, (mean, std) in enumerate(zip(means, stds)):
            constant = bool(std < 1e-7)
            passed = abs(float(mean)) < 1e-3 and (constant or abs(float(std) - 1.0) < 1e-2)
            rows.append(
                {
                    "split_name": split_name,
                    "setting": setting,
                    "input_group": prefix,
                    "feature": f"channel_{index}",
                    "transformed_mean": float(mean),
                    "transformed_std": float(std),
                    "fitted_scale": float(scales[index]),
                    "constant": constant,
                    "status": "pass" if passed else "fail",
                }
            )

    if "x_train" in arrays:
        add_sequence("vibration_current", "x_train", "std")
    else:
        add_sequence("vibration", "vib_train", "vib_std")
        add_sequence("current", "cur_train", "cur_std")
        scalars = np.asarray(arrays["scal_train"], dtype=np.float64)
        scalar_names = list(arrays["scalar_cols"]) or ["placeholder_no_scalar"]
        scales = np.asarray(arrays["scal_std"], dtype=np.float64).reshape(-1)
        for index, name in enumerate(scalar_names):
            mean = float(scalars[:, index].mean())
            std = float(scalars[:, index].std())
            constant = bool(std < 1e-7)
            passed = abs(mean) < 1e-3 and (constant or abs(std - 1.0) < 1e-2)
            rows.append(
                {
                    "split_name": split_name,
                    "setting": setting,
                    "input_group": "scalar",
                    "feature": name,
                    "transformed_mean": mean,
                    "transformed_std": std,
                    "fitted_scale": float(scales[index]),
                    "constant": constant,
                    "status": "pass" if passed else "fail",
                }
            )
    return rows


def save_normalization_artifacts(paths: RunPaths, arrays: dict[str, object], split_name: str, setting: str) -> None:
    target = ensure_dir(paths.root / "normalization")
    if "x_train" in arrays:
        np.savez_compressed(target / f"{split_name}_{setting}_sequence_stats.npz", mean=arrays["mean"], scale=arrays["std"])
        return
    np.savez_compressed(
        target / f"{split_name}_{setting}_stats.npz",
        vibration_mean=arrays["vib_mean"],
        vibration_scale=arrays["vib_std"],
        current_mean=arrays["cur_mean"],
        current_scale=arrays["cur_std"],
        scalar_mean=arrays["scal_mean"],
        scalar_scale=arrays["scal_std"],
    )
    (target / f"{split_name}_{setting}_scalar_schema.json").write_text(
        json.dumps(
            {
                "columns": list(arrays["scalar_cols"]),
                "median": arrays["scal_median"],
                "statistics_source": "training_only",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def merge_normalization_audit(paths: RunPaths, rows: list[dict[str, object]]) -> pd.DataFrame:
    """Merge incremental audits so a focused rerun cannot erase prior settings."""
    audit_path = paths.tables / "normalization_audit.csv"
    frames: list[pd.DataFrame] = []
    if audit_path.exists():
        frames.append(pd.read_csv(audit_path))
    if rows:
        frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    keys = ["split_name", "setting", "input_group", "feature"]
    combined = combined.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)
    combined.to_csv(audit_path, index=False)
    return combined


def append_result(path: Path, row: dict[str, object]) -> pd.DataFrame:
    current = pd.read_csv(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([current, pd.DataFrame([row])], ignore_index=True)
    combined.to_csv(path, index=False)
    return combined


def completed_keys(path: Path, columns: list[str]) -> set[tuple[object, ...]]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    return {tuple(row[column] for column in columns) for _, row in frame.iterrows()}


def run_raw_matrix(
    paths: RunPaths,
    windows: pd.DataFrame,
    frozen_splits: dict[str, pd.DataFrame],
    *,
    models: list[str],
    seeds: list[int],
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    fixed_length: int,
    normalization_rows: list[dict[str, object]],
) -> pd.DataFrame:
    out = paths.tables / "corrected_raw_dl_by_seed.csv"
    done = completed_keys(out, ["split_name", "model", "seed", "sequence_length"])
    for split_name, split in frozen_splits.items():
        arrays = prepare_raw_arrays(windows, split, seed=42, fixed_length=fixed_length)
        normalization_rows.extend(_normalization_rows(arrays, split_name, f"raw_{fixed_length}"))
        save_normalization_artifacts(paths, arrays, split_name, f"raw_{fixed_length}")
        for model in models:
            for seed in seeds:
                key = (split_name, model, seed, fixed_length)
                prediction = paths.root / "predictions" / f"formal_dl_{model}_vib_current_{split_name}_length{fixed_length}_seed{seed}_predictions.csv"
                if key in done and prediction.exists():
                    continue
                print(f"[postfix raw] split={split_name} model={model} seed={seed} length={fixed_length}", flush=True)
                row = train_raw_one(
                    arrays,
                    model,
                    split_name,
                    f"mcc5_postfix_{split_name}_split.csv",
                    seed,
                    paths,
                    epochs,
                    patience,
                    batch_size,
                    learning_rate,
                    run_tag=f"length{fixed_length}",
                )
                row["evidence_status"] = "corrected_formal_train_only_zscore"
                append_result(out, row)
                done.add(key)
        del arrays
        gc.collect()
        torch.cuda.empty_cache()
    return pd.read_csv(out)


def run_length_sensitivity(
    paths: RunPaths,
    windows: pd.DataFrame,
    cross_rpm_split: pd.DataFrame,
    *,
    seeds: list[int],
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    normalization_rows: list[dict[str, object]],
) -> pd.DataFrame:
    out = paths.tables / "corrected_input_length_sensitivity_by_seed.csv"
    done = completed_keys(out, ["split_name", "model", "seed", "sequence_length"])
    for length in [8192, 12800]:
        arrays = prepare_raw_arrays(windows, cross_rpm_split, seed=42, fixed_length=length)
        normalization_rows.extend(_normalization_rows(arrays, "cross_rpm", f"raw_length_{length}"))
        save_normalization_artifacts(paths, arrays, "cross_rpm", f"raw_length_{length}")
        for seed in seeds:
            key = ("cross_rpm", "cnn", seed, length)
            if key in done:
                continue
            print(f"[postfix length] length={length} seed={seed}", flush=True)
            row = train_raw_one(
                arrays,
                "cnn",
                "cross_rpm",
                "mcc5_postfix_cross_rpm_split.csv",
                seed,
                paths,
                epochs,
                patience,
                batch_size,
                learning_rate,
                run_tag=f"sensitivity{length}",
            )
            row["evidence_status"] = "corrected_input_length_sensitivity"
            append_result(out, row)
            done.add(key)
        del arrays
        gc.collect()
        torch.cuda.empty_cache()
    frame = pd.read_csv(out)
    summary = summarize_results(frame, ["split_name", "model", "sequence_length"])
    summary.to_csv(paths.tables / "corrected_input_length_sensitivity.csv", index=False)
    return frame


def run_fusion_matrix(
    paths: RunPaths,
    windows: pd.DataFrame,
    features: pd.DataFrame,
    frozen_splits: dict[str, pd.DataFrame],
    *,
    settings: list[str],
    seeds: list[int],
    focused_seeds: list[int],
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    fixed_length: int,
    normalization_rows: list[dict[str, object]],
) -> pd.DataFrame:
    out = paths.tables / "corrected_fusion_by_seed.csv"
    done = completed_keys(out, ["split_name", "setting", "seed"])
    for split_name, split in frozen_splits.items():
        run_seeds = focused_seeds if split_name == "cross_rpm" else seeds
        for setting in settings:
            arrays = prepare_fusion_arrays(windows, features, split, seed=42, fixed_length=fixed_length, setting=setting)
            normalization_rows.extend(_normalization_rows(arrays, split_name, setting))
            save_normalization_artifacts(paths, arrays, split_name, setting)
            scalar_count = len(arrays["scalar_cols"])
            expected = {
                "vibration_current": 0,
                "vibration_current_rpm_load_only": 2,
                "vibration_current_auxiliary_only": 26,
                "vibration_current_rpm_load": 28,
                "full_engineered_features_254": 254,
            }.get(setting)
            if expected is not None and scalar_count != expected:
                raise RuntimeError(f"Scalar schema mismatch for {setting}: expected {expected}, got {scalar_count}.")
            for seed in run_seeds:
                key = (split_name, setting, seed)
                if key in done:
                    continue
                print(f"[postfix fusion] split={split_name} setting={setting} seed={seed}", flush=True)
                row = train_fusion_one(
                    arrays,
                    split_name,
                    f"mcc5_postfix_{split_name}_split.csv",
                    seed,
                    paths,
                    epochs,
                    patience,
                    batch_size,
                    learning_rate,
                    weight_decay,
                    setting,
                    "corrected_fusion",
                )
                row["setting_label"] = SETTING_LABELS[setting]
                row["evidence_status"] = (
                    "posthoc_exploratory_same_holdout"
                    if setting in {"vibration_current_auxiliary_only", "vibration_current_rpm_load"}
                    else "corrected_formal_train_only_zscore"
                )
                append_result(out, row)
                done.add(key)
            del arrays
            gc.collect()
            torch.cuda.empty_cache()
    return pd.read_csv(out)


def build_summaries(paths: RunPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(paths.tables / "corrected_raw_dl_by_seed.csv")
    fusion = pd.read_csv(paths.tables / "corrected_fusion_by_seed.csv")
    fusion["setting_label"] = fusion["setting"].map(SETTING_LABELS).fillna(fusion["setting_label"])
    fusion.to_csv(paths.tables / "corrected_fusion_by_seed.csv", index=False)
    raw_summary = summarize_results(raw, ["split_name", "model", "sequence_length", "evidence_status"])
    fusion_summary = summarize_results(
        fusion,
        ["split_name", "setting", "setting_label", "model", "scalar_feature_count", "evidence_status"],
    )
    raw_summary.to_csv(paths.tables / "corrected_raw_dl_summary.csv", index=False)
    fusion_summary.to_csv(paths.tables / "corrected_fusion_summary.csv", index=False)

    focused = fusion[fusion["split_name"] == "cross_rpm"].copy()
    pivot = focused.pivot_table(index="seed", columns="setting_label", values=["macro_f1", "worst_class_recall"])
    delta_pairs = [
        ("context_2", "no_scalars"),
        ("auxiliary_26", "no_scalars"),
        ("auxiliary_context_28", "auxiliary_26"),
        ("full_fusion_legacy_262", "no_scalars"),
        ("full_engineered_254", "no_scalars"),
    ]
    delta_rows = []
    for metric in ["macro_f1", "worst_class_recall"]:
        for left, right in delta_pairs:
            if (metric, left) not in pivot or (metric, right) not in pivot:
                continue
            delta = (pivot[(metric, left)] - pivot[(metric, right)]).dropna()
            for seed, value in delta.items():
                delta_rows.append(
                    {
                        "metric": metric,
                        "comparison": f"{left}_minus_{right}",
                        "seed": int(seed),
                        "delta": float(value),
                    }
                )
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(paths.tables / "corrected_fusion_matched_deltas_by_seed.csv", index=False)
    if not deltas.empty:
        delta_summary = deltas.groupby(["metric", "comparison"])["delta"].agg(["count", "mean", "std", "min", "max"]).reset_index()
        delta_summary.to_csv(paths.tables / "corrected_fusion_matched_deltas.csv", index=False)
    return raw_summary, fusion_summary


def _metrics_from_confusion(matrix: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(matrix, dtype=np.float64)
    total = matrix.sum()
    tp = np.diag(matrix)
    precision = tp / np.maximum(matrix.sum(axis=0), 1.0)
    recall = tp / np.maximum(matrix.sum(axis=1), 1.0)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    support = matrix.sum(axis=1)
    supported = support > 0
    return {
        "accuracy": float(tp.sum() / total) if total else np.nan,
        "macro_f1": float(f1[supported].mean()) if supported.any() else np.nan,
        "worst_class_recall": float(recall[supported].min()) if supported.any() else np.nan,
    }


def source_metrics_and_bootstrap(paths: RunPaths, *, replicates: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    source_dir = ensure_dir(paths.root / "predictions" / "source_level")

    metadata_by_prediction: dict[str, dict[str, object]] = {}
    result_tables = [
        paths.tables / "corrected_raw_dl_by_seed.csv",
        paths.tables / "corrected_input_length_sensitivity_by_seed.csv",
        paths.tables / "corrected_fusion_by_seed.csv",
    ]
    metadata_columns = [
        "split_name",
        "model",
        "setting",
        "seed",
        "run_tag",
        "sequence_length",
        "scalar_feature_count",
        "setting_label",
        "evidence_status",
    ]
    for result_table in result_tables:
        if not result_table.exists():
            continue
        result_frame = pd.read_csv(result_table)
        for _, result_row in result_frame.iterrows():
            prediction_value = result_row.get("prediction_path")
            if pd.isna(prediction_value):
                continue
            metadata_by_prediction[Path(str(prediction_value)).name] = {
                column: result_row[column]
                for column in metadata_columns
                if column in result_frame.columns and not pd.isna(result_row[column])
            }

    for prediction_path in sorted((paths.root / "predictions").glob("*_predictions.csv")):
        frame = pd.read_csv(prediction_path)
        probability_columns = [column for column in frame.columns if column.startswith("prob__")]
        labels = [column.removeprefix("prob__") for column in probability_columns]
        if not probability_columns or "source_file" not in frame:
            continue
        source = frame.groupby("source_file", as_index=False).agg(
            true_label=("true_label", "first"),
            **{column: (column, "mean") for column in probability_columns},
        )
        probability_values = source[probability_columns].to_numpy()
        source["predicted_label"] = [labels[index] for index in probability_values.argmax(axis=1)]
        majority = (
            frame.groupby("source_file")["predicted_label"]
            .agg(lambda values: sorted(values.value_counts()[lambda counts: counts == counts.max()].index)[0])
            .rename("majority_label")
        )
        source = source.merge(majority, on="source_file", how="left")
        run_meta = {key: frame[key].iloc[0] for key in ["split_name", "model", "setting", "seed"] if key in frame}
        run_meta.update(metadata_by_prediction.get(prediction_path.name, {}))
        run_meta.setdefault("run_tag", "")
        run_meta.setdefault("sequence_length", np.nan)
        run_meta.setdefault("scalar_feature_count", np.nan)
        run_meta.setdefault("setting_label", run_meta.get("setting", ""))
        run_meta.setdefault("evidence_status", "corrected_predefined_protocol")
        for aggregation, prediction_column in [("mean_probability", "predicted_label"), ("majority_vote", "majority_label")]:
            metrics = classification_metrics(source["true_label"], source[prediction_column], labels=labels)
            metric_rows.append(
                {
                    "prediction_file": prediction_path.name,
                    "aggregation": aggregation,
                    "source_count": len(source),
                    **run_meta,
                    **metrics,
                }
            )
        source.to_csv(source_dir / f"source_{prediction_path.name}", index=False)

        label_to_index = {label: index for index, label in enumerate(labels)}
        source_confusions = []
        majority_confusions = []
        source_labels = []
        window_confusions = []
        for source_name, group in frame.groupby("source_file", sort=False):
            true_label = str(group["true_label"].iloc[0])
            source_labels.append(true_label)
            mean_pred = str(source.loc[source["source_file"] == source_name, "predicted_label"].iloc[0])
            source_cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
            source_cm[label_to_index[true_label], label_to_index[mean_pred]] += 1
            source_confusions.append(source_cm)
            majority_pred = str(source.loc[source["source_file"] == source_name, "majority_label"].iloc[0])
            majority_cm = np.zeros_like(source_cm)
            majority_cm[label_to_index[true_label], label_to_index[majority_pred]] += 1
            majority_confusions.append(majority_cm)
            window_cm = np.zeros_like(source_cm)
            for predicted in group["predicted_label"].astype(str):
                window_cm[label_to_index[true_label], label_to_index[predicted]] += 1
            window_confusions.append(window_cm)

        source_confusions_array = np.stack(source_confusions)
        majority_confusions_array = np.stack(majority_confusions)
        window_confusions_array = np.stack(window_confusions)
        source_labels_array = np.asarray(source_labels)
        rng = np.random.default_rng(seed + int(hashlib.sha256(prediction_path.name.encode()).hexdigest()[:8], 16))
        bootstrap_metrics = {
            "source_mean_probability": [],
            "source_majority_vote": [],
            "window_clustered": [],
        }
        for _ in range(replicates):
            sampled_indices: list[int] = []
            for label in labels:
                candidates = np.flatnonzero(source_labels_array == label)
                if len(candidates):
                    sampled_indices.extend(rng.choice(candidates, size=len(candidates), replace=True).tolist())
            sampled = np.asarray(sampled_indices, dtype=int)
            bootstrap_metrics["source_mean_probability"].append(_metrics_from_confusion(source_confusions_array[sampled].sum(axis=0)))
            bootstrap_metrics["source_majority_vote"].append(_metrics_from_confusion(majority_confusions_array[sampled].sum(axis=0)))
            bootstrap_metrics["window_clustered"].append(_metrics_from_confusion(window_confusions_array[sampled].sum(axis=0)))
        point_metrics = {
            "source_mean_probability": _metrics_from_confusion(source_confusions_array.sum(axis=0)),
            "source_majority_vote": _metrics_from_confusion(majority_confusions_array.sum(axis=0)),
            "window_clustered": _metrics_from_confusion(window_confusions_array.sum(axis=0)),
        }
        for aggregation, values in bootstrap_metrics.items():
            for metric in ["accuracy", "macro_f1", "worst_class_recall"]:
                samples = np.asarray([row[metric] for row in values], dtype=float)
                bootstrap_rows.append(
                    {
                        "prediction_file": prediction_path.name,
                        "aggregation": aggregation,
                        "metric": metric,
                        "point_estimate": float(point_metrics[aggregation][metric]),
                        "bootstrap_mean": float(np.nanmean(samples)),
                        "ci_low_95": float(np.nanpercentile(samples, 2.5)),
                        "ci_high_95": float(np.nanpercentile(samples, 97.5)),
                        "bootstrap_replicates": replicates,
                        "bootstrap_unit": "source_recording_stratified_by_class",
                        "source_recording_count": len(source),
                        **run_meta,
                    }
                )
    metrics_df = pd.DataFrame(metric_rows)
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    metrics_df.to_csv(paths.tables / "source_recording_metrics_by_seed.csv", index=False)
    if not metrics_df.empty:
        summary_groups = [
            "split_name",
            "model",
            "setting",
            "run_tag",
            "sequence_length",
            "scalar_feature_count",
            "setting_label",
            "evidence_status",
            "aggregation",
        ]
        summary = metrics_df.groupby(summary_groups, dropna=False)[
            ["accuracy", "macro_f1", "worst_class_recall"]
        ].agg(["mean", "std"]).reset_index()
        summary.columns = ["_".join(str(value) for value in column if value) for column in summary.columns.to_flat_index()]
        summary.to_csv(paths.tables / "source_recording_metrics_summary.csv", index=False)
    bootstrap_df.to_csv(paths.tables / "recording_cluster_bootstrap.csv", index=False)
    return metrics_df, bootstrap_df


def write_claim_decision(paths: RunPaths, raw_summary: pd.DataFrame, fusion_summary: pd.DataFrame) -> None:
    raw_cross = raw_summary[(raw_summary["split_name"] == "cross_rpm") & (raw_summary["model"] == "cnn")]
    fusion_cross = fusion_summary[fusion_summary["split_name"] == "cross_rpm"]
    lines = [
        "# Corrected Claim Decision Matrix",
        "",
        f"Run ID: `{paths.run_id}`",
        "",
        "## Evidence Rules",
        "",
        "- Old centered-only neural/fusion numbers are withdrawn from final claims.",
        "- The 26/28-scalar settings remain post-hoc exploratory on the same held-out set.",
        "- Source-recording separation does not establish physical bearing-instance independence.",
        "- The recommended title remains recording-grouped unless specimen IDs become verifiable.",
        "",
        "## Corrected Cross-RPM Raw CNN",
        "",
        dataframe_markdown(raw_cross) if not raw_cross.empty else "No corrected cross-RPM CNN result.",
        "",
        "## Corrected Cross-RPM Fusion-Input Audit",
        "",
        dataframe_markdown(fusion_cross) if not fusion_cross.empty else "No corrected cross-RPM fusion result.",
        "",
        "## Allowed Interpretation",
        "",
        "Only the tested compact architecture and frozen protocol may be discussed. No result establishes broad multisource-fusion superiority, specimen independence, or non-MCC5 speed-transfer generalization.",
    ]
    (paths.docs / f"{paths.run_id}_claim_decision_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", default=now_run_id())
    parser.add_argument("--mode", choices=["audit", "full", "postprocess"], default="full")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--settings", default=",".join(DEFAULT_SETTINGS))
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--focused_seeds", default="42,43,44,45,46")
    parser.add_argument("--validation_seed", type=int, default=42)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--fixed_length", type=int, default=8192)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--bootstrap_replicates", type=int, default=2000)
    args = parser.parse_args(argv)

    start = time.time()
    paths = make_paths(args.run_id)
    env = require_cuda()
    env["arguments"] = vars(args)
    write_preflight(paths, env)
    windows, features, original_splits = load_formal_tables()
    frozen_splits, split_audit = freeze_grouped_splits(
        paths,
        windows,
        original_splits,
        validation_seed=args.validation_seed,
        validation_fraction=args.validation_fraction,
    )
    if split_audit.filter(regex="_overlap$").to_numpy().sum() != 0 or split_audit["fallback_used"].any():
        raise RuntimeError("Grouped internal-validation gate failed.")

    normalization_rows: list[dict[str, object]] = []
    models = parse_list(args.models)
    settings = parse_list(args.settings)
    seeds = parse_int_list(args.seeds)
    focused_seeds = parse_int_list(args.focused_seeds)

    if args.mode == "audit":
        raw_arrays = prepare_raw_arrays(windows, frozen_splits["cross_rpm"], seed=42, fixed_length=args.fixed_length)
        normalization_rows.extend(_normalization_rows(raw_arrays, "cross_rpm", f"raw_{args.fixed_length}"))
        save_normalization_artifacts(paths, raw_arrays, "cross_rpm", f"raw_{args.fixed_length}")
        del raw_arrays
        for setting in settings:
            arrays = prepare_fusion_arrays(windows, features, frozen_splits["cross_rpm"], 42, args.fixed_length, setting)
            normalization_rows.extend(_normalization_rows(arrays, "cross_rpm", setting))
            save_normalization_artifacts(paths, arrays, "cross_rpm", setting)
            del arrays
            gc.collect()
        audit = merge_normalization_audit(paths, normalization_rows)
        if not audit["status"].eq("pass").all():
            raise RuntimeError("Normalization audit failed.")
    elif args.mode == "full":
        run_raw_matrix(
            paths,
            windows,
            frozen_splits,
            models=models,
            seeds=seeds,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            fixed_length=args.fixed_length,
            normalization_rows=normalization_rows,
        )
        run_length_sensitivity(
            paths,
            windows,
            frozen_splits["cross_rpm"],
            seeds=seeds,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            normalization_rows=normalization_rows,
        )
        run_fusion_matrix(
            paths,
            windows,
            features,
            frozen_splits,
            settings=settings,
            seeds=seeds,
            focused_seeds=focused_seeds,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            fixed_length=args.fixed_length,
            normalization_rows=normalization_rows,
        )
        audit = merge_normalization_audit(paths, normalization_rows)
        if not audit["status"].eq("pass").all():
            raise RuntimeError("Normalization audit failed after formal reruns.")
        raw_summary, fusion_summary = build_summaries(paths)
        source_metrics_and_bootstrap(paths, replicates=args.bootstrap_replicates, seed=20260710)
        write_claim_decision(paths, raw_summary, fusion_summary)
    else:
        raw_summary, fusion_summary = build_summaries(paths)
        source_metrics_and_bootstrap(paths, replicates=args.bootstrap_replicates, seed=20260710)
        write_claim_decision(paths, raw_summary, fusion_summary)

    status = {
        "run_id": paths.run_id,
        "mode": args.mode,
        "status": "completed",
        "elapsed_sec": time.time() - start,
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "arguments": vars(args),
    }
    (paths.logs / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
