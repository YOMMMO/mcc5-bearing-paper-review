"""Formal GPU DL and fusion run for MCC5.

This runner is separate from the classical formal pipeline. It never
overwrites the formal classical evidence from
``formal_20260706_215358`` and writes GPU/DL artifacts into a fresh
``results/formal_runs/<run_id>`` directory plus the explicit summary CSVs
requested by the formal DL/fusion prompt.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.experiments.exp03_dl_baselines import _make_model
from src.models.fusion_net import OrderNormalizedMultisourceFusionNet, count_parameters
from src.signal.preprocessing import resample_or_pad
from src.utils.io import ensure_dir
from src.utils.metrics import classification_metrics, save_confusion_matrix, save_per_class_metrics
from src.utils.plotting import save_bar_plot, save_training_curve
from src.utils.seed import set_global_seed
from src.utils.tables import read_table
from src.utils.train_only_normalization import (
    standardize_scalar_frames,
    standardize_sequence_splits,
)


WINDOWS_PATH = Path("data/processed/windows/mcc5_windows_formal.parquet")
FEATURES_PATH = Path("data/processed/features/mcc5_features_formal.parquet")
SPLIT_PATHS = {
    "source_file": Path("data/processed/splits/mcc5_formal_source_file_split.csv"),
    "cross_condition": Path("data/processed/splits/mcc5_formal_cross_condition_split.csv"),
    "cross_load": Path("data/processed/splits/mcc5_formal_cross_load_split.csv"),
    "cross_rpm": Path("data/processed/splits/mcc5_formal_cross_rpm_split.csv"),
}
CLASSICAL_RESULTS = Path("results/tables/mcc5_formal_ml_baselines_all_splits.csv")
RAW_MODELS = ["cnn", "tcn", "transformer"]
DEFAULT_SEEDS = [42, 43, 44]
DEFAULT_ABLATION_SEEDS = [42]
ID_COLS = {
    "window_id",
    "source_file",
    "sample_id",
    "label_group",
    "label_raw",
    "condition_type",
    "npz_path",
    "split",
    "role",
    "split_type",
    "reason",
}
FULL_ENGINEERED_EXCLUDE = {
    "start_index",
    "end_index",
    "start_time",
    "end_time",
    "has_vibration",
    "has_current",
    "has_torque",
    "has_key_phase",
}
ABLATION_SETTINGS = [
    "vibration_only",
    "current_only",
    "scalar_physical_features_only",
    "vibration_current",
    "vibration_current_rpm_load_only",
    "vibration_current_auxiliary_only",
    "vibration_current_rpm_load",
    "vibration_current_order_features",
    "vibration_current_order_features_rpm_load",
    "full_multisource_fusion",
]


@dataclass
class RunPaths:
    run_id: str
    root: Path
    tables: Path
    figures: Path
    logs: Path
    checkpoints: Path
    docs: Path


def now_run_id() -> str:
    return datetime.now().strftime("formal_dl_fusion_%Y%m%d_%H%M%S")


def make_run_paths(run_id: str) -> RunPaths:
    root = ensure_dir(Path("results/formal_runs") / run_id)
    paths = RunPaths(
        run_id=run_id,
        root=root,
        tables=ensure_dir(root / "tables"),
        figures=ensure_dir(root / "figures"),
        logs=ensure_dir(root / "logs"),
        checkpoints=ensure_dir(root / "checkpoints"),
        docs=ensure_dir("docs/formal_runs"),
    )
    Path("results/formal_runs/latest_dl_fusion_run_id.txt").write_text(run_id + "\n", encoding="utf-8")
    return paths


def require_cuda() -> dict[str, str | bool | int]:
    info = {
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda",
    }
    for key, value in info.items():
        print(f"{key}= {value}")
    if not info["cuda_available"]:
        raise RuntimeError("CUDA is unavailable; formal DL/fusion must not run.")
    return info


def write_environment_report(paths: RunPaths, env_info: dict[str, str | bool | int]) -> None:
    lines = [
        f"# GPU/PyTorch Environment Report",
        "",
        f"Run ID: `{paths.run_id}`",
        f"Time: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Environment",
        "",
        f"- Python: `{env_info['python']}`",
        f"- Torch: `{env_info['torch']}`",
        f"- Torch CUDA: `{env_info['torch_cuda']}`",
        f"- CUDA available: `{env_info['cuda_available']}`",
        f"- Device count: `{env_info['device_count']}`",
        f"- Device: `{env_info['device']}`",
        "",
        "Gate status: passed.",
    ]
    (paths.docs / f"{paths.run_id}_gpu_environment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_inputs() -> None:
    missing = [p for p in [WINDOWS_PATH, FEATURES_PATH, *SPLIT_PATHS.values()] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing formal input files: " + ", ".join(str(p) for p in missing))


def load_formal_tables() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    windows = read_table(WINDOWS_PATH)
    features = read_table(FEATURES_PATH)
    splits = {name: pd.read_csv(path) for name, path in SPLIT_PATHS.items()}
    return windows, features, splits


def derive_roles(windows: pd.DataFrame, split: pd.DataFrame, seed: int) -> pd.DataFrame:
    df = windows[["window_id", "source_file", "label_group"]].merge(split[["window_id", "split"]], on="window_id", how="inner")
    df["role"] = df["split"].astype(str)
    if (df["role"] == "val").any():
        return df[["window_id", "role"]]

    train = df[df["role"] == "train"].copy()
    if train.empty:
        return df[["window_id", "role"]]

    rng = np.random.default_rng(seed)
    val_sources: set[str] = set()
    groups = train.groupby("source_file", sort=False)["label_group"].first().reset_index()
    for _label, label_groups in groups.groupby("label_group", sort=True):
        sources = label_groups["source_file"].astype(str).to_numpy()
        if len(sources) <= 1:
            continue
        n_val = max(1, int(round(0.2 * len(sources))))
        n_val = min(n_val, len(sources) - 1)
        chosen = rng.choice(sources, size=n_val, replace=False)
        val_sources.update(str(x) for x in chosen)
    if not val_sources:
        by_label = train.groupby("label_group", group_keys=False)
        sampled = by_label.apply(lambda x: x.sample(max(1, int(round(0.2 * len(x)))), random_state=seed))
        val_ids = set(sampled["window_id"])
        df.loc[df["window_id"].isin(val_ids) & (df["role"] == "train"), "role"] = "val"
    else:
        df.loc[df["source_file"].astype(str).isin(val_sources) & (df["role"] == "train"), "role"] = "val"
    return df[["window_id", "role"]]


def audit_inputs_and_splits(paths: RunPaths, windows: pd.DataFrame, features: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> None:
    input_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []

    for split_name, split in splits.items():
        roles = derive_roles(windows, split, seed=42)
        merged = windows.merge(split[["window_id", "split"]], on="window_id", how="inner").merge(roles, on="window_id", how="left")
        counts = merged["role"].value_counts().to_dict()
        original_counts = merged["split"].value_counts().to_dict()
        label_counts = merged.groupby(["role", "label_group"]).size().reset_index(name="count")
        for _, row in label_counts.iterrows():
            class_rows.append(
                {
                    "split_name": split_name,
                    "role": row["role"],
                    "label_group": row["label_group"],
                    "count": int(row["count"]),
                }
            )
        role_sets = {role: set(x["window_id"]) for role, x in merged.groupby("role")}
        source_sets = {role: set(x["source_file"].astype(str)) for role, x in merged.groupby("role")}
        window_overlap = 0
        source_overlap = 0
        roles_present = sorted(role_sets)
        for i, left in enumerate(roles_present):
            for right in roles_present[i + 1 :]:
                window_overlap += len(role_sets[left] & role_sets[right])
                source_overlap += len(source_sets[left] & source_sets[right])
        train_labels = set(merged.loc[merged["role"] == "train", "label_group"].astype(str))
        test_labels = set(merged.loc[merged["role"] == "test", "label_group"].astype(str))
        ok = (
            window_overlap == 0
            and source_overlap == 0
            and bool(train_labels)
            and bool(test_labels)
            and train_labels.issuperset(test_labels)
        )
        input_rows.append(
            {
                "split_name": split_name,
                "windows_shape": str(tuple(windows.shape)),
                "features_shape": str(tuple(features.shape)),
                "split_path": str(SPLIT_PATHS[split_name]),
                "original_split_counts_json": json.dumps(original_counts, ensure_ascii=False, sort_keys=True),
                "derived_role_counts_json": json.dumps(counts, ensure_ascii=False, sort_keys=True),
                "class_count_rows": int(len(label_counts)),
            }
        )
        audit_rows.append(
            {
                "split_name": split_name,
                "split_path": str(SPLIT_PATHS[split_name]),
                "source_file_leakage_count": int(source_overlap),
                "window_role_overlap_count": int(window_overlap),
                "train_label_count": int(len(train_labels)),
                "test_label_count": int(len(test_labels)),
                "test_labels_missing_from_train": ",".join(sorted(test_labels - train_labels)),
                "status": "ok" if ok else "fail",
            }
        )

    input_df = pd.DataFrame(input_rows)
    audit_df = pd.DataFrame(audit_rows)
    class_df = pd.DataFrame(class_rows)
    input_df.to_csv(paths.tables / "formal_dl_input_audit.csv", index=False)
    audit_df.to_csv(paths.tables / "formal_dl_split_audit.csv", index=False)
    class_df.to_csv(paths.tables / "formal_dl_split_class_counts.csv", index=False)
    (paths.docs / f"{paths.run_id}_formal_dl_input_audit.md").write_text(
        "\n".join(
            [
                f"# Formal DL Input Audit",
                "",
                f"Run ID: `{paths.run_id}`",
                "",
                f"- Windows: `{WINDOWS_PATH}` shape `{tuple(windows.shape)}`",
                f"- Features: `{FEATURES_PATH}` shape `{tuple(features.shape)}`",
                "",
                "## Split Audit",
                "",
                "```text",
                audit_df.to_string(index=False),
                "```",
                "",
                "## Input Counts",
                "",
                "```text",
                input_df.to_string(index=False),
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if not audit_df["status"].eq("ok").all():
        raise RuntimeError(f"Formal split audit failed:\n{audit_df.to_string(index=False)}")


def _load_npz(path: str | Path, input_mode: str, fixed_length: int) -> np.ndarray:
    p = Path(str(path))
    with np.load(p, allow_pickle=False) as data:
        vib = data.get("vibration", np.empty((0, 0), dtype=np.float32))
        cur = data.get("current", np.empty((0, 0), dtype=np.float32))
    vib = np.asarray(vib, dtype=np.float32)
    cur = np.asarray(cur, dtype=np.float32)
    if input_mode == "vibration_only":
        arr = vib
    elif input_mode == "current_only":
        arr = cur
    else:
        parts = [x for x in [vib, cur] if x.size]
        arr = np.concatenate(parts, axis=0) if parts else np.empty((0, 0), dtype=np.float32)
    if arr.size == 0:
        arr = np.zeros((1, fixed_length), dtype=np.float32)
    return resample_or_pad(arr, fixed_length).astype(np.float32)


def stack_rows(df: pd.DataFrame, input_mode: str, fixed_length: int, channels: int | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    arrays: list[np.ndarray] = []
    rows: list[pd.Series] = []
    for _, row in df.iterrows():
        path = Path(str(row.get("npz_path", "")))
        if not path.exists():
            continue
        arr = _load_npz(path, input_mode, fixed_length)
        if channels is None:
            channels = arr.shape[0]
        if arr.shape[0] < channels:
            pad = np.zeros((channels - arr.shape[0], fixed_length), dtype=np.float32)
            arr = np.vstack([arr, pad])
        elif arr.shape[0] > channels:
            arr = arr[:channels]
        arrays.append(arr)
        rows.append(row)
    if not arrays:
        return np.empty((0, channels or 1, fixed_length), dtype=np.float32), pd.DataFrame()
    return np.stack(arrays).astype(np.float32), pd.DataFrame(rows).reset_index(drop=True)


def normalize_sequence(train: np.ndarray, val: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_val, x_test, stats = standardize_sequence_splits(train, val, test)
    return x_train, x_val, x_test, stats.mean, stats.scale


def prepare_role_frames(windows: pd.DataFrame, split: pd.DataFrame, seed: int) -> pd.DataFrame:
    roles = derive_roles(windows, split, seed=seed)
    df = windows.merge(roles, on="window_id", how="inner")
    return df


def prepare_raw_arrays(windows: pd.DataFrame, split: pd.DataFrame, seed: int, fixed_length: int) -> dict[str, object]:
    df = prepare_role_frames(windows, split, seed)
    train_df = df[df["role"] == "train"].copy()
    val_df = df[df["role"] == "val"].copy()
    test_df = df[df["role"] == "test"].copy()
    labels = sorted(df["label_group"].astype(str).unique())
    label_to_idx = {label: i for i, label in enumerate(labels)}
    x_train, train_df = stack_rows(train_df, "vib_current", fixed_length)
    channels = x_train.shape[1]
    x_val, val_df = stack_rows(val_df, "vib_current", fixed_length, channels)
    x_test, test_df = stack_rows(test_df, "vib_current", fixed_length, channels)
    if x_train.size == 0 or x_val.size == 0 or x_test.size == 0:
        raise RuntimeError("No loadable train/val/test windows for raw DL.")
    x_train, x_val, x_test, mean, std = normalize_sequence(x_train, x_val, x_test)
    y_train = train_df["label_group"].astype(str).map(label_to_idx).to_numpy(dtype=np.int64, copy=True)
    y_val = val_df["label_group"].astype(str).map(label_to_idx).to_numpy(dtype=np.int64, copy=True)
    y_test = test_df["label_group"].astype(str).map(label_to_idx).to_numpy(dtype=np.int64, copy=True)
    return {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "labels": labels,
        "channels": channels,
        "mean": mean,
        "std": std,
        "test_meta": test_df[
            [
                column
                for column in ["window_id", "source_file", "label_group", "condition_type", "rpm_nominal", "load_nm"]
                if column in test_df.columns
            ]
        ].reset_index(drop=True),
    }


def predict_sequence_outputs(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    dataset = TensorDataset(torch.from_numpy(x))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    preds: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    start = time.time()
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(device, non_blocking=True))
            preds.append(logits.argmax(dim=1).detach().cpu().numpy())
            probabilities.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    elapsed = time.time() - start
    return np.concatenate(preds), np.concatenate(probabilities), elapsed / max(1, len(x))


def predict_sequence(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> tuple[np.ndarray, float]:
    preds, _probabilities, elapsed = predict_sequence_outputs(model, x, device, batch_size)
    return preds, elapsed


def save_prediction_table(
    meta: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
    out_path: Path,
    **run_fields: object,
) -> Path:
    """Save test predictions with source-recording provenance."""

    if len(meta) != len(y_true) or len(y_true) != len(y_pred) or probabilities.shape[0] != len(y_true):
        raise ValueError("Prediction rows are not aligned with test metadata.")
    frame = meta.reset_index(drop=True).copy()
    frame["true_index"] = y_true
    frame["predicted_index"] = y_pred
    frame["true_label"] = [labels[int(value)] for value in y_true]
    frame["predicted_label"] = [labels[int(value)] for value in y_pred]
    for index, label in enumerate(labels):
        frame[f"prob__{label}"] = probabilities[:, index]
    for key, value in run_fields.items():
        frame[key] = value
    ensure_dir(out_path.parent)
    frame.to_csv(out_path, index=False)
    return out_path


def train_raw_one(
    arrays: dict[str, object],
    model_name: str,
    split_name: str,
    split_file: str,
    seed: int,
    paths: RunPaths,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    run_tag: str = "",
) -> dict[str, object]:
    set_global_seed(seed)
    device = torch.device("cuda")
    x_train = arrays["x_train"]
    x_val = arrays["x_val"]
    x_test = arrays["x_test"]
    y_train = arrays["y_train"]
    y_val = arrays["y_val"]
    y_test = arrays["y_test"]
    labels = arrays["labels"]
    channels = int(arrays["channels"])
    num_classes = len(labels)

    model = _make_model(model_name, channels, num_classes).to(device)
    counts = np.bincount(y_train, minlength=num_classes)
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    history = {"train_loss": [], "val_macro_f1": []}
    best_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    train_start = time.time()
    epochs_ran = 0
    for epoch in range(epochs):
        epochs_ran = epoch + 1
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(yb)
        history["train_loss"].append(total / max(1, len(y_train)))
        val_pred, _ = predict_sequence(model, x_val, device, batch_size)
        val_metrics = classification_metrics(y_val, val_pred, labels=list(range(num_classes)))
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if patience > 0 and epochs_without_improvement >= patience:
            break

    train_time = time.time() - train_start
    if best_state is not None:
        model.load_state_dict(best_state)
    pred, probabilities, inference_sec = predict_sequence_outputs(model, x_test, device, batch_size)
    metrics = classification_metrics(y_test, pred, labels=list(range(num_classes)))
    tag = f"_{run_tag}" if run_tag else ""
    prefix = f"formal_dl_{model_name}_vib_current_{split_name}{tag}_seed{seed}"
    cm_path = paths.figures / f"{prefix}_confusion_matrix.png"
    curve_path = paths.figures / f"{prefix}_training_curve.png"
    save_confusion_matrix(y_test, pred, cm_path, labels=list(range(num_classes)))
    save_per_class_metrics(y_test, pred, paths.tables / f"{prefix}_per_class_metrics.csv", labels=list(range(num_classes)))
    save_training_curve(history, curve_path)
    prediction_path = save_prediction_table(
        arrays["test_meta"],
        y_test,
        pred,
        probabilities,
        labels,
        ensure_dir(paths.root / "predictions") / f"{prefix}_predictions.csv",
        split_name=split_name,
        model=model_name,
        setting="vib_current",
        seed=seed,
        run_tag=run_tag,
        sequence_length=int(x_train.shape[2]),
    )
    checkpoint_path = paths.checkpoints / f"{prefix}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "labels": labels,
            "mean": arrays["mean"],
            "std": arrays["std"],
            "epochs_ran": epochs_ran,
            "best_val_macro_f1": best_f1,
            "split_name": split_name,
            "seed": seed,
        },
        checkpoint_path,
    )
    torch.cuda.empty_cache()
    return {
        "split_name": split_name,
        "split_file": split_file,
        "seed": seed,
        "model": model_name,
        "input_mode": "vib_current",
        "run_tag": run_tag,
        "sequence_length": int(x_train.shape[2]),
        **metrics,
        "epochs_ran": epochs_ran,
        "best_val_macro_f1": best_f1,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "train_time_sec": train_time,
        "inference_time_ms_per_sample": inference_sec * 1000.0,
        "prediction_path": str(prediction_path),
        "checkpoint": str(checkpoint_path),
    }


def summarize_results(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metric_cols = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "worst_class_recall",
        "best_val_macro_f1",
        "train_time_sec",
        "inference_time_ms_per_sample",
    ]
    summary = df.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join([str(x) for x in col if x]) for col in summary.columns.to_flat_index()]
    return summary


def run_raw_baselines(
    paths: RunPaths,
    windows: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    seeds: list[int],
    epochs: int,
    patience: int,
    batch_size: int,
    fixed_length: int,
    learning_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for split_name, split in splits.items():
        print(f"[raw] loading arrays for {split_name}")
        arrays = prepare_raw_arrays(windows, split, seed=42, fixed_length=fixed_length)
        for model_name in RAW_MODELS:
            for seed in seeds:
                print(f"[raw] split={split_name} model={model_name} seed={seed}")
                rows.append(
                    train_raw_one(
                        arrays=arrays,
                        model_name=model_name,
                        split_name=split_name,
                        split_file=SPLIT_PATHS[split_name].name,
                        seed=seed,
                        paths=paths,
                        epochs=epochs,
                        patience=patience,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                    )
                )
        del arrays
        torch.cuda.empty_cache()
    by_seed = pd.DataFrame(rows)
    summary = summarize_results(by_seed, ["split_name", "split_file", "model", "input_mode"])
    by_seed.to_csv(paths.tables / "formal_dl_baselines_by_seed.csv", index=False)
    summary.to_csv(paths.tables / "formal_dl_baselines_summary.csv", index=False)
    by_seed.to_csv("results/tables/mcc5_formal_dl_baselines_by_seed.csv", index=False)
    summary.to_csv("results/tables/mcc5_formal_dl_baselines_summary.csv", index=False)
    return by_seed, summary


def feature_augmented_frames(windows: pd.DataFrame, features: pd.DataFrame, split: pd.DataFrame, seed: int) -> pd.DataFrame:
    drop_meta = ["source_file", "sample_id", "label_group", "label_raw", "condition_type", "rpm_nominal", "load_nm"]
    feature_cols = [c for c in features.columns if c not in drop_meta]
    df = windows.merge(features[feature_cols], on="window_id", how="inner")
    roles = derive_roles(windows, split, seed)
    return df.merge(roles, on="window_id", how="inner")


def select_scalar_columns(df: pd.DataFrame, setting: str = "full_multisource_fusion") -> list[str]:
    numeric = df.drop(columns=[c for c in ID_COLS if c in df.columns], errors="ignore").select_dtypes(include=[np.number, "bool"])
    columns = list(numeric.columns)
    if setting in {"vibration_only", "current_only", "vibration_current"}:
        return []
    if setting == "vibration_current_rpm_load_only":
        return [c for c in ["rpm_nominal", "load_nm"] if c in columns]
    if setting == "vibration_current_auxiliary_only":
        patterns = ["torque_", "key_phase_", "has_torque", "has_key_phase"]
        return [c for c in columns if any(pattern in c for pattern in patterns)]
    if setting in {"scalar_physical_features_only", "vibration_current_rpm_load"}:
        # Legacy setting name: this includes auxiliary
        # torque/key-phase summaries in addition to RPM and load.
        patterns = ["rpm_nominal", "load_nm", "torque_", "key_phase_", "has_torque", "has_key_phase"]
        return [c for c in columns if any(p in c for p in patterns)]
    if setting == "vibration_current_order_features":
        return [c for c in columns if "order" in c]
    if setting == "vibration_current_order_features_rpm_load":
        return [c for c in columns if "order" in c or c in {"rpm_nominal", "load_nm"}]
    if setting == "full_engineered_features_254":
        return [c for c in columns if c not in FULL_ENGINEERED_EXCLUDE]
    return columns


def modalities_for_setting(setting: str) -> tuple[bool, bool]:
    if setting == "vibration_only":
        return True, False
    if setting in {"current_only", "scalar_physical_features_only"}:
        return False, setting == "current_only"
    full_settings = {"full_multisource_fusion", "full_engineered_features_254"}
    return "vibration" in setting or setting in full_settings, "current" in setting or setting in full_settings


def scalar_matrix(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, pd.Series]:
    if not columns:
        return (
            np.zeros((len(train_df), 1), dtype=np.float32),
            np.zeros((len(val_df), 1), dtype=np.float32),
            np.zeros((len(test_df), 1), dtype=np.float32),
            [],
            np.zeros((1, 1), dtype=np.float32),
            np.ones((1, 1), dtype=np.float32),
            pd.Series(dtype=float),
        )
    x_train, x_val, x_test, stats = standardize_scalar_frames(train_df, val_df, test_df, columns)
    return x_train, x_val, x_test, columns, stats.mean, stats.scale, stats.median


def prepare_fusion_arrays(
    windows: pd.DataFrame,
    features: pd.DataFrame,
    split: pd.DataFrame,
    seed: int,
    fixed_length: int,
    setting: str,
) -> dict[str, object]:
    df = feature_augmented_frames(windows, features, split, seed)
    train_df = df[df["role"] == "train"].copy()
    val_df = df[df["role"] == "val"].copy()
    test_df = df[df["role"] == "test"].copy()
    labels = sorted(df["label_group"].astype(str).unique())
    label_to_idx = {label: i for i, label in enumerate(labels)}
    use_vib, use_cur = modalities_for_setting(setting)

    if use_vib:
        vib_train, train_df = stack_rows(train_df, "vibration_only", fixed_length)
        vib_ch = vib_train.shape[1]
        vib_val, val_df = stack_rows(val_df, "vibration_only", fixed_length, vib_ch)
        vib_test, test_df = stack_rows(test_df, "vibration_only", fixed_length, vib_ch)
    else:
        vib_train = np.zeros((len(train_df), 1, fixed_length), dtype=np.float32)
        vib_val = np.zeros((len(val_df), 1, fixed_length), dtype=np.float32)
        vib_test = np.zeros((len(test_df), 1, fixed_length), dtype=np.float32)
        vib_ch = 1
    if use_cur:
        cur_train, train_df_cur = stack_rows(train_df, "current_only", fixed_length)
        cur_ch = cur_train.shape[1]
        cur_val, _ = stack_rows(val_df, "current_only", fixed_length, cur_ch)
        cur_test, _ = stack_rows(test_df, "current_only", fixed_length, cur_ch)
        train_df = train_df_cur.reset_index(drop=True)
    else:
        cur_train = np.zeros((len(train_df), 1, fixed_length), dtype=np.float32)
        cur_val = np.zeros((len(val_df), 1, fixed_length), dtype=np.float32)
        cur_test = np.zeros((len(test_df), 1, fixed_length), dtype=np.float32)
        cur_ch = 1
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError(f"No loadable train/val/test windows for fusion setting {setting}.")
    vib_train, vib_val, vib_test, vib_mean, vib_std = normalize_sequence(vib_train, vib_val, vib_test)
    cur_train, cur_val, cur_test, cur_mean, cur_std = normalize_sequence(cur_train, cur_val, cur_test)
    scalar_cols = select_scalar_columns(df, setting)
    scal_train, scal_val, scal_test, scalar_cols, scal_mean, scal_std, scal_median = scalar_matrix(train_df, val_df, test_df, scalar_cols)
    y_train = train_df["label_group"].astype(str).map(label_to_idx).to_numpy(dtype=np.int64, copy=True)
    y_val = val_df["label_group"].astype(str).map(label_to_idx).to_numpy(dtype=np.int64, copy=True)
    y_test = test_df["label_group"].astype(str).map(label_to_idx).to_numpy(dtype=np.int64, copy=True)
    return {
        "vib_train": vib_train,
        "vib_val": vib_val,
        "vib_test": vib_test,
        "cur_train": cur_train,
        "cur_val": cur_val,
        "cur_test": cur_test,
        "scal_train": scal_train,
        "scal_val": scal_val,
        "scal_test": scal_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "labels": labels,
        "vib_ch": vib_ch,
        "cur_ch": cur_ch,
        "scalar_cols": scalar_cols,
        "vib_mean": vib_mean,
        "vib_std": vib_std,
        "cur_mean": cur_mean,
        "cur_std": cur_std,
        "scal_mean": scal_mean,
        "scal_std": scal_std,
        "scal_median": scal_median.to_dict(),
        "setting": setting,
        "test_meta": test_df[
            [
                column
                for column in ["window_id", "source_file", "label_group", "condition_type", "rpm_nominal", "load_nm"]
                if column in test_df.columns
            ]
        ].reset_index(drop=True),
    }


def predict_fusion_outputs(
    model: nn.Module,
    arrays: dict[str, object],
    role: str,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    vib = arrays[f"vib_{role}"]
    cur = arrays[f"cur_{role}"]
    scal = arrays[f"scal_{role}"]
    dataset = TensorDataset(torch.from_numpy(vib), torch.from_numpy(cur), torch.from_numpy(scal))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    preds: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    start = time.time()
    model.eval()
    with torch.no_grad():
        for vb, cb, sb in loader:
            logits = model(vb.to(device, non_blocking=True), cb.to(device, non_blocking=True), sb.to(device, non_blocking=True))
            preds.append(logits.argmax(dim=1).detach().cpu().numpy())
            probabilities.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    elapsed = time.time() - start
    return np.concatenate(preds), np.concatenate(probabilities), elapsed / max(1, len(vib))


def predict_fusion(model: nn.Module, arrays: dict[str, object], role: str, device: torch.device, batch_size: int) -> tuple[np.ndarray, float]:
    preds, _probabilities, elapsed = predict_fusion_outputs(model, arrays, role, device, batch_size)
    return preds, elapsed


def train_fusion_one(
    arrays: dict[str, object],
    split_name: str,
    split_file: str,
    seed: int,
    paths: RunPaths,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    setting: str,
    prefix_kind: str,
) -> dict[str, object]:
    set_global_seed(seed)
    device = torch.device("cuda")
    y_train = arrays["y_train"]
    y_val = arrays["y_val"]
    y_test = arrays["y_test"]
    labels = arrays["labels"]
    num_classes = len(labels)
    model = OrderNormalizedMultisourceFusionNet(
        int(arrays["vib_ch"]),
        int(arrays["cur_ch"]),
        int(arrays["scal_train"].shape[1]),
        num_classes,
    ).to(device)
    counts = np.bincount(y_train, minlength=num_classes)
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    dataset = TensorDataset(
        torch.from_numpy(arrays["vib_train"]),
        torch.from_numpy(arrays["cur_train"]),
        torch.from_numpy(arrays["scal_train"]),
        torch.from_numpy(y_train),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    history = {"train_loss": [], "val_macro_f1": []}
    best_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    train_start = time.time()
    epochs_ran = 0
    for epoch in range(epochs):
        epochs_ran = epoch + 1
        model.train()
        total = 0.0
        for vb, cb, sb, yb in loader:
            vb = vb.to(device, non_blocking=True)
            cb = cb.to(device, non_blocking=True)
            sb = sb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(vb, cb, sb), yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(yb)
        history["train_loss"].append(total / max(1, len(y_train)))
        val_pred, _ = predict_fusion(model, arrays, "val", device, batch_size)
        val_metrics = classification_metrics(y_val, val_pred, labels=list(range(num_classes)))
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if patience > 0 and epochs_without_improvement >= patience:
            break
    train_time = time.time() - train_start
    if best_state is not None:
        model.load_state_dict(best_state)
    pred, probabilities, inference_sec = predict_fusion_outputs(model, arrays, "test", device, batch_size)
    metrics = classification_metrics(y_test, pred, labels=list(range(num_classes)))
    prefix = f"{prefix_kind}_{setting}_{split_name}_seed{seed}"
    cm_path = paths.figures / f"{prefix}_confusion_matrix.png"
    curve_path = paths.figures / f"{prefix}_training_curve.png"
    save_confusion_matrix(y_test, pred, cm_path, labels=list(range(num_classes)))
    save_per_class_metrics(y_test, pred, paths.tables / f"{prefix}_per_class_metrics.csv", labels=list(range(num_classes)))
    save_training_curve(history, curve_path)
    prediction_path = save_prediction_table(
        arrays["test_meta"],
        y_test,
        pred,
        probabilities,
        labels,
        ensure_dir(paths.root / "predictions") / f"{prefix}_predictions.csv",
        split_name=split_name,
        model="OrderNormalizedMultisourceFusionNet",
        setting=setting,
        seed=seed,
    )
    checkpoint_path = paths.checkpoints / f"{prefix}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "labels": labels,
            "scalar_cols": arrays["scalar_cols"],
            "normalization": {
                "vib_mean": arrays["vib_mean"],
                "vib_std": arrays["vib_std"],
                "cur_mean": arrays["cur_mean"],
                "cur_std": arrays["cur_std"],
                "scal_mean": arrays["scal_mean"],
                "scal_std": arrays["scal_std"],
                "scal_median": arrays["scal_median"],
            },
            "epochs_ran": epochs_ran,
            "best_val_macro_f1": best_f1,
            "split_name": split_name,
            "seed": seed,
            "setting": setting,
            "parameter_count": count_parameters(model),
        },
        checkpoint_path,
    )
    torch.cuda.empty_cache()
    return {
        "split_name": split_name,
        "split_file": split_file,
        "seed": seed,
        "model": "OrderNormalizedMultisourceFusionNet",
        "setting": setting,
        **metrics,
        "epochs_ran": epochs_ran,
        "best_val_macro_f1": best_f1,
        "parameter_count": count_parameters(model),
        "scalar_feature_count": len(arrays["scalar_cols"]),
        "train_time_sec": train_time,
        "inference_time_ms_per_sample": inference_sec * 1000.0,
        "prediction_path": str(prediction_path),
        "checkpoint": str(checkpoint_path),
    }


def run_fusion(
    paths: RunPaths,
    windows: pd.DataFrame,
    features: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    seeds: list[int],
    epochs: int,
    patience: int,
    batch_size: int,
    fixed_length: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for split_name, split in splits.items():
        for seed in seeds:
            print(f"[fusion] split={split_name} seed={seed}")
            arrays = prepare_fusion_arrays(windows, features, split, seed, fixed_length, "full_multisource_fusion")
            rows.append(
                train_fusion_one(
                    arrays,
                    split_name,
                    SPLIT_PATHS[split_name].name,
                    seed,
                    paths,
                    epochs,
                    patience,
                    batch_size,
                    learning_rate,
                    weight_decay,
                    "full_multisource_fusion",
                    "formal_fusion",
                )
            )
            del arrays
            torch.cuda.empty_cache()
    by_seed = pd.DataFrame(rows)
    summary = summarize_results(by_seed, ["split_name", "split_file", "model", "setting"])
    by_seed.to_csv(paths.tables / "formal_fusion_by_seed.csv", index=False)
    summary.to_csv(paths.tables / "formal_fusion_summary.csv", index=False)
    by_seed.to_csv("results/tables/mcc5_formal_fusion_by_seed.csv", index=False)
    summary.to_csv("results/tables/mcc5_formal_fusion_summary.csv", index=False)
    write_model_summary(paths, by_seed)
    return by_seed, summary


def write_model_summary(paths: RunPaths, fusion_rows: pd.DataFrame) -> None:
    if fusion_rows.empty:
        return
    row = fusion_rows.iloc[0]
    lines = [
        "OrderNormalizedMultisourceFusionNet formal summary",
        f"run_id: {paths.run_id}",
        f"parameter_count: {row.get('parameter_count')}",
        f"scalar_feature_count: {row.get('scalar_feature_count')}",
        f"mean_inference_time_ms_per_sample: {fusion_rows['inference_time_ms_per_sample'].mean()}",
    ]
    (paths.logs / "model_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ablation(
    paths: RunPaths,
    windows: pd.DataFrame,
    features: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    seeds: list[int],
    epochs: int,
    patience: int,
    batch_size: int,
    fixed_length: int,
    learning_rate: float,
    weight_decay: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name, split in splits.items():
        for setting in ABLATION_SETTINGS:
            for seed in seeds:
                print(f"[ablation] split={split_name} setting={setting} seed={seed}")
                arrays = prepare_fusion_arrays(windows, features, split, seed, fixed_length, setting)
                rows.append(
                    train_fusion_one(
                        arrays,
                        split_name,
                        SPLIT_PATHS[split_name].name,
                        seed,
                        paths,
                        epochs,
                        patience,
                        batch_size,
                        learning_rate,
                        weight_decay,
                        setting,
                        "formal_fusion_ablation",
                    )
                )
                del arrays
                torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(paths.tables / "formal_fusion_ablation.csv", index=False)
    df.to_csv("results/tables/mcc5_formal_fusion_ablation.csv", index=False)
    if not df.empty:
        avg = df.groupby("setting", as_index=False)[["macro_f1", "worst_class_recall"]].mean()
        save_bar_plot(avg["setting"], avg["macro_f1"], paths.figures / "formal_fusion_ablation_macro_f1.png", ylabel="Macro F1")
        save_bar_plot(avg["setting"], avg["worst_class_recall"], paths.figures / "formal_fusion_ablation_worst_recall.png", ylabel="Worst Recall")
    return df


def best_by_split(summary: pd.DataFrame, name_col: str) -> pd.DataFrame:
    idx = summary.groupby("split_file")["macro_f1_mean"].idxmax()
    return summary.loc[idx, ["split_file", name_col, "macro_f1_mean", "macro_f1_std", "worst_class_recall_mean", "worst_class_recall_std"]].copy()


def compare_with_classical(paths: RunPaths, raw_summary: pd.DataFrame, fusion_summary: pd.DataFrame) -> pd.DataFrame:
    classical = pd.read_csv(CLASSICAL_RESULTS)
    classical_best = classical.loc[classical.groupby("split_file")["macro_f1"].idxmax()].copy()
    raw_best = best_by_split(raw_summary, "model").rename(
        columns={
            "model": "best_raw_dl_model",
            "macro_f1_mean": "best_raw_dl_macro_f1_mean",
            "macro_f1_std": "best_raw_dl_macro_f1_std",
            "worst_class_recall_mean": "best_raw_dl_worst_class_recall_mean",
            "worst_class_recall_std": "best_raw_dl_worst_class_recall_std",
        }
    )
    fusion = fusion_summary.copy().rename(
        columns={
            "macro_f1_mean": "fusion_macro_f1_mean",
            "macro_f1_std": "fusion_macro_f1_std",
            "worst_class_recall_mean": "fusion_worst_class_recall_mean",
            "worst_class_recall_std": "fusion_worst_class_recall_std",
        }
    )
    rows = []
    for _, c in classical_best.iterrows():
        split_file = c["split_file"]
        raw = raw_best[raw_best["split_file"] == split_file]
        fus = fusion[fusion["split_file"] == split_file]
        if raw.empty or fus.empty:
            continue
        raw_row = raw.iloc[0]
        fus_row = fus.iloc[0]
        rows.append(
            {
                "split_file": split_file,
                "best_classical_model": c["model"],
                "best_classical_macro_f1": c["macro_f1"],
                "best_classical_worst_class_recall": c["worst_class_recall"],
                "best_raw_dl_model": raw_row["best_raw_dl_model"],
                "best_raw_dl_macro_f1_mean": raw_row["best_raw_dl_macro_f1_mean"],
                "best_raw_dl_macro_f1_std": raw_row["best_raw_dl_macro_f1_std"],
                "best_raw_dl_worst_class_recall_mean": raw_row["best_raw_dl_worst_class_recall_mean"],
                "fusion_macro_f1_mean": fus_row["fusion_macro_f1_mean"],
                "fusion_macro_f1_std": fus_row["fusion_macro_f1_std"],
                "fusion_worst_class_recall_mean": fus_row["fusion_worst_class_recall_mean"],
                "fusion_minus_classical_macro_f1": fus_row["fusion_macro_f1_mean"] - c["macro_f1"],
                "fusion_minus_raw_dl_macro_f1": fus_row["fusion_macro_f1_mean"] - raw_row["best_raw_dl_macro_f1_mean"],
                "fusion_worst_class_recall_minus_classical": fus_row["fusion_worst_class_recall_mean"] - c["worst_class_recall"],
                "fusion_worst_class_recall_minus_raw_dl": fus_row["fusion_worst_class_recall_mean"] - raw_row["best_raw_dl_worst_class_recall_mean"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(paths.tables / "formal_fusion_vs_classical.csv", index=False)
    df.to_csv("results/tables/mcc5_formal_fusion_vs_classical.csv", index=False)
    return df


def decide_title(paths: RunPaths, fusion_by_seed: pd.DataFrame, ablation: pd.DataFrame, comparison: pd.DataFrame) -> tuple[str, str, list[str], list[str]]:
    complete_splits = set(fusion_by_seed["split_file"].unique()) == {p.name for p in SPLIT_PATHS.values()}
    split_audit = pd.read_csv(paths.tables / "formal_dl_split_audit.csv")
    audit_ok = split_audit["status"].eq("ok").all()
    cross_rpm = comparison[comparison["split_file"] == "mcc5_formal_cross_rpm_split.csv"]
    other = comparison[comparison["split_file"].isin(["mcc5_formal_source_file_split.csv", "mcc5_formal_cross_condition_split.csv", "mcc5_formal_cross_load_split.csv"])]
    cross_rpm_help = False
    if not cross_rpm.empty:
        row = cross_rpm.iloc[0]
        cross_rpm_help = (row["fusion_minus_classical_macro_f1"] >= 0.03) or (row["fusion_worst_class_recall_minus_classical"] >= 0.05)
    not_worse_other = (not other.empty) and (other["fusion_minus_classical_macro_f1"] >= -0.02).all()
    seed_stable = fusion_by_seed.groupby("split_file")["macro_f1"].std().fillna(0).le(0.03).all()
    full_beats_single = False
    if not ablation.empty:
        avg = ablation.groupby("setting")["macro_f1"].mean()
        if "full_multisource_fusion" in avg.index:
            single = avg.reindex(["vibration_only", "current_only", "scalar_physical_features_only"]).dropna()
            if not single.empty:
                full_beats_single = (avg["full_multisource_fusion"] - single.max()) >= 0.03
    fusion_beats_raw = (not comparison.empty) and (comparison["fusion_minus_raw_dl_macro_f1"] > 0).all()
    helps_cross_rpm = (not cross_rpm.empty) and (
        (cross_rpm.iloc[0]["fusion_minus_classical_macro_f1"] > 0)
        or (cross_rpm.iloc[0]["fusion_worst_class_recall_minus_classical"] > 0)
    )
    below_classical_most = (not comparison.empty) and (comparison["fusion_minus_classical_macro_f1"] < -0.02).sum() >= 3

    if complete_splits and audit_ok and cross_rpm_help and not_worse_other and full_beats_single and seed_stable:
        decision = "A"
        title = "Cross-Condition Bearing Fault Diagnosis of Electric-Drive Systems Based on Order Normalization and Multisource Signal Fusion"
    elif fusion_beats_raw and full_beats_single and helps_cross_rpm:
        decision = "B"
        title = "Order-Normalized Multisource Feature Fusion for Cross-Condition Bearing Fault Diagnosis in Electric-Drive Systems"
    elif below_classical_most or not helps_cross_rpm or not full_beats_single:
        decision = "C"
        title = "Recording-Grouped Bearing Fault Diagnosis under Directional Operating-Condition Shifts: An Audited MCC5 Electric-Drive Benchmark"
    else:
        decision = "B"
        title = "Order-Normalized Multisource Feature Fusion for Cross-Condition Bearing Fault Diagnosis in Electric-Drive Systems"

    allowed = [
        "CUDA-enabled formal DL/fusion was run on the MCC5 formal windows.",
        "Report validation-selected test metrics as mean/std over seeds for raw DL and full fusion.",
        "Use the selected title decision only with the corresponding evidence thresholds.",
    ]
    forbidden = [
        "Do not claim fusion superiority over classical baselines unless the comparison table supports it.",
        "Do not use previous skipped CPU-only DL/fusion rows as evidence.",
        "Do not use preliminary subset results as final paper results.",
    ]
    lines = [
        f"# Fusion Title Decision",
        "",
        f"Run ID: `{paths.run_id}`",
        "",
        f"Decision: `{decision}`",
        "",
        f"Recommended title: {title}",
        "",
        "## Criteria",
        "",
        f"- Complete all formal splits: `{complete_splits}`",
        f"- Split audit ok: `{audit_ok}`",
        f"- Cross-RPM helps classical threshold: `{cross_rpm_help}`",
        f"- Not worse than classical by >0.02 on other splits: `{not_worse_other}`",
        f"- Full fusion beats single modalities by >=0.03 average macro-F1: `{full_beats_single}`",
        f"- Seed trend stable: `{seed_stable}`",
        "",
        "## Allowed Claims",
        "",
        *[f"- {x}" for x in allowed],
        "",
        "## Forbidden Claims",
        "",
        *[f"- {x}" for x in forbidden],
    ]
    (paths.docs / f"{paths.run_id}_fusion_title_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision, title, allowed, forbidden


def write_final_report(
    paths: RunPaths,
    env_info: dict[str, str | bool | int],
    raw_summary: pd.DataFrame,
    fusion_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    ablation: pd.DataFrame,
    decision: str,
    title: str,
    allowed: list[str],
    forbidden: list[str],
) -> None:
    split_audit = pd.read_csv(paths.tables / "formal_dl_split_audit.csv")
    best_raw = best_by_split(raw_summary, "model") if not raw_summary.empty else pd.DataFrame()
    lines = [
        f"# Formal DL/Fusion Final Report",
        "",
        f"Run ID: `{paths.run_id}`",
        "",
        "## 1. GPU/PyTorch Environment",
        "",
        f"- Python: `{env_info['python']}`",
        f"- Torch: `{env_info['torch']}`",
        f"- Torch CUDA: `{env_info['torch_cuda']}`",
        f"- Device: `{env_info['device']}`",
        "",
        "## 2. Formal DL/Fusion Run Status",
        "",
        "Completed raw DL baselines, full fusion, seed-42 fusion ablation, comparison, and title decision.",
        "",
        "## 3. Split Audit Status",
        "",
        "```text",
        split_audit.to_string(index=False),
        "```",
        "",
        "## 4. Best Raw DL Model Per Split",
        "",
        "```text\n" + best_raw.to_string(index=False) + "\n```" if not best_raw.empty else "No raw DL summary available.",
        "",
        "## 5. Fusion Result Per Split",
        "",
        "```text",
        fusion_summary.to_string(index=False),
        "```",
        "",
        "## 6. Fusion vs Classical Comparison",
        "",
        "```text",
        comparison.to_string(index=False),
        "```",
        "",
        "## 7. Ablation Conclusion",
        "",
        "```text\n"
        + ablation.groupby("setting")[["macro_f1", "worst_class_recall"]].mean().to_string()
        + "\n```"
        if not ablation.empty
        else "No ablation table available.",
        "",
        "## 8. Title Decision",
        "",
        f"Decision `{decision}`: {title}",
        "",
        "## 9. Allowed Claims",
        "",
        *[f"- {x}" for x in allowed],
        "",
        "## 10. Forbidden Claims",
        "",
        *[f"- {x}" for x in forbidden],
        "",
        "## 11. Manuscript Update Status",
        "",
        "Pending manual patch after reviewing the title decision table in this run. No fusion improvement was fabricated.",
        "",
        "## 12. Next Human Actions",
        "",
        "- Review the title decision and manuscript claim calibration.",
        "- Insert verified literature references before submission.",
    ]
    (paths.docs / f"{paths.run_id}_dl_fusion_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_setup_docs(env_info: dict[str, str | bool | int]) -> None:
    audit = Path("docs/formal_runs/gpu_environment_setup_audit.md")
    if not audit.exists():
        return
    note = [
        "",
        "## Resolved GPU Environment Setup",
        "",
        f"Resolved time: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "- Miniconda installed at `C:\\Users\\A\\miniconda3`.",
        "- User PATH was updated with `C:\\Users\\A\\miniconda3\\condabin` only, so `conda` is discoverable while plain `python` is not redirected to Miniconda.",
        "- Conda environment `thesis2-gpu` was created with Python 3.12 from conda-forge.",
        "- Project dependencies were installed from `requirements_no_torch.txt`.",
        "- CUDA PyTorch was installed from `https://download.pytorch.org/whl/cu128`.",
        f"- Verified Python: `{env_info['python']}`",
        f"- Verified Torch: `{env_info['torch']}`",
        f"- Verified Torch CUDA: `{env_info['torch_cuda']}`",
        f"- Verified CUDA available: `{env_info['cuda_available']}`",
        f"- Verified device: `{env_info['device']}`",
    ]
    text = audit.read_text(encoding="utf-8")
    if "## Resolved GPU Environment Setup" not in text:
        audit.write_text(text.rstrip() + "\n" + "\n".join(note) + "\n", encoding="utf-8")


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--ablation_seeds", default="42")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--early_stopping_patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--fixed_length", type=int, default=8192)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    args = parser.parse_args(argv)

    run_id = args.run_id or now_run_id()
    paths = make_run_paths(run_id)
    env_info = require_cuda()
    write_environment_report(paths, env_info)
    update_setup_docs(env_info)
    verify_inputs()
    windows, features, splits = load_formal_tables()
    audit_inputs_and_splits(paths, windows, features, splits)

    seeds = parse_int_list(args.seeds)
    ablation_seeds = parse_int_list(args.ablation_seeds)
    raw_by_seed, raw_summary = run_raw_baselines(
        paths,
        windows,
        splits,
        seeds=seeds,
        epochs=args.epochs,
        patience=args.early_stopping_patience,
        batch_size=args.batch_size,
        fixed_length=args.fixed_length,
        learning_rate=args.learning_rate,
    )
    fusion_by_seed, fusion_summary = run_fusion(
        paths,
        windows,
        features,
        splits,
        seeds=seeds,
        epochs=args.epochs,
        patience=args.early_stopping_patience,
        batch_size=args.batch_size,
        fixed_length=args.fixed_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    ablation = run_ablation(
        paths,
        windows,
        features,
        splits,
        seeds=ablation_seeds,
        epochs=args.epochs,
        patience=args.early_stopping_patience,
        batch_size=args.batch_size,
        fixed_length=args.fixed_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    comparison = compare_with_classical(paths, raw_summary, fusion_summary)
    decision, title, allowed, forbidden = decide_title(paths, fusion_by_seed, ablation, comparison)
    write_final_report(paths, env_info, raw_summary, fusion_summary, comparison, ablation, decision, title, allowed, forbidden)
    (paths.logs / "run_status.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "completed",
                "raw_rows": int(len(raw_by_seed)),
                "fusion_rows": int(len(fusion_by_seed)),
                "ablation_rows": int(len(ablation)),
                "title_decision": decision,
                "title": title,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print("FORMAL_DL_FUSION_RUN_COMPLETED")
    print(f"run_id={run_id}")
    print(f"title_decision={decision}")
    print(f"title={title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
