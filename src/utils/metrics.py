"""Metrics for classification, regression, and confusion matrices."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir
from src.utils.plotting import save_confusion_matrix_plot

try:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_recall_fscore_support,
        precision_score,
        r2_score,
        recall_score,
    )
except Exception:  # pragma: no cover - depends on runtime extras
    accuracy_score = None
    confusion_matrix = None
    f1_score = None
    mean_absolute_error = None
    mean_squared_error = None
    precision_recall_fscore_support = None
    precision_score = None
    r2_score = None
    recall_score = None


def classification_metrics(y_true, y_pred, labels=None) -> dict[str, float]:
    """Return common classification metrics."""
    if len(y_true) == 0:
        return {
            "accuracy": np.nan,
            "macro_precision": np.nan,
            "macro_recall": np.nan,
            "macro_f1": np.nan,
            "weighted_f1": np.nan,
            "worst_class_recall": np.nan,
        }
    if accuracy_score is None:
        return _classification_metrics_numpy(y_true, y_pred, labels)
    unique_labels = labels if labels is not None else sorted(pd.Series(y_true).dropna().unique())
    support = pd.Series(y_true).value_counts()
    recalls = recall_score(
        y_true, y_pred, labels=unique_labels, average=None, zero_division=0
    )
    supported_recalls = [
        rec for label, rec in zip(unique_labels, recalls) if support.get(label, 0) > 0
    ]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "worst_class_recall": float(np.min(supported_recalls)) if supported_recalls else np.nan,
    }


def save_confusion_matrix(y_true, y_pred, out_path, labels=None) -> np.ndarray:
    """Save a confusion matrix plot and return the raw matrix."""
    labels = labels if labels is not None else sorted(pd.Series(list(y_true) + list(y_pred)).unique())
    cm = _confusion_matrix_numpy(y_true, y_pred, labels) if confusion_matrix is None else confusion_matrix(y_true, y_pred, labels=labels)
    save_confusion_matrix_plot(cm, labels, out_path)
    return cm


def save_per_class_metrics(y_true, y_pred, out_path, labels=None) -> pd.DataFrame:
    """Save precision/recall/f1 by class."""
    labels = labels if labels is not None else sorted(pd.Series(list(y_true) + list(y_pred)).unique())
    if precision_recall_fscore_support is None:
        cm = _confusion_matrix_numpy(y_true, y_pred, labels)
        support = cm.sum(axis=1)
        tp = np.diag(cm)
        precision = tp / np.maximum(cm.sum(axis=0), 1)
        recall = tp / np.maximum(support, 1)
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    else:
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
    df = pd.DataFrame(
        {
            "label": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )
    out = Path(out_path)
    ensure_dir(out.parent)
    df.to_csv(out, index=False)
    return df


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return robust regression metrics."""
    if len(y_true) == 0:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan}
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))
    if mean_absolute_error is None:
        ss_res = float(np.sum((y_true_arr - y_pred_arr) ** 2))
        ss_tot = float(np.sum((y_true_arr - np.mean(y_true_arr)) ** 2))
        return {
            "mae": float(np.mean(np.abs(y_true_arr - y_pred_arr))),
            "rmse": rmse,
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        }
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def _confusion_matrix_numpy(y_true, y_pred, labels) -> np.ndarray:
    label_to_idx = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for truth, pred in zip(y_true, y_pred):
        if truth in label_to_idx and pred in label_to_idx:
            cm[label_to_idx[truth], label_to_idx[pred]] += 1
    return cm


def _classification_metrics_numpy(y_true, y_pred, labels=None) -> dict[str, float]:
    labels = labels if labels is not None else sorted(pd.Series(list(y_true) + list(y_pred)).unique())
    cm = _confusion_matrix_numpy(y_true, y_pred, labels)
    total = cm.sum()
    tp = np.diag(cm).astype(float)
    precision = tp / np.maximum(cm.sum(axis=0), 1)
    recall = tp / np.maximum(cm.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    support = cm.sum(axis=1)
    weighted_f1 = float(np.sum(f1 * support) / max(1, support.sum()))
    supported_recall = recall[support > 0]
    return {
        "accuracy": float(tp.sum() / total) if total else np.nan,
        "macro_precision": float(np.mean(precision)) if len(precision) else np.nan,
        "macro_recall": float(np.mean(recall)) if len(recall) else np.nan,
        "macro_f1": float(np.mean(f1)) if len(f1) else np.nan,
        "weighted_f1": weighted_f1,
        "worst_class_recall": float(np.min(supported_recall)) if len(supported_recall) else np.nan,
    }
