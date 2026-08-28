"""Run recording-level partition and session-proxy sensitivity controls."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.exp02_ml_baselines import _feature_matrix
from src.experiments.exp34_comprehensive_review_audit import (
    exact_groups,
    feature_block,
    source_file_split,
)
from src.experiments.exp38_classical_recording_metrics import (
    _aggregate_recordings,
    _fit_probability_table,
)
from src.utils.metrics import classification_metrics
from src.utils.tables import read_table


DATE_PATTERN = re.compile(r"(\d{6})\d{6}[A-Za-z]*$")
CLASSICAL_CONTROLS = ["random_forest", "xgboost"]


def acquisition_date(source_file: str) -> str:
    match = DATE_PATTERN.search(Path(str(source_file)).stem)
    if not match:
        raise ValueError(f"Cannot parse acquisition date from {source_file}")
    return match.group(1)


def all_feature_columns(features: pd.DataFrame) -> list[str]:
    numeric_columns = list(_feature_matrix(features).columns)
    membership = pd.DataFrame(
        [{"feature": column, "block": feature_block(column)} for column in numeric_columns]
    )
    return exact_groups(numeric_columns, membership)["all_features"]


def source_role_map(features: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    roles = features[["window_id", "source_file"]].merge(
        split[["window_id", "split"]], on="window_id", how="inner"
    )
    counts = roles.groupby("source_file")["split"].nunique()
    if int(counts.max()) != 1:
        raise RuntimeError("A source recording crosses roles in a recording-grouped protocol")
    return roles.drop_duplicates("source_file")[["source_file", "split"]]


def run_within_date_control(
    features: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = features["source_file"].astype(str).map(acquisition_date)
    subset = features.loc[
        (dates == "250707") & (features["label_group"].astype(str) != "healthy")
    ].copy()
    sources: list[pd.DataFrame] = []
    metrics: list[pd.DataFrame] = []
    for split_seed in range(42, 52):
        split = source_file_split(subset, split_seed)
        for model_name in CLASSICAL_CONTROLS:
            windows, labels = _fit_probability_table(
                subset, split, columns, model_name, seed=42
            )
            source_table, metric_table = _aggregate_recordings(
                windows,
                labels,
                "within_date_250707_three_fault_classes",
                model_name,
            )
            source_table.insert(0, "split_seed", split_seed)
            metric_table.insert(0, "split_seed", split_seed)
            sources.append(source_table)
            metrics.append(metric_table)
    source_table = pd.concat(sources, ignore_index=True)
    metric_table = pd.concat(metrics, ignore_index=True)
    summary = (
        metric_table.groupby(["model", "aggregation"])[
            ["accuracy", "macro_f1", "worst_class_recall"]
        ]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    return source_table, metric_table, summary


def _metadata_pipeline(
    categorical: list[str],
    numeric: list[str],
    model_name: str,
) -> Pipeline:
    transformers = []
    if categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            )
        )
    if numeric:
        transformers.append(("numeric", StandardScaler(), numeric))
    preprocessing = ColumnTransformer(transformers, remainder="drop")
    if model_name == "decision_tree":
        classifier = DecisionTreeClassifier(
            max_depth=4, class_weight="balanced", random_state=42
        )
    elif model_name == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        )
    elif model_name == "dummy_most_frequent":
        classifier = DummyClassifier(strategy="most_frequent")
    else:
        raise ValueError(model_name)
    return Pipeline([("preprocess", preprocessing), ("model", classifier)])


def run_training_only_metadata_baselines(
    features: pd.DataFrame,
    split_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_table = features[
        ["source_file", "label_group", "condition_type", "rpm_nominal", "load_nm"]
    ].drop_duplicates("source_file")
    source_table = source_table.copy()
    source_table["acquisition_date"] = source_table["source_file"].astype(str).map(
        acquisition_date
    )
    definitions = {
        "date_only": (["acquisition_date"], []),
        "operating_context_only": (["condition_type"], ["rpm_nominal", "load_nm"]),
        "date_and_context": (
            ["acquisition_date", "condition_type"],
            ["rpm_nominal", "load_nm"],
        ),
    }
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    labels = sorted(source_table["label_group"].astype(str).unique())
    for split_name in ["source_file", "cross_condition", "cross_load", "cross_rpm"]:
        split = pd.read_csv(split_dir / f"mcc5_formal_{split_name}_split.csv")
        roles = source_role_map(features, split)
        framed = source_table.merge(roles, on="source_file", how="inner")
        train = framed[framed["split"] == "train"].copy()
        test = framed[framed["split"] == "test"].copy()
        for feature_set, (categorical, numeric) in definitions.items():
            columns = categorical + numeric
            for model_name in [
                "decision_tree",
                "logistic_regression",
                "dummy_most_frequent",
            ]:
                model = _metadata_pipeline(categorical, numeric, model_name)
                model.fit(train[columns], train["label_group"].astype(str))
                predictions = model.predict(test[columns]).astype(str)
                metrics = classification_metrics(
                    test["label_group"].astype(str), predictions, labels=labels
                )
                metric_rows.append(
                    {
                        "split": split_name,
                        "feature_set": feature_set,
                        "model": model_name,
                        "train_recordings": len(train),
                        "test_recordings": len(test),
                        "fit_scope": "training_recordings_only",
                        **metrics,
                    }
                )
                prediction_table = test[
                    [
                        "source_file",
                        "label_group",
                        "acquisition_date",
                        "condition_type",
                        "rpm_nominal",
                        "load_nm",
                    ]
                ].copy()
                prediction_table.insert(0, "split_name", split_name)
                prediction_table.insert(1, "feature_set", feature_set)
                prediction_table.insert(2, "model", model_name)
                prediction_table["prediction"] = predictions
                prediction_rows.append(prediction_table)
    return pd.DataFrame(metric_rows), pd.concat(prediction_rows, ignore_index=True)


def run_outer_date_predictability(
    features: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = features[features["label_group"].astype(str) == "bearing_outer"].copy()
    subset["label_group"] = subset["source_file"].astype(str).map(acquisition_date)
    recordings = subset[["source_file", "label_group"]].drop_duplicates("source_file")
    splitter = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
    metric_rows: list[pd.DataFrame] = []
    source_rows: list[pd.DataFrame] = []
    for fold_index, (train_index, test_index) in enumerate(
        splitter.split(recordings["source_file"], recordings["label_group"]), start=1
    ):
        train_sources = set(recordings.iloc[train_index]["source_file"].astype(str))
        test_sources = set(recordings.iloc[test_index]["source_file"].astype(str))
        split = pd.DataFrame(
            {
                "window_id": subset["window_id"],
                "split": np.where(
                    subset["source_file"].astype(str).isin(test_sources), "test", "train"
                ),
            }
        )
        if train_sources & test_sources:
            raise RuntimeError("Outer-race date-prediction fold has source overlap")
        for feature_set, columns in feature_sets.items():
            for model_name in CLASSICAL_CONTROLS:
                windows, labels = _fit_probability_table(
                    subset, split, columns, model_name, seed=42
                )
                source_table, metrics = _aggregate_recordings(
                    windows,
                    labels,
                    "outer_race_acquisition_date_prediction",
                    model_name,
                )
                source_table.insert(0, "fold", fold_index)
                source_table.insert(1, "feature_set", feature_set)
                metrics.insert(0, "fold", fold_index)
                metrics.insert(1, "feature_set", feature_set)
                source_rows.append(source_table)
                metric_rows.append(metrics)
    metric_table = pd.concat(metric_rows, ignore_index=True)
    summary = (
        metric_table.groupby(["feature_set", "model", "aggregation"])[
            ["accuracy", "macro_f1", "worst_class_recall"]
        ]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    return pd.concat(source_rows, ignore_index=True), summary


def _stratified_window_split(features: pd.DataFrame, seed: int) -> pd.DataFrame:
    indices = np.arange(len(features))
    train_index, remainder_index = train_test_split(
        indices,
        train_size=0.70,
        random_state=seed,
        stratify=features["label_group"].astype(str),
    )
    validation_index, test_index = train_test_split(
        remainder_index,
        test_size=0.50,
        random_state=seed,
        stratify=features.iloc[remainder_index]["label_group"].astype(str),
    )
    roles = np.full(len(features), "validation", dtype=object)
    roles[train_index] = "train"
    roles[validation_index] = "validation"
    roles[test_index] = "test"
    return pd.DataFrame({"window_id": features["window_id"], "split": roles})


def run_partition_controls(
    features: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    order = features["window_id"].astype(str).str.extract(r"_w(\d+)$")[0].astype(int)
    nonoverlapping = features.loc[(order % 2) == 0].copy()
    rows: list[dict[str, object]] = []
    for split_seed in range(42, 52):
        definitions = {
            "random_overlapping_windows": (
                features,
                _stratified_window_split(features, split_seed),
                False,
            ),
            "random_nonoverlapping_windows": (
                nonoverlapping,
                _stratified_window_split(nonoverlapping, split_seed),
                False,
            ),
            "recording_grouped_overlapping_windows": (
                features,
                source_file_split(features, split_seed),
                True,
            ),
            "recording_grouped_nonoverlapping_windows": (
                nonoverlapping,
                source_file_split(nonoverlapping, split_seed),
                True,
            ),
        }
        for protocol, (frame, split, recording_metric_valid) in definitions.items():
            windows, labels = _fit_probability_table(
                frame, split, columns, "xgboost", seed=42
            )
            window_metrics = classification_metrics(
                windows["label_group"], windows["window_prediction"], labels=labels
            )
            merged = frame[["window_id", "source_file"]].merge(
                split[["window_id", "split"]], on="window_id", how="inner"
            )
            train_sources = set(
                merged.loc[merged["split"] == "train", "source_file"].astype(str)
            )
            test_sources = set(
                merged.loc[merged["split"] == "test", "source_file"].astype(str)
            )
            row: dict[str, object] = {
                "split_seed": split_seed,
                "model_seed": 42,
                "protocol": protocol,
                "model": "xgboost",
                "uses_overlapping_windows": "nonoverlapping" not in protocol,
                "recording_grouped": protocol.startswith("recording_grouped"),
                "train_windows": int((merged["split"] == "train").sum()),
                "test_windows": int((merged["split"] == "test").sum()),
                "train_recordings": len(train_sources),
                "test_recordings": len(test_sources),
                "source_recording_overlap": len(train_sources & test_sources),
                "window_accuracy": window_metrics["accuracy"],
                "window_macro_f1": window_metrics["macro_f1"],
                "window_worst_class_recall": window_metrics["worst_class_recall"],
                "recording_metric_valid": recording_metric_valid,
                "recording_accuracy": np.nan,
                "recording_macro_f1": np.nan,
                "recording_worst_class_recall": np.nan,
            }
            if recording_metric_valid:
                _source_table, source_metrics = _aggregate_recordings(
                    windows, labels, protocol, "xgboost"
                )
                metric = source_metrics[
                    source_metrics["aggregation"] == "mean_probability"
                ].iloc[0]
                row.update(
                    {
                        "recording_accuracy": metric["accuracy"],
                        "recording_macro_f1": metric["macro_f1"],
                        "recording_worst_class_recall": metric[
                            "worst_class_recall"
                        ],
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", default="data/processed/features/mcc5_features_formal.parquet"
    )
    parser.add_argument("--split-dir", default="data/processed/splits")
    parser.add_argument("--out", default="evidence")
    args = parser.parse_args(argv)

    features = read_table(args.features)
    columns = all_feature_columns(features)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    within_sources, within_metrics, within_summary = run_within_date_control(
        features, columns
    )
    within_sources.to_csv(out / "within_date_250707_predictions.csv", index=False)
    within_metrics.to_csv(out / "within_date_250707_metrics.csv", index=False)
    within_summary.to_csv(out / "within_date_250707_summary.csv", index=False)

    metadata_metrics, metadata_predictions = run_training_only_metadata_baselines(
        features, Path(args.split_dir)
    )
    metadata_metrics.to_csv(out / "training_only_metadata_baselines.csv", index=False)
    metadata_predictions.to_csv(
        out / "training_only_metadata_predictions.csv", index=False
    )

    numeric_columns = list(_feature_matrix(features).columns)
    membership = pd.DataFrame(
        [{"feature": column, "block": feature_block(column)} for column in numeric_columns]
    )
    groups = exact_groups(numeric_columns, membership)
    outer_sources, outer_summary = run_outer_date_predictability(
        features,
        {
            "all_features_254": groups["all_features"],
            "all_signals_without_order": groups["all_signals_without_order"],
            "vibration_current_no_order": groups["vibration_current_no_order"],
        },
    )
    outer_sources.to_csv(out / "outer_race_date_predictions.csv", index=False)
    outer_summary.to_csv(out / "outer_race_date_summary.csv", index=False)

    partition_controls = run_partition_controls(features, columns)
    partition_controls.to_csv(out / "partition_leakage_controls.csv", index=False)
    partition_summary = (
        partition_controls.groupby(
            ["protocol", "uses_overlapping_windows", "recording_grouped"],
            dropna=False,
        )[
            [
                "source_recording_overlap",
                "window_accuracy",
                "window_macro_f1",
                "window_worst_class_recall",
                "recording_accuracy",
                "recording_macro_f1",
                "recording_worst_class_recall",
            ]
        ]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    partition_summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in partition_summary.columns
    ]
    partition_summary.to_csv(
        out / "partition_leakage_controls_summary.csv", index=False
    )

    confusion_rows = []
    for (model, split_seed), frame in within_sources.groupby(
        ["model", "split_seed"], sort=True
    ):
        labels = sorted(frame["true_label"].astype(str).unique())
        matrix = confusion_matrix(
            frame["true_label"], frame["mean_probability_prediction"], labels=labels
        )
        confusion_rows.append(
            {
                "model": model,
                "split_seed": split_seed,
                "labels": "|".join(labels),
                "confusion_matrix": ";".join(
                    ",".join(str(int(value)) for value in row) for row in matrix
                ),
            }
        )
    pd.DataFrame(confusion_rows).to_csv(
        out / "within_date_250707_confusion_matrices.csv", index=False
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
