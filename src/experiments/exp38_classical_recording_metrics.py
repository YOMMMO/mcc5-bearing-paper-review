"""Generate recording-level classical metrics from frozen MCC5 protocols."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.exp02_ml_baselines import _feature_matrix
from src.experiments.exp34_comprehensive_review_audit import (
    exact_groups,
    feature_block,
    make_model,
    majority_vote,
)
from src.utils.metrics import classification_metrics
from src.utils.tables import read_table


PRIMARY_MODELS = {
    "source_file": "xgboost",
    "cross_condition": "random_forest",
    "cross_load": "random_forest",
    "cross_rpm": "random_forest",
}


def _fit_probability_table(
    features: pd.DataFrame,
    split: pd.DataFrame,
    columns: list[str],
    model_name: str,
    seed: int,
) -> tuple[pd.DataFrame, list[str]]:
    merged = features.merge(split[["window_id", "split"]], on="window_id", how="inner")
    train = merged[merged["split"] == "train"].copy()
    test = merged[merged["split"] == "test"].copy()
    x_train = _feature_matrix(train).reindex(columns=columns).dropna(axis=1, how="all")
    x_test = _feature_matrix(test).reindex(columns=x_train.columns, fill_value=np.nan)
    y_train = train["label_group"].astype(str)
    labels = sorted(pd.concat([y_train, test["label_group"].astype(str)]).unique())
    model = make_model(model_name, seed)

    if model_name == "xgboost":
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(y_train)
        model.fit(x_train, encoded)
        probabilities = model.predict_proba(x_test)
        probability_labels = [str(label) for label in encoder.classes_]
    else:
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)
        probability_labels = [str(label) for label in model.named_steps["model"].classes_]

    table = test[
        ["window_id", "source_file", "label_group", "condition_type", "rpm_nominal", "load_nm"]
    ].copy()
    for index, label in enumerate(probability_labels):
        table[f"probability_{label}"] = probabilities[:, index]
    table["window_prediction"] = [
        probability_labels[index] for index in np.argmax(probabilities, axis=1)
    ]
    return table, labels


def _aggregate_recordings(
    window_predictions: pd.DataFrame,
    labels: list[str],
    split_name: str,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    probability_columns = [f"probability_{label}" for label in labels]
    source_rows: list[dict[str, object]] = []
    for source, frame in window_predictions.groupby("source_file", sort=True):
        mean_probabilities = frame[probability_columns].mean(axis=0)
        mean_label = labels[int(np.argmax(mean_probabilities.to_numpy()))]
        vote_label = majority_vote(frame["window_prediction"])
        row: dict[str, object] = {
            "split": split_name,
            "model": model_name,
            "source_file": source,
            "true_label": str(frame["label_group"].iloc[0]),
            "mean_probability_prediction": mean_label,
            "majority_vote_prediction": vote_label,
            "window_count": len(frame),
            "condition_type": str(frame["condition_type"].iloc[0]),
            "rpm_nominal": float(frame["rpm_nominal"].iloc[0]),
            "load_nm": float(frame["load_nm"].iloc[0]),
        }
        for label in labels:
            row[f"mean_probability_{label}"] = float(mean_probabilities[f"probability_{label}"])
        source_rows.append(row)
    source_table = pd.DataFrame(source_rows)

    metric_rows = []
    for aggregation, prediction_column in [
        ("mean_probability", "mean_probability_prediction"),
        ("majority_vote", "majority_vote_prediction"),
    ]:
        metrics = classification_metrics(
            source_table["true_label"], source_table[prediction_column], labels=labels
        )
        metric_rows.append(
            {
                "split": split_name,
                "model": model_name,
                "aggregation": aggregation,
                "source_recording_count": len(source_table),
                **metrics,
            }
        )
    return source_table, pd.DataFrame(metric_rows)


def _stratified_bootstrap(
    source_table: pd.DataFrame,
    prediction_column: str,
    labels: list[str],
    seed: int,
    replicates: int,
) -> dict[str, float]:
    groups = {
        label: frame.reset_index(drop=True)
        for label, frame in source_table.groupby("true_label", sort=True)
    }
    rng = np.random.default_rng(seed)
    macro_f1: list[float] = []
    accuracy: list[float] = []
    worst_recall: list[float] = []
    for _ in range(replicates):
        sampled = []
        for frame in groups.values():
            indices = rng.integers(0, len(frame), size=len(frame))
            sampled.append(frame.iloc[indices])
        bootstrap = pd.concat(sampled, ignore_index=True)
        metrics = classification_metrics(
            bootstrap["true_label"], bootstrap[prediction_column], labels=labels
        )
        macro_f1.append(metrics["macro_f1"])
        accuracy.append(metrics["accuracy"])
        worst_recall.append(metrics["worst_class_recall"])
    return {
        "accuracy_ci_low": float(np.quantile(accuracy, 0.025)),
        "accuracy_ci_high": float(np.quantile(accuracy, 0.975)),
        "macro_f1_ci_low": float(np.quantile(macro_f1, 0.025)),
        "macro_f1_ci_high": float(np.quantile(macro_f1, 0.975)),
        "worst_class_recall_ci_low": float(np.quantile(worst_recall, 0.025)),
        "worst_class_recall_ci_high": float(np.quantile(worst_recall, 0.975)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", default="data/processed/features/mcc5_features_formal.parquet"
    )
    parser.add_argument("--split-dir", default="data/processed/splits")
    parser.add_argument("--out", default="evidence")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args(argv)

    features = read_table(args.features)
    numeric_columns = list(_feature_matrix(features).columns)
    membership = pd.DataFrame(
        [{"feature": column, "block": feature_block(column)} for column in numeric_columns]
    )
    columns = exact_groups(numeric_columns, membership)["all_features"]
    split_dir = Path(args.split_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_sources: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []
    bootstrap_rows: list[dict[str, object]] = []
    for split_name, model_name in PRIMARY_MODELS.items():
        split_path = split_dir / f"mcc5_formal_{split_name}_split.csv"
        window_table, labels = _fit_probability_table(
            features, pd.read_csv(split_path), columns, model_name, args.seed
        )
        source_table, metrics = _aggregate_recordings(
            window_table, labels, split_name, model_name
        )
        all_sources.append(source_table)
        all_metrics.append(metrics)
        for row in metrics.to_dict(orient="records"):
            prediction_column = (
                "mean_probability_prediction"
                if row["aggregation"] == "mean_probability"
                else "majority_vote_prediction"
            )
            bootstrap_rows.append(
                {
                    **row,
                    "bootstrap_unit": "source_recording",
                    "bootstrap_replicates": args.bootstrap,
                    **_stratified_bootstrap(
                        source_table,
                        prediction_column,
                        labels,
                        args.seed,
                        args.bootstrap,
                    ),
                }
            )

    pd.concat(all_sources, ignore_index=True).to_csv(
        out_dir / "classical_source_recording_predictions.csv", index=False
    )
    pd.concat(all_metrics, ignore_index=True).to_csv(
        out_dir / "classical_source_recording_metrics.csv", index=False
    )
    pd.DataFrame(bootstrap_rows).to_csv(
        out_dir / "classical_source_recording_bootstrap.csv", index=False
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
