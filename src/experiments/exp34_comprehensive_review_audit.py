"""Run evidence audits requested during the comprehensive manuscript review.

The outputs are independent of the formal result tables. They diagnose legacy
feature-group contamination, test directional robustness with exact feature
groups, quantify repeated grouped-split variability, and estimate uncertainty
by bootstrapping source files rather than individual windows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.exp02_ml_baselines import _feature_matrix
from src.utils.metrics import classification_metrics
from src.utils.tables import read_table


FEATURES_PATH = Path("data/processed/features/mcc5_features_formal.parquet")
WINDOWS_PATH = Path("data/processed/windows/mcc5_windows_formal.parquet")
FORMAL_SPLIT_DIR = Path("data/processed/splits")
DEFAULT_OUT = Path("results/review_runs/review_20260710/classical")

TIME_SUFFIXES = {
    "mean",
    "std",
    "rms",
    "peak",
    "peak_to_peak",
    "skewness",
    "kurtosis",
    "crest_factor",
    "impulse_factor",
    "shape_factor",
    "clearance_factor",
    "energy",
}

LEGACY_GROUPS = {
    "time_only": [
        "mean",
        "std",
        "rms",
        "peak",
        "peak_to_peak",
        "skewness",
        "kurtosis",
        "crest_factor",
        "impulse_factor",
        "shape_factor",
        "clearance_factor",
        "energy",
    ],
    "time_frequency": [
        "mean",
        "std",
        "rms",
        "peak",
        "spectral",
        "dominant_frequency",
        "band_energy",
        "high_frequency_ratio",
    ],
    "time_frequency_envelope": [
        "mean",
        "std",
        "rms",
        "peak",
        "spectral",
        "dominant_frequency",
        "band_energy",
        "envelope",
    ],
    "time_frequency_envelope_order": [
        "mean",
        "std",
        "rms",
        "peak",
        "spectral",
        "dominant_frequency",
        "band_energy",
        "envelope",
        "order",
    ],
    "order_only": ["order"],
    "non_order_physical": ["rpm_nominal", "load_nm", "torque", "key_phase", "vib_", "cur_"],
    "all_physical": [""],
}


def feature_block(column: str) -> str:
    if column in {"rpm_nominal", "load_nm"}:
        return "operating_context"
    if column.startswith("vib_"):
        if "_keyphase_" in column:
            return "vibration_keyphase_order"
        suffix = column.split("_", 2)[-1]
        if suffix in TIME_SUFFIXES:
            return "vibration_time"
        if "_envelope_" in column:
            return "vibration_envelope"
        if "_order_" in column or suffix.startswith("peak_order") or suffix == "order_entropy":
            return "vibration_nominal_order"
        return "vibration_frequency"
    if column.startswith("cur_"):
        suffix = column.split("_", 2)[-1]
        if suffix in TIME_SUFFIXES:
            return "current_time"
        if "_order_" in column or suffix.startswith("peak_order") or suffix == "order_entropy":
            return "current_nominal_order"
        return "current_frequency"
    if column.startswith("torque_"):
        return "torque_time"
    if column.startswith("key_phase_"):
        return "key_phase_time"
    return "unclassified"


def exact_groups(columns: list[str], membership: pd.DataFrame) -> dict[str, list[str]]:
    by_block = {
        block: membership.loc[membership["block"] == block, "feature"].tolist()
        for block in membership["block"].unique()
    }

    def combine(*blocks: str) -> list[str]:
        selected = set()
        for block in blocks:
            selected.update(by_block.get(block, []))
        return [column for column in columns if column in selected]

    vib_t = "vibration_time"
    vib_f = "vibration_frequency"
    vib_e = "vibration_envelope"
    vib_o = "vibration_nominal_order"
    vib_k = "vibration_keyphase_order"
    cur_t = "current_time"
    cur_f = "current_frequency"
    cur_o = "current_nominal_order"
    torque = "torque_time"
    key = "key_phase_time"
    ctx = "operating_context"

    return {
        "context_only": combine(ctx),
        "vibration_time_only": combine(vib_t),
        "vibration_time_frequency": combine(vib_t, vib_f),
        "vibration_time_frequency_envelope": combine(vib_t, vib_f, vib_e),
        "nominal_order_only": combine(vib_o, cur_o),
        "keyphase_order_only": combine(vib_k),
        "all_order_only": combine(vib_o, cur_o, vib_k),
        "vibration_current_no_order": combine(vib_t, vib_f, vib_e, cur_t, cur_f),
        "vibration_current_plus_nominal_order": combine(vib_t, vib_f, vib_e, cur_t, cur_f, vib_o, cur_o),
        "vibration_current_all_order": combine(vib_t, vib_f, vib_e, cur_t, cur_f, vib_o, cur_o, vib_k),
        "all_signals_without_order": combine(vib_t, vib_f, vib_e, cur_t, cur_f, torque, key),
        "all_signals_without_order_plus_context": combine(vib_t, vib_f, vib_e, cur_t, cur_f, torque, key, ctx),
        "all_signals_plus_nominal_order_context": combine(vib_t, vib_f, vib_e, cur_t, cur_f, torque, key, vib_o, cur_o, ctx),
        "all_signals_all_order_no_context": combine(vib_t, vib_f, vib_e, cur_t, cur_f, torque, key, vib_o, cur_o, vib_k),
        "all_features": list(columns),
    }


def make_model(name: str, seed: int):
    if name == "random_forest":
        estimator = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    elif name == "xgboost":
        from xgboost import XGBClassifier

        estimator = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(name)
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])


def fit_predict(
    features: pd.DataFrame,
    split: pd.DataFrame,
    columns: list[str],
    model_name: str,
    seed: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    merged = features.merge(split[["window_id", "split"]], on="window_id", how="inner")
    train = merged[merged["split"] == "train"].copy()
    test = merged[merged["split"] == "test"].copy()
    if test.empty:
        test = merged[merged["split"] == "val"].copy()
    x_train = _feature_matrix(train).reindex(columns=columns).dropna(axis=1, how="all")
    x_test = _feature_matrix(test).reindex(columns=x_train.columns, fill_value=np.nan)
    y_train = train["label_group"].astype(str)
    y_test = test["label_group"].astype(str)
    model = make_model(model_name, seed)
    if model_name == "xgboost":
        encoder = LabelEncoder()
        y_fit = encoder.fit_transform(y_train)
        model.fit(x_train, y_fit)
        prediction = encoder.inverse_transform(np.asarray(model.predict(x_test), dtype=int))
    else:
        model.fit(x_train, y_train)
        prediction = np.asarray(model.predict(x_test), dtype=str)
    labels = sorted(pd.concat([y_train, y_test]).unique())
    metrics = classification_metrics(y_test.to_numpy(), prediction, labels=labels)
    prediction_table = test[["window_id", "source_file", "label_group"]].copy()
    prediction_table["prediction"] = prediction
    return metrics, prediction_table


def make_directional_splits(features: pd.DataFrame, out_dir: Path) -> dict[str, pd.DataFrame]:
    definitions = {
        "condition_speed_to_torque": ("condition_type", "torque_circulation"),
        "condition_torque_to_speed": ("condition_type", "speed_circulation"),
        "load_20_to_40": ("load_nm", 40.0),
        "load_40_to_20": ("load_nm", 20.0),
        "rpm_1000_holdout": ("rpm_nominal", 1000.0),
        "rpm_2000_holdout": ("rpm_nominal", 2000.0),
        "rpm_3000_holdout": ("rpm_nominal", 3000.0),
    }
    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, (column, test_value) in definitions.items():
        values = features[column].astype(str) if column == "condition_type" else pd.to_numeric(features[column], errors="coerce")
        mask = values == test_value
        split = pd.DataFrame(
            {
                "window_id": features["window_id"],
                "split": np.where(mask, "test", "train"),
                "split_type": name,
                "reason": f"hold_out_{column}_{test_value}",
            }
        )
        split.to_csv(split_dir / f"{name}.csv", index=False)
        outputs[name] = split
    return outputs


def source_file_split(features: pd.DataFrame, seed: int) -> pd.DataFrame:
    groups = features[["source_file", "label_group"]].drop_duplicates("source_file")
    rng = np.random.default_rng(seed)
    mapping: dict[str, str] = {}
    for _label, frame in groups.groupby("label_group", sort=True):
        shuffled = frame.iloc[rng.permutation(len(frame))]
        n = len(shuffled)
        n_train = max(1, int(round(n * 0.70)))
        n_val = max(1, int(round(n * 0.15)))
        if n_train + n_val >= n:
            n_train = n - 2
            n_val = 1
        for source in shuffled.iloc[:n_train]["source_file"].astype(str):
            mapping[source] = "train"
        for source in shuffled.iloc[n_train : n_train + n_val]["source_file"].astype(str):
            mapping[source] = "val"
        for source in shuffled.iloc[n_train + n_val :]["source_file"].astype(str):
            mapping[source] = "test"
    return pd.DataFrame(
        {
            "window_id": features["window_id"],
            "split": features["source_file"].astype(str).map(mapping),
            "split_type": "repeated_source_file_split",
            "reason": f"grouped_by_source_file_seed_{seed}",
        }
    )


def majority_vote(values: pd.Series) -> str:
    counts = values.astype(str).value_counts()
    return sorted(counts[counts == counts.max()].index)[0]


def cluster_metrics(predictions: pd.DataFrame, seed: int, n_boot: int) -> tuple[dict[str, float], dict[str, float]]:
    labels = sorted(predictions["label_group"].astype(str).unique())
    file_rows = []
    for source, frame in predictions.groupby("source_file", sort=True):
        file_rows.append(
            {
                "source_file": source,
                "label_group": str(frame["label_group"].iloc[0]),
                "prediction": majority_vote(frame["prediction"]),
            }
        )
    file_df = pd.DataFrame(file_rows)
    file_level = classification_metrics(file_df["label_group"], file_df["prediction"], labels=labels)

    label_to_index = {label: index for index, label in enumerate(labels)}
    groups_by_label: dict[str, list[np.ndarray]] = {}
    for _, frame in predictions.groupby("source_file", sort=True):
        label = str(frame["label_group"].iloc[0])
        matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
        true_index = label_to_index[label]
        for prediction, count in frame["prediction"].astype(str).value_counts().items():
            matrix[true_index, label_to_index[prediction]] += int(count)
        groups_by_label.setdefault(label, []).append(matrix)
    groups = [matrix for label_groups in groups_by_label.values() for matrix in label_groups]
    rng = np.random.default_rng(seed)
    macro_f1 = []
    worst_recall = []
    for _ in range(n_boot):
        sampled_groups: list[np.ndarray] = []
        for label_groups in groups_by_label.values():
            sampled = rng.integers(0, len(label_groups), size=len(label_groups))
            sampled_groups.extend(label_groups[i] for i in sampled)
        matrix = np.sum(sampled_groups, axis=0)
        true_positive = np.diag(matrix).astype(float)
        precision = true_positive / np.maximum(matrix.sum(axis=0), 1)
        recall = true_positive / np.maximum(matrix.sum(axis=1), 1)
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        macro_f1.append(float(np.mean(f1)))
        worst_recall.append(float(np.min(recall)))
    cluster_ci = {
        "source_file_count": len(groups),
        "macro_f1_cluster_boot_low": float(np.quantile(macro_f1, 0.025)),
        "macro_f1_cluster_boot_high": float(np.quantile(macro_f1, 0.975)),
        "worst_recall_cluster_boot_low": float(np.quantile(worst_recall, 0.025)),
        "worst_recall_cluster_boot_high": float(np.quantile(worst_recall, 0.975)),
    }
    return file_level, cluster_ci


def legacy_contamination(columns: list[str], membership: pd.DataFrame) -> pd.DataFrame:
    block_map = membership.set_index("feature")["block"].to_dict()
    rows = []
    for group, patterns in LEGACY_GROUPS.items():
        if group == "non_order_physical":
            selected = [c for c in columns if "order" not in c.lower() and any(p in c.lower() for p in patterns)]
        else:
            selected = [c for c in columns if any(p in c.lower() for p in patterns)]
        counts = pd.Series([block_map[c] for c in selected]).value_counts().to_dict()
        rows.append(
            {
                "legacy_group": group,
                "feature_count": len(selected),
                "block_counts_json": json.dumps(counts, sort_keys=True),
                "contains_order_features": any("order" in block_map[c] for c in selected),
            }
        )
    return pd.DataFrame(rows)


def write_markdown(df: pd.DataFrame, path: Path, title: str) -> None:
    try:
        table = df.to_markdown(index=False)
    except Exception:
        table = df.to_csv(index=False)
    path.write_text(f"# {title}\n\n{table}\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    features = read_table(FEATURES_PATH)
    windows = read_table(WINDOWS_PATH)
    numeric_columns = list(_feature_matrix(features).columns)
    membership = pd.DataFrame(
        [{"feature": column, "block": feature_block(column)} for column in numeric_columns]
    )
    if (membership["block"] == "unclassified").any():
        unknown = membership.loc[membership["block"] == "unclassified", "feature"].tolist()
        raise RuntimeError(f"Unclassified numeric features: {unknown}")
    if len(membership) != 254:
        raise RuntimeError(f"Expected 254 model features, found {len(membership)}")
    membership.to_csv(out_dir / "exact_feature_membership.csv", index=False)
    block_audit = membership.groupby("block").size().reset_index(name="feature_count").sort_values("block")
    block_audit.to_csv(out_dir / "exact_feature_block_audit.csv", index=False)
    contamination = legacy_contamination(numeric_columns, membership)
    contamination.to_csv(out_dir / "legacy_feature_group_contamination.csv", index=False)

    groups = exact_groups(numeric_columns, membership)
    group_audit = pd.DataFrame(
        [{"feature_group": name, "feature_count": len(columns)} for name, columns in groups.items()]
    )
    group_audit.to_csv(out_dir / "exact_feature_group_audit.csv", index=False)

    formal_splits = {
        path.stem.replace("mcc5_formal_", "").replace("_split", ""): pd.read_csv(path)
        for path in sorted(FORMAL_SPLIT_DIR.glob("mcc5_formal_*_split.csv"))
    }
    representation_rows = []
    for split_name, split in formal_splits.items():
        for group_name, columns in groups.items():
            for model_name in ["random_forest", "xgboost"]:
                metrics, _ = fit_predict(features, split, columns, model_name, seed=42)
                representation_rows.append(
                    {
                        "split": split_name,
                        "feature_group": group_name,
                        "feature_count": len(columns),
                        "model": model_name,
                        **metrics,
                    }
                )
    representation = pd.DataFrame(representation_rows)
    representation.to_csv(out_dir / "exact_representation_ablation_by_model.csv", index=False)
    best_idx = representation.groupby(["split", "feature_group"])["macro_f1"].idxmax()
    representation_best = representation.loc[best_idx].sort_values(["split", "macro_f1"], ascending=[True, False])
    representation_best.to_csv(out_dir / "exact_representation_ablation_best.csv", index=False)

    directional_splits = make_directional_splits(features, out_dir)
    directional_groups = [
        "vibration_current_no_order",
        "all_signals_without_order_plus_context",
        "all_features",
    ]
    directional_rows = []
    split_audit_rows = []
    for split_name, split in directional_splits.items():
        merged = windows.merge(split[["window_id", "split"]], on="window_id", how="inner")
        train_sources = set(merged.loc[merged["split"] == "train", "source_file"].astype(str))
        test_sources = set(merged.loc[merged["split"] == "test", "source_file"].astype(str))
        split_audit_rows.append(
            {
                "split": split_name,
                "train_windows": int((merged["split"] == "train").sum()),
                "test_windows": int((merged["split"] == "test").sum()),
                "train_sources": len(train_sources),
                "test_sources": len(test_sources),
                "source_file_overlap": len(train_sources & test_sources),
            }
        )
        for group_name in directional_groups:
            for model_name in ["random_forest", "xgboost"]:
                metrics, _ = fit_predict(features, split, groups[group_name], model_name, seed=42)
                directional_rows.append(
                    {
                        "split": split_name,
                        "feature_group": group_name,
                        "feature_count": len(groups[group_name]),
                        "model": model_name,
                        **metrics,
                    }
                )
    directional = pd.DataFrame(directional_rows)
    directional.to_csv(out_dir / "directional_robustness_classical.csv", index=False)
    pd.DataFrame(split_audit_rows).to_csv(out_dir / "directional_split_audit.csv", index=False)

    repeated_rows = []
    for seed in range(42, 52):
        split = source_file_split(features, seed)
        for model_name in ["random_forest", "xgboost"]:
            metrics, _ = fit_predict(features, split, groups["all_features"], model_name, seed=seed)
            repeated_rows.append({"seed": seed, "model": model_name, **metrics})
    repeated = pd.DataFrame(repeated_rows)
    repeated.to_csv(out_dir / "repeated_source_file_split_by_seed.csv", index=False)
    repeated_summary = repeated.groupby("model")[["accuracy", "macro_f1", "worst_class_recall"]].agg(["mean", "std", "min", "max"])
    repeated_summary.columns = ["_".join(column) for column in repeated_summary.columns]
    repeated_summary.reset_index().to_csv(out_dir / "repeated_source_file_split_summary.csv", index=False)

    main_models = {
        "source_file": "xgboost",
        "cross_condition": "random_forest",
        "cross_load": "random_forest",
        "cross_rpm": "random_forest",
    }
    uncertainty_rows = []
    file_rows = []
    for split_name, model_name in main_models.items():
        metrics, predictions = fit_predict(
            features,
            formal_splits[split_name],
            groups["all_features"],
            model_name,
            seed=42,
        )
        file_metrics, cluster_ci = cluster_metrics(predictions, seed=42, n_boot=args.bootstrap)
        uncertainty_rows.append({"split": split_name, "model": model_name, **metrics, **cluster_ci})
        file_rows.append({"split": split_name, "model": model_name, "source_file_count": predictions["source_file"].nunique(), **file_metrics})
    uncertainty = pd.DataFrame(uncertainty_rows)
    file_level = pd.DataFrame(file_rows)
    uncertainty.to_csv(out_dir / "source_file_cluster_bootstrap.csv", index=False)
    file_level.to_csv(out_dir / "source_file_majority_vote_metrics.csv", index=False)

    write_markdown(contamination, out_dir / "legacy_feature_group_contamination.md", "Legacy Feature-Group Contamination")
    write_markdown(representation_best, out_dir / "exact_representation_ablation_best.md", "Exact Representation Ablation")
    write_markdown(directional, out_dir / "directional_robustness_classical.md", "Directional Robustness: Classical Models")
    write_markdown(repeated_summary.reset_index(), out_dir / "repeated_source_file_split_summary.md", "Repeated Source-File Splits")
    write_markdown(uncertainty, out_dir / "source_file_cluster_bootstrap.md", "Source-File Cluster Bootstrap")
    print(f"output={out_dir}")
    print(f"feature_blocks={block_audit.to_dict(orient='records')}")
    print(f"representation_rows={len(representation)}")
    print(f"directional_rows={len(directional)}")
    print(f"repeated_rows={len(repeated)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
