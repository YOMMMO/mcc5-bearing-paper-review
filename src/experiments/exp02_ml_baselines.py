"""Leakage-safe classical ML baselines over extracted features."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    SKLEARN_AVAILABLE = True
    SKLEARN_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on installed packages
    joblib = None
    SKLEARN_AVAILABLE = False
    SKLEARN_IMPORT_ERROR = exc

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.io import ensure_dir
from src.utils.logger import get_logger
from src.utils.metrics import classification_metrics, save_confusion_matrix, save_per_class_metrics
from src.utils.tables import read_table


ID_COLS = {
    "window_id",
    "source_file",
    "sample_id",
    "label_group",
    "label_raw",
    "condition_type",
    "npz_path",
}


def _models(seed: int = 42) -> dict:
    if not SKLEARN_AVAILABLE:
        return {}
    models = {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        ),
        "svm": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", C=3.0, gamma="scale", random_state=seed)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)),
            ]
        ),
        "mlp": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, random_state=seed)),
            ]
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="mlogloss",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    except Exception:
        pass
    return models


def _feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.drop(columns=[c for c in ID_COLS if c in df.columns], errors="ignore")
    numeric = numeric.select_dtypes(include=[np.number, "bool"]).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return numeric


def _load_splits(split_arg: str | Path, split_glob: str = "*.csv") -> list[Path]:
    p = Path(split_arg)
    if p.is_dir():
        return sorted(p.glob(split_glob))
    return [p]


def run_baselines(
    features: pd.DataFrame,
    split_path: Path,
    out_prefix: Path,
    seed: int = 42,
    max_train_samples: int | None = None,
) -> pd.DataFrame:
    """Train and evaluate all classical baselines for one split file."""
    logger = get_logger("ml_baselines")
    if not SKLEARN_AVAILABLE:
        logger.warning("Skipping ML baselines: scikit-learn/joblib unavailable: %s", SKLEARN_IMPORT_ERROR)
        return pd.DataFrame([{"status": "skipped", "reason": f"missing sklearn/joblib: {SKLEARN_IMPORT_ERROR}"}])
    split = pd.read_csv(split_path)
    df = features.merge(split[["window_id", "split"]], on="window_id", how="inner")
    if df.empty:
        logger.warning("No overlapping windows for split %s", split_path)
        return pd.DataFrame()

    train_df = df[df["split"] == "train"].copy()
    eval_df = df[df["split"] == "test"].copy()
    if eval_df.empty:
        eval_df = df[df["split"] == "val"].copy()
    if eval_df.empty:
        eval_df = train_df.copy()
    if max_train_samples and len(train_df) > max_train_samples:
        train_df = train_df.sample(max_train_samples, random_state=seed)

    y_train = train_df["label_group"].astype(str)
    y_eval = eval_df["label_group"].astype(str)
    if y_train.nunique() < 2 or y_eval.empty:
        logger.warning("Skipping %s: need >=2 train classes and non-empty eval", split_path)
        return pd.DataFrame()
    X_train = _feature_matrix(train_df)
    X_train = X_train.dropna(axis=1, how="all")
    X_eval = _feature_matrix(eval_df).reindex(columns=X_train.columns, fill_value=np.nan)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    rows = []
    labels = sorted(pd.concat([y_train, y_eval]).unique())
    model_dir = ensure_dir("results/checkpoints")
    figure_dir = ensure_dir("results/figures")
    table_dir = ensure_dir("results/tables")
    split_name = split_path.stem

    for name, model in _models(seed).items():
        try:
            start = time.time()
            model.fit(X_train, y_train_encoded)
            train_time = time.time() - start
            pred_start = time.time()
            y_pred_encoded = np.asarray(model.predict(X_eval), dtype=int)
            y_pred = pd.Series(label_encoder.inverse_transform(y_pred_encoded), index=y_eval.index)
            infer_ms = (time.time() - pred_start) * 1000 / max(1, len(X_eval))
            metrics = classification_metrics(y_eval, y_pred, labels=labels)
            row = {
                "split_file": split_path.name,
                "model": name,
                **metrics,
                "train_time_sec": train_time,
                "inference_time_ms_per_sample": infer_ms,
            }
            rows.append(row)
            cm_path = figure_dir / f"{split_name}_{name}_confusion_matrix.png"
            save_confusion_matrix(y_eval, y_pred, cm_path, labels=labels)
            save_per_class_metrics(
                y_eval,
                y_pred,
                table_dir / f"{split_name}_{name}_per_class_metrics.csv",
                labels=labels,
            )
            if joblib is not None:
                joblib.dump(
                    {
                        "model": model,
                        "label_encoder": label_encoder,
                        "feature_columns": list(X_train.columns),
                    },
                    model_dir / f"{split_name}_{name}.joblib",
                )
        except Exception as exc:
            logger.warning("Model %s failed on %s: %s", name, split_path.name, exc)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--split_glob", default="*.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)

    logger = get_logger("ml_baselines")
    if not SKLEARN_AVAILABLE:
        out = Path(args.out)
        ensure_dir(out.parent)
        pd.DataFrame([{"status": "skipped", "reason": f"missing sklearn/joblib: {SKLEARN_IMPORT_ERROR}"}]).to_csv(out, index=False)
        logger.warning("Skipping ML baselines: scikit-learn/joblib unavailable: %s", SKLEARN_IMPORT_ERROR)
        return 0
    if not Path(args.features).exists() and not Path(args.features).with_suffix(".csv").exists():
        logger.warning("Features file missing: %s", args.features)
        return 0
    features = read_table(args.features)
    if args.max_windows:
        features = features.head(args.max_windows)
    if features.empty:
        logger.warning("Features table is empty.")
        return 0
    if args.dry_run:
        logger.info("Dry run: would train on %d feature rows", len(features))
        return 0

    all_rows = []
    for split_path in _load_splits(args.split, args.split_glob):
        if split_path.exists():
            all_rows.append(
                run_baselines(
                    features,
                    split_path,
                    Path(args.out),
                    seed=args.seed,
                    max_train_samples=args.max_train_samples,
                )
            )
    results = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    out = Path(args.out)
    ensure_dir(out.parent)
    results.to_csv(out, index=False)
    logger.info("Saved ML baseline results: %s (%d rows)", out, len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
