"""Run feature ablation with a leakage-safe RandomForest baseline."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    SKLEARN_AVAILABLE = True
    SKLEARN_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on installed packages
    SKLEARN_AVAILABLE = False
    SKLEARN_IMPORT_ERROR = exc

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.exp02_ml_baselines import _feature_matrix
from src.utils.io import ensure_dir
from src.utils.metrics import classification_metrics
from src.utils.plotting import save_bar_plot
from src.utils.tables import read_table


SETTINGS = {
    "vibration_only": ["vib_"],
    "current_only": ["cur_"],
    "scalar_physical_features_only": ["rpm_nominal", "load_nm", "torque_", "key_phase_"],
    "vibration_current": ["vib_", "cur_"],
    "vibration_current_rpm_load": ["vib_", "cur_", "rpm_nominal", "load_nm"],
    "vibration_current_order_features": ["vib_", "cur_", "order"],
    "vibration_current_order_features_rpm_load": ["vib_", "cur_", "order", "rpm_nominal", "load_nm"],
    "full_multisource_fusion": [""],
}


def _table_exists(path: str | Path) -> bool:
    p = Path(path)
    return p.exists() or (p.suffix == ".parquet" and p.with_suffix(".csv").exists())


def _cols(df: pd.DataFrame, patterns: list[str]) -> list[str]:
    if patterns == [""]:
        return list(df.columns)
    return [c for c in df.columns if any(p in c for p in patterns)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="data/processed/features/mcc5_features.parquet")
    parser.add_argument("--split", default="data/processed/splits/mcc5_source_file_split.csv")
    parser.add_argument("--out", default="results/tables/mcc5_ablation.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)
    if not SKLEARN_AVAILABLE:
        out = Path(args.out)
        ensure_dir(out.parent)
        pd.DataFrame([{"status": "skipped", "reason": f"missing scikit-learn: {SKLEARN_IMPORT_ERROR}"}]).to_csv(out, index=False)
        print(f"Skipping ablation: scikit-learn unavailable: {SKLEARN_IMPORT_ERROR}")
        return 0
    if not _table_exists(args.features) or not Path(args.split).exists():
        print("Missing features or split; skipping ablation.")
        return 0
    features = read_table(args.features)
    if args.max_windows:
        features = features.head(args.max_windows)
    split = pd.read_csv(args.split)
    df = features.merge(split[["window_id", "split"]], on="window_id", how="inner")
    if df.empty or df["label_group"].nunique() < 2 or args.dry_run:
        print("No runnable ablation data.")
        return 0
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    if test.empty:
        test = df[df["split"] == "val"]
    if test.empty:
        test = train
    X_all = _feature_matrix(df)
    rows = []
    for setting, patterns in SETTINGS.items():
        cols = _cols(X_all, patterns)
        if not cols:
            continue
        X_train = _feature_matrix(train).reindex(columns=cols)
        X_test = _feature_matrix(test).reindex(columns=cols)
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(n_estimators=120, random_state=args.seed, n_jobs=-1)),
            ]
        )
        start = time.time()
        model.fit(X_train, train["label_group"].astype(str))
        pred_start = time.time()
        pred = model.predict(X_test)
        metrics = classification_metrics(test["label_group"].astype(str), pred)
        rows.append(
            {
                "setting": setting,
                **metrics,
                "train_time_sec": time.time() - start,
                "inference_time": (time.time() - pred_start) / max(1, len(X_test)),
            }
        )
    out_df = pd.DataFrame(rows)
    out = Path(args.out)
    ensure_dir(out.parent)
    out_df.to_csv(out, index=False)
    if not out_df.empty:
        figure_dir = ensure_dir("results/figures")
        save_bar_plot(out_df["setting"], out_df["macro_f1"], figure_dir / f"{out.stem}_macro_f1.png", ylabel="Macro F1")
        save_bar_plot(out_df["setting"], out_df["worst_class_recall"], figure_dir / f"{out.stem}_worst_recall.png", ylabel="Worst Recall")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
