"""Compare feature representations with a common RandomForest classifier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.exp05_ablation import _cols
from src.experiments.exp02_ml_baselines import _feature_matrix
from src.utils.io import ensure_dir
from src.utils.metrics import classification_metrics
from src.utils.plotting import save_bar_plot
from src.utils.tables import read_table


REPRESENTATIONS = {
    "time_features_only": ["_mean", "_std", "_rms", "_kurtosis", "_energy", "_peak"],
    "time_frequency": ["_mean", "_std", "_rms", "frequency", "spectral", "band_energy"],
    "time_frequency_envelope": ["_mean", "_std", "_rms", "frequency", "spectral", "band_energy", "envelope"],
    "time_frequency_envelope_order": ["_mean", "_std", "_rms", "frequency", "spectral", "band_energy", "envelope", "order"],
}


def _table_exists(path: str | Path) -> bool:
    p = Path(path)
    return p.exists() or (p.suffix == ".parquet" and p.with_suffix(".csv").exists())


def _optional_result_row(path: str | Path, representation: str) -> dict:
    """Return a representation row from an optional experiment result table."""
    p = Path(path)
    if not p.exists():
        return {"representation": representation, "macro_f1": pd.NA}
    df = pd.read_csv(p)
    if df.empty or "macro_f1" not in df.columns:
        return {"representation": representation, "macro_f1": pd.NA}
    computed = df
    if "status" in computed.columns:
        computed = computed[computed["status"].fillna("computed") != "skipped"]
    computed = computed.dropna(subset=["macro_f1"])
    if computed.empty:
        return {"representation": representation, "macro_f1": pd.NA}
    row = computed.sort_values("macro_f1", ascending=False).iloc[0].to_dict()
    return {
        "representation": representation,
        "accuracy": row.get("accuracy", pd.NA),
        "macro_precision": row.get("macro_precision", pd.NA),
        "macro_recall": row.get("macro_recall", pd.NA),
        "macro_f1": row.get("macro_f1", pd.NA),
        "weighted_f1": row.get("weighted_f1", pd.NA),
        "worst_class_recall": row.get("worst_class_recall", pd.NA),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="data/processed/features/mcc5_features.parquet")
    parser.add_argument("--split", default="data/processed/splits/mcc5_source_file_split.csv")
    parser.add_argument("--out", default="results/tables/mcc5_representation_comparison.csv")
    parser.add_argument("--dl_results", default="results/tables/dl_cnn_vib_current_results.csv")
    parser.add_argument("--fusion_results", default="results/tables/mcc5_fusion_results.csv")
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)
    if not _table_exists(args.features) or not Path(args.split).exists():
        print("Missing features or split; skipping representation comparison.")
        return 0
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
    except Exception as exc:
        out = Path(args.out)
        ensure_dir(out.parent)
        pd.DataFrame([{"status": "skipped", "reason": f"missing scikit-learn: {exc}"}]).to_csv(out, index=False)
        print(f"Skipping representation comparison: scikit-learn unavailable: {exc}")
        return 0

    features = read_table(args.features)
    if args.max_windows:
        features = features.head(args.max_windows)
    df = features.merge(pd.read_csv(args.split)[["window_id", "split"]], on="window_id", how="inner")
    if df.empty or args.dry_run:
        return 0
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    if test.empty:
        test = df[df["split"] == "val"]
    if test.empty:
        test = train
    all_x = _feature_matrix(df)
    rows = []
    for name, patterns in REPRESENTATIONS.items():
        cols = _cols(all_x, patterns)
        if not cols:
            continue
        model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1))])
        model.fit(_feature_matrix(train).reindex(columns=cols), train["label_group"].astype(str))
        pred = model.predict(_feature_matrix(test).reindex(columns=cols))
        rows.append({"representation": name, **classification_metrics(test["label_group"].astype(str), pred)})
    rows.append(_optional_result_row(args.dl_results, "raw_deep_learning_cnn"))
    rows.append(_optional_result_row(args.fusion_results, "proposed_fusion"))
    out_df = pd.DataFrame(rows)
    out = Path(args.out)
    ensure_dir(out.parent)
    out_df.to_csv(out, index=False)
    plot_df = out_df.dropna(subset=["macro_f1"])
    if not plot_df.empty:
        figure_dir = ensure_dir("results/figures")
        save_bar_plot(plot_df["representation"], plot_df["macro_f1"], figure_dir / f"{out.stem}.png", ylabel="Macro F1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
