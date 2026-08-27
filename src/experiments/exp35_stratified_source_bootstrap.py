"""Recompute main-model uncertainty with class-stratified source-file bootstrap."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.exp02_ml_baselines import _feature_matrix
from src.experiments.exp34_comprehensive_review_audit import (
    FEATURES_PATH,
    FORMAL_SPLIT_DIR,
    cluster_metrics,
    exact_groups,
    feature_block,
    fit_predict,
    write_markdown,
)
from src.utils.tables import read_table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/review_runs/review_20260710/classical")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    features = read_table(FEATURES_PATH)
    columns = list(_feature_matrix(features).columns)
    membership = pd.DataFrame([{"feature": c, "block": feature_block(c)} for c in columns])
    groups = exact_groups(columns, membership)
    splits = {
        path.stem.replace("mcc5_formal_", "").replace("_split", ""): pd.read_csv(path)
        for path in sorted(FORMAL_SPLIT_DIR.glob("mcc5_formal_*_split.csv"))
    }
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
            splits[split_name],
            groups["all_features"],
            model_name,
            seed=42,
        )
        file_metrics, cluster_ci = cluster_metrics(predictions, seed=42, n_boot=args.bootstrap)
        uncertainty_rows.append({"split": split_name, "model": model_name, **metrics, **cluster_ci})
        file_rows.append(
            {
                "split": split_name,
                "model": model_name,
                "source_file_count": predictions["source_file"].nunique(),
                **file_metrics,
            }
        )
    uncertainty = pd.DataFrame(uncertainty_rows)
    file_level = pd.DataFrame(file_rows)
    uncertainty.to_csv(out / "source_file_cluster_bootstrap_stratified.csv", index=False)
    file_level.to_csv(out / "source_file_majority_vote_metrics_stratified.csv", index=False)
    write_markdown(
        uncertainty,
        out / "source_file_cluster_bootstrap_stratified.md",
        "Class-Stratified Source-File Bootstrap",
    )
    print(uncertainty.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
