"""Measure source-file partition sensitivity with a fixed model random state."""

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
    exact_groups,
    feature_block,
    fit_predict,
    source_file_split,
    write_markdown,
)
from src.utils.tables import read_table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/review_runs/review_20260710/classical")
    parser.add_argument("--split_seeds", default="42,43,44,45,46,47,48,49,50,51")
    parser.add_argument("--model_seed", type=int, default=42)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    split_seeds = [int(value) for value in args.split_seeds.split(",") if value.strip()]
    features = read_table(FEATURES_PATH)
    columns = list(_feature_matrix(features).columns)
    membership = pd.DataFrame([{"feature": c, "block": feature_block(c)} for c in columns])
    groups = exact_groups(columns, membership)

    rows = []
    for split_seed in split_seeds:
        split = source_file_split(features, split_seed)
        for model_name in ["random_forest", "xgboost"]:
            metrics, _ = fit_predict(
                features,
                split,
                groups["all_features"],
                model_name,
                seed=args.model_seed,
            )
            rows.append(
                {
                    "split_seed": split_seed,
                    "model_seed": args.model_seed,
                    "model": model_name,
                    **metrics,
                }
            )
    by_split = pd.DataFrame(rows)
    summary = by_split.groupby("model")[["accuracy", "macro_f1", "worst_class_recall"]].agg(
        ["mean", "std", "min", "max"]
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    by_split.to_csv(out / "repeated_source_file_split_fixed_model_seed.csv", index=False)
    summary.to_csv(out / "repeated_source_file_split_fixed_model_seed_summary.csv", index=False)
    write_markdown(
        summary,
        out / "repeated_source_file_split_fixed_model_seed_summary.md",
        "Repeated Source-File Splits With Fixed Model Seed",
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
