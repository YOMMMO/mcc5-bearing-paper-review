from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_formal_dl_fusion_gpu import (
    RunPaths,
    SPLIT_PATHS,
    load_formal_tables,
    prepare_raw_arrays,
    require_cuda,
    summarize_results,
    train_raw_one,
)


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def run_paths(root: Path, fixed_length: int) -> RunPaths:
    length_root = root / f"length_{fixed_length}"
    return RunPaths(
        run_id=f"review_raw_length_{fixed_length}",
        root=length_root,
        tables=length_root / "tables",
        figures=length_root / "figures",
        logs=length_root / "logs",
        checkpoints=length_root / "checkpoints",
        docs=root,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched raw-CNN sensitivity to retained sample length.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/review_runs/review_20260710/raw_length_sensitivity"),
    )
    parser.add_argument("--lengths", default="8192,12800")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    args = parser.parse_args()

    require_cuda()
    lengths = parse_ints(args.lengths)
    seeds = parse_ints(args.seeds)
    args.out.mkdir(parents=True, exist_ok=True)
    windows, _, splits = load_formal_tables()
    split_name = "cross_rpm"
    split = splits[split_name]

    rows: list[dict[str, object]] = []
    for fixed_length in lengths:
        paths = run_paths(args.out, fixed_length)
        for directory in [paths.root, paths.tables, paths.figures, paths.logs, paths.checkpoints]:
            directory.mkdir(parents=True, exist_ok=True)
        print(f"[raw-length] loading fixed_length={fixed_length}")
        arrays = prepare_raw_arrays(windows, split, seed=42, fixed_length=fixed_length)
        for seed in seeds:
            print(f"[raw-length] fixed_length={fixed_length} seed={seed}")
            row = train_raw_one(
                arrays=arrays,
                model_name="cnn",
                split_name=split_name,
                split_file=SPLIT_PATHS[split_name].name,
                seed=seed,
                paths=paths,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
            )
            row["fixed_length"] = fixed_length
            row["retained_seconds"] = fixed_length / 12800.0
            rows.append(row)
        del arrays
        gc.collect()
        torch.cuda.empty_cache()

    by_seed = pd.DataFrame(rows)
    summary = summarize_results(by_seed, ["fixed_length", "retained_seconds", "model", "split_name"])
    by_seed.to_csv(args.out / "raw_length_sensitivity_by_seed.csv", index=False)
    summary.to_csv(args.out / "raw_length_sensitivity_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
