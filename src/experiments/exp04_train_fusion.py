"""Train the proposed multisource fusion model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import torch
    import yaml
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from src.experiments.exp03_dl_baselines import _device, _stack
    from src.models.fusion_net import MultisourceFusionNet, save_model_summary

    FUSION_DEPS_AVAILABLE = True
    FUSION_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on installed packages
    torch = None
    yaml = None
    nn = None
    DataLoader = None
    TensorDataset = None
    MultisourceFusionNet = None
    save_model_summary = None
    FUSION_DEPS_AVAILABLE = False
    FUSION_IMPORT_ERROR = exc

from src.utils.io import ensure_dir
from src.utils.logger import get_logger
from src.utils.metrics import classification_metrics, save_confusion_matrix
from src.utils.plotting import save_training_curve
from src.utils.seed import set_global_seed
from src.utils.tables import read_table


ID_COLS = {"window_id", "source_file", "sample_id", "label_group", "label_raw", "condition_type", "npz_path", "split"}


def _scalar_matrix(df: pd.DataFrame, columns: list[str] | None = None):
    numeric = df.drop(columns=[c for c in ID_COLS if c in df.columns], errors="ignore").select_dtypes(include=[np.number, "bool"])
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    if columns is None:
        columns = list(numeric.columns)
    numeric = numeric.reindex(columns=columns)
    return numeric, columns


def _upsert_result(path: str | Path, row: pd.DataFrame, split_file: str) -> None:
    """Write one fusion split result, replacing the previous row for that split."""
    out = Path(path)
    ensure_dir(out.parent)
    if out.exists():
        old = pd.read_csv(out)
        if "split_file" in old.columns:
            old = old[old["split_file"] != split_file]
            row = pd.concat([old, row], ignore_index=True)
    row.to_csv(out, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)

    logger = get_logger("fusion_train")
    if not FUSION_DEPS_AVAILABLE:
        ensure_dir("results/tables")
        pd.DataFrame([{"status": "skipped", "reason": f"missing torch/pyyaml/model imports: {FUSION_IMPORT_ERROR}", "model": "fusion"}]).to_csv(
            "results/tables/mcc5_fusion_results.csv",
            index=False,
        )
        logger.warning("Skipping fusion training: dependencies unavailable: %s", FUSION_IMPORT_ERROR)
        return 0
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) if Path(args.config).exists() else {}
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)
    for path in [args.windows, args.features, args.split]:
        p = Path(path)
        if not p.exists() and not (p.suffix == ".parquet" and p.with_suffix(".csv").exists()):
            logger.warning("Missing required file: %s", path)
            return 0
    windows = read_table(args.windows)
    features = read_table(args.features)
    split = pd.read_csv(args.split)
    df = windows.merge(features, on=["window_id", "source_file", "sample_id", "label_group", "label_raw", "condition_type", "rpm_nominal", "load_nm"], how="inner")
    df = df.merge(split[["window_id", "split"]], on="window_id", how="inner")
    if args.max_windows:
        df = df.head(args.max_windows)
    if args.max_files and "source_file" in df:
        keep = df["source_file"].drop_duplicates().head(args.max_files)
        df = df[df["source_file"].isin(keep)]
    if df.empty or df["label_group"].nunique() < 2:
        logger.warning("Not enough data for fusion training.")
        return 0
    if args.dry_run:
        logger.info("Dry run: would train fusion on %d windows", len(df))
        return 0

    fixed_length = int(cfg.get("fixed_length", 8192))
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    if val_df.empty:
        val_df = test_df if not test_df.empty else train_df
    if test_df.empty:
        test_df = val_df
    if args.max_train_samples and len(train_df) > args.max_train_samples:
        train_df = train_df.sample(args.max_train_samples, random_state=seed)

    labels = sorted(df["label_group"].astype(str).unique())
    label_to_idx = {label: i for i, label in enumerate(labels)}
    vib_train, train_df = _stack(train_df, "vibration_only", fixed_length)
    cur_train, _ = _stack(train_df, "current_only", fixed_length)
    if vib_train.size == 0:
        logger.warning("No loadable fusion windows.")
        return 0
    vib_ch, cur_ch = vib_train.shape[1], cur_train.shape[1]
    vib_val, val_df = _stack(val_df, "vibration_only", fixed_length, vib_ch)
    cur_val, _ = _stack(val_df, "current_only", fixed_length, cur_ch)
    vib_test, test_df = _stack(test_df, "vibration_only", fixed_length, vib_ch)
    cur_test, _ = _stack(test_df, "current_only", fixed_length, cur_ch)

    def norm(train, *others):
        mean = train.mean(axis=(0, 2), keepdims=True)
        std = train.std(axis=(0, 2), keepdims=True)
        std[std < 1e-8] = 1.0
        return [(x - mean) / std for x in (train, *others)], mean, std

    (vib_train, vib_val, vib_test), vib_mean, vib_std = norm(vib_train, vib_val, vib_test)
    (cur_train, cur_val, cur_test), cur_mean, cur_std = norm(cur_train, cur_val, cur_test)
    scal_train_df, scalar_cols = _scalar_matrix(train_df)
    scal_val_df, _ = _scalar_matrix(val_df, scalar_cols)
    scal_test_df, _ = _scalar_matrix(test_df, scalar_cols)
    med = scal_train_df.median()
    scal_train = scal_train_df.fillna(med).to_numpy(dtype=np.float32)
    scal_val = scal_val_df.fillna(med).to_numpy(dtype=np.float32)
    scal_test = scal_test_df.fillna(med).to_numpy(dtype=np.float32)
    mean = scal_train.mean(axis=0, keepdims=True)
    std = scal_train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    scal_train = (scal_train - mean) / std
    scal_val = (scal_val - mean) / std
    scal_test = (scal_test - mean) / std

    y_train = train_df["label_group"].astype(str).map(label_to_idx).to_numpy()
    y_val = val_df["label_group"].astype(str).map(label_to_idx).to_numpy()
    y_test = test_df["label_group"].astype(str).map(label_to_idx).to_numpy()
    device = _device(cfg.get("device", "auto"))
    model = MultisourceFusionNet(vib_ch, cur_ch, scal_train.shape[1], len(labels)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.get("learning_rate", 1e-3)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    class_weights = None
    if cfg.get("use_class_weights", True):
        counts = np.bincount(y_train, minlength=len(labels))
        class_weights = torch.tensor(counts.sum() / np.maximum(counts, 1), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    dataset = TensorDataset(
        torch.tensor(vib_train),
        torch.tensor(cur_train),
        torch.tensor(scal_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=int(cfg.get("batch_size", 64)), shuffle=True)
    epochs = args.epochs or int(cfg.get("epochs", 30))
    patience = args.early_stopping_patience
    if patience is None:
        patience = int(cfg.get("early_stopping_patience", 8))
    history = {"train_loss": [], "val_macro_f1": []}
    best_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    epochs_ran = 0
    for _ in range(epochs):
        epochs_ran += 1
        model.train()
        total = 0.0
        for vb, cb, sb, yb in loader:
            vb, cb, sb, yb = vb.to(device), cb.to(device), sb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(vb, cb, sb), yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(yb)
        history["train_loss"].append(total / max(1, len(dataset)))
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(vib_val).to(device), torch.tensor(cur_val).to(device), torch.tensor(scal_val, dtype=torch.float32).to(device)).argmax(1).cpu().numpy()
        metrics = classification_metrics(y_val, pred, labels=list(range(len(labels))))
        history["val_macro_f1"].append(metrics["macro_f1"])
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if patience > 0 and epochs_without_improvement >= patience:
            logger.info("Early stopping after %d epochs; best val macro_f1=%.4f", epochs_ran, best_f1)
            break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(vib_test).to(device), torch.tensor(cur_test).to(device), torch.tensor(scal_test, dtype=torch.float32).to(device)).argmax(1).cpu().numpy()
    split_name = Path(args.split).stem
    split_file = Path(args.split).name
    result = pd.DataFrame(
        [
            {
                **classification_metrics(y_test, pred, labels=list(range(len(labels)))),
                "model": "fusion",
                "split_file": split_file,
                "epochs_ran": epochs_ran,
                "best_val_macro_f1": best_f1,
            }
        ]
    )
    ensure_dir("results/tables")
    ensure_dir("results/figures")
    ensure_dir("results/checkpoints")
    _upsert_result("results/tables/mcc5_fusion_results.csv", result, split_file)
    save_confusion_matrix(y_test, pred, f"results/figures/mcc5_fusion_{split_name}_confusion_matrix.png", labels=list(range(len(labels))))
    save_training_curve(history, f"results/figures/mcc5_fusion_{split_name}_training_curve.png")
    checkpoint = {
        "model_state": model.state_dict(),
        "labels": labels,
        "scalar_cols": scalar_cols,
        "vib_mean": vib_mean,
        "vib_std": vib_std,
        "cur_mean": cur_mean,
        "cur_std": cur_std,
        "epochs_ran": epochs_ran,
        "best_val_macro_f1": best_f1,
    }
    torch.save(checkpoint, f"results/checkpoints/mcc5_fusion_{split_name}_best.pt")
    if split_name == "mcc5_source_file_split":
        save_confusion_matrix(y_test, pred, "results/figures/mcc5_fusion_confusion_matrix.png", labels=list(range(len(labels))))
        save_training_curve(history, "results/figures/mcc5_fusion_training_curve.png")
        torch.save(checkpoint, "results/checkpoints/mcc5_fusion_best.pt")
    save_model_summary("results/logs/model_summary.txt")
    logger.info("Saved fusion model results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
