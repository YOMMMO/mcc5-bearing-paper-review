"""Tiny deep-learning baselines on saved NPZ windows."""

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
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from src.models.cnn1d import CurrentOnlyCNN, VibCurrentCNN, VibrationOnlyCNN
    from src.models.tcn import SimpleTCN
    from src.models.transformer1d import Transformer1DClassifier

    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on installed packages
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    CurrentOnlyCNN = VibCurrentCNN = VibrationOnlyCNN = SimpleTCN = Transformer1DClassifier = None
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = exc

from src.signal.preprocessing import resample_or_pad
from src.utils.io import ensure_dir
from src.utils.logger import get_logger
from src.utils.metrics import classification_metrics, save_confusion_matrix
from src.utils.plotting import save_training_curve
from src.utils.seed import set_global_seed
from src.utils.tables import read_table


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load_npz(path: str | Path, input_mode: str, fixed_length: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        vib = data.get("vibration", np.empty((0, 0)))
        cur = data.get("current", np.empty((0, 0)))
    if input_mode in {"vibration", "vibration_only"}:
        arr = vib
    elif input_mode in {"current", "current_only"}:
        arr = cur
    else:
        arr = np.concatenate([vib, cur], axis=0) if vib.size or cur.size else np.empty((0, 0))
    if arr.size == 0:
        arr = np.zeros((1, fixed_length), dtype=np.float32)
    return resample_or_pad(arr, fixed_length).astype(np.float32)


def _stack(df: pd.DataFrame, input_mode: str, fixed_length: int, channels: int | None = None):
    arrays = []
    ok_rows = []
    for _, row in df.iterrows():
        path = Path(str(row.get("npz_path", "")))
        if not path.exists():
            continue
        arr = _load_npz(path, input_mode, fixed_length)
        if channels is None:
            channels = arr.shape[0]
        if arr.shape[0] < channels:
            arr = np.vstack([arr, np.zeros((channels - arr.shape[0], fixed_length), dtype=np.float32)])
        elif arr.shape[0] > channels:
            arr = arr[:channels]
        arrays.append(arr)
        ok_rows.append(row)
    if not arrays:
        return np.empty((0, channels or 1, fixed_length), dtype=np.float32), pd.DataFrame()
    return np.stack(arrays), pd.DataFrame(ok_rows)


def _make_model(model_name: str, in_channels: int, num_classes: int):
    if model_name == "tcn":
        return SimpleTCN(in_channels, num_classes)
    if model_name == "transformer":
        return Transformer1DClassifier(in_channels, num_classes)
    if in_channels <= 3:
        return VibrationOnlyCNN(in_channels, num_classes)
    if model_name == "current":
        return CurrentOnlyCNN(in_channels, num_classes)
    return VibCurrentCNN(in_channels, num_classes)


def _upsert_result(path: str | Path, row: pd.DataFrame, split_file: str) -> None:
    """Write one split result, replacing the previous row for the same split."""
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
    parser.add_argument("--split", required=True)
    parser.add_argument("--input_mode", default="vib_current")
    parser.add_argument("--model", default="cnn", choices=["cnn", "tcn", "transformer", "current"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--early_stopping_patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--fixed_length", type=int, default=8192)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)

    set_global_seed(args.seed)
    logger = get_logger("dl_baselines")
    if not TORCH_AVAILABLE:
        ensure_dir("results/tables")
        pd.DataFrame(
            [
                {
                    "status": "skipped",
                    "reason": f"missing torch or model imports: {TORCH_IMPORT_ERROR}",
                    "model": args.model,
                    "input_mode": args.input_mode,
                }
            ]
        ).to_csv(f"results/tables/dl_{args.model}_{args.input_mode}_results.csv", index=False)
        logger.warning("Skipping DL baseline: torch unavailable: %s", TORCH_IMPORT_ERROR)
        return 0
    if not Path(args.windows).exists() and not Path(args.windows).with_suffix(".csv").exists():
        logger.warning("Missing windows file.")
        return 0
    if not Path(args.split).exists():
        logger.warning("Missing windows or split file.")
        return 0
    windows = read_table(args.windows)
    split = pd.read_csv(args.split)
    df = windows.merge(split[["window_id", "split"]], on="window_id", how="inner")
    if args.max_windows:
        df = df.head(args.max_windows)
    if args.max_files and "source_file" in df:
        keep = df["source_file"].drop_duplicates().head(args.max_files)
        df = df[df["source_file"].isin(keep)]
    if df.empty or df["label_group"].nunique() < 2:
        logger.warning("Not enough labeled windows for DL baseline.")
        return 0
    if args.dry_run:
        logger.info("Dry run: would train on %d windows", len(df))
        return 0

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    if val_df.empty:
        val_df = test_df if not test_df.empty else train_df
    if test_df.empty:
        test_df = val_df
    if args.max_train_samples and len(train_df) > args.max_train_samples:
        train_df = train_df.sample(args.max_train_samples, random_state=args.seed)

    labels = sorted(df["label_group"].astype(str).unique())
    label_to_idx = {label: i for i, label in enumerate(labels)}
    X_train, train_df = _stack(train_df, args.input_mode, args.fixed_length)
    if X_train.size == 0:
        logger.warning("No loadable train NPZ windows.")
        return 0
    channels = X_train.shape[1]
    X_val, val_df = _stack(val_df, args.input_mode, args.fixed_length, channels)
    X_test, test_df = _stack(test_df, args.input_mode, args.fixed_length, channels)
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std[std < 1e-8] = 1.0
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std
    y_train = train_df["label_group"].astype(str).map(label_to_idx).to_numpy()
    y_val = val_df["label_group"].astype(str).map(label_to_idx).to_numpy()
    y_test = test_df["label_group"].astype(str).map(label_to_idx).to_numpy()

    device = _device(args.device)
    model = _make_model(args.model, channels, len(labels)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    history = {"train_loss": [], "val_macro_f1": []}
    best_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    epochs_ran = 0
    for _epoch in range(args.epochs):
        epochs_ran += 1
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
        history["train_loss"].append(total / max(1, len(X_train)))
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.tensor(X_val).to(device))
            val_pred = val_logits.argmax(dim=1).cpu().numpy()
        val_metrics = classification_metrics(y_val, val_pred, labels=list(range(len(labels))))
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            logger.info("Early stopping after %d epochs; best val macro_f1=%.4f", epochs_ran, best_f1)
            break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_test).to(device)).argmax(dim=1).cpu().numpy()
    split_name = Path(args.split).stem
    split_file = Path(args.split).name
    metrics = classification_metrics(y_test, pred, labels=list(range(len(labels))))
    result = pd.DataFrame(
        [
            {
                **metrics,
                "model": args.model,
                "input_mode": args.input_mode,
                "split_file": split_file,
                "epochs_ran": epochs_ran,
                "best_val_macro_f1": best_f1,
            }
        ]
    )
    ensure_dir("results/tables")
    _upsert_result(f"results/tables/dl_{args.model}_{args.input_mode}_results.csv", result, split_file)
    save_confusion_matrix(y_test, pred, f"results/figures/dl_{args.model}_{args.input_mode}_{split_name}_confusion_matrix.png", labels=list(range(len(labels))))
    save_training_curve(history, f"results/figures/dl_{args.model}_{args.input_mode}_{split_name}_training_curve.png")
    ensure_dir("results/checkpoints")
    torch.save(
        {
            "model_state": model.state_dict(),
            "labels": labels,
            "mean": mean,
            "std": std,
            "epochs_ran": epochs_ran,
            "best_val_macro_f1": best_f1,
        },
        f"results/checkpoints/dl_{args.model}_{args.input_mode}_{split_name}.pt",
    )
    logger.info("Saved tiny DL baseline results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
