#!/usr/bin/env python3
"""Train an LSTM to predict PLTR closing prices using multi-ticker historical data."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray, index: int) -> np.ndarray:
        return values * self.std[index] + self.mean[index]


class SequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:  # pragma: no cover
        return self.features.shape[0]

    def __getitem__(self, idx: int):  # pragma: no cover
        return self.features[idx], self.targets[idx]


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        output = output[:, -1, :]
        return self.fc(output).squeeze(-1)


def parse_feature_arg(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Feature argument must be TICKER=PATH")
    ticker, path = value.split("=", 1)
    if not ticker:
        raise argparse.ArgumentTypeError("Ticker symbol cannot be empty")
    return ticker.upper(), Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an LSTM to forecast PLTR close using multi-ticker history.",
    )
    parser.add_argument("pltr_csv", type=Path, help="CSV with PLTR OHLCV history (Date, Open, High, Low, Close, Volume)")
    parser.add_argument(
        "--feature",
        action="append",
        type=parse_feature_arg,
        metavar="TICKER=PATH",
        required=True,
        help="Additional ticker CSVs to use as features (e.g. --feature VOO=voo_3650_days.csv)",
    )
    parser.add_argument("--lookback", type=int, default=60, help="Days per input sequence (default 60)")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs (default 60)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default 64)")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate (default 1e-3)")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda")
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/multiticker_lstm.pt"),
        help="Path to save trained weights",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("models/multiticker_lstm.json"),
        help="Path to save training metadata",
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("predictions/multiticker_validation_predictions.csv"),
        help="Where to store validation predictions",
    )
    return parser.parse_args()


def load_price_csv(path: Path, prefix: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df[list(required)].copy()
    df.rename(columns={
        "Date": "date",
        "Open": f"{prefix}_open",
        "High": f"{prefix}_high",
        "Low": f"{prefix}_low",
        "Close": f"{prefix}_close",
        "Volume": f"{prefix}_volume",
    }, inplace=True)
    df.sort_values("date", inplace=True)
    return df


def assemble_dataset(pltr_path: Path, feature_paths: List[Tuple[str, Path]]) -> pd.DataFrame:
    pltr_df = load_price_csv(pltr_path, "pltr")
    frames = [pltr_df]
    for ticker, path in feature_paths:
        frames.append(load_price_csv(path, ticker.lower()))

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="inner")

    merged.set_index("date", inplace=True)
    merged.sort_index(inplace=True)

    # Feature engineering: log returns and normalized columns for each ticker
    for column in list(merged.columns):
        if column.endswith("_close"):
            name = column.replace("_close", "")
            merged[f"{name}_log_return"] = np.log(merged[column]).diff()
        if column.endswith("_volume"):
            name = column.replace("_volume", "")
            merged[f"{name}_volume_z"] = (
                merged[column] - merged[column].rolling(window=20).mean()
            ) / merged[column].rolling(window=20).std().replace(0, np.nan)

    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    merged.dropna(inplace=True)
    if merged.empty:
        raise RuntimeError("Merged dataset is empty after cleaning. Check overlapping date ranges.")
    return merged


def build_sequences(df: pd.DataFrame, lookback: int) -> Tuple[np.ndarray, np.ndarray, Scaler, List[str]]:
    feature_cols = [col for col in df.columns if col != "pltr_close"]
    feature_cols.insert(0, "pltr_close")  # guarantee target close is first for convenience

    array = df[feature_cols].to_numpy(dtype=np.float64)
    if array.shape[0] <= lookback:
        raise RuntimeError("Not enough rows for the chosen lookback window.")

    mean = array.mean(axis=0)
    std = array.std(axis=0)
    std[std == 0] = 1.0
    scaler = Scaler(mean=mean, std=std)
    scaled = scaler.transform(array)

    close_index = feature_cols.index("pltr_close")
    X, y = [], []
    for start in range(0, len(scaled) - lookback):
        end = start + lookback
        target_idx = end
        if target_idx >= len(scaled):
            break
        X.append(scaled[start:end])
        y.append(scaled[target_idx, close_index])

    return np.stack(X), np.array(y), scaler, feature_cols, close_index


def split_dataset(X: np.ndarray, y: np.ndarray, ratio: float = 0.8):
    split = int(len(X) * ratio)
    split = min(max(split, 1), len(X) - 1)
    return SequenceDataset(X[:split], y[:split]), SequenceDataset(X[split:], y[split:])


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer, epochs: int, device: torch.device):
    criterion = nn.MSELoss()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * batch_X.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"[train] Epoch {epoch:03d} | train_loss={train_loss:.6f} val_loss={val_loss:.6f}")


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, scaler: Scaler, close_index: int) -> Dict[str, float]:
    preds, actuals = [], []
    model.eval()
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            preds.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.numpy())

    preds = np.array(preds)
    actuals = np.array(actuals)
    preds_real = scaler.inverse_transform(preds, close_index)
    actuals_real = scaler.inverse_transform(actuals, close_index)

    mse = np.mean((preds_real - actuals_real) ** 2)
    rmse = math.sqrt(mse)
    mae = np.mean(np.abs(preds_real - actuals_real))
    denom = np.where(np.abs(actuals_real) < 1e-6, np.nan, actuals_real)
    mape = float(np.nan_to_num(np.mean(np.abs((preds_real - actuals_real) / denom) * 100), nan=0.0))

    return {"rmse": rmse, "mae": mae, "mape": mape}


def save_predictions(loader: DataLoader, model: nn.Module, device: torch.device, scaler: Scaler, close_index: int, df: pd.DataFrame, lookback: int, path: Path):
    preds = []
    actuals = []
    model.eval()
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            preds.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())

    preds = scaler.inverse_transform(np.array(preds), close_index)
    actuals = scaler.inverse_transform(np.array(actuals), close_index)

    index = df.index[-len(actuals):]
    result_df = pd.DataFrame({
        "date": index,
        "actual_close": actuals,
        "predicted_close": preds,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(path, index=False)
    print(f"[train] Saved validation predictions to {path}")


def save_artifacts(model: nn.Module, model_path: Path, metadata_path: Path, scaler: Scaler, feature_cols: List[str], metrics: Dict[str, float], args: argparse.Namespace):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "feature_cols": feature_cols, "lookback": args.lookback}, model_path)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookback": args.lookback,
        "feature_mean": scaler.mean.tolist(),
        "feature_std": scaler.std.tolist(),
        "metrics": metrics,
        "features": feature_cols,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2))


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)

    dataset = assemble_dataset(args.pltr_csv, args.feature)
    features, targets, scaler, feature_cols, close_index = build_sequences(dataset, args.lookback)
    train_ds, val_ds = split_dataset(features, targets)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = LSTMRegressor(input_size=len(feature_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train(model, train_loader, val_loader, optimizer, args.epochs, device)

    metrics = evaluate(model, val_loader, device, scaler, close_index)
    print(f"[train] Validation metrics: {json.dumps(metrics, indent=2)}")

    save_predictions(val_loader, model, device, scaler, close_index, dataset.iloc[-len(val_ds):], args.lookback, args.predictions_csv)
    save_artifacts(model, args.model_output, args.metadata_output, scaler, feature_cols, metrics, args)

    print("[train] Training complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
