#!/usr/bin/env python3
"""Train an LSTM-based forecaster on daily OHLCV data."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.features.shape[0]

    def __getitem__(self, idx: int):  # pragma: no cover - trivial
        return self.features[idx], self.targets[idx]


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - exercised indirectly
        output, _ = self.lstm(x)
        output = output[:, -1, :]
        return self.fc(output).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LSTM forecaster for stock prices.")
    parser.add_argument("--ticker", default="PLTR", help="Ticker symbol (used for metadata only).")
    parser.add_argument("--price-csv", type=Path, required=True, help="CSV with Date, Open, High, Low, Close, Volume.")
    parser.add_argument(
        "--sentiment-csv",
        type=Path,
        help="Optional CSV with Date and additional features (e.g., Fear & Greed Rating).",
    )
    parser.add_argument("--lookback", type=int, default=60, help="Number of past days per training sequence.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument(
        "--forecast-days", type=int, default=30, help="Number of future trading days to forecast iteratively."
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("models/lstm_pltr.pt"),
        help="Path to save the trained model weights.",
    )
    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=Path("models/lstm_pltr.json"),
        help="Path to save scaler and training metadata.",
    )
    parser.add_argument(
        "--forecast-json",
        type=Path,
        default=Path("web/data/ai_forecast.json"),
        help="Where to store the generated forecast for the dashboard.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device spec (default cpu). Override to cuda if GPU is available.",
    )
    return parser.parse_args()


def load_price_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    expected_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Price CSV is missing columns: {sorted(missing)}")
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_sentiment(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    if not path.exists():
        print(f"[train] Sentiment file {path} not found; continuing without it.", file=sys.stderr)
        return None
    df = pd.read_csv(path, parse_dates=["Date"])
    value_columns = [col for col in df.columns if col != "Date"]
    df[value_columns] = df[value_columns].apply(pd.to_numeric, errors="coerce")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return None
    df = df[["Date", *numeric_cols]].copy()
    return df


def engineer_features(price_df: pd.DataFrame, sentiment_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = price_df.copy()
    df["log_return"] = np.log(df["Close"]).diff().fillna(0.0)
    df["close_open_pct"] = ((df["Close"] - df["Open"]) / df["Open"].replace(0, np.nan)).fillna(0.0)
    df["high_low_pct"] = ((df["High"] - df["Low"]) / df["Close"].replace(0, np.nan)).fillna(0.0)

    # Volume features
    rolling_mean = df["Volume"].rolling(window=20, min_periods=1).mean()
    rolling_std = df["Volume"].rolling(window=20, min_periods=1).std().replace(0, np.nan)
    df["volume_z"] = ((df["Volume"] - rolling_mean) / rolling_std).fillna(0.0)

    if sentiment_df is not None and not sentiment_df.empty:
        df = df.merge(sentiment_df, on="Date", how="left")
        df = df.ffill()

    df = df.fillna(0.0)
    return df


def build_sequences(
    df: pd.DataFrame, lookback: int, close_index: int, feature_cols: List[str]
) -> tuple[np.ndarray, np.ndarray, Scaler]:
    if len(df) <= lookback:
        raise SystemExit("Not enough rows to build sequences with the requested lookback.")

    feature_array = df[feature_cols].to_numpy(dtype=np.float64)
    mean = feature_array.mean(axis=0)
    std = feature_array.std(axis=0)
    std[std == 0] = 1.0
    scaler = Scaler(mean=mean, std=std)
    scaled = scaler.transform(feature_array)

    X: List[np.ndarray] = []
    y: List[float] = []
    for start in range(0, len(scaled) - lookback):
        end = start + lookback
        target_idx = end
        if target_idx >= len(scaled):
            break
        X.append(scaled[start:end])
        y.append(scaled[target_idx, close_index])

    features = np.stack(X)
    targets = np.array(y, dtype=np.float64)
    return features, targets, scaler


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    criterion = nn.MSELoss()
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_loss = 0.0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            preds = model(features)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * features.size(0)
        epoch_train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                preds = model(features)
                loss = criterion(preds, targets)
                val_loss += loss.item() * features.size(0)
        val_loss /= len(val_loader.dataset)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(val_loss)

        print(f"[train] Epoch {epoch:03d} | train_loss={epoch_train_loss:.6f} val_loss={val_loss:.6f}")

    return {
        "train_loss": history["train_loss"][-1],
        "val_loss": history["val_loss"][-1],
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler: Scaler,
    close_index: int,
) -> Dict[str, float]:
    model.eval()
    preds: List[float] = []
    actuals: List[float] = []
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            outputs = model(features).cpu().numpy()
            preds.extend(outputs.tolist())
            actuals.extend(targets.numpy().tolist())

    preds = np.array(preds)
    actuals = np.array(actuals)

    preds_real = scaler.inverse_transform(preds, close_index)
    actuals_real = scaler.inverse_transform(actuals, close_index)

    mse = np.mean((preds_real - actuals_real) ** 2)
    rmse = math.sqrt(mse)
    mae = np.mean(np.abs(preds_real - actuals_real))
    denom = np.where(np.abs(actuals_real) < 1e-6, np.nan, actuals_real)
    mape_array = np.abs((preds_real - actuals_real) / denom) * 100
    mape = float(np.nan_to_num(np.nanmean(mape_array), nan=0.0))

    return {"rmse": rmse, "mae": mae, "mape": mape}


def iterative_forecast(
    model: nn.Module,
    data_df: pd.DataFrame,
    scaler: Scaler,
    feature_cols: List[str],
    lookback: int,
    forecast_days: int,
    close_index: int,
    device: torch.device,
) -> List[Dict[str, float]]:
    if forecast_days <= 0:
        return []

    data_df = data_df.copy()
    feature_array = data_df[feature_cols].to_numpy(dtype=np.float64)
    scaled = scaler.transform(feature_array)
    window = scaled[-lookback:].copy()

    last_date = data_df["Date"].iloc[-1]
    if hasattr(last_date, "to_pydatetime"):
        last_date = last_date.to_pydatetime()
    predictions: List[Dict[str, float]] = []

    for step in range(1, forecast_days + 1):
        tensor = torch.as_tensor(window[np.newaxis, :, :], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_scaled = model(tensor).cpu().item()
        pred_close = float(scaler.inverse_transform(np.array([pred_scaled]), close_index)[0])

        next_date = next_market_day(last_date)
        predictions.append({"date": next_date.strftime("%Y-%m-%d"), "close": round(pred_close, 4)})
        last_date = next_date

        new_row = window[-1].copy()
        new_row[close_index] = pred_scaled
        # Reset derived features where appropriate since we lack future context
        for name_index, name in enumerate(feature_cols):
            if name in {"log_return", "close_open_pct", "high_low_pct", "volume_z"}:
                new_row[name_index] = 0.0
        window = np.vstack([window[1:], new_row])

    return predictions


def next_market_day(date) -> datetime:
    candidate = date + pd.Timedelta(days=1)
    while candidate.weekday() >= 5:  # skip weekends
        candidate += pd.Timedelta(days=1)
    if hasattr(candidate, "to_pydatetime"):
        return candidate.to_pydatetime()
    return candidate


def save_artifacts(
    model: nn.Module,
    model_path: Path,
    metadata_path: Path,
    scaler: Scaler,
    feature_cols: List[str],
    args: argparse.Namespace,
    metrics: Dict[str, float],
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "feature_cols": feature_cols}, model_path)

    metadata = {
        "ticker": args.ticker,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookback": args.lookback,
        "feature_mean": scaler.mean.tolist(),
        "feature_std": scaler.std.tolist(),
        "metrics": metrics,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2))


def save_forecast(
    path: Path,
    ticker: str,
    predictions: Sequence[Dict[str, float]],
    metrics: Dict[str, float],
    lookback: int,
    forecast_days: int,
) -> None:
    payload = {
        "ticker": ticker,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookback": lookback,
        "forecast_days": forecast_days,
        "metrics": metrics,
        "predictions": list(predictions),
        "source": "lstm",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"[train] Wrote forecast to {path}")


def main() -> int:
    args = parse_args()

    device = torch.device(args.device)

    price_df = load_price_data(args.price_csv)
    sentiment_df = load_sentiment(args.sentiment_csv)
    feature_df = engineer_features(price_df, sentiment_df)

    feature_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "log_return",
        "close_open_pct",
        "high_low_pct",
        "volume_z",
    ]

    if sentiment_df is not None:
        feature_cols.extend(col for col in sentiment_df.columns if col != "Date")

    features, targets, scaler = build_sequences(feature_df, args.lookback, feature_cols.index("Close"), feature_cols)

    split = int(len(features) * 0.8)
    if split <= 0 or split >= len(features):
        split = max(1, len(features) - 1)
    train_ds = SequenceDataset(features[:split], targets[:split])
    val_ds = SequenceDataset(features[split:], targets[split:])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = LSTMRegressor(input_size=len(feature_cols)).to(device)
    optimizer = torch.optim.Adam(modㄟel.parameters(), lr=args.learning_rate)

    train_metrics = train_model(model, train_loader, val_loader, args.epochs, optimizer, device)

    eval_metrics = evaluate(model, val_loader, device, scaler, feature_cols.index("Close"))
    print(f"[train] Evaluation metrics: {json.dumps(eval_metrics, indent=2)}")

    predictions = iterative_forecast(
        model,
        feature_df,
        scaler,
        feature_cols,
        args.lookback,
        args.forecast_days,
        feature_cols.index("Close"),
        device,
    )

    save_artifacts(model, args.output_model, args.output_metadata, scaler, feature_cols, args, eval_metrics)
    save_forecast(args.forecast_json, args.ticker, predictions, eval_metrics, args.lookback, args.forecast_days)

    print("[train] Training run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
