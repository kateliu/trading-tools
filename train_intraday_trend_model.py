#!/usr/bin/env python3
"""Train a short-horizon Palantir trend model using 1-minute data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

FEATURE_WINDOWS_RETURN = [1, 5, 10, 15]
FEATURE_WINDOWS_STD = [5, 10, 20]
EMA_WINDOWS = [3, 8, 20, 50]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/validate Palantir intraday trend model.")
    parser.add_argument(
        "data_dir",
        type=Path,
        default=Path("PLTR-one-minute"),
        help="Directory containing daily 1-minute CSV files.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Prediction horizon in minutes (default: 10).",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/intraday_linear_model.json"),
        help="Where to save model coefficients and scaler stats (default: models/intraday_linear_model.json)",
    )
    return parser.parse_args()


def load_minutes(data_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for csv_path in sorted(data_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.sort_values("timestamp", inplace=True)
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No CSV files found in {data_dir}")
    combined = pd.concat(frames, ignore_index=True)
    combined.rename(
        columns={"open": "Open", "close": "Close", "high": "High", "low": "Low", "volume": "Volume"},
        inplace=True,
    )
    combined.set_index("timestamp", inplace=True)
    combined = combined.sort_index()
    return combined


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Log returns
    for window in FEATURE_WINDOWS_RETURN:
        df[f"log_return_{window}"] = np.log(df["Close"] / df["Close"].shift(window))

    # Price range features
    df["range"] = df["High"] - df["Low"]
    df["close_open"] = df["Close"] - df["Open"]
    df["upper_wick"] = df["High"] - df[["Close", "Open"]].max(axis=1)
    df["lower_wick"] = df[["Close", "Open"]].min(axis=1) - df["Low"]

    # Rolling std / Bollinger components
    for window in FEATURE_WINDOWS_STD:
        std = df["Close"].rolling(window).std()
        df[f"std_{window}"] = std
    window = 20
    mid = df["Close"].rolling(window).mean()
    std20 = df["Close"].rolling(window).std()
    df["boll_mid"] = mid
    df["boll_upper"] = mid + 2 * std20
    df["boll_lower"] = mid - 2 * std20
    df["boll_width"] = df["boll_upper"] - df["boll_lower"]
    df["boll_percent_b"] = (df["Close"] - df["boll_lower"]) / df["boll_width"]

    # Exponential moving averages and slopes
    for window in EMA_WINDOWS:
        ema = df["Close"].ewm(span=window, adjust=False).mean()
        df[f"ema_{window}"] = ema
        df[f"ema_delta_{window}"] = ema - ema.shift(1)

    # MACD (12,26,9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_line"] = macd
    df["macd_signal"] = signal
    df["macd_hist"] = macd - signal

    # RSI (14)
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain, index=df.index).rolling(14).mean()
    avg_loss = pd.Series(loss, index=df.index).rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi14"] = 100 - (100 / (1 + rs))

    # Volume features
    df["volume_ma_20"] = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_ma_20"]

    # VWAP distance approximation (rolling)
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    vwap = (typical_price * df["Volume"]).rolling(30).sum() / df["Volume"].rolling(30).sum()
    df["vwap"] = vwap
    df["vwap_gap"] = df["Close"] - vwap

    df.dropna(inplace=True)
    return df


def create_labels(df: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df["future_log_return"] = np.log(df["Close"].shift(-horizon) / df["Close"])
    df.dropna(inplace=True)
    future = df.pop("future_log_return")
    return df, future


def prepare_dataset(data_dir: Path, horizon: int) -> Tuple[pd.DataFrame, pd.Series]:
    raw = load_minutes(data_dir)
    feats = add_features(raw)
    feats, labels = create_labels(feats, horizon)
    return feats, labels


def standardize(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0
    train_scaled = (train - mean) / std
    test_scaled = (test - mean) / std
    return train_scaled, test_scaled, mean, std


def serialize_model(path: Path, feature_names: List[str], coef: np.ndarray, mean: np.ndarray, std: np.ndarray, horizon: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "horizon": horizon,
        "feature_names": feature_names,
        "intercept": float(coef[0]),
        "coefficients": coef[1:].tolist(),
        "scaler_mean": mean.tolist(),
        "scaler_std": std.tolist(),
    }
    import json

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved model parameters to {path}")


def run_training(features: pd.DataFrame, labels: pd.Series, horizon: int, model_output: Path) -> None:
    feature_names = features.columns.tolist()
    X = features.values
    y = labels.values

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    X_train_scaled, X_test_scaled, mean, std = standardize(X_train, X_test)

    X_train_aug = np.hstack([np.ones((X_train_scaled.shape[0], 1)), X_train_scaled])
    X_test_aug = np.hstack([np.ones((X_test_scaled.shape[0], 1)), X_test_scaled])

    # Ordinary least squares solution
    coef, *_ = np.linalg.lstsq(X_train_aug, y_train, rcond=None)

    train_pred = X_train_aug @ coef
    test_pred = X_test_aug @ coef

    def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> None:
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else float('nan')
        print(f"{label} -> RMSE: {rmse:.6f}, MAE: {mae:.6f}, R^2: {r2:.4f}")

    print("Training performance:")
    regression_metrics(y_train, train_pred, "Train")
    print("Validation performance (last 20% of samples):")
    regression_metrics(y_test, test_pred, "Test")

    print("\nFeature set used ({} features):".format(len(feature_names)))
    for name in feature_names:
        print(f" - {name}")

    serialize_model(model_output, feature_names, coef, mean, std, horizon)


def main() -> None:
    args = parse_args()
    features, labels = prepare_dataset(args.data_dir, args.horizon)
    run_training(features, labels, args.horizon, args.model_output)


if __name__ == "__main__":
    main()
