#!/usr/bin/env python3
"""Inference helper for the intraday linear trend model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from train_intraday_trend_model import add_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict next-horizon log return using the saved intraday model.",
    )
    parser.add_argument(
        "data",
        type=Path,
        help="Path to 1-minute CSV (timestamp, open, high, low, close, volume) or directory of such CSVs.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/intraday_linear_model.json"),
        help="Path to saved model JSON (default: models/intraday_linear_model.json)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=500,
        help="Number of most recent rows to use when multiple rows are available (default: 500).",
    )
    return parser.parse_args()


def load_minutes(path: Path, window: int) -> pd.DataFrame:
    if path.is_dir():
        frames = []
        for csv_path in sorted(path.glob("*.csv")):
            df = pd.read_csv(csv_path)
            frames.append(df)
        if not frames:
            raise RuntimeError(f"No CSV files found in directory {path}")
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(path)
        if df.empty:
            raise RuntimeError(f"File {path} is empty")

    df = df.rename(
        columns={"timestamp": "timestamp", "open": "Open", "close": "Close", "high": "High", "low": "Low", "volume": "Volume"}
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values("timestamp", inplace=True)
    df.set_index("timestamp", inplace=True)

    if window and len(df) > window:
        df = df.iloc[-window:]
    return df


def load_model(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return json.loads(path.read_text())


def prepare_latest_features(df: pd.DataFrame, feature_names: List[str]) -> pd.Series:
    engineered = add_features(df)
    if engineered.empty:
        raise RuntimeError("Not enough rows after feature engineering (increase --window).")
    missing = [name for name in feature_names if name not in engineered.columns]
    if missing:
        raise RuntimeError(f"Engineered features missing required columns: {missing}")
    return engineered[feature_names].iloc[-1]


def predict(feature_vector: pd.Series, model_payload: dict) -> float:
    mean = np.array(model_payload["scaler_mean"])
    std = np.array(model_payload["scaler_std"])
    std[std == 0] = 1.0

    scaled = (feature_vector.values - mean) / std
    x_aug = np.hstack([[1.0], scaled])
    coef = np.array([model_payload["intercept"], *model_payload["coefficients"]])
    return float(x_aug @ coef)


def main() -> None:
    args = parse_args()
    df = load_minutes(args.data, args.window)
    model_payload = load_model(args.model)
    feature_vector = prepare_latest_features(df, model_payload["feature_names"])
    prediction = predict(feature_vector, model_payload)

    close_now = df["Close"].iloc[-1]
    predicted_pct = (np.exp(prediction) - 1.0) * 100
    horizon = model_payload.get("horizon", "?")

    print("Latest timestamp:", df.index[-1])
    print("Current close price:", close_now)
    print(f"Predicted log return for next {horizon} minutes: {prediction:.6f}")
    print(f"Approximate percentage move: {predicted_pct:.4f}%")


if __name__ == "__main__":
    main()
