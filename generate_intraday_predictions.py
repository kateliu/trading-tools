#!/usr/bin/env python3
"""Generate predictions for PLTR intraday data and visualize them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from string import Template
from typing import Dict, List

import numpy as np
import pandas as pd

from train_intraday_trend_model import add_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Produce next-horizon predictions for intraday data and render a Plotly page.",
    )
    parser.add_argument(
        "data",
        type=Path,
        help="CSV file or directory containing 1-minute bars (timestamp, open, high, low, close, volume).",
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
        default=2000,
        help="Number of most recent rows to use when concatenating files (default: 2000).",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("predictions/intraday_predictions.csv"),
        help="Where to save the predictions CSV (default: predictions/intraday_predictions.csv)",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=Path("web/intraday_predictions.html"),
        help="Where to save the Plotly HTML page (default: web/intraday_predictions.html)",
    )
    return parser.parse_args()


def load_minutes(path: Path, window: int) -> pd.DataFrame:
    if path.is_dir():
        frames = [pd.read_csv(csv_path) for csv_path in sorted(path.glob("*.csv"))]
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


def load_model(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return json.loads(path.read_text())


def compute_predictions(df: pd.DataFrame, model_payload: Dict[str, object]) -> pd.DataFrame:
    features = add_features(df)
    feature_names: List[str] = model_payload["feature_names"]
    missing = [name for name in feature_names if name not in features.columns]
    if missing:
        raise RuntimeError(f"Engineered features missing required columns: {missing}")

    feature_matrix = features[feature_names]
    mean = np.array(model_payload["scaler_mean"])
    std = np.array(model_payload["scaler_std"])
    std[std == 0] = 1.0
    scaled = (feature_matrix.values - mean) / std

    coef = np.array([model_payload["intercept"], *model_payload["coefficients"]])
    intercept = coef[0]
    weights = coef[1:]
    predictions = intercept + scaled @ weights
    predictions = np.asarray(predictions, dtype=float)

    horizon = model_payload.get("horizon", 10)

    result = pd.DataFrame(index=feature_matrix.index)
    result["close"] = df.loc[result.index, "Close"]
    result["predicted_log_return"] = predictions
    result["predicted_future_price"] = result["close"] * np.exp(predictions)
    result["actual_future_price"] = df["Close"].shift(-horizon).loc[result.index]
    result["predicted_timestamp"] = result.index + pd.to_timedelta(horizon, unit="m")
    return result


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    print(f"Saved predictions to {path}")


def render_html(df: pd.DataFrame, horizon: int, path: Path) -> None:
    df = df.dropna(subset=["actual_future_price"])
    if df.empty:
        raise RuntimeError("No rows with actual future price available for HTML plot.")

    actual_times = df["predicted_timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    predicted_series = df["predicted_future_price"].tolist()
    actual_series = df["actual_future_price"].tolist()

    template = Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Intraday Predictions vs Actuals</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
    body { font-family: 'Segoe UI', sans-serif; margin: 0; background: #f9fafb; color: #111827; }
    header { padding: 1rem 1.5rem; background: #1f2937; color: #f9fafb; }
    #chart { width: 100%; height: calc(100vh - 160px); }
</style>
</head>
<body>
<header>
    <h1>Palantir ${horizon}-Minute Forecast</h1>
    <p>Comparing predicted future close vs actual close at the forecast horizon.</p>
</header>
<div id="chart"></div>
<script>
const times = $times;
const predicted = $predicted;
const actual = $actual;
const tracePred = {
    x: times,
    y: predicted,
    mode: 'lines',
    name: 'Predicted Future Close',
    line: { color: '#f97316', width: 2 }
};
const traceAct = {
    x: times,
    y: actual,
    mode: 'lines',
    name: 'Actual Future Close',
    line: { color: '#2563eb', width: 2 }
};
const layout = {
    margin: { l: 60, r: 40, t: 60, b: 50 },
    hovermode: 'x unified',
    xaxis: { title: 'Timestamp (UTC)' },
    yaxis: { title: 'Price (USD)' }
};
Plotly.newPlot('chart', [tracePred, traceAct], layout, { displaylogo: false, responsive: true });
</script>
</body>
</html>
""")
    html = template.substitute(
        times=json.dumps(actual_times.tolist()),
        predicted=json.dumps(predicted_series),
        actual=json.dumps(actual_series),
        horizon=horizon,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    print(f"Saved comparison chart to {path}")


def main() -> None:
    args = parse_args()
    df = load_minutes(args.data, args.window)
    model_payload = load_model(args.model)
    predictions = compute_predictions(df, model_payload)
    save_csv(predictions, args.csv_output)
    render_html(predictions, model_payload.get("horizon", 10), args.html_output)


if __name__ == "__main__":
    main()
