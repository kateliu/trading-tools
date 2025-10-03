#!/usr/bin/env python3
"""Visualize correlations among multiple ticker time series."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and plot correlation between multiple ticker price histories.",
    )
    parser.add_argument(
        "--ticker",
        action="append",
        required=True,
        metavar="TICKER=PATH",
        help="CSV file per ticker (Date, Open, High, Low, Close, Volume). e.g. --ticker PLTR=pltr_3650_days.csv",
    )
    parser.add_argument(
        "--metric",
        choices=["close", "log_return", "pct_change"],
        default="log_return",
        help="Value used for correlation (default: log_return)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("correlation_heatmap.png"),
        help="Path to save the heatmap (default: correlation_heatmap.png)",
    )
    return parser.parse_args()


def load_ticker_csv(argument: str) -> pd.DataFrame:
    if "=" not in argument:
        raise ValueError("Ticker argument must be TICKER=PATH")
    ticker, path = argument.split("=", 1)
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path_obj, parse_dates=["Date"])
    required = {"Date", "Close"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    df = df[["Date", "Close"]].copy()
    df.rename(columns={"Close": ticker.upper()}, inplace=True)
    df.sort_values("Date", inplace=True)
    return df.set_index("Date")


def compute_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric == "close":
        return df
    if metric == "log_return":
        return np.log(df).diff().dropna()
    if metric == "pct_change":
        return df.pct_change().dropna()
    raise ValueError(f"Unsupported metric: {metric}")


def main() -> int:
    args = parse_args()
    frames: List[pd.DataFrame] = []
    for spec in args.ticker:
        frames.append(load_ticker_csv(spec))

    merged = pd.concat(frames, axis=1, join="inner")
    merged.dropna(inplace=True)
    if merged.empty:
        raise SystemExit("No overlapping data between provided tickers.")

    values = compute_metric(merged, args.metric)
    corr = values.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True)
    plt.title(f"Ticker Correlation ({args.metric})")
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150)
    print(f"Saved correlation heatmap to {args.output}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
