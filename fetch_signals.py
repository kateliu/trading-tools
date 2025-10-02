#!/usr/bin/env python3
"""Fetch daily OHLCV data for a ticker and engineer technical signals."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import fetch_stock_prices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download OHLCV data and compute technical indicators for a ticker.",
    )
    parser.add_argument("ticker", help="Ticker symbol, e.g. PLTR")
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of calendar days of history to request (default: 365)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("signals.csv"),
        help="File path to write the engineered feature table (default: signals.csv)",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        help="Optional path to a local CSV containing Date, Open, High, Low, Close, Volume columns.",
    )
    return parser.parse_args()


def fetch_ohlcv(ticker: str, days: int) -> pd.DataFrame:
    records = fetch_stock_prices.fetch_history(ticker, days)
    if not records:
        raise RuntimeError("No price data returned from Yahoo Finance.")
    return records_to_frame(records)


def load_from_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"Input CSV '{path}' not found")

    df = pd.read_csv(path)
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise RuntimeError(f"Input CSV missing required columns: {sorted(missing)}")

    df = df[list(required)].copy()
    df.rename(columns={col: col.lower() for col in df.columns}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    df = df.dropna()
    if df.empty:
        raise RuntimeError("Filtered CSV has no usable rows.")
    return df


def records_to_frame(records: List[Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df.sort_values("date", inplace=True)
    df.set_index("date", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]]
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Returns & momentum
    out["log_return"] = np.log(out["close"]).diff()
    out["pct_change"] = out["close"].pct_change()
    out["sma_20"] = out["close"].rolling(window=20).mean()
    out["sma_50"] = out["close"].rolling(window=50).mean()
    out["ema_12"] = out["close"].ewm(span=12, adjust=False).mean()
    out["ema_26"] = out["close"].ewm(span=26, adjust=False).mean()
    out["macd"] = out["ema_12"] - out["ema_26"]
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()

    # Volatility & ranges
    out["true_range"] = np.maximum.reduce(
        [
            out["high"] - out["low"],
            (out["high"] - out["close"].shift()).abs(),
            (out["low"] - out["close"].shift()).abs(),
        ]
    )
    out["atr_14"] = out["true_range"].rolling(window=14).mean()
    out["rolling_volatility_30"] = out["pct_change"].rolling(window=30).std() * np.sqrt(252)
    out["bollinger_mid"] = out["sma_20"]
    rolling_std_20 = out["close"].rolling(window=20).std()
    out["bollinger_upper"] = out["bollinger_mid"] + 2 * rolling_std_20
    out["bollinger_lower"] = out["bollinger_mid"] - 2 * rolling_std_20

    # Relative strength index (RSI)
    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # Price shape
    out["close_open_pct"] = (out["close"] - out["open"]) / out["open"].replace(0, np.nan)
    out["high_low_pct"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
    out["gap_pct"] = out["open"].pct_change()

    # Volume analytics
    out["volume_sma_20"] = out["volume"].rolling(window=20).mean()
    out["volume_roc"] = out["volume"].pct_change()
    volume_std = out["volume"].rolling(window=20).std().replace(0, np.nan)
    out["volume_z"] = (out["volume"] - out["volume_sma_20"]) / volume_std

    # Market structure (support/resistance approximations)
    out["rolling_high_50"] = out["high"].rolling(window=50).max()
    out["rolling_low_50"] = out["low"].rolling(window=50).min()

    out.drop(columns=["true_range"], inplace=True)
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out = out.dropna().copy()

    return out


def main() -> int:
    args = parse_args()
    if args.days <= 0:
        print("--days must be a positive integer", file=sys.stderr)
        return 1

    ticker = args.ticker.strip().upper()
    try:
        if args.input_csv:
            price_df = load_from_csv(args.input_csv)
        else:
            price_df = fetch_ohlcv(ticker, args.days)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    enriched = compute_indicators(price_df)
    if enriched.empty:
        print("Insufficient data to compute indicators (consider increasing --days)", file=sys.stderr)
        return 1

    enriched.insert(0, "ticker", ticker)
    enriched.index.name = "date"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.output, float_format="%.6f")
    print(f"Saved {len(enriched)} rows with technical signals to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
