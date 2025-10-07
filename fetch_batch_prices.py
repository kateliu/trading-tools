#!/usr/bin/env python3
"""Fetch OHLCV data for multiple tickers listed on stdin or a file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import fetch_stock_prices


DEFAULT_DAYS = 365


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch OHLC prices and volume for multiple tickers.",
    )
    parser.add_argument(
        "tickers",
        type=Path,
        nargs="?",
        help="Optional path to a text file containing one ticker per line. Defaults to stdin.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help="Number of calendar days to fetch from Yahoo Finance (default: 365).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("10062025"),
        help="Directory where CSVs will be stored (default: 10062025)",
    )
    return parser.parse_args()


def read_tickers(path: Path | None) -> List[str]:
    if path:
        if not path.exists():
            raise FileNotFoundError(f"Ticker list file not found: {path}")
        content = path.read_text(encoding="utf-8")
    else:
        content = sys.stdin.read()
    tickers = [line.strip().upper() for line in content.splitlines() if line.strip()]
    if not tickers:
        raise ValueError("No tickers provided.")
    return tickers


def write_csv(records, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fetch_stock_prices.write_csv(records, path)


def fetchTicker(ticker: str, days: int, dest_dir: Path) -> None:
    try:
        records = fetch_stock_prices.fetch_history(ticker, days)
    except RuntimeError as exc:
        print(f"[error] {ticker}: {exc}", file=sys.stderr)
        return

    output_path = dest_dir / f"{ticker}.csv"
    try:
        write_csv(records, output_path)
    except OSError as exc:
        print(f"[error] Failed to write {output_path}: {exc}", file=sys.stderr)
        return

    print(f"Saved {len(records)} rows for {ticker} to {output_path}")


def main() -> int:
    args = parse_args()
    if args.days <= 0:
        print("--days must be a positive integer", file=sys.stderr)
        return 1

    try:
        tickers = read_tickers(args.tickers)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    output_dir = args.output_dir
    for ticker in tickers:
        fetchTicker(ticker, args.days, output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
