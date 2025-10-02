#!/usr/bin/env python3
"""Download daily OHLC prices for a ticker over the requested lookback."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_history(ticker: str, days: int):
    """Return list of OHLC records for the ticker."""
    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=days)
    start = start.replace(hour=0, minute=0, second=0, microsecond=0)

    params = urllib.parse.urlencode(
        {
            "period1": str(int(start.timestamp())),
            "period2": str(int(end.timestamp())),
            "interval": "1d",
            "includePrePost": "false",
        }
    )

    url = f"{BASE_URL.format(ticker=urllib.parse.quote(ticker))}?{params}"
    request = urllib.request.Request(url, headers=HEADERS)

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"HTTP {exc.code} error received while fetching data: {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach Yahoo Finance: {exc.reason}") from exc

    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(f"Yahoo Finance returned an error: {chart['error']}")

    results = chart.get("result") or []
    if not results:
        raise RuntimeError("Yahoo Finance returned no chart data.")

    series = results[0]
    timestamps = series.get("timestamp") or []
    quote = (series.get("indicators", {}).get("quote") or [{}])[0]

    opens = quote.get("open") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    closes = quote.get("close") or []
    volumes = quote.get("volume") or []

    records = []
    for ts, o, h, l, c, v in zip(timestamps, opens, highs, lows, closes, volumes):
        if None in (ts, o, h, l, c, v):
            continue
        date_str = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).date().isoformat()
        records.append(
            {
                "date": date_str,
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": float(v),
            }
        )

    if not records:
        raise RuntimeError("No usable OHLC records returned by Yahoo Finance.")

    return records


def write_csv(records, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Date", "Open", "High", "Low", "Close", "Volume"])
        for record in records:
            writer.writerow(
                [
                    record["date"],
                    f"{record['open']:.2f}",
                    f"{record['high']:.2f}",
                    f"{record['low']:.2f}",
                    f"{record['close']:.2f}",
                    str(int(round(record["volume"]))),
                ]
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch daily OHLC stock prices from Yahoo Finance."
    )
    parser.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    parser.add_argument("days", type=int, help="Number of past days to fetch (positive integer)")
    parser.add_argument("output", type=Path, help="Path to output CSV file")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.days <= 0:
        parser.error("days must be a positive integer")

    ticker = args.ticker.strip().upper()

    try:
        records = fetch_history(ticker, args.days)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    try:
        write_csv(records, args.output)
    except OSError as exc:
        print(f"Failed to write CSV file: {exc}", file=sys.stderr)
        return 1

    print(f"Saved {len(records)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
