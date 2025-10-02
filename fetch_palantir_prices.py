#!/usr/bin/env python3
"""Download Palantir daily OHLC data for the past 365 days from Yahoo Finance."""

import datetime as dt
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
HEADERS = {"User-Agent": "Mozilla/5.0"}

def fetch_history(ticker: str, days: int = 365):
    """Return a list of daily OHLC records for the given ticker and lookback."""
    # Yahoo Finance expects UNIX timestamps delimiting the requested window.
    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=days)
    start = start.replace(hour=0, minute=0, second=0, microsecond=0)
    period1 = int(start.timestamp())
    period2 = int(end.timestamp())

    query = urllib.parse.urlencode(
        {
            "period1": str(period1),
            "period2": str(period2),
            "interval": "1d",
            "includePrePost": "false",
        }
    )

    url = f"{BASE_URL.format(ticker=ticker)}?{query}"
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

    return records

def main():
    ticker = "PLTR"
    output_path = "palantir_prices.csv"
    try:
        records = fetch_history(ticker)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    try:
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["Date", "Open", "High", "Low", "Close"])
            for record in records:
                writer.writerow(
                    [
                        record["date"],
                        f"{record['open']:.2f}",
                        f"{record['high']:.2f}",
                        f"{record['low']:.2f}",
                        f"{record['close']:.2f}",
                    ]
                )
    except OSError as exc:
        print(f"Failed to write CSV file: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Date        Open      High      Low       Close     Volume")
    for record in records:
        print(
            f"{record['date']}  "
            f"{record['open']:<9.2f}{record['high']:<9.2f}{record['low']:<9.2f}{record['close']:<9.2f}{int(round(record['volume'])):>11}"
        )

if __name__ == "__main__":
    main()
