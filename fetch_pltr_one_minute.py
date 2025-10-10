#!/usr/bin/env python3
"""Fetch the last 30 days of PLTR 1-minute aggregates from Polygon.io."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, Iterable, List


POLYGON_BASE_URL = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start}/{end}"
TICKER = "PLTR"
OUTPUT_FOLDER = pathlib.Path("PLTR-one-minute")
DEFAULT_LOOKBACK_DAYS = 30
MAX_CALLS_PER_MINUTE = 5
ENV_FILE = pathlib.Path(".env_dev")
ENV_KEY = "POLYGON_API_KEY"


class PolygonRateLimitError(RuntimeError):
    """Raised when Polygon returns a 429 rate limit response."""


class PolygonDataUnavailableError(RuntimeError):
    """Raised when Polygon denies access to the requested data."""


def load_polygon_api_key(env_key: str = ENV_KEY, env_file: pathlib.Path = ENV_FILE) -> str:
    """Return the Polygon API key from the environment or the .env file."""

    if env_key in os.environ and os.environ[env_key].strip():
        return os.environ[env_key].strip()

    if not env_file.exists():
        raise RuntimeError(
            f"Environment file {env_file} not found and {env_key} not set in environment."
        )

    try:
        with env_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() == env_key:
                    return value.strip().strip('"').strip("'")
    except OSError as exc:
        raise RuntimeError(f"Failed to read {env_file}: {exc}") from exc

    raise RuntimeError(f"{env_key} not found in {env_file} or environment variables.")


def daterange_days(end_date: dt.date, days: int) -> List[dt.date]:
    """Return a list of calendar dates from oldest to newest covering `days`."""

    return [end_date - dt.timedelta(days=offset) for offset in range(days - 1, -1, -1)]


def build_request_url(ticker: str, day: dt.date, api_key: str) -> str:
    """Construct the Polygon aggregates URL for the given ticker and date."""

    start = day.isoformat()
    end = day.isoformat()
    base_url = POLYGON_BASE_URL.format(ticker=ticker, start=start, end=end)
    query = urllib.parse.urlencode({"adjusted": "true", "sort": "asc", "limit": "50000", "apiKey": api_key})
    return f"{base_url}?{query}"


def fetch_polygon_day(day: dt.date, api_key: str) -> List[Dict[str, float]]:
    """Fetch all minute aggregates for the given day."""

    url = build_request_url(TICKER, day, api_key)
    request = urllib.request.Request(url)

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise PolygonRateLimitError("Received 429 rate limit from Polygon") from exc
        message = _extract_http_error_message(exc)
        if exc.code in {403, 404}:
            raise PolygonDataUnavailableError(
                f"Polygon returned HTTP {exc.code} for {day}: {message}"
            ) from exc
        raise RuntimeError(f"HTTP {exc.code} error while fetching {day}: {message}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach Polygon for {day}: {exc.reason}") from exc

    status = payload.get("status")
    if status != "OK":
        message = payload.get("message") or "Polygon returned a non-OK status"
        raise RuntimeError(f"Polygon error for {day}: {message}")

    results = payload.get("results") or []
    records: List[Dict[str, float]] = []
    for item in results:
        ts_ms = item.get("t")
        if ts_ms is None:
            continue
        timestamp = dt.datetime.utcfromtimestamp(ts_ms / 1000).replace(tzinfo=dt.timezone.utc)
        records.append(
            {
                "timestamp": timestamp.isoformat(),
                "open": float(item.get("o", 0.0)),
                "close": float(item.get("c", 0.0)),
                "low": float(item.get("l", 0.0)),
                "high": float(item.get("h", 0.0)),
                "volume": float(item.get("v", 0.0)),
            }
        )

    return records


def write_day_csv(day: dt.date, records: Iterable[Dict[str, float]], folder: pathlib.Path) -> None:
    """Write the day's minute records to a CSV file."""

    folder.mkdir(parents=True, exist_ok=True)
    output_path = folder / f"{day.isoformat()}.csv"

    try:
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "open", "close", "low", "high", "volume"])
            for record in records:
                writer.writerow(
                    [
                        record["timestamp"],
                        f"{record['open']:.4f}",
                        f"{record['close']:.4f}",
                        f"{record['low']:.4f}",
                        f"{record['high']:.4f}",
                        f"{record['volume']:.0f}",
                    ]
                )
    except OSError as exc:
        raise RuntimeError(f"Failed to write CSV for {day}: {exc}") from exc


def throttle_sleep(counter: int, window_start: float) -> float:
    """Sleep if we reached the per-minute call quota and return the new window start."""

    if counter < MAX_CALLS_PER_MINUTE:
        return window_start

    elapsed = time.monotonic() - window_start
    if elapsed < 60:
        time.sleep(60 - elapsed)
    return time.monotonic()


def _extract_http_error_message(exc: urllib.error.HTTPError) -> str:
    """Attempt to derive a helpful error message from the HTTPError payload."""

    try:
        payload = exc.read() or b""
    except Exception:
        return exc.reason

    if not payload:
        return exc.reason

    try:
        data = json.loads(payload.decode("utf-8"))
        if isinstance(data, dict) and data.get("message"):
            return str(data["message"])
    except Exception:
        try:
            return payload.decode("utf-8", errors="ignore").strip() or exc.reason
        except Exception:
            return exc.reason

    return exc.reason


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch recent PLTR 1-minute aggregates from Polygon.io."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help="Number of calendar days to fetch (default: 30).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.days < 1:
        print("--days must be a positive integer", file=sys.stderr)
        return 1

    try:
        api_key = load_polygon_api_key()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    end_date = dt.date.today() - dt.timedelta(days=1)
    days = daterange_days(end_date, args.days)

    call_counter = 0
    window_start = time.monotonic()

    for index, day in enumerate(days, start=1):
        while True:
            try:
                records = fetch_polygon_day(day, api_key)
                break
            except PolygonRateLimitError:
                # Back off for a full minute when Polygon signals throttling.
                time.sleep(60)
            except PolygonDataUnavailableError as exc:
                print(f"{exc}; writing empty CSV", file=sys.stderr)
                records = []
                break
            except RuntimeError as exc:
                print(exc, file=sys.stderr)
                return 1

        write_day_csv(day, records, OUTPUT_FOLDER)
        call_counter += 1
        window_start = throttle_sleep(call_counter, window_start)
        if call_counter == MAX_CALLS_PER_MINUTE:
            call_counter = 0

        print(f"Fetched {day.isoformat()} ({index}/{len(days)}) with {len(records)} rows")

    return 0


if __name__ == "__main__":
    sys.exit(main())
