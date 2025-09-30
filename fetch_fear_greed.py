#!/usr/bin/env python3
"""Download CNN Fear & Greed index values for the past 365 days."""

import csv
import datetime as dt
import json
import sys
import urllib.error
import urllib.request

API_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.cnn.com",
    "Referer": "https://www.cnn.com/",
    "Connection": "keep-alive",
}


def fetch_index_history(days: int = 365):
    """Return a list of fear & greed index readings within the lookback window."""
    request = urllib.request.Request(API_URL, headers=HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            detail = ""
        message = f"HTTP {exc.code} error received while fetching data: {exc.reason}"
        if detail:
            message += f" | Response: {detail.strip()[:200]}"
        raise RuntimeError(message) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach CNN endpoint: {exc.reason}") from exc

    history = payload.get("fear_and_greed_historical")
    if not history:
        raise RuntimeError("CNN Fear & Greed data response did not include history.")

    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
    records = []

    if isinstance(history, dict):
        entries = history.get("data") or history.get("values") or history.get("points") or []
    else:
        entries = history

    for entry in entries:
        parsed = _parse_entry(entry)
        if not parsed:
            continue

        date = parsed["date"]
        score = parsed["score"]
        description = parsed["description"]

        if date < cutoff:
            continue

        records.append(
            {
                "date": date.date().isoformat(),
                "rating": score,
                "description": description,
            }
        )

    # Sort just in case the API does not provide chronological order.
    records.sort(key=lambda row: row["date"])
    return records


def _parse_entry(entry):
    """Normalize a single API entry into datetime, score, and description."""
    if isinstance(entry, dict):
        timestamp = entry.get("timestamp") or entry.get("time") or entry.get("x") or entry.get("date")
        score = entry.get("score") or entry.get("value") or entry.get("y")
        description = (
            entry.get("ratingDescription")
            or entry.get("rating")
            or entry.get("classification")
            or entry.get("text")
            or ""
        )
    elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
        timestamp = entry[0]
        score = entry[1]
        description = entry[2] if len(entry) > 2 else ""
    else:
        return None

    resolved_dt = _resolve_datetime(timestamp)
    if resolved_dt is None:
        return None

    score_value = _resolve_score(score)
    if score_value is None:
        return None

    return {"date": resolved_dt, "score": score_value, "description": description or ""}


def _resolve_datetime(value):
    """Convert assorted timestamp formats to a timezone-aware datetime."""
    if value is None:
        return None

    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)

    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds > 1_000_000_000_000:  # assume milliseconds
            seconds /= 1000
        elif seconds > 10_000_000_000:  # assume milliseconds with fewer digits
            seconds /= 1000
        return dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            numeric = float(text)
        except ValueError:
            # Try ISO-8601 style date strings.
            for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
                try:
                    parsed = dt.datetime.strptime(text, fmt)
                except ValueError:
                    continue
                if fmt == "%Y-%m-%d":
                    parsed = dt.datetime.combine(parsed.date(), dt.time(0, 0), tzinfo=dt.timezone.utc)
                else:
                    parsed = parsed.replace(tzinfo=dt.timezone.utc)
                return parsed
            return None
        else:
            return _resolve_datetime(numeric)

    return None


def _resolve_score(value):
    """Normalize numeric score values."""
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return int(round(float(value)))

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(round(float(text)))
        except ValueError:
            # If the rating is textual (e.g., "Fear"), there may be a numeric
            # counterpart stored under a different key that we already
            # considered. Without a number we cannot chart the index, so skip.
            return None

    return None


def write_csv(records, path: str) -> None:
    """Write the fear & greed records to a CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Date", "Rating", "Description"])
        for record in records:
            writer.writerow([record["date"], record["rating"], record["description"]])


def main():
    output_path = "fear_greed_index.csv"
    try:
        records = fetch_index_history()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    try:
        write_csv(records, output_path)
    except OSError as exc:
        print(f"Failed to write CSV file: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Date        Rating  Description")
    for record in records:
        description = record["description"] or "-"
        print(f"{record['date']}  {record['rating']:<7} {description}")

    print(f"\nSaved {len(records)} rows to {output_path}.")


if __name__ == "__main__":
    main()
