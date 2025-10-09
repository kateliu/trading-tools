#!/usr/bin/env python3
"""Generate an interactive web page highlighting large Palantir log-return days."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from string import Template
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Palantir event browser from OHLC data.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("10062025/PLTR.csv"),
        help="Path to Palantir OHLC CSV (default: 10062025/PLTR.csv)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Absolute log-return threshold for event detection (default: 0.05)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("web/palantir_events.html"),
        help="Destination HTML file (default: web/palantir_events.html)",
    )
    return parser.parse_args()


def load_price_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    required = {"Date", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"CSV is missing required columns: {sorted(missing)}")
    df = df.sort_values("Date").reset_index(drop=True)
    df["Close"] = df["Close"].astype(float)
    df["log_return"] = np.log(df["Close"].div(df["Close"].shift(1)))
    df.dropna(subset=["log_return"], inplace=True)
    return df


def select_events(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    mask = df["log_return"].abs() >= threshold
    return df.loc[mask].copy()


def build_payload(df: pd.DataFrame, events: pd.DataFrame) -> Dict[str, object]:
    all_points = [
        {
            "date": row.Date.strftime("%Y-%m-%d"),
            "close": row.Close,
            "log_return": row.log_return,
        }
        for row in df.itertuples()
    ]

    event_rows = []
    for row in events.itertuples():
        date_str = row.Date.strftime("%Y-%m-%d")
        event_rows.append(
            {
                "date": date_str,
                "close": row.Close,
                "log_return": row.log_return,
                "search_url": f"https://news.google.com/search?q=Palantir%20Technologies%20({date_str})",
            }
        )

    return {"series": all_points, "events": event_rows}


def render_html(payload: Dict[str, object], threshold: float) -> str:
    data_json = json.dumps(payload)
    template = Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Palantir Event Explorer</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
    body { font-family: 'Segoe UI', sans-serif; background: #f9fafb; margin: 0; color: #111827; }
    header { background: #111827; color: #f9fafb; padding: 1.25rem 1.75rem; }
    header h1 { margin: 0 0 0.4rem; font-size: 1.75rem; }
    header p { margin: 0; font-size: 0.95rem; opacity: 0.85; }
    section { padding: 1.25rem 1.75rem; }
    #chart { width: 100%; height: 420px; }
    table { width: 100%; border-collapse: collapse; margin-top: 1.25rem; }
    th, td { padding: 0.65rem 0.75rem; border-bottom: 1px solid #d1d5db; text-align: left; font-size: 0.95rem; }
    th { background: #e5e7eb; font-weight: 600; }
    tr:hover { background: #f3f4f6; }
    a { color: #2563eb; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .badge { display: inline-block; padding: 0.2rem 0.45rem; border-radius: 0.5rem; font-size: 0.85rem; color: #fff; }
    .badge.pos { background: #16a34a; }
    .badge.neg { background: #dc2626; }
    footer { background: #e5e7eb; padding: 1rem 1.75rem; font-size: 0.9rem; }
</style>
</head>
<body>
<header>
    <h1>Palantir Log-Return Events</h1>
    <p>Highlighting daily log returns with |value| ≥ $threshold</p>
</header>
<section>
    <div id="chart"></div>
    <h2>Significant Move Dates</h2>
    <table>
        <thead>
            <tr>
                <th>Date</th>
                <th>Close Price (USD)</th>
                <th>Log Return</th>
                <th>News</th>
            </tr>
        </thead>
        <tbody id="event-table"></tbody>
    </table>
</section>
<footer>News links open a Google News query for Palantir on the selected day.</footer>
<script>
const data = $payload;
const threshold = $threshold;

const traceAll = {
    x: data.series.map(pt => pt.date),
    y: data.series.map(pt => pt.close),
    mode: 'lines',
    name: 'Close',
    line: { color: '#2563eb', width: 2 }
};

const eventDates = data.events.map(pt => pt.date);
const eventClose = data.events.map(pt => pt.close);
const eventReturns = data.events.map(pt => pt.log_return);
const traceEvents = {
    x: eventDates,
    y: eventClose,
    mode: 'markers',
    name: 'Event',
    marker: { size: 10, color: eventReturns.map(v => v > 0 ? '#16a34a' : '#dc2626') },
};

const layout = {
    margin: { l: 60, r: 40, t: 50, b: 60 },
    hovermode: 'x unified',
    yaxis: { title: 'Close Price (USD)' },
    xaxis: { title: 'Date' },
};

Plotly.newPlot('chart', [traceAll, traceEvents], layout, {
    displaylogo: false,
    responsive: true,
    modeBarButtonsToRemove: ['lasso2d', 'select2d']
});

const tableBody = document.getElementById('event-table');
if (data.events.length === 0) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 4;
    cell.textContent = 'No dates exceeded the selected log-return threshold.';
    row.appendChild(cell);
    tableBody.appendChild(row);
} else {
    for (const event of data.events) {
        const row = document.createElement('tr');

        const dateCell = document.createElement('td');
        dateCell.textContent = event.date;
        row.appendChild(dateCell);

        const closeCell = document.createElement('td');
        closeCell.textContent = event.close.toFixed(2);
        row.appendChild(closeCell);

        const lrCell = document.createElement('td');
        const badge = document.createElement('span');
        badge.className = 'badge ' + (event.log_return >= 0 ? 'pos' : 'neg');
        badge.textContent = event.log_return.toFixed(4);
        lrCell.appendChild(badge);
        row.appendChild(lrCell);

        const linkCell = document.createElement('td');
        const anchor = document.createElement('a');
        anchor.href = event.search_url;
        anchor.target = '_blank';
        anchor.rel = 'noopener';
        anchor.textContent = 'View News';
        linkCell.appendChild(anchor);
        row.appendChild(linkCell);

        tableBody.appendChild(row);
    }
}
</script>
</body>
</html>
""")
    return template.substitute(payload=data_json, threshold=f"{threshold:.2f}")


def main() -> int:
    args = parse_args()
    df = load_price_data(args.csv)
    events = select_events(df, args.threshold)
    payload = build_payload(df, events)
    html = render_html(payload, args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Saved Palantir event explorer to {args.output}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
