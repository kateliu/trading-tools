#!/usr/bin/env python3
"""Build an interactive web page to explore ticker time series from a folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an interactive Plotly page for ticker time series.",
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Directory with ticker CSVs (Date, Open, High, Low, Close, Volume).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("web/ticker_explorer.html"),
        help="Destination HTML file (default: web/ticker_explorer.html)",
    )
    return parser.parse_args()


def load_ticker_csvs(folder: Path) -> Dict[str, pd.DataFrame]:
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    data: Dict[str, pd.DataFrame] = {}
    for csv_path in sorted(folder.glob("*.csv")):
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        required = {"Date", "Close"}
        if not required.issubset(df.columns):
            continue
        ticker = csv_path.stem.upper()
        df = df.sort_values("Date").set_index("Date")
        df = df[["Close"]]
        data[ticker] = df

    if not data:
        raise RuntimeError("No valid CSV files with 'Close' column found in folder.")
    return data


def compute_series(df: pd.DataFrame) -> Dict[str, pd.Series]:
    close = df["Close"].copy()
    log_return = np.log(close).diff().dropna()
    pct_change = close.pct_change().dropna()
    return {
        "close": close.dropna(),
        "log_return": log_return,
        "pct_change": pct_change,
    }


def build_dataset(folder: Path) -> Dict[str, Dict[str, List[Dict[str, float]]]]:
    raw = load_ticker_csvs(folder)
    dataset: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
    for ticker, df in raw.items():
        series_dict = compute_series(df)
        dataset[ticker] = {
            metric: [
                {"date": date.strftime("%Y-%m-%d"), "value": float(value)}
                for date, value in series.items()
            ]
            for metric, series in series_dict.items()
            if not series.empty
        }
    return dataset


def render_html(dataset: Dict[str, Dict[str, List[Dict[str, float]]]]) -> str:
    payload = json.dumps(dataset)
    template = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Ticker Series Explorer</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
    body { font-family: 'Segoe UI', sans-serif; margin: 0; background: #f9fafb; color: #111827; }
    header { padding: 1rem 1.5rem; background: #1f2937; color: #f9fafb; }
    .controls { padding: 1rem 1.5rem; display: flex; flex-wrap: wrap; gap: 1.5rem; background: #e5e7eb; }
    .controls label { display: flex; flex-direction: column; font-size: 0.95rem; gap: 0.35rem; }
    .controls select { padding: 0.35rem 0.6rem; border-radius: 6px; border: 1px solid #cbd5f5; font-size: 0.95rem; }
    #chart { width: 100%; height: calc(60vh); }
    #hist { width: 100%; height: calc(25vh); }
    @media (max-height: 700px) {
        #chart { height: 360px; }
        #hist { height: 220px; }
    }
</style>
</head>
<body>
<header>
    <h1>Ticker Series Explorer</h1>
    <p>Select a ticker and metric to view its history.</p>
</header>
<section class="controls">
    <label>Ticker
        <select id="ticker"></select>
    </label>
    <label>Metric
        <select id="metric">
            <option value="close">Close</option>
            <option value="log_return">Log Return</option>
            <option value="pct_change">Percent Change</option>
        </select>
    </label>
</section>
<div id="chart"></div>
<div id="hist"></div>
<script>
const dataset = __PAYLOAD__;
const tickers = Object.keys(dataset);
const tickerSelect = document.getElementById('ticker');
const metricSelect = document.getElementById('metric');
const chartDiv = document.getElementById('chart');
const histDiv = document.getElementById('hist');

function populateTickers() {
    tickers.forEach(ticker => {
        const option = document.createElement('option');
        option.value = ticker;
        option.text = ticker;
        tickerSelect.appendChild(option);
    });
}

function getSeries(ticker, metric) {
    const series = (dataset[ticker] || {})[metric] || [];
    return {
        x: series.map(point => point.date),
        y: series.map(point => point.value),
        name: ticker + ' (' + metric + ')',
        mode: 'lines',
        line: { width: 2, color: '#2563eb' },
    };
}

function renderChart() {
    const ticker = tickerSelect.value;
    const metric = metricSelect.value;
    const trace = getSeries(ticker, metric);
    const layout = {
        margin: { l: 60, r: 40, t: 60, b: 50 },
        title: ticker + ' – ' + metric,
        xaxis: { title: 'Date', type: 'date' },
        yaxis: { title: metric },
        hovermode: 'x unified',
    };
    Plotly.react(chartDiv, [trace], layout, { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'] });

    const values = (dataset[ticker] || {})[metric] || [];
    const histTrace = {
        x: values.map(point => point.value),
        type: 'histogram',
        marker: { color: '#9333ea' },
        opacity: 0.75,
        autobinx: true,
    };
    const histLayout = {
        margin: { l: 60, r: 40, t: 40, b: 60 },
        title: ticker + ' – ' + metric + ' Distribution',
        xaxis: { title: metric },
        yaxis: { title: 'Count' },
    };
    Plotly.react(histDiv, [histTrace], histLayout, { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'] });
}

populateTickers();
if (tickers.length > 0) {
    tickerSelect.value = tickers[0];
    renderChart();
}

tickerSelect.addEventListener('change', renderChart);
metricSelect.addEventListener('change', renderChart);
</script>
</body>
</html>
"""
    return template.replace("__PAYLOAD__", payload)


def main() -> int:
    args = parse_args()
    dataset = build_dataset(args.folder)
    html = render_html(dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Saved ticker explorer to {args.output}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
