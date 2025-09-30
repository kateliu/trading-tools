#!/usr/bin/env python3
"""Plot Palantir closing prices alongside CNN Fear & Greed index."""

import csv
import datetime as dt
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

PALANTIR_CSV = Path("palantir_prices.csv")
FEAR_GREED_CSV = Path("fear_greed_index.csv")


def read_palantir(path: Path):
    dates = []
    closes = []
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                date_str = row.get("Date")
                close_str = row.get("Close")
                if not date_str or close_str is None:
                    continue
                try:
                    date = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
                    close = float(close_str)
                except ValueError:
                    continue
                dates.append(date)
                closes.append(close)
    except FileNotFoundError:
        raise RuntimeError(f"Missing Palantir price file: {path}")
    if not dates:
        raise RuntimeError("No Palantir price data available to plot.")
    return dates, closes


def read_fear_greed(path: Path):
    dates = []
    ratings = []
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                date_str = row.get("Date")
                rating_str = row.get("Rating")
                if not date_str or rating_str is None:
                    continue
                try:
                    date = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
                    rating = int(rating_str)
                except ValueError:
                    continue
                dates.append(date)
                ratings.append(rating)
    except FileNotFoundError:
        raise RuntimeError(f"Missing Fear & Greed index file: {path}")
    if not dates:
        raise RuntimeError("No Fear & Greed data available to plot.")
    return dates, ratings


def main():
    try:
        price_dates, closings = read_palantir(PALANTIR_CSV)
        fg_dates, ratings = read_fear_greed(FEAR_GREED_CSV)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    fig, ax_price = plt.subplots(figsize=(12, 6))

    ax_price.plot(price_dates, closings, color="tab:blue", label="PLTR Close")
    ax_price.set_ylabel("Palantir Close (USD)", color="tab:blue")
    ax_price.tick_params(axis="y", labelcolor="tab:blue")

    ax_sentiment = ax_price.twinx()
    ax_sentiment.plot(fg_dates, ratings, color="tab:red", label="Fear & Greed")
    ax_sentiment.set_ylabel("Fear & Greed Index", color="tab:red")
    ax_sentiment.tick_params(axis="y", labelcolor="tab:red")
    ax_sentiment.set_ylim(0, 100)

    ax_price.set_title("Palantir Closing Price vs. CNN Fear & Greed Index (Past 365 Days)")
    ax_price.set_xlabel("Date")

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax_price.xaxis.set_major_locator(locator)
    ax_price.xaxis.set_major_formatter(formatter)

    lines = ax_price.get_lines() + ax_sentiment.get_lines()
    labels = [line.get_label() for line in lines]
    ax_price.legend(lines, labels, loc="upper left")

    fig.tight_layout()
    output_path = Path("palantir_vs_fear_greed.png")
    fig.savefig(output_path, dpi=150)
    print(f"Saved chart to {output_path}")


if __name__ == "__main__":
    main()
