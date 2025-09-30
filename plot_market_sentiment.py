#!/usr/bin/env python3
"""Plot Palantir daily OHLC bars alongside CNN Fear & Greed index."""

import csv
import datetime as dt
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

PALANTIR_CSV = Path("palantir_prices.csv")
FEAR_GREED_CSV = Path("fear_greed_index.csv")


def read_palantir(path: Path):
    records = []
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                date_str = row.get("Date")
                open_str = row.get("Open")
                high_str = row.get("High")
                low_str = row.get("Low")
                close_str = row.get("Close")
                if not date_str:
                    continue
                try:
                    date = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
                    open_price = float(open_str)
                    high_price = float(high_str)
                    low_price = float(low_str)
                    close = float(close_str)
                except ValueError:
                    continue
                records.append(
                    {
                        "date": date,
                        "open": open_price,
                        "high": high_price,
                        "low": low_price,
                        "close": close,
                    }
                )
    except FileNotFoundError:
        raise RuntimeError(f"Missing Palantir price file: {path}")
    if not records:
        raise RuntimeError("No Palantir price data available to plot.")
    records.sort(key=lambda entry: entry["date"])
    return records


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
        price_records = read_palantir(PALANTIR_CSV)
        fg_dates, ratings = read_fear_greed(FEAR_GREED_CSV)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    fig, ax_price = plt.subplots(figsize=(12, 6))

    date_nums = mdates.date2num([record["date"] for record in price_records])
    lows = [record["low"] for record in price_records]
    highs = [record["high"] for record in price_records]

    candle_width = 0.6
    min_body_height = 1e-3

    for x, record in zip(date_nums, price_records):
        color = "tab:green" if record["close"] >= record["open"] else "tab:red"
        ax_price.vlines(x, record["low"], record["high"], color=color, linewidth=1)

        body_bottom = min(record["open"], record["close"])
        body_height = max(abs(record["close"] - record["open"]), min_body_height)
        rect = Rectangle(
            (x - candle_width / 2, body_bottom),
            candle_width,
            body_height,
            facecolor=color,
            edgecolor=color,
            alpha=0.6,
        )
        ax_price.add_patch(rect)

    ax_price.set_ylabel("Palantir Price (USD)", color="tab:blue")
    ax_price.tick_params(axis="y", labelcolor="tab:blue")
    ax_price.set_ylim(min(lows) * 0.97, max(highs) * 1.03)
    ax_price.set_xlim(date_nums[0] - 1, date_nums[-1] + 1)
    ax_price.xaxis_date()

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

    price_handle = Line2D([], [], color="tab:green", label="PLTR OHLC")
    sentiment_handle = ax_sentiment.get_lines()[0]
    ax_price.legend([price_handle, sentiment_handle], ["PLTR OHLC", "Fear & Greed"], loc="upper left")

    fig.tight_layout()
    output_path = Path("palantir_vs_fear_greed.png")
    fig.savefig(output_path, dpi=150)
    print(f"Saved chart to {output_path}")


if __name__ == "__main__":
    main()
