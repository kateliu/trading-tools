#!/usr/bin/env python3
"""Orchestrate daily data refresh and artifacts for Palantir analytics."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

try:
    import fetch_stock_prices
    import fetch_fear_greed
except ImportError as exc:  # pragma: no cover
    print(f"Failed to import project modules: {exc}", file=sys.stderr)
    raise SystemExit(1) from exc


DEFAULT_TICKER = "PLTR"
DEFAULT_DAYS = 365
DEFAULT_PRICE_CSV = Path("palantir_prices.csv")
DEFAULT_SENTIMENT_CSV = Path("fear_greed_index.csv")
WEB_DATA_DIR = Path("web/data")
DEFAULT_MODEL_PATH = Path("models/lstm_pltr.pt")
DEFAULT_METADATA_PATH = Path("models/lstm_pltr.json")
DEFAULT_FORECAST_JSON = WEB_DATA_DIR / "ai_forecast.json"


def log(message: str) -> None:
    print(f"[workflow] {message}")


def fetch_prices(ticker: str, days: int, destination: Path) -> Path:
    log(f"Fetching {days} days of {ticker} pricing data from Yahoo Finance …")
    records = fetch_stock_prices.fetch_history(ticker, days)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fetch_stock_prices.write_csv(records, destination)
    log(f"Wrote {len(records)} rows to {destination}")
    return destination


def fetch_sentiment(days: int, destination: Path) -> Path:
    log(f"Fetching {days} days of CNN Fear & Greed index data …")
    records = fetch_fear_greed.fetch_index_history(days)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fetch_fear_greed.write_csv(records, destination)
    log(f"Wrote {len(records)} rows to {destination}")
    return destination


def sync_to_web(paths: Iterable[Path], web_dir: Path = WEB_DATA_DIR) -> None:
    web_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path is None:
            continue
        target = web_dir / path.name
        shutil.copy2(path, target)
        log(f"Synced {path} → {target}")


def run_plot_script(python_executable: Path | str = sys.executable) -> None:
    log("Rendering sentiment comparison chart …")
    try:
        subprocess.run(
            [str(python_executable), "plot_market_sentiment.py"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        log("plot_market_sentiment.py failed; see stderr output below:")
        sys.stderr.write(exc.stderr.decode("utf-8", errors="ignore"))
        raise SystemExit(exc.returncode) from exc


def run_training(
    python_executable: Path | str,
    ticker: str,
    price_csv: Path,
    sentiment_csv: Optional[Path],
    args: argparse.Namespace,
) -> None:
    log("Training neural network forecaster …")
    cmd = [
        str(python_executable),
        "train_lstm.py",
        "--ticker",
        ticker,
        "--price-csv",
        str(price_csv),
        "--lookback",
        str(args.lookback),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--forecast-days",
        str(args.forecast_days),
        "--output-model",
        str(args.model_output),
        "--output-metadata",
        str(args.metadata_output),
        "--forecast-json",
        str(args.forecast_json),
        "--device",
        args.device,
    ]

    if sentiment_csv is not None:
        cmd.extend(["--sentiment-csv", str(sentiment_csv)])

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        log("train_lstm.py failed; see details above.")
        raise SystemExit(exc.returncode) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="End-to-end workflow for fetching Palantir data and updating artifacts.",
    )
    parser.add_argument("--ticker", default=DEFAULT_TICKER, help="Ticker symbol to fetch (default: PLTR)")
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help="Number of calendar days to request from Yahoo Finance and CNN (default: 365)",
    )
    parser.add_argument(
        "--price-csv",
        type=Path,
        default=DEFAULT_PRICE_CSV,
        help="Destination CSV for price data (default: palantir_prices.csv)",
    )
    parser.add_argument(
        "--sentiment-csv",
        type=Path,
        default=DEFAULT_SENTIMENT_CSV,
        help="Destination CSV for Fear & Greed data (default: fear_greed_index.csv)",
    )
    parser.add_argument(
        "--skip-sentiment",
        action="store_true",
        help="Skip downloading the Fear & Greed index.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip running plot_market_sentiment.py",
    )
    parser.add_argument(
        "--no-web-sync",
        action="store_true",
        help="Do not copy refreshed CSVs into web/data/",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter to use when spawning helper scripts.",
    )
    parser.add_argument(
        "--train-model",
        action="store_true",
        help="Train the neural network forecaster after refreshing data.",
    )
    parser.add_argument("--lookback", type=int, default=60, help="Lookback window for model training (default: 60)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs for the model (default: 50)")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Mini-batch size for training (default: 32)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Optimizer learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--forecast-days", type=int, default=30, help="Forecast horizon to generate (default: 30 days)"
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to store the trained model weights.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Where to store training metadata/scaler information.",
    )
    parser.add_argument(
        "--forecast-json",
        type=Path,
        default=DEFAULT_FORECAST_JSON,
        help="Path for saving AI forecast JSON for the dashboard.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device spec to pass through to the trainer (default: cpu).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.days <= 0:
        parser.error("--days must be a positive integer")

    ticker = args.ticker.upper().strip()

    price_csv = fetch_prices(ticker, args.days, args.price_csv)

    sentiment_csv: Path | None = None
    if not args.skip_sentiment:
        sentiment_csv = fetch_sentiment(args.days, args.sentiment_csv)

    if not args.no_web_sync:
        sync_targets = [price_csv]
        if sentiment_csv is not None:
            sync_targets.append(sentiment_csv)
        sync_to_web(sync_targets)

    if args.train_model:
        run_training(args.python, ticker, price_csv, sentiment_csv, args)

    if not args.skip_plot:
        run_plot_script(args.python)

    log("Workflow completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
