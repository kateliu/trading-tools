# Trading Tools Workflow

This repository collects a few small utilities for analyzing Palantir (PLTR) and related market data. The `workflow.py` driver stitches the individual scripts together so you can fetch fresh data, train an ML forecaster, and regenerate downstream artifacts in one command.

## Prerequisites

- Python 3.9+
- Dependencies from `requirements.txt` (create and activate a virtualenv, then run `pip install -r requirements.txt`).
  * The file lists CPU-only PyTorch; if you want GPU builds, install the appropriate wheel separately and remove/adjust the torch line.
- Outbound network access so the Yahoo Finance and CNN endpoints can be reached

## Running the workflow

```bash
source .venv/bin/activate  # if you use a virtual environment
python workflow.py --ticker PLTR --days 365 --train-model
```

This will:

1. Download `--days` of daily OHLC data for `--ticker` using Yahoo Finance and write it to `palantir_prices.csv` (or the path you pass to `--price-csv`).
2. Download the same lookback window of CNN Fear & Greed index values and store them in `fear_greed_index.csv` (skipped if you pass `--skip-sentiment`).
3. Copy the refreshed CSVs into `web/data/` so the Plotly dashboard can load them directly (skip with `--no-web-sync`).
4. If `--train-model` is supplied, call `train_lstm.py` to fit an LSTM on the latest OHLCV + sentiment features and emit `web/data/ai_forecast.json` for the dashboard.
5. Regenerate `palantir_vs_fear_greed.png` by calling `plot_market_sentiment.py` unless you pass `--skip-plot`.

Use `python workflow.py --help` to see the full list of options, including:

- `--train-model` / `--lookback` / `--epochs` / `--forecast-days` for the neural forecaster
- `--model-output`, `--metadata-output`, `--forecast-json` to control artifact locations
- overriding output paths or specifying a different Python interpreter for helper scripts.

You can also invoke the trainer directly if you want to experiment with hyper-parameters:

```bash
python train_lstm.py --price-csv palantir_prices.csv \
    --sentiment-csv fear_greed_index.csv \
    --lookback 90 --epochs 80 --forecast-days 45 --device cuda
```

## Updating the dashboard

After running the workflow, serve the `web/` folder locally to view the Plotly chart that now includes SMA/EMA/RSI indicators and a simple AI forecast overlay:

```bash
cd web
python -m http.server 8000
# open http://localhost:8000/palantir_chart.html?v=3 in your browser
```

The dashboard reads its price data from `web/data/palantir_prices.csv`, which is maintained automatically by the workflow. Selecting “AI Forecast (30 days)” now displays the predictions produced by `train_lstm.py` (if available); otherwise it falls back to a local linear trendline.

## Troubleshooting

- **Network failures** – Yahoo Finance or CNN occasionally rate-limit requests. Re-run the workflow; if failures persist, add your own retry logic around the fetch functions.
- **Plotting errors** – `plot_market_sentiment.py` requires `matplotlib`. If the workflow exits complaining about imports, install the missing packages or re-run with `--skip-plot`.
- **Web chart is blank** – Ensure the CSV exists under `web/data/` and hard-refresh the browser (Ctrl/⌘+Shift+R) so the latest `script.js` is loaded.

Feel free to extend `workflow.py` with additional steps (e.g., generating AI forecasts in advance, pushing artifacts to cloud storage, or alerting yourself via email).
