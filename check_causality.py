#!/usr/bin/env python3
"""Assess directional causality between two numeric series via Granger tests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover
    print("This script requires numpy. Install it with 'pip install numpy'.", file=sys.stderr)
    raise SystemExit(1) from exc

try:
    import pandas as pd
except ModuleNotFoundError as exc:  # pragma: no cover
    print("This script requires pandas. Install it with 'pip install pandas'.", file=sys.stderr)
    raise SystemExit(1) from exc

try:
    from statsmodels.tsa.stattools import grangercausalitytests
except ModuleNotFoundError as exc:  # pragma: no cover
    print(
        "This script requires statsmodels. Install it with 'pip install statsmodels'.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


def load_series(path: Path, column: str | None) -> pd.Series:
    """Load a numeric series from a CSV file."""
    try:
        frame = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise SystemExit(f"File not found: {path}") from exc
    except Exception as exc:  # pragma: no cover - surface the error details
        raise SystemExit(f"Failed to read '{path}': {exc}") from exc

    if column:
        if column not in frame.columns:
            raise SystemExit(f"Column '{column}' not found in {path} (columns: {list(frame.columns)})")
        series = frame[column]
    else:
        if frame.shape[1] == 1:
            series = frame.iloc[:, 0]
        else:
            raise SystemExit(
                f"File {path} has multiple columns; specify one with --column-a/--column-b."
            )

    try:
        series = pd.to_numeric(series, errors="coerce")
    except Exception as exc:
        raise SystemExit(f"Could not convert data in '{path}' to numeric values: {exc}") from exc

    series = series.dropna()
    if series.empty:
        raise SystemExit(f"Series extracted from {path} contains no numeric data after dropping NaNs.")

    return series.reset_index(drop=True)


def align_series(a: pd.Series, b: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Align two series to the same length using their overlapping suffix."""
    n = min(len(a), len(b))
    if n == 0:
        raise SystemExit("At least one of the series is empty after cleaning.")
    return a.iloc[-n:].to_numpy(), b.iloc[-n:].to_numpy()


def select_maxlag(length: int, user_lag: int | None) -> int:
    """Pick an appropriate maximum lag for Granger testing."""
    if user_lag is not None:
        if user_lag < 1:
            raise SystemExit("--max-lag must be a positive integer.")
        if user_lag >= length:
            raise SystemExit("--max-lag must be smaller than the series length.")
        return user_lag

    # A common heuristic: max lag ≈ sqrt(n) but capped to ensure enough degrees of freedom.
    heuristic = int(round(length ** 0.5))
    max_lag = max(1, min(heuristic, length // 4))
    return max_lag


def run_granger(x: np.ndarray, y: np.ndarray, max_lag: int) -> Iterable[Tuple[int, float]]:
    """Yield (lag, p-value) pairs for tests where x helps predict y."""
    stacked = np.column_stack([y, x])
    results = grangercausalitytests(stacked, maxlag=max_lag, verbose=False)
    for lag, outcome in results.items():
        if isinstance(outcome, tuple):
            outcome = outcome[0]
        ssr_result = outcome.get("ssr_ftest") if isinstance(outcome, dict) else None
        if not ssr_result:
            continue
        try:
            p_value = float(ssr_result[1])
        except (TypeError, ValueError):
            continue
        yield lag, p_value


def summarize(direction: str, p_values: Iterable[Tuple[int, float]], alpha: float) -> str:
    """Create a textual summary from Granger test results."""
    p_values = list(p_values)
    if not p_values:
        return f"No valid Granger causality results for {direction}."

    best_lag, best_p = min(p_values, key=lambda item: item[1])
    decision = "suggests" if best_p < alpha else "fails to show"
    return (
        f"{direction}: Minimum p-value {best_p:.4f} at lag {best_lag} -- "
        f"test {decision} predictive causality at the {alpha:.2f} level."
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assess whether one numeric time series Granger-causes another."
    )
    parser.add_argument("file_a", type=Path, help="CSV containing the first series (potential cause).")
    parser.add_argument("file_b", type=Path, help="CSV containing the second series (potential effect).")
    parser.add_argument(
        "--column-a",
        dest="column_a",
        help="Column name in file_a (required if the file has multiple columns).",
    )
    parser.add_argument(
        "--column-b",
        dest="column_b",
        help="Column name in file_b (required if the file has multiple columns).",
    )
    parser.add_argument(
        "--max-lag",
        dest="max_lag",
        type=int,
        help="Maximum lag (in observations) to evaluate. Default is a heuristic based on series length.",
    )
    parser.add_argument(
        "--alpha",
        dest="alpha",
        type=float,
        default=0.05,
        help="Significance level for reporting (default: 0.05).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.alpha <= 0 or args.alpha >= 1:
        parser.error("--alpha must be between 0 and 1 (exclusive).")

    series_a = load_series(args.file_a, args.column_a)
    series_b = load_series(args.file_b, args.column_b)

    aligned_a, aligned_b = align_series(series_a, series_b)
    length = len(aligned_a)
    max_lag = select_maxlag(length, args.max_lag)

    if length < max_lag + 2:
        parser.error("Series are too short for the requested max lag. Provide longer data or reduce --max-lag.")

    forward_results = list(run_granger(aligned_a, aligned_b, max_lag))
    reverse_results = list(run_granger(aligned_b, aligned_a, max_lag))

    print(f"Evaluating Granger causality with max lag = {max_lag} on {length} paired observations.")
    print(summarize("A → B", forward_results, args.alpha))
    print(summarize("B → A", reverse_results, args.alpha))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
