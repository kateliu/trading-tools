#!/usr/bin/env python3
"""Compute and plot correlations among all ticker CSVs in a directory."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot correlations for all ticker CSVs in a folder.",
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Directory containing one CSV per ticker (Date, Open, High, Low, Close, Volume)",
    )
    parser.add_argument(
        "--metric",
        choices=["close", "log_return", "pct_change"],
        default="log_return",
        help="Value to correlate: close, log_return (default), or pct_change",
    )
    parser.add_argument(
        "--heatmap-output",
        type=Path,
        default=Path("10062025_correlation_heatmap.png"),
        help="Path to save the correlation heatmap.",
    )
    parser.add_argument(
        "--graph-output",
        type=Path,
        default=Path("10062025_correlation_graph.png"),
        help="Path to save the correlation graph.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Edge threshold: connect tickers whose absolute correlation exceeds this value (default: 0.5)",
    )
    return parser.parse_args()


def load_folder(folder: Path) -> pd.DataFrame:
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    frames: List[pd.DataFrame] = []
    for csv_path in sorted(folder.glob("*.csv")):
        ticker = csv_path.stem.upper()
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        if "Close" not in df.columns:
            print(f"[warn] {csv_path} missing 'Close' column; skipping")
            continue
        frame = df[["Date", "Close"]].copy()
        frame.rename(columns={"Close": ticker}, inplace=True)
        frame.sort_values("Date", inplace=True)
        frames.append(frame.set_index("Date"))

    if not frames:
        raise RuntimeError("No valid CSV files with 'Close' column found.")

    merged = pd.concat(frames, axis=1, join="inner")
    merged.dropna(inplace=True)
    if merged.empty:
        raise RuntimeError("No overlapping data among tickers.")
    return merged


def compute_metric(values: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric == "close":
        return values
    if metric == "log_return":
        return np.log(values).diff().dropna()
    if metric == "pct_change":
        return values.pct_change().dropna()
    raise ValueError(metric)


def plot_heatmap(corr: pd.DataFrame, metric: str, output: Path) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True)
    plt.title(f"Ticker Correlation ({metric})")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)
    print(f"Saved correlation heatmap to {output}")


def plot_graph(corr: pd.DataFrame, threshold: float, output: Path) -> None:
    graph = nx.Graph()
    tickers = corr.columns.tolist()
    graph.add_nodes_from(tickers)

    for i, source in enumerate(tickers):
        for j in range(i + 1, len(tickers)):
            target = tickers[j]
            value = corr.iloc[i, j]
            if abs(value) >= threshold:
                graph.add_edge(source, target, weight=value)

    pos = nx.spring_layout(graph, seed=42)
    edge_colors = [graph[u][v]["weight"] for u, v in graph.edges]
    edge_widths = [abs(weight) * 2 for weight in edge_colors]

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(graph, pos, node_color="skyblue", node_size=1200)
    nx.draw_networkx_labels(graph, pos, font_size=12)
    edges = nx.draw_networkx_edges(
        graph,
        pos,
        edge_color=edge_colors,
        edge_cmap=plt.cm.coolwarm,
        edge_vmin=-1,
        edge_vmax=1,
        width=edge_widths,
    )
    plt.colorbar(edges, shrink=0.75, label="Correlation")
    plt.title(f"Ticker Correlation Graph (|corr| >= {threshold:.2f})")
    plt.axis("off")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)
    print(f"Saved correlation graph to {output}")


def main() -> int:
    args = parse_args()
    values = load_folder(args.folder)
    metric_values = compute_metric(values, args.metric)
    corr = metric_values.corr()
    plot_heatmap(corr, args.metric, args.heatmap_output)
    plot_graph(corr, args.threshold, args.graph_output)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
