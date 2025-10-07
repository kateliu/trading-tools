#!/usr/bin/env python3
"""Generate an interactive HTML dashboard visualizing ticker correlations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an interactive web page showing ticker correlation heatmap and graph.",
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Directory containing ticker CSV files (Date, Open, High, Low, Close, Volume).",
    )
    parser.add_argument(
        "--metric",
        choices=["close", "log_return", "pct_change"],
        default="log_return",
        help="Value used for correlation computation (default: log_return).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Only draw graph edges where |correlation| >= threshold (default: 0.5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("web/correlation_dashboard.html"),
        help="Destination HTML file (default: web/correlation_dashboard.html).",
    )
    return parser.parse_args()


def load_folder(folder: Path) -> pd.DataFrame:
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    frames: List[pd.DataFrame] = []
    for csv_path in sorted(folder.glob("*.csv")):
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        if "Close" not in df.columns:
            continue
        ticker = csv_path.stem.upper()
        frame = df[["Date", "Close"]].copy()
        frame.rename(columns={"Close": ticker}, inplace=True)
        frame.sort_values("Date", inplace=True)
        frames.append(frame.set_index("Date"))

    if not frames:
        raise RuntimeError("No suitable CSVs (with Close column) found in folder.")

    merged = pd.concat(frames, axis=1, join="inner")
    merged.dropna(inplace=True)
    if merged.empty:
        raise RuntimeError("No overlapping history among provided tickers.")
    return merged


def compute_metric(values: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric == "close":
        return values
    if metric == "log_return":
        return np.log(values).diff().dropna()
    if metric == "pct_change":
        return values.pct_change().dropna()
    raise ValueError(metric)


def build_graph_traces(corr: pd.DataFrame, threshold: float) -> Tuple[List[go.Scatter], go.Scatter]:
    graph = nx.Graph()
    tickers = corr.columns.tolist()
    graph.add_nodes_from(tickers)

    positive_edges = []
    negative_edges = []
    for i, source in enumerate(tickers):
        for j in range(i + 1, len(tickers)):
            target = tickers[j]
            weight = corr.iloc[i, j]
            if abs(weight) >= threshold:
                graph.add_edge(source, target, weight=weight)
                if weight >= 0:
                    positive_edges.append((source, target, weight))
                else:
                    negative_edges.append((source, target, weight))

    pos = nx.spring_layout(graph, seed=42)

    def edge_trace(edges: List[Tuple[str, str, float]], color: str, name: str) -> go.Scatter | None:
        if not edges:
            return None
        edge_x, edge_y, text = [], [], []
        for src, dst, weight in edges:
            x0, y0 = pos[src]
            x1, y1 = pos[dst]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        return go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color=color, width=max(1.5, 4 * np.mean([abs(w) for _, _, w in edges]))),
            hoverinfo="none",
            showlegend=True,
            name=name,
        )

    node_x = [pos[t][0] for t in tickers]
    node_y = [pos[t][1] for t in tickers]
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=tickers,
        textposition="middle center",
        marker=dict(color="skyblue", size=20, line=dict(color="darkslategray", width=1)),
        hoverinfo="text",
        showlegend=False,
    )

    traces = []
    pos_trace = edge_trace(positive_edges, "#2ECC40", "Positive")
    neg_trace = edge_trace(negative_edges, "#FF4136", "Negative")
    if pos_trace:
        traces.append(pos_trace)
    if neg_trace:
        traces.append(neg_trace)
    traces.append(node_trace)
    return traces


def build_figure(corr: pd.DataFrame, metric: str, threshold: float) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Correlation Heatmap", "Correlation Graph"))

    heatmap = go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="RdBu",
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Correlation"),
    )
    fig.add_trace(heatmap, row=1, col=1)

    for trace in build_graph_traces(corr, threshold):
        fig.add_trace(trace, row=1, col=2)

    fig.update_xaxes(title_text="Ticker", row=1, col=1)
    fig.update_yaxes(title_text="Ticker", row=1, col=1)
    fig.update_xaxes(showgrid=False, showticklabels=False, row=1, col=2)
    fig.update_yaxes(showgrid=False, showticklabels=False, row=1, col=2)
    fig.update_layout(
        title=f"Ticker Correlation Dashboard (metric={metric}, threshold={threshold:.2f})",
        legend=dict(x=0.55, y=1.0),
        margin=dict(l=50, r=50, t=60, b=40),
    )
    return fig


def main() -> int:
    args = parse_args()
    values = load_folder(args.folder)
    metric_values = compute_metric(values, args.metric)
    corr = metric_values.corr()

    fig = build_figure(corr, args.metric, args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output, include_plotlyjs="cdn")
    print(f"Saved interactive dashboard to {args.output}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
