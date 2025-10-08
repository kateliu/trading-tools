#!/usr/bin/env python3
"""Generate an interactive correlation graph webpage for tickers in a folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
from string import Template

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an interactive HTML graph showing ticker correlations.",
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Directory containing ticker CSVs (Date, Open, High, Low, Close, Volume).",
    )
    parser.add_argument(
        "--metric",
        choices=["close", "log_return", "pct_change"],
        default="log_return",
        help="Correlation basis (default: log_return).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Include edges where absolute correlation >= threshold (default: 0.5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("web/correlation_graph.html"),
        help="Destination HTML file (default: web/correlation_graph.html).",
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
        raise RuntimeError("No usable CSVs (must contain 'Close' column) in folder.")

    merged = pd.concat(frames, axis=1, join="inner")
    merged.dropna(inplace=True)
    if merged.empty:
        raise RuntimeError("No overlapping data among provided tickers.")
    return merged


def compute_metric(values: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric == "close":
        return values
    if metric == "log_return":
        return np.log(values).diff().dropna()
    if metric == "pct_change":
        return values.pct_change().dropna()
    raise ValueError(metric)


def build_graph_data(corr: pd.DataFrame, threshold: float) -> Dict[str, List[Dict[str, object]]]:
    nodes = [{"id": ticker, "label": ticker} for ticker in corr.columns]
    links: List[Dict[str, object]] = []
    for i, src in enumerate(corr.columns):
        for j in range(i + 1, len(corr.columns)):
            dst = corr.columns[j]
            weight = float(corr.iloc[i, j])
            if abs(weight) >= threshold:
                links.append({"source": src, "target": dst, "weight": weight})
    return {"nodes": nodes, "links": links}


def render_html(data: Dict[str, List[Dict[str, object]]], metric: str, threshold: float) -> str:
    payload = json.dumps(data)
    template = Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Ticker Correlation Graph</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
    body { font-family: 'Segoe UI', sans-serif; margin: 0; background: #f5f5f5; }
    header { padding: 1rem 1.5rem; background: #1f2937; color: white; }
    #chart { width: 100%; height: calc(100vh - 140px); }
    svg { width: 100%; height: 100%; }
    .node circle { stroke: #1f2937; stroke-width: 1.5px; cursor: grab; }
    .node text { pointer-events: none; font-size: 13px; font-weight: 600; fill: #1f2937; }
    .link { stroke-opacity: 0.4; }
    .highlight-node circle { stroke: #111827; stroke-width: 3px; }
    .dimmed { opacity: 0.15; }
    .legend { margin-top: 0.75rem; display: flex; gap: 1.25rem; flex-wrap: wrap; font-size: 0.9rem; }
    .legend-item { display: flex; align-items: center; gap: 0.4rem; }
    .legend-swatch { width: 16px; height: 16px; border-radius: 4px; border: 1px solid rgba(17, 24, 39, 0.4); }
    .legend-swatch.positive { background: linear-gradient(90deg, #b7e4c7, #1b4332); }
    .legend-swatch.negative { background: #ff6b6b; opacity: 0.7; }
    footer { padding: 0.75rem 1.5rem; background: #e5e7eb; font-size: 0.9rem; }
</style>
</head>
<body>
<header>
    <h1>Ticker Correlation Graph</h1>
    <p>metric=$metric, edges shown where |corr| ≥ $threshold</p>
    <div class="legend">
        <span class="legend-item"><span class="legend-swatch positive"></span>Positive correlation (darker = stronger)</span>
        <span class="legend-item"><span class="legend-swatch negative"></span>Negative correlation</span>
    </div>
</header>
<div id="chart"></div>
<footer>Click a ticker to highlight its direct correlations. Drag nodes to reposition them.</footer>
<script>
const data = $payload;
const threshold = $threshold;
const width = document.getElementById('chart').clientWidth;
const height = document.getElementById('chart').clientHeight;

const svg = d3.select('#chart').append('svg')
    .attr('viewBox', '0 0 ' + width + ' ' + height)
    .call(d3.zoom().on('zoom', function(event) { svgGroup.attr('transform', event.transform); }));

const svgGroup = svg.append('g');

const strokeScale = d3.scaleLinear()
    .domain([threshold, 1])
    .range([0.4, 0.9]);
const colorScale = d3.scaleLinear()
    .domain([threshold, 1])
    .range(['#b7e4c7', '#1b4332']);

const link = svgGroup.selectAll('.link')
    .data(data.links)
    .enter()
    .append('line')
    .attr('class', function(d) { return 'link ' + (d.weight >= 0 ? 'positive' : 'negative'); })
    .attr('stroke-width', function(d) { return Math.max(1.5, Math.abs(d.weight) * 4); })
    .attr('stroke', function(d) {
        var magnitude = Math.abs(d.weight);
        return d.weight >= 0 ? colorScale(Math.min(magnitude, 1)) : '#ff6b6b';
    })
    .attr('stroke-opacity', function(d) {
        var magnitude = Math.abs(d.weight);
        return d.weight >= 0 ? strokeScale(Math.min(magnitude, 1)) : 0.6;
    });

const nodeGroup = svgGroup.selectAll('.node')
    .data(data.nodes)
    .enter()
    .append('g')
    .attr('class', 'node')
    .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended));

nodeGroup.append('circle')
    .attr('r', 20)
    .attr('fill', '#60a5fa');

nodeGroup.append('text')
    .attr('text-anchor', 'middle')
    .attr('dy', '0.35em')
    .text(function(d) { return d.label; });

const simulation = d3.forceSimulation(data.nodes)
    .force('link', d3.forceLink(data.links).id(function(d) { return d.id; }).distance(160).strength(0.4))
    .force('charge', d3.forceManyBody().strength(-500))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .on('tick', ticked);

const adjacency = new Map();
for (const node of data.nodes) { adjacency.set(node.id, new Set()); }
for (const edge of data.links) {
    adjacency.get(edge.source).add(edge.target);
    adjacency.get(edge.target).add(edge.source);
}

nodeGroup.on('click', function(event, node) { highlight(node.id); });
svg.on('click', function(event) {
    if (event.target.tagName === 'svg') highlight(null);
});

function highlight(selectedId) {
    if (!selectedId) {
        nodeGroup.classed('dimmed highlight-node', false);
        link.classed('dimmed', false);
        return;
    }
    const neighbors = adjacency.get(selectedId) || new Set();
    nodeGroup.classed('highlight-node', function(d) { return d.id === selectedId; })
        .classed('dimmed', function(d) { return d.id !== selectedId && !neighbors.has(d.id); });
    link.classed('dimmed', function(d) { return !(d.source.id === selectedId || d.target.id === selectedId); });
}

function ticked() {
    link
        .attr('x1', function(d) { return d.source.x; })
        .attr('y1', function(d) { return d.source.y; })
        .attr('x2', function(d) { return d.target.x; })
        .attr('y2', function(d) { return d.target.y; });

    nodeGroup.attr('transform', function(d) { return 'translate(' + d.x + ',' + d.y + ')'; });
}

function dragstarted(event, d) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
}

function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
}

function dragended(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
}
</script>
</body>
</html>
""")
    return template.substitute(payload=payload, metric=metric, threshold=f"{threshold:.2f}")


def main() -> int:
    args = parse_args()
    history = load_folder(args.folder)
    values = compute_metric(history, args.metric)
    corr = values.corr()
    data = build_graph_data(corr, args.threshold)

    html = render_html(data, args.metric, args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Saved interactive correlation graph to {args.output}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
