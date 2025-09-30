(async function init() {
    const statusEl = document.getElementById("status");
    const rangeEl = document.getElementById("range");
    const indicatorEl = document.getElementById("indicators" );
    let fullData = [];
    let sourcePath = "";

    try {
        statusEl.textContent = "Loading palantir_prices.csv...";
        const result = await loadStockData();
        fullData = result.rows;
        sourcePath = result.path;
        statusEl.textContent = `Loaded ${fullData.length} rows from ${sourcePath}.`;
    } catch (error) {
        console.error(error);
        statusEl.textContent = `Failed to load data: ${error.message}`;
        return;
    }

    function applyRangeSelection() {
        const value = rangeEl.value;
        let subset = fullData;
        if (value !== "all") {
            const days = Number.parseInt(value, 10);
            const cutoff = fullData[fullData.length - 1].dateObj;
            const startDate = new Date(cutoff);
            startDate.setDate(startDate.getDate() - days);
            subset = fullData.filter(row => row.dateObj >= startDate);
        }
        if (!subset.length) {
            statusEl.textContent = `No data in selected range (source: ${sourcePath}).`;
            clearChart();
        } else {
            statusEl.textContent = `Showing ${subset.length} rows from ${sourcePath}.`;
            renderChart(subset, value, indicatorEl.value);
        }
    }

    rangeEl.addEventListener("change", applyRangeSelection);
    indicatorEl.addEventListener("change", applyRangeSelection);
    applyRangeSelection();
})();

async function loadCsv(path) {
    const response = await fetch(path, {
        cache: "no-store",
        headers: { "Cache-Control": "no-cache" },
    });
    if (!response.ok && response.status !== 304) {
        throw new Error(`${response.status} ${response.statusText}`);
    }
    if (response.status === 304) {
        // Retry with a cache-busting query to force fresh contents.
        const cacheBusted = `${path}?t=${Date.now()}`;
        const retry = await fetch(cacheBusted, { cache: "no-store" });
        if (!retry.ok) {
            throw new Error(`${retry.status} ${retry.statusText}`);
        }
        return parseCsv(await retry.text());
    }
    const text = await response.text();
    return parseCsv(text);
}

async function loadStockData() {
    const candidates = [
        "data/palantir_prices.csv",
        "palantir_prices.csv",
        "../palantir_prices.csv",
    ];

    const errors = [];
    for (const path of candidates) {
        try {
            const rows = await loadCsv(path);
            return { rows, path };
        } catch (err) {
            errors.push(`${path}: ${err.message}`);
        }
    }

    throw new Error(
        `palantir_prices.csv not found. Place the CSV next to this page or in a data/ subfolder.\n` +
        errors.join("\n")
    );
}

function parseCsv(text) {
    const lines = text.trim().split(/\r?\n/);
    if (lines.length <= 1) {
        throw new Error("CSV does not contain data rows");
    }
    const header = lines.shift().split(",").map(s => s.trim());
    const idx = {
        date: header.indexOf("Date"),
        open: header.indexOf("Open"),
        high: header.indexOf("High"),
        low: header.indexOf("Low"),
        close: header.indexOf("Close"),
    };
    if (Object.values(idx).some(i => i === -1)) {
        throw new Error("CSV missing required columns (Date, Open, High, Low, Close)");
    }

    const rows = [];
    for (const line of lines) {
        if (!line.trim()) continue;
        const cells = line.split(",");
        const dateStr = cells[idx.date];
        const open = Number.parseFloat(cells[idx.open]);
        const high = Number.parseFloat(cells[idx.high]);
        const low = Number.parseFloat(cells[idx.low]);
        const close = Number.parseFloat(cells[idx.close]);
        if (!dateStr || Number.isNaN(open) || Number.isNaN(high) || Number.isNaN(low) || Number.isNaN(close)) {
            continue;
        }
        const dateObj = new Date(dateStr);
        if (Number.isNaN(dateObj.getTime())) continue;
        rows.push({ date: dateStr, dateObj, open, high, low, close });
    }
    if (!rows.length) {
        throw new Error("No valid rows parsed from CSV");
    }
    // Ensure chronological order
    rows.sort((a, b) => a.dateObj - b.dateObj);
    return rows;
}

function renderChart(data, rangeLabel, indicator) {
    const titleSuffix = rangeLabel === "all" ? "All Data" : `Last ${rangeLabel} Days`;
    const trace = {
        type: "candlestick",
        x: data.map(row => row.date),
        open: data.map(row => row.open),
        high: data.map(row => row.high),
        low: data.map(row => row.low),
        close: data.map(row => row.close),
        increasing: { line: { color: "#2ca02c" } },
        decreasing: { line: { color: "#d62728" } },
        name: "Palantir (PLTR)",
    };

    const overlays = buildIndicatorTraces(data, indicator);

    const layout = {
        title: `Palantir Stock Price — ${titleSuffix}`,
        margin: { l: 60, r: 60, t: 60, b: 40 },
        xaxis: {
            range: [data[0].date, data[data.length - 1].date],
            rangeslider: { visible: true },
            type: "date",
        },
        yaxis: {
            title: "Price (USD)",
            fixedrange: false,
        },
        plot_bgcolor: "rgba(0,0,0,0)",
        paper_bgcolor: "rgba(0,0,0,0)",
    };

    if (indicator === "rsi") {
        layout.yaxis2 = {
            title: "RSI",
            overlaying: "y",
            side: "right",
            range: [0, 100],
            showgrid: false,
            zeroline: false,
        };
        layout.shapes = [
            {
                type: "line",
                xref: "paper",
                x0: 0,
                x1: 1,
                yref: "y2",
                y0: 70,
                y1: 70,
                line: { color: "rgba(200, 100, 100, 0.5)", dash: "dot" },
            },
            {
                type: "line",
                xref: "paper",
                x0: 0,
                x1: 1,
                yref: "y2",
                y0: 30,
                y1: 30,
                line: { color: "rgba(100, 150, 200, 0.5)", dash: "dot" },
            },
        ];
    }

    const config = {
        responsive: true,
        displaylogo: false,
        modeBarButtonsToRemove: [
            "autoScale2d",
            "hoverCompareCartesian",
            "toggleSpikelines",
        ],
    };

    try {
        Plotly.newPlot("chart", [trace, ...overlays], layout, config);
    } catch (error) {
        console.error("Failed to render chart", error);
        const statusEl = document.getElementById("status");
        statusEl.textContent = `Plotly render failed: ${error.message}`;
    }
}

function buildIndicatorTraces(data, indicator) {
    if (!indicator || indicator === "none") {
        return [];
    }

    switch (indicator) {
        case "sma":
            return [
                makeLineTrace("SMA (20)", movingAverage(data, 20), "#1f77b4"),
                makeLineTrace("SMA (50)", movingAverage(data, 50), "#ff7f0e"),
            ];
        case "ema":
            return [
                makeLineTrace("EMA (12)", exponentialMovingAverage(data, 12), "#9467bd"),
                makeLineTrace("EMA (26)", exponentialMovingAverage(data, 26), "#8c564b"),
            ];
        case "rsi":
            return [relativeStrengthTrace(data, 14)];
        default:
            return [];
    }
}

function makeLineTrace(name, series, color) {
    return {
        type: "scatter",
        mode: "lines",
        x: series.map(point => point.date),
        y: series.map(point => point.value),
        name,
        line: { color, width: 1.5 },
        yaxis: "y",
    };
}

function movingAverage(data, windowSize) {
    const results = [];
    let sum = 0;
    for (let i = 0; i < data.length; i += 1) {
        sum += data[i].close;
        if (i >= windowSize) {
            sum -= data[i - windowSize].close;
        }
        if (i >= windowSize - 1) {
            results.push({ date: data[i].date, value: sum / windowSize });
        }
    }
    return results;
}

function exponentialMovingAverage(data, span) {
    const k = 2 / (span + 1);
    const ema = [];
    let prev = data[0].close;
    ema.push({ date: data[0].date, value: prev });
    for (let i = 1; i < data.length; i += 1) {
        const value = data[i].close * k + prev * (1 - k);
        ema.push({ date: data[i].date, value });
        prev = value;
    }
    return ema;
}

function relativeStrengthTrace(data, period) {
    const gains = [];
    const losses = [];
    for (let i = 1; i < data.length; i += 1) {
        const change = data[i].close - data[i - 1].close;
        gains.push(Math.max(change, 0));
        losses.push(Math.max(-change, 0));
    }

    const rsis = [];
    let avgGain = average(gains.slice(0, period));
    let avgLoss = average(losses.slice(0, period));
    for (let i = period; i < gains.length; i += 1) {
        avgGain = (avgGain * (period - 1) + gains[i]) / period;
        avgLoss = (avgLoss * (period - 1) + losses[i]) / period;
        const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
        const rsi = avgLoss === 0 ? 100 : 100 - 100 / (1 + rs);
        rsis.push({ date: data[i + 1].date, value: rsi });
    }

    return {
        type: "scatter",
        mode: "lines",
        x: rsis.map(point => point.date),
        y: rsis.map(point => point.value),
        name: "RSI (14)",
        line: { color: "#2ca02c", width: 1.5 },
        yaxis: "y2",
    };
}

function average(values) {
    if (!values.length) return 0;
    return values.reduce((total, value) => total + value, 0) / values.length;
}

function clearChart() {
    const chart = document.getElementById("chart");
    if (chart && chart.data) {
        Plotly.purge(chart);
    }
}
