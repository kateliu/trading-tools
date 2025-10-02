(async function init() {
    const statusEl = document.getElementById("status");
    const rangeEl = document.getElementById("range");
    const indicatorEl = document.getElementById("indicators");
    let fullData = [];
    let sourcePath = "";
    let aiForecast = null;

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

    try {
        aiForecast = await loadAiForecast();
        if (aiForecast && aiForecast.predictions.length) {
            console.info(`[chart] Loaded AI forecast with ${aiForecast.predictions.length} points.`);
        }
    } catch (error) {
        console.warn(`[chart] Unable to load AI forecast: ${error.message}`);
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
            renderChart(subset, value, indicatorEl ? indicatorEl.value : "none", aiForecast);
        }
    }

    rangeEl.addEventListener("change", applyRangeSelection);
    if (indicatorEl) {
        indicatorEl.addEventListener("change", applyRangeSelection);
    }
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

async function loadAiForecast() {
    const path = "data/ai_forecast.json";
    try {
        const response = await fetch(path, { cache: "no-store" });
        if (response.status === 404) {
            return null;
        }
        if (!response.ok) {
            throw new Error(`${response.status} ${response.statusText}`);
        }
        const payload = await response.json();
        if (!Array.isArray(payload.predictions)) {
            throw new Error("Forecast payload missing predictions array");
        }
        const cleaned = payload.predictions
            .map((entry) => {
                const date = entry.date;
                const close = Number.parseFloat(entry.close ?? entry.value ?? entry.next_close);
                const dateObj = date ? new Date(date) : null;
                return { date, close, dateObj };
            })
            .filter((entry) => entry.date && entry.dateObj && Number.isFinite(entry.close));
        return { ...payload, predictions: cleaned };
    } catch (error) {
        throw new Error(`Unable to load ${path}: ${error.message}`);
    }
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
        volume: header.indexOf("Volume"),
    };
    if (["date", "open", "high", "low", "close"].some((key) => idx[key] === -1)) {
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
        const volume = idx.volume !== -1 ? Number.parseFloat(cells[idx.volume]) : null;
        if (!dateStr || Number.isNaN(open) || Number.isNaN(high) || Number.isNaN(low) || Number.isNaN(close)) {
            continue;
        }
        const dateObj = new Date(dateStr);
        if (Number.isNaN(dateObj.getTime())) continue;
        rows.push({ date: dateStr, dateObj, open, high, low, close, volume });
    }
    if (!rows.length) {
        throw new Error("No valid rows parsed from CSV");
    }
    // Ensure chronological order
    rows.sort((a, b) => a.dateObj - b.dateObj);
    return rows;
}

function renderChart(data, rangeLabel, indicator, aiForecast) {
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
        name: "PLTR Price",
    };

    const indicatorResult = buildIndicatorTraces(data, indicator, aiForecast);
    const overlays = indicatorResult.traces;
    const xAxisEnd = indicatorResult.extendX || data[data.length - 1].date;

    const volumeTrace = buildVolumeTrace(data);

    const layout = {
        title: `Palantir Stock Price — ${titleSuffix}`,
        margin: { l: 60, r: 60, t: 60, b: 40 },
        xaxis: {
            range: [data[0].date, xAxisEnd],
            rangeslider: { visible: true },
            type: "date",
        },
        yaxis: {
            title: "Price (USD)",
            fixedrange: false,
        },
        yaxis2: {
            title: "Volume",
            overlaying: "y",
            side: "right",
            showgrid: false,
            rangemode: "tozero",
            tickformat: ",",
        },
        plot_bgcolor: "rgba(0,0,0,0)",
        paper_bgcolor: "rgba(0,0,0,0)",
    };

    if (indicatorResult.secondaryAxis) {
        layout.yaxis3 = indicatorResult.secondaryAxis;
    }

    if (indicatorResult.shapes && indicatorResult.shapes.length) {
        layout.shapes = (layout.shapes || []).concat(indicatorResult.shapes);
    }

    if (indicatorResult.annotations && indicatorResult.annotations.length) {
        layout.annotations = indicatorResult.annotations;
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
        const traces = [trace, ...overlays];
        if (volumeTrace) {
            traces.push(volumeTrace);
        }
        Plotly.newPlot("chart", traces, layout, config);
    } catch (error) {
        console.error("Failed to render chart", error);
        const statusEl = document.getElementById("status");
        statusEl.textContent = `Plotly render failed: ${error.message}`;
    }
}

function buildIndicatorTraces(data, indicator, aiForecast) {
    const result = {
        traces: [],
        extendX: null,
        secondaryAxis: null,
        shapes: [],
        annotations: [],
    };

    if (!indicator || indicator === "none") {
        return result;
    }

    switch (indicator) {
        case "sma":
            result.traces = [
                makeLineTrace("SMA (20)", movingAverage(data, 20), "#1f77b4"),
                makeLineTrace("SMA (50)", movingAverage(data, 50), "#ff7f0e"),
            ];
            return result;
        case "ema":
            result.traces = [
                makeLineTrace("EMA (12)", exponentialMovingAverage(data, 12), "#9467bd"),
                makeLineTrace("EMA (26)", exponentialMovingAverage(data, 26), "#8c564b"),
            ];
            return result;
        case "rsi":
            result.traces = [relativeStrengthTrace(data, 14)];
            result.secondaryAxis = {
                title: "RSI",
                overlaying: "y",
                side: "right",
                range: [0, 100],
                showgrid: false,
                zeroline: false,
                position: 1.0,
            };
            result.shapes = [
                {
                    type: "line",
                    xref: "paper",
                    x0: 0,
                    x1: 1,
                    yref: "y3",
                    y0: 70,
                    y1: 70,
                    line: { color: "rgba(200, 100, 100, 0.5)", dash: "dot" },
                },
                {
                    type: "line",
                    xref: "paper",
                    x0: 0,
                    x1: 1,
                    yref: "y3",
                    y0: 30,
                    y1: 30,
                    line: { color: "rgba(100, 150, 200, 0.5)", dash: "dot" },
                },
            ];
            return result;
        case "ai":
            if (aiForecast && Array.isArray(aiForecast.predictions) && aiForecast.predictions.length) {
                addAiForecastTrace(data, aiForecast, result);
            } else {
                addLinearForecastTrace(data, result);
            }
            return result;
        default:
            return result;
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

function addAiForecastTrace(data, forecast, result) {
    const lastActual = data[data.length - 1];
    const sorted = [...forecast.predictions].sort(
        (a, b) => new Date(a.date) - new Date(b.date),
    );
    if (!sorted.length) {
        addLinearForecastTrace(data, result);
        return;
    }

    const upcoming = sorted.filter((entry) => entry.dateObj ? entry.dateObj > lastActual.dateObj : new Date(entry.date) > lastActual.dateObj);
    const sequence = upcoming.length ? upcoming : sorted;

    const x = [lastActual.date, ...sequence.map((item) => item.date)];
    const y = [lastActual.close, ...sequence.map((item) => Number(item.close))];

    const name = forecast.source ? `AI Forecast (${forecast.source})` : "AI Forecast";
    result.traces.push({
        type: "scatter",
        mode: "lines",
        x,
        y,
        name,
        line: { color: "#ff1493", width: 2, dash: "dash" },
        hovertemplate: "%{x}<br>%{y:.2f} USD<extra>AI Forecast</extra>",
        yaxis: "y",
    });

    const lastForecast = sequence[sequence.length - 1];
    result.extendX = lastForecast.date;
    result.shapes.push({
        type: "rect",
        xref: "x",
        x0: lastActual.date,
        x1: lastForecast.date,
        yref: "paper",
        y0: 0,
        y1: 1,
        fillcolor: "rgba(255, 20, 147, 0.07)",
        line: { width: 0 },
    });

    const metrics = forecast.metrics || {};
    const rmse = typeof metrics.rmse === "number" ? metrics.rmse.toFixed(2) : "n/a";
    const mae = typeof metrics.mae === "number" ? metrics.mae.toFixed(2) : "n/a";
    const startClose = Number(sequence[0].close);
    const endClose = Number(lastForecast.close);
    const slope = (endClose - startClose) / Math.max(sequence.length, 1);
    const trend = buildTrendText(slope);
    const midPoint = sequence[Math.floor((sequence.length - 1) / 2)] || lastForecast;

    result.annotations.push({
        x: midPoint.date,
        y: Number(midPoint.close),
        xref: "x",
        yref: "y",
        text: `${name}<br>${trend}<br>RMSE ${rmse} | MAE ${mae}`,
        showarrow: false,
        bgcolor: "rgba(255, 20, 147, 0.08)",
        bordercolor: "rgba(255, 20, 147, 0.25)",
        borderwidth: 1,
        font: { color: "#ff1493", size: 12 },
        align: "left",
    });
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
        yaxis: "y3",
    };
}

function average(values) {
    if (!values.length) return 0;
    return values.reduce((total, value) => total + value, 0) / values.length;
}

function addLinearForecastTrace(data, result) {
    if (data.length < 2) {
        return;
    }

    const horizon = 30;
    const forecast = linearRegressionForecast(data, horizon);
    if (!forecast.points.length) {
        return;
    }

    const lastActual = data[data.length - 1];
    const forecastTrace = makeForecastTrace(lastActual, forecast.points);
    result.traces.push(forecastTrace);
    result.extendX = forecast.points[forecast.points.length - 1].date;

    result.shapes.push({
        type: "rect",
        xref: "x",
        x0: lastActual.date,
        x1: result.extendX,
        yref: "paper",
        y0: 0,
        y1: 1,
        fillcolor: "rgba(255, 20, 147, 0.07)",
        line: { width: 0 },
    });

    const trendText = buildTrendText(forecast.slope);
    const midPoint = forecast.points[Math.floor(forecast.points.length / 2)];
    result.annotations.push({
        x: midPoint.date,
        y: midPoint.value,
        xref: "x",
        yref: "y",
        text: `AI trend: ${trendText}`,
        showarrow: false,
        bgcolor: "rgba(255, 20, 147, 0.08)",
        bordercolor: "rgba(255, 20, 147, 0.25)",
        borderwidth: 1,
        font: { color: "#ff1493", size: 12 },
        align: "left",
    });
}

function makeForecastTrace(lastPoint, forecastPoints) {
    const x = [lastPoint.date, ...forecastPoints.map(point => point.date)];
    const y = [lastPoint.close, ...forecastPoints.map(point => point.value)];
    return {
        type: "scatter",
        mode: "lines",
        x,
        y,
        name: "AI Forecast (30d)",
        line: { color: "#ff1493", width: 2, dash: "dash" },
        hovertemplate: "%{x}<br>%{y:.2f} USD<extra>AI Forecast</extra>",
        yaxis: "y",
    };
}

function linearRegressionForecast(data, horizon) {
    const n = data.length;
    if (n < 2) {
        return { points: [], slope: 0, intercept: data[0] ? data[0].close : 0 };
    }

    let sumX = 0;
    let sumY = 0;
    let sumXY = 0;
    let sumX2 = 0;
    for (let i = 0; i < n; i += 1) {
        const x = i;
        const y = data[i].close;
        sumX += x;
        sumY += y;
        sumXY += x * y;
        sumX2 += x * x;
    }

    const denominator = n * sumX2 - sumX * sumX;
    const slope = denominator === 0 ? 0 : (n * sumXY - sumX * sumY) / denominator;
    const intercept = (sumY - slope * sumX) / n;

    const points = [];
    let cursor = new Date(data[data.length - 1].dateObj.getTime());
    for (let i = 1; i <= horizon; i += 1) {
        cursor = nextMarketDay(cursor);
        const futureIndex = n - 1 + i;
        const estimated = slope * futureIndex + intercept;
        points.push({ date: toISODate(cursor), value: estimated });
    }

    return { points, slope, intercept };
}

function nextMarketDay(date) {
    const candidate = new Date(date.getTime());
    candidate.setDate(candidate.getDate() + 1);
    let day = candidate.getUTCDay();
    if (day === 6) {
        candidate.setDate(candidate.getDate() + 2);
    } else if (day === 0) {
        candidate.setDate(candidate.getDate() + 1);
    }
    return candidate;
}

function toISODate(date) {
    return date.toISOString().split("T")[0];
}

function buildTrendText(slope) {
    const dailyChange = slope;
    const direction = dailyChange > 0.05 ? "↑" : dailyChange < -0.05 ? "↓" : "→";
    return `${direction} ${dailyChange.toFixed(2)} USD/day`;
}

function clearChart() {
    const chart = document.getElementById("chart");
    if (chart && chart.data) {
        Plotly.purge(chart);
    }
}

function buildVolumeTrace(data) {
    const volumes = data.map((row) => row.volume).filter((value) => Number.isFinite(value));
    if (!volumes.length) {
        return null;
    }

    return {
        type: "bar",
        x: data.map((row) => row.date),
        y: data.map((row) => Number.isFinite(row.volume) ? row.volume : null),
        name: "Volume",
        marker: { color: "rgba(100, 149, 237, 0.4)" },
        yaxis: "y2",
        opacity: 0.6,
    };
}
