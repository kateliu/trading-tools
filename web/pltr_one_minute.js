const statusEl = document.getElementById("status");
const sessionPicker = document.getElementById("session-picker");
const sessionRange = document.getElementById("session-range");

let availableSessions = [];
let currentSession = null;
let fullRecords = [];

init().catch((error) => {
    console.error(error);
    statusEl.textContent = `Failed to initialise dashboard: ${error.message}`;
});

async function init() {
    statusEl.textContent = "Loading available sessions...";
    try {
        const response = await fetch("../PLTR-one-minute/available_dates.json", { cache: "no-store" });
        if (!response.ok) {
            throw new Error(`HTTP ${response.status} ${response.statusText}`);
        }
        const payload = await response.json();
        availableSessions = (payload.dates || []).sort();
    } catch (error) {
        throw new Error(`Unable to load session list: ${error.message}`);
    }

    if (!availableSessions.length) {
        statusEl.textContent = "No sessions found in PLTR-one-minute.";
        return;
    }

    populateSessionPicker();

    sessionPicker.addEventListener("change", () => {
        loadSelectedSession().catch((error) => {
            console.error(error);
            statusEl.textContent = `Failed to load session: ${error.message}`;
        });
    });

    sessionRange.addEventListener("change", () => {
        renderCurrentSession();
    });

    await loadSelectedSession();
}

function populateSessionPicker() {
    sessionPicker.innerHTML = "";
    [...availableSessions].reverse().forEach((sessionDate, index) => {
        const option = document.createElement("option");
        option.value = sessionDate;
        option.textContent = sessionDate;
        if (index === 0) {
            option.selected = true;
        }
        sessionPicker.append(option);
    });
}

async function loadSelectedSession() {
    const selected = sessionPicker.value;
    if (!selected) {
        return;
    }

    statusEl.textContent = `Loading ${selected}.csv...`;
    const path = `../PLTR-one-minute/${selected}.csv`;

    let csvText;
    try {
        const response = await fetch(path, { cache: "no-store" });
        if (!response.ok) {
            throw new Error(`HTTP ${response.status} ${response.statusText}`);
        }
        csvText = await response.text();
    } catch (error) {
        throw new Error(`Unable to fetch ${path}: ${error.message}`);
    }

    const parsed = parseCsv(csvText);
    if (!parsed.length) {
        statusEl.textContent = `${selected}.csv contains no rows.`;
        Plotly.purge("chart");
        fullRecords = [];
        currentSession = selected;
        return;
    }

    fullRecords = parsed;
    currentSession = selected;
    renderCurrentSession();
}

function renderCurrentSession() {
    if (!fullRecords.length) {
        return;
    }

    const scope = sessionRange.value;
    const filtered = scope === "regular" ? filterRegularHours(fullRecords) : fullRecords;

    if (!filtered.length) {
        statusEl.textContent = "No bars available for the selected range.";
        Plotly.purge("chart");
        return;
    }

    const closes = filtered.map((row) => row.close);
    const highs = filtered.map((row) => row.high);
    const lows = filtered.map((row) => row.low);
    const volumes = filtered.map((row) => row.volume);
    const times = filtered.map((row) => row.timestamp);

    const ma5 = sma(closes, 5);
    const ema5 = ema(closes, 5);
    const bollinger = bollingerBands(closes, 20, 2);
    const macdResult = macd(closes);
    const volumeSma = sma(volumes, 5);
    const ewo = ewoOscillator(closes);

    const candlestickTrace = {
        type: "candlestick",
        name: "Price",
        x: times,
        open: filtered.map((row) => row.open),
        high: highs,
        low: lows,
        close: closes,
        xaxis: "x1",
        yaxis: "y1",
        increasing: { line: { color: "#2ecc71" } },
        decreasing: { line: { color: "#e74c3c" } },
        hovertemplate:
            "%{x|%Y-%m-%d %H:%M}<br>Open: %{open:.2f}<br>High: %{high:.2f}<br>Low: %{low:.2f}<br>Close: %{close:.2f}<br>Volume: %{customdata}<extra></extra>",
        customdata: filtered.map((row) => formatVolume(row.volume)),
        showlegend: false,
    };

    const maTrace = {
        type: "scatter",
        mode: "lines",
        name: "MA 5",
        x: times,
        y: ma5,
        xaxis: "x1",
        yaxis: "y1",
        line: { color: "#f1c40f", width: 1.6 },
    };

    const emaTrace = {
        type: "scatter",
        mode: "lines",
        name: "EMA 5",
        x: times,
        y: ema5,
        xaxis: "x1",
        yaxis: "y1",
        line: { color: "#3498db", width: 1.4, dash: "solid" },
    };

    const bbUpperTrace = {
        type: "scatter",
        mode: "lines",
        name: "BOLL Upper",
        x: times,
        y: bollinger.upper,
        xaxis: "x1",
        yaxis: "y1",
        line: { color: "#95a5a6", width: 1 },
        hovertemplate: "Upper %{y:.2f}<extra></extra>",
        showlegend: true,
    };

    const bbMidTrace = {
        type: "scatter",
        mode: "lines",
        name: "BOLL Mid",
        x: times,
        y: bollinger.mid,
        xaxis: "x1",
        yaxis: "y1",
        line: { color: "#bdc3c7", width: 1, dash: "dot" },
        hovertemplate: "Mid %{y:.2f}<extra></extra>",
        showlegend: true,
    };

    const bbLowerTrace = {
        type: "scatter",
        mode: "lines",
        name: "BOLL Lower",
        x: times,
        y: bollinger.lower,
        xaxis: "x1",
        yaxis: "y1",
        line: { color: "#95a5a6", width: 1 },
        hovertemplate: "Lower %{y:.2f}<extra></extra>",
        showlegend: true,
    };

    const macdHistogramTrace = {
        type: "bar",
        name: "MACD Histogram",
        x: times,
        y: macdResult.histogram,
        xaxis: "x2",
        yaxis: "y2",
        marker: {
            color: macdResult.histogram.map((value) => (value ?? 0) >= 0 ? "#16a085" : "#c0392b"),
        },
        hovertemplate: "%{x|%H:%M} | Hist %{y:.4f}<extra></extra>",
    };

    const macdLineTrace = {
        type: "scatter",
        mode: "lines",
        name: "MACD",
        x: times,
        y: macdResult.macd,
        xaxis: "x2",
        yaxis: "y2",
        line: { color: "#e67e22", width: 1.5 },
    };

    const macdSignalTrace = {
        type: "scatter",
        mode: "lines",
        name: "Signal",
        x: times,
        y: macdResult.signal,
        xaxis: "x2",
        yaxis: "y2",
        line: { color: "#2980b9", width: 1.5 },
    };

    const volumeTrace = {
        type: "bar",
        name: "Volume",
        x: times,
        y: volumes,
        xaxis: "x3",
        yaxis: "y3",
        marker: {
            color: filtered.map((row, index) => (row.close >= row.open ? "#1abc9c" : "#e74c3c")),
        },
        hovertemplate: "%{x|%H:%M} | Vol %{y:,.0f}<extra></extra>",
    };

    const volumeAvgTrace = {
        type: "scatter",
        mode: "lines",
        name: "VMA 5",
        x: times,
        y: volumeSma,
        xaxis: "x3",
        yaxis: "y3",
        line: { color: "#3498db", width: 1.5 },
    };

    const ewoTrace = {
        type: "bar",
        name: "EWO",
        x: times,
        y: ewo,
        xaxis: "x4",
        yaxis: "y4",
        marker: {
            color: ewo.map((value) => (value ?? 0) >= 0 ? "#f1c40f" : "#2980b9"),
        },
        hovertemplate: "%{x|%H:%M} | EWO %{y:.2f}<extra></extra>",
    };

    const data = [
        candlestickTrace,
        maTrace,
        emaTrace,
        bbUpperTrace,
        bbMidTrace,
        bbLowerTrace,
        macdHistogramTrace,
        macdLineTrace,
        macdSignalTrace,
        volumeTrace,
        volumeAvgTrace,
        ewoTrace,
    ];

    const layout = {
        legend: { orientation: "h", x: 0, y: 1.08 },
        margin: { l: 60, r: 35, t: 40, b: 40 },
        dragmode: "pan",
        showlegend: true,
        xaxis: {
            title: `${currentSession} (UTC)`,
            rangeslider: { visible: false },
            type: "date",
        },
        yaxis: {
            title: "Price",
            tickprefix: "$",
        },
        xaxis2: {
            showticklabels: false,
            type: "date",
        },
        yaxis2: {
            title: "MACD",
            zerolinecolor: "rgba(255,255,255,0.25)",
        },
        xaxis3: {
            showticklabels: false,
            type: "date",
        },
        yaxis3: {
            title: "Volume",
        },
        xaxis4: {
            type: "date",
        },
        yaxis4: {
            title: "EWO",
            zerolinecolor: "rgba(255,255,255,0.25)",
        },
        grid: {
            rows: 4,
            columns: 1,
            pattern: "independent",
            roworder: "top to bottom",
            rowheights: [0.55, 0.18, 0.17, 0.10],
        },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        colorway: ["#f1c40f", "#3498db", "#95a5a6", "#ecf0f1"],
    };

    Plotly.newPlot("chart", data, layout, {
        responsive: true,
        displaylogo: false,
        modeBarButtonsToRemove: ["autoScale2d", "select2d", "lasso2d"],
    });

    const scopeLabel = sessionRange.value === "regular" ? "regular session" : "full session";
    statusEl.textContent = `Showing ${filtered.length} bars for ${currentSession} (${scopeLabel}).`;
}

function parseCsv(csvText) {
    const lines = csvText.trim().split(/\r?\n/);
    if (lines.length <= 1) {
        return [];
    }

    const records = [];
    for (let i = 1; i < lines.length; i += 1) {
        const parts = lines[i].split(",");
        if (parts.length < 6) {
            continue;
        }
        const [timestamp, open, close, low, high, volume] = parts;
        const parsedTime = new Date(timestamp);
        if (!Number.isFinite(parsedTime.getTime())) {
            continue;
        }
        records.push({
            timestamp: parsedTime,
            isoTimestamp: timestamp,
            open: Number.parseFloat(open),
            close: Number.parseFloat(close),
            low: Number.parseFloat(low),
            high: Number.parseFloat(high),
            volume: Number.parseFloat(volume),
        });
    }
    return records;
}

function filterRegularHours(rows) {
    const formatter = new Intl.DateTimeFormat("en-US", {
        timeZone: "America/New_York",
        hour12: false,
        hour: "2-digit",
        minute: "2-digit",
    });

    return rows.filter((row) => {
        const parts = formatter.formatToParts(row.timestamp).reduce((acc, part) => {
            acc[part.type] = part.value;
            return acc;
        }, {});
        const hour = Number.parseInt(parts.hour, 10);
        const minute = Number.parseInt(parts.minute, 10);
        const minutesSinceMidnight = hour * 60 + minute;
        return minutesSinceMidnight >= 9 * 60 + 30 && minutesSinceMidnight <= 16 * 60;
    });
}

function sma(values, period) {
    const result = new Array(values.length).fill(null);
    let sum = 0;
    for (let i = 0; i < values.length; i += 1) {
        const value = values[i];
        sum += value;
        if (i >= period) {
            sum -= values[i - period];
        }
        if (i >= period - 1) {
            result[i] = sum / period;
        }
    }
    return result;
}

function ema(values, period) {
    const result = new Array(values.length).fill(null);
    const smoothing = 2 / (period + 1);
    let emaPrev = null;
    let warmup = 0;
    let accumulator = 0;

    for (let i = 0; i < values.length; i += 1) {
        const value = values[i];
        if (!Number.isFinite(value)) {
            continue;
        }

        if (warmup < period) {
            warmup += 1;
            accumulator += value;
            if (warmup === period) {
                emaPrev = accumulator / period;
                result[i] = emaPrev;
            }
            continue;
        }

        emaPrev = value * smoothing + emaPrev * (1 - smoothing);
        result[i] = emaPrev;
    }

    return result;
}

function bollingerBands(values, period = 20, multiplier = 2) {
    const mid = sma(values, period);
    const upper = new Array(values.length).fill(null);
    const lower = new Array(values.length).fill(null);

    let sum = 0;
    let sumSq = 0;
    for (let i = 0; i < values.length; i += 1) {
        const value = values[i];
        sum += value;
        sumSq += value * value;
        if (i >= period) {
            const dropped = values[i - period];
            sum -= dropped;
            sumSq -= dropped * dropped;
        }
        if (i >= period - 1) {
            const mean = sum / period;
            const variance = sumSq / period - mean * mean;
            const stdDev = Math.sqrt(Math.max(variance, 0));
            upper[i] = mean + multiplier * stdDev;
            lower[i] = mean - multiplier * stdDev;
        }
    }

    return { mid, upper, lower };
}

function macd(values, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
    const fastEma = ema(values, fastPeriod);
    const slowEma = ema(values, slowPeriod);
    const macdLine = values.map((_, index) => {
        const fast = fastEma[index];
        const slow = slowEma[index];
        if (fast == null || slow == null) {
            return null;
        }
        return fast - slow;
    });

    const signalLine = ema(macdLine, signalPeriod);
    const histogram = macdLine.map((value, index) => {
        const signal = signalLine[index];
        if (value == null || signal == null) {
            return null;
        }
        return value - signal;
    });

    return {
        macd: macdLine,
        signal: signalLine,
        histogram,
    };
}

function ewoOscillator(values, shortPeriod = 5, longPeriod = 35) {
    const short = ema(values, shortPeriod);
    const long = ema(values, longPeriod);
    return values.map((_, index) => {
        if (short[index] == null || long[index] == null) {
            return null;
        }
        return short[index] - long[index];
    });
}

function formatVolume(volume) {
    if (!Number.isFinite(volume)) {
        return "";
    }
    if (volume >= 1_000_000) {
        return `${(volume / 1_000_000).toFixed(2)}M`;
    }
    if (volume >= 1_000) {
        return `${(volume / 1_000).toFixed(1)}K`;
    }
    return volume.toFixed(0);
}
