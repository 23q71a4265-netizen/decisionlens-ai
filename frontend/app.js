// DecisionLens AI — frontend controller
// Talks to the FastAPI backend at API_BASE. Same-origin by default (the
// backend serves this file too), so this works unmodified when deployed.

const API_BASE = window.location.origin.includes("localhost") || window.location.origin.includes("127.0.0.1")
  ? window.location.origin
  : window.location.origin; // same origin in production; change if frontend/backend are split

let state = {
  columns: [],
  numericColumns: [],
  groupCol: null,
  groupValues: [],
};

let forecastChart = null;

const el = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

function setStatus(live, label) {
  el("statusDot").classList.toggle("live", live);
  el("statusText").textContent = label;
}

async function checkHealth() {
  try {
    const h = await api("/api/health");
    setStatus(h.dataset_loaded, h.dataset_loaded ? "LIVE" : "READY — NO DATA");
  } catch (e) {
    setStatus(false, "BACKEND UNREACHABLE");
  }
}

// ---------------- data intake ----------------
el("loadSampleBtn").addEventListener("click", async () => {
  setIntakeMeta("Loading demo dataset…");
  try {
    const data = await api("/api/load-sample", { method: "POST" });
    onDatasetLoaded(data);
  } catch (e) {
    setIntakeMeta(`Error: ${e.message}`);
  }
});

el("csvInput").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  setIntakeMeta(`Uploading ${file.name}…`);
  const form = new FormData();
  form.append("file", file);
  try {
    const res = await fetch(`${API_BASE}/api/upload`, { method: "POST", body: form });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || "Upload failed");
    }
    const data = await res.json();
    onDatasetLoaded(data);
  } catch (err) {
    setIntakeMeta(`Error: ${err.message}`);
  }
});

function setIntakeMeta(text) {
  el("intakeMeta").textContent = text;
}

async function onDatasetLoaded(data) {
  el("datasetLabel").textContent = `${data.dataset_name} · ${data.rows} rows · ${data.columns.length} cols`;
  setIntakeMeta(`Loaded ${data.dataset_name} — ${data.rows} rows, group by "${data.group_col || "none"}".`);
  setStatus(true, "LIVE");
  state.columns = data.columns;
  state.groupCol = data.group_col;

  await Promise.all([refreshSummary(), refreshInsights(), refreshAlerts(), refreshGroups()]);
  populateForecastColumns();
  await runForecast();
}

async function refreshGroups() {
  try {
    const g = await api("/api/groups");
    state.groupValues = g.values || [];
    const sel = el("forecastGroup");
    sel.innerHTML = '<option value="">All</option>' + state.groupValues.map(v => `<option value="${v}">${v}</option>`).join("");
  } catch (e) { /* non-fatal */ }
}

// ---------------- KPI summary ----------------
async function refreshSummary() {
  try {
    const s = await api("/api/summary");
    const metrics = Object.entries(s.metrics || {});
    state.numericColumns = metrics.map(([k]) => k);
    if (metrics.length === 0) {
      el("kpiRow").innerHTML = '<div class="kpi-placeholder">No numeric columns found in this dataset.</div>';
      return;
    }
    el("kpiRow").innerHTML = metrics.map(([col, m]) => {
      const changeClass = m.pct_change_latest > 0 ? "up" : m.pct_change_latest < 0 ? "down" : "";
      const arrow = m.pct_change_latest > 0 ? "▲" : m.pct_change_latest < 0 ? "▼" : "•";
      return `
        <div class="kpi-card">
          <div class="kpi-label">${escapeHtml(col)}</div>
          <div class="kpi-value">${formatNum(m.latest)}</div>
          <div class="kpi-change ${changeClass} mono">${arrow} ${Math.abs(m.pct_change_latest)}% · avg ${formatNum(m.mean)}</div>
        </div>`;
    }).join("");
  } catch (e) {
    el("kpiRow").innerHTML = `<div class="kpi-placeholder">Could not load summary: ${escapeHtml(e.message)}</div>`;
  }
}

// ---------------- insights / anomalies ----------------
async function refreshInsights() {
  try {
    const data = await api("/api/insights");
    renderInsights(data.anomalies || []);
    renderTicker(data.anomalies || []);
  } catch (e) {
    el("insightsList").innerHTML = `<p class="empty-state">Could not load insights: ${escapeHtml(e.message)}</p>`;
  }
}

function renderInsights(anomalies) {
  if (!anomalies.length) {
    el("insightsList").innerHTML = '<p class="empty-state">No anomalies detected — data looks within normal range.</p>';
    return;
  }
  el("insightsList").innerHTML = anomalies.slice(0, 30).map(a => {
    const groupTxt = a.group ? ` in <strong>${escapeHtml(String(a.group))}</strong>` : "";
    let detail;
    if (a.type === "zscore") {
      detail = `<strong>${escapeHtml(a.column)}</strong>${groupTxt}: value ${formatNum(a.value)} (z=${a.z_score})`;
    } else {
      detail = `Multivariate pattern flagged${groupTxt} (score ${a.anomaly_score})`;
    }
    return `<div class="insight-row"><span>${detail}</span><span class="tag ${a.severity}">${a.severity}</span></div>`;
  }).join("");
}

function renderTicker(anomalies) {
  const track = el("tickerTrack");
  if (!anomalies.length) {
    track.innerHTML = '<span class="ticker-item mono">All signals nominal — no anomalies in the active dataset.</span>';
    return;
  }
  const items = anomalies.slice(0, 12).map(a => {
    const cls = a.severity === "high" ? "tick-critical" : "tick-warning";
    const groupTxt = a.group ? ` · ${a.group}` : "";
    const label = a.type === "zscore"
      ? `${a.column}${groupTxt}: ${formatNum(a.value)} (z=${a.z_score})`
      : `multivariate anomaly${groupTxt} (score ${a.anomaly_score})`;
    return `<span class="ticker-item mono ${cls}">⚠ ${escapeHtml(label)}</span>`;
  });
  // duplicate the list so the CSS marquee (translateX -50%) loops seamlessly
  track.innerHTML = items.join("") + items.join("");
}

// ---------------- forecast ----------------
function populateForecastColumns() {
  const sel = el("forecastColumn");
  sel.innerHTML = state.numericColumns.map(c => `<option value="${c}">${c}</option>`).join("");
}

el("runForecastBtn").addEventListener("click", runForecast);

async function runForecast() {
  const column = el("forecastColumn").value;
  const periods = el("forecastPeriods").value;
  const group = el("forecastGroup").value;
  if (!column) return;
  try {
    const q = new URLSearchParams({ column, periods });
    if (group) q.append("group", group);
    const data = await api(`/api/forecast?${q.toString()}`);
    if (data.error) {
      el("forecastNote").textContent = data.error;
      return;
    }
    drawForecastChart(data);
    el("forecastNote").textContent =
      `Trend: ${data.trend_direction} (slope ${data.trend_slope}/period) · shaded band = 95% confidence interval`;
  } catch (e) {
    el("forecastNote").textContent = `Error: ${e.message}`;
  }
}

function drawForecastChart(data) {
  const ctx = el("forecastChart").getContext("2d");
  const histLen = data.history.length;
  const foreLen = data.forecast.length;
  const labels = [
    ...Array.from({ length: histLen }, (_, i) => `t-${histLen - i}`),
    ...Array.from({ length: foreLen }, (_, i) => `t+${i + 1}`),
  ];
  const historyData = [...data.history, ...Array(foreLen).fill(null)];
  const forecastData = [...Array(histLen - 1).fill(null), data.history[histLen - 1], ...data.forecast];
  const upperData = [...Array(histLen).fill(null), ...data.upper_bound];
  const lowerData = [...Array(histLen).fill(null), ...data.lower_bound];

  if (forecastChart) forecastChart.destroy();
  forecastChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Upper bound", data: upperData, borderWidth: 0,
          pointRadius: 0, backgroundColor: "rgba(47,224,196,0.08)", fill: "+1",
        },
        {
          label: "Lower bound", data: lowerData, borderWidth: 0,
          pointRadius: 0, backgroundColor: "rgba(47,224,196,0.08)", fill: false,
        },
        {
          label: "History", data: historyData, borderColor: "#7C8A99",
          backgroundColor: "transparent", pointRadius: 0, borderWidth: 2, tension: 0.15,
        },
        {
          label: "Forecast", data: forecastData, borderColor: "#2FE0C4",
          backgroundColor: "transparent", pointRadius: 0, borderWidth: 2, borderDash: [4, 3], tension: 0.15,
        },
      ],
    },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { color: "#7C8A99", filter: (item) => item.text === "History" || item.text === "Forecast" } },
      },
      scales: {
        x: { ticks: { color: "#7C8A99", maxTicksLimit: 10 }, grid: { color: "#1F2830" } },
        y: { ticks: { color: "#7C8A99" }, grid: { color: "#1F2830" } },
      },
    },
  });
}

// ---------------- NL query ----------------
el("queryForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = el("queryInput").value.trim();
  if (!question) return;
  el("queryAnswer").innerHTML = '<span class="muted mono">Analyzing…</span>';
  try {
    const data = await api("/api/query", { method: "POST", body: JSON.stringify({ question }) });
    el("queryAnswer").innerHTML = `${escapeHtml(data.answer)}<br/><span class="source-tag">${data.powered_by}</span>`;
  } catch (err) {
    el("queryAnswer").innerHTML = `<span class="muted">Error: ${escapeHtml(err.message)}</span>`;
  }
});

// ---------------- alerts ----------------
el("alertThreshold").addEventListener("input", (e) => {
  el("alertThresholdVal").textContent = e.target.value;
});
el("alertThreshold").addEventListener("change", refreshAlerts);

async function refreshAlerts() {
  const z = el("alertThreshold").value;
  try {
    const data = await api(`/api/alerts?z_threshold=${z}`);
    renderAlerts(data.alerts || []);
  } catch (e) {
    el("alertsList").innerHTML = `<p class="empty-state">Could not load alerts: ${escapeHtml(e.message)}</p>`;
  }
}

function renderAlerts(alerts) {
  if (!alerts.length) {
    el("alertsList").innerHTML = '<p class="empty-state">No alerts at this sensitivity level.</p>';
    return;
  }
  el("alertsList").innerHTML = alerts.map(a => {
    const cls = a.severity === "critical" ? "critical" : "warning";
    const groupTxt = a.group ? `${escapeHtml(String(a.group))} · ` : "";
    return `<div class="alert-row ${cls}">
      <span>${groupTxt}<strong>${escapeHtml(a.column)}</strong> ${a.direction} to ${formatNum(a.latest_value)}</span>
      <span class="z mono">z=${a.z_score}</span>
    </div>`;
  }).join("");
}

// ---------------- utils ----------------
function formatNum(n) {
  if (n === null || n === undefined) return "—";
  const num = Number(n);
  if (Math.abs(num) >= 1000) return num.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return num.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ---------------- boot ----------------
checkHealth();
