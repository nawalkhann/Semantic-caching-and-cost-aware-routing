"""Self-contained HTML dashboard for the per-request cost/latency log.

`render_dashboard(store)` returns one HTML page (no external assets) with KPI
tiles, cumulative cost and rolling p95 latency per run, the route mix, and the
most recent requests. The API serves it at /dashboard; the benchmark writes it
to benchmarks/results/dashboard.html.
"""

from __future__ import annotations

import html
import json

import numpy as np

from .metrics import MetricsStore, summarize

_MAX_POINTS = 300
_P95_WINDOW = 200


def _downsample(xs: list[int], ys: list[float]) -> list[list[float]]:
    if len(xs) <= _MAX_POINTS:
        return [[x, y] for x, y in zip(xs, ys)]
    idx = np.linspace(0, len(xs) - 1, _MAX_POINTS).round().astype(int)
    return [[xs[i], ys[i]] for i in idx]


def dashboard_data(store: MetricsStore, recent: int = 50) -> dict:
    runs = []
    for run in store.runs():
        recs = store.records(run)
        cost = np.cumsum([r["cost_usd"] for r in recs])
        lat = np.array([r["latency_ms"] for r in recs])
        n = np.arange(1, len(recs) + 1)
        p95 = [float(np.percentile(lat[max(0, i - _P95_WINDOW):i], 95)) for i in n]
        s = summarize(recs)
        runs.append({
            "name": run,
            "summary": s,
            "cum_cost": _downsample(n.tolist(), cost.tolist()),
            "rolling_p95": _downsample(n.tolist(), p95),
            "routes": {k: v["requests"] for k, v in s["by_route"].items()},
            "recent": [
                {k: r[k] for k in ("id", "query", "route", "cost_usd", "latency_ms",
                                   "similarity", "complexity", "correct")}
                for r in recs[-recent:][::-1]],
        })
    return {"runs": runs}


def render_dashboard(store: MetricsStore, title: str = "RAG cost & latency",
                     benchmark: dict | None = None) -> str:
    data = dashboard_data(store)
    data["benchmark"] = benchmark
    payload = json.dumps(data).replace("</", "<\\/")
    return (_TEMPLATE.replace("__TITLE__", html.escape(title))
            .replace("__DATA__", payload))


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  color-scheme: light;
  --surface-0: #f4f3f0; --surface-1: #fcfcfb; --border: #e2e1dc;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #77766f;
  --grid: #e8e7e2;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --s6: #008300;
  --good: #0ca30c; --critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface-0: #111110; --surface-1: #1a1a19; --border: #2e2e2b;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #94938a;
    --grid: #2a2a27;
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #008300;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #111110; --surface-1: #1a1a19; --border: #2e2e2b;
  --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #94938a;
  --grid: #2a2a27;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #008300;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface-0); color: var(--text-primary);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 0 0 12px; }
.sub { color: var(--text-secondary); margin: 0 0 20px; }
.controls { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 16px; }
select { font: inherit; padding: 6px 10px; border-radius: 8px; border: 1px solid var(--border);
  background: var(--surface-1); color: var(--text-primary); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px;
  margin-bottom: 16px; }
.tile, .card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; }
.tile { padding: 14px 16px; }
.tile .label { color: var(--text-secondary); font-size: 12px; }
.tile .value { font-size: 24px; font-weight: 600; font-variant-numeric: tabular-nums; }
.tile .delta { font-size: 12px; color: var(--text-muted); }
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 12px;
  margin-bottom: 16px; }
.card { padding: 16px; min-width: 0; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 12px;
  color: var(--text-secondary); margin-bottom: 8px; }
.legend span::before { content: ""; display: inline-block; width: 10px; height: 10px;
  border-radius: 3px; margin-right: 6px; background: var(--c); vertical-align: -1px; }
svg { display: block; width: 100%; height: auto; overflow: visible; }
svg text { fill: var(--text-muted); font-size: 11px; }
.tip { position: fixed; pointer-events: none; background: var(--surface-1);
  border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; font-size: 12px;
  box-shadow: 0 4px 16px rgba(0,0,0,.12); display: none; z-index: 10; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); }
th { color: var(--text-secondary); font-weight: 500; }
td.num, th.num { text-align: right; white-space: nowrap; }
td.q { max-width: 420px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.pill { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 12px;
  border: 1px solid var(--border); color: var(--text-secondary); }
.pill::before { content: ""; display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  margin-right: 5px; background: var(--c); }
.ok { color: var(--good); } .bad { color: var(--critical); }
.empty { color: var(--text-secondary); padding: 40px 0; text-align: center; }
</style>
</head>
<body>
<main>
  <h1>__TITLE__</h1>
  <p class="sub">Per-request cost and latency from the semantic cache and model router.</p>
  <div id="app"></div>
</main>
<div class="tip" id="tip"></div>
<script>
const DATA = __DATA__;
const app = document.getElementById("app");
const tip = document.getElementById("tip");
const SERIES = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6"];
const ROUTE_COLOR = { cache: "--s3", small: "--s1", large: "--s2" };
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const usd = (v, d = 2) => "$" + v.toFixed(d);
const ms = v => v >= 1000 ? (v / 1000).toFixed(2) + " s" : Math.round(v) + " ms";
const pct = v => (v * 100).toFixed(1) + "%";
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const runColor = name => SERIES[DATA.runs.findIndex(r => r.name === name) % SERIES.length];

function showTip(evt, htmlText) {
  tip.innerHTML = htmlText; tip.style.display = "block";
  const x = Math.min(evt.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
  tip.style.left = x + "px"; tip.style.top = (evt.clientY + 14) + "px";
}
function hideTip() { tip.style.display = "none"; }

function niceMax(v) {
  if (v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p;
}

function lineChart(series, { fmtY, xLabel }) {
  const W = 560, H = 240, L = 56, R = 12, T = 8, B = 30;
  const xMax = Math.max(...series.flatMap(s => s.points.map(p => p[0])));
  const yMax = niceMax(Math.max(...series.flatMap(s => s.points.map(p => p[1]))));
  const x = v => L + (v / xMax) * (W - L - R), y = v => T + (1 - v / yMax) * (H - T - B);
  let g = "";
  for (let i = 0; i <= 4; i++) {
    const v = yMax * i / 4;
    g += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="${css("--grid")}"/>`;
    g += `<text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${fmtY(v)}</text>`;
  }
  g += `<text x="${W - R}" y="${H - 6}" text-anchor="end">${xLabel}</text>`;
  g += `<text x="${L}" y="${H - 6}">0</text>`;
  for (const s of series) {
    const d = s.points.map((p, i) => (i ? "L" : "M") + x(p[0]).toFixed(1) + "," + y(p[1]).toFixed(1)).join("");
    g += `<path d="${d}" fill="none" stroke="${css(s.color)}" stroke-width="2" stroke-linejoin="round"/>`;
  }
  g += `<line class="xh" y1="${T}" y2="${H - B}" stroke="${css("--text-muted")}" stroke-dasharray="3 3" visibility="hidden"/>`;
  g += `<rect class="hit" x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}" fill="transparent"/>`;
  const el = document.createElement("div");
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img">${g}</svg>`;
  const svg = el.firstChild, xh = svg.querySelector(".xh");
  svg.querySelector(".hit").addEventListener("mousemove", evt => {
    const box = svg.getBoundingClientRect();
    const vx = ((evt.clientX - box.left) / box.width * W - L) / (W - L - R) * xMax;
    const rows = series.map(s => {
      let best = s.points[0];
      for (const p of s.points) if (Math.abs(p[0] - vx) < Math.abs(best[0] - vx)) best = p;
      return { s, p: best };
    });
    const px = x(rows[0].p[0]);
    xh.setAttribute("x1", px); xh.setAttribute("x2", px); xh.setAttribute("visibility", "visible");
    showTip(evt, `<b>request ${rows[0].p[0]}</b><br>` + rows.map(r =>
      `<span class="pill" style="--c:${css(r.s.color)}">${esc(r.s.name)}</span> ${fmtY(r.p[1])}`).join("<br>"));
  });
  svg.querySelector(".hit").addEventListener("mouseleave", () => { hideTip(); xh.setAttribute("visibility", "hidden"); });
  return svg;
}

function routeBars(runs) {
  const keys = ["cache", "small", "large"];
  const W = 1100, rowH = 30, L = 150, R = 12, H = runs.length * rowH + 8;
  let g = "";
  runs.forEach((run, i) => {
    const total = Object.values(run.routes).reduce((a, b) => a + b, 0);
    let x0 = L;
    g += `<text x="${L - 10}" y="${i * rowH + 20}" text-anchor="end" style="font-size:13px">${esc(run.name)}</text>`;
    for (const k of keys) {
      const n = run.routes[k] || 0;
      if (!n) continue;
      const w = n / total * (W - L - R);
      g += `<rect data-run="${esc(run.name)}" data-k="${k}" data-n="${n}" data-t="${total}" x="${x0}" y="${i * rowH + 5}" width="${Math.max(w - 2, 1)}" height="18" rx="4" fill="${css(ROUTE_COLOR[k])}"/>`;
      x0 += w;
    }
  });
  const el = document.createElement("div");
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img">${g}</svg>`;
  const svg = el.firstChild;
  svg.querySelectorAll("rect").forEach(r => {
    r.addEventListener("mousemove", evt => showTip(evt,
      `<b>${r.dataset.run}</b><br>${r.dataset.k}: ${r.dataset.n} of ${r.dataset.t} (${pct(r.dataset.n / r.dataset.t)})`));
    r.addEventListener("mouseleave", hideTip);
  });
  return svg;
}

function card(title, node, legend) {
  const c = document.createElement("div");
  c.className = "card";
  c.innerHTML = `<h2>${title}</h2>` + (legend ? `<div class="legend">${legend}</div>` : "");
  c.appendChild(node);
  return c;
}

function legendFor(items) {
  return items.map(([name, color]) => `<span style="--c:${css(color)}">${esc(name)}</span>`).join("");
}

function render() {
  app.innerHTML = "";
  if (!DATA.runs.length) { app.innerHTML = `<div class="empty">No requests logged yet.</div>`; return; }
  const controls = document.createElement("div");
  controls.className = "controls";
  controls.innerHTML = `<label>Run <select id="run">${DATA.runs.map(r =>
    `<option ${r.name === state.run ? "selected" : ""}>${esc(r.name)}</option>`).join("")}</select></label>`;
  app.appendChild(controls);
  controls.querySelector("select").addEventListener("change", e => { state.run = e.target.value; render(); });

  const run = DATA.runs.find(r => r.name === state.run);
  const base = DATA.runs[0].summary, s = run.summary;
  const delta = (a, b) => run === DATA.runs[0] ? "" : `${((a / b - 1) * 100).toFixed(1)}% vs ${esc(DATA.runs[0].name)}`;
  const tiles = document.createElement("div");
  tiles.className = "tiles";
  tiles.innerHTML = [
    ["Requests", s.requests.toLocaleString(), ""],
    ["Total cost", usd(s.total_cost_usd), delta(s.total_cost_usd, base.total_cost_usd)],
    ["Cost / 1k requests", usd(s.cost_per_1k_requests_usd, 3), ""],
    ["p50 latency", ms(s.p50_latency_ms), delta(s.p50_latency_ms, base.p50_latency_ms)],
    ["p95 latency", ms(s.p95_latency_ms), delta(s.p95_latency_ms, base.p95_latency_ms)],
    ["Cache hit rate", pct(s.cache_hit_rate), ""],
    ["Accuracy", s.accuracy == null ? "n/a" : pct(s.accuracy), s.accuracy == null || base.accuracy == null || run === DATA.runs[0] ? "" :
      `${((s.accuracy - base.accuracy) * 100).toFixed(1)} pts vs ${esc(DATA.runs[0].name)}`],
  ].map(([l, v, d]) => `<div class="tile"><div class="label">${l}</div><div class="value">${v}</div><div class="delta">${d}</div></div>`).join("");
  app.appendChild(tiles);

  const multi = DATA.runs.length > 1;
  const series = DATA.runs.map(r => ({ name: r.name, color: runColor(r.name) }));
  const grid = document.createElement("div");
  grid.className = "grid2";
  grid.appendChild(card("Cumulative cost", lineChart(
    DATA.runs.map((r, i) => ({ ...series[i], points: r.cum_cost })), { fmtY: v => usd(v), xLabel: "requests" }),
    multi ? legendFor(series.map(x => [x.name, x.color])) : ""));
  grid.appendChild(card(`Rolling p95 latency (last ${200} requests)`, lineChart(
    DATA.runs.map((r, i) => ({ ...series[i], points: r.rolling_p95 })), { fmtY: ms, xLabel: "requests" }),
    multi ? legendFor(series.map(x => [x.name, x.color])) : ""));
  app.appendChild(grid);

  app.appendChild(card("Where requests were answered", routeBars(DATA.runs),
    legendFor([["cache", ROUTE_COLOR.cache], ["small model", ROUTE_COLOR.small], ["large model", ROUTE_COLOR.large]])));
  app.lastChild.style.marginBottom = "16px";

  if (DATA.benchmark) app.appendChild(benchmarkTable(DATA.benchmark));

  const t = document.createElement("div");
  t.className = "card";
  t.innerHTML = `<h2>Latest requests: ${esc(run.name)}</h2><div class="table-wrap"><table>
    <thead><tr><th>#</th><th>Query</th><th>Route</th><th class="num">Cost</th><th class="num">Latency</th>
    <th class="num">Similarity</th><th class="num">Complexity</th><th>Correct</th></tr></thead><tbody>
    ${run.recent.map(r => `<tr><td>${r.id}</td><td class="q" title="${esc(r.query)}">${esc(r.query)}</td>
      <td><span class="pill" style="--c:${css(ROUTE_COLOR[r.route] || "--s4")}">${esc(r.route)}</span></td>
      <td class="num">${usd(r.cost_usd, 5)}</td><td class="num">${ms(r.latency_ms)}</td>
      <td class="num">${r.similarity == null ? "" : r.similarity.toFixed(3)}</td>
      <td class="num">${r.complexity == null ? "" : r.complexity.toFixed(2)}</td>
      <td>${r.correct == null ? "" : r.correct ? '<span class="ok">✓ yes</span>' : '<span class="bad">✗ no</span>'}</td></tr>`).join("")}
    </tbody></table></div>`;
  app.appendChild(t);
}

function benchmarkTable(b) {
  const names = Object.keys(b.strategies), base = b.strategies.baseline;
  const m = (a, k) => a[k] ? a[k].mean : null;
  const d = (a, k) => base && m(base, k) ? `${((m(a, k) / m(base, k) - 1) * 100).toFixed(1)}%` : "";
  const c = document.createElement("div");
  c.className = "card"; c.style.marginBottom = "16px";
  c.innerHTML = `<h2>Benchmark: ${b.config.requests.toLocaleString()} requests × ${b.config.seeds} seeds (means)</h2>
  <div class="table-wrap"><table><thead><tr><th>Strategy</th><th class="num">Cost / 1k</th><th class="num">Δ cost</th>
  <th class="num">p95</th><th class="num">Δ p95</th><th class="num">Accuracy</th><th class="num">Cache hits</th><th class="num">False hits</th></tr></thead><tbody>
  ${names.map(n => { const a = b.strategies[n]; return `<tr><td>${esc(n)}</td>
    <td class="num">${usd(m(a, "cost_per_1k_requests_usd"), 3)}</td><td class="num">${d(a, "cost_per_1k_requests_usd")}</td>
    <td class="num">${ms(m(a, "p95_latency_ms"))}</td><td class="num">${d(a, "p95_latency_ms")}</td>
    <td class="num">${pct(m(a, "accuracy"))}</td><td class="num">${pct(m(a, "cache_hit_rate"))}</td>
    <td class="num">${a.false_hit_rate ? pct(a.false_hit_rate.mean) : "-"}</td></tr>`; }).join("")}
  </tbody></table></div>`;
  return c;
}

const state = { run: DATA.runs.length ? DATA.runs[DATA.runs.length - 1].name : null };
render();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
</script>
</body>
</html>
"""
