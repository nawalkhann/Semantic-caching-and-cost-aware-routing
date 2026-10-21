"""Build the static project website into site/ (deployed by Vercel from that folder).

    python scripts/build_site.py

Benchmark numbers are read from benchmarks/results/results.json, so re-running the
benchmark and then this script keeps the site in sync with the repo.
"""

from __future__ import annotations

import html
import json
import re
import shutil
from datetime import date
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
BASE_URL = "https://semantic-cache-router.vercel.app"
REPO_URL = "https://github.com/nawalkhann/semantic-caching-and-cost-aware-routing"
AUTHOR = "Nawal"
SITE_NAME = "Semantic Cache Router"
TODAY = date.today().isoformat()

R = json.loads((ROOT / "benchmarks/results/results.json").read_text())
S = R["strategies"]


def m(strategy: str, metric: str) -> float:
    return S[strategy][metric]["mean"]


def pct_change(strategy: str, metric: str) -> float:
    return (m(strategy, metric) / m("baseline", metric) - 1) * 100


NUM = {
    "cost_drop": f"{-pct_change('cache+router', 'cost_per_1k_requests_usd'):.0f}%",
    "p95_drop": f"{-pct_change('cache+router', 'p95_latency_ms'):.0f}%",
    "base_cost": f"${m('baseline', 'cost_per_1k_requests_usd'):.2f}",
    "best_cost": f"${m('cache+router', 'cost_per_1k_requests_usd'):.2f}",
    "base_p95": f"{m('baseline', 'p95_latency_ms') / 1000:.1f} s",
    "best_p95": f"{m('cache+router', 'p95_latency_ms') / 1000:.1f} s",
    "base_acc": f"{m('baseline', 'accuracy') * 100:.1f}%",
    "best_acc": f"{m('cache+router', 'accuracy') * 100:.1f}%",
    "acc_std": f"{S['cache+router']['accuracy']['std'] * 100:.1f}",
    "naive_false": f"{m('semantic-unguarded', 'false_hit_rate') * 100:.0f}%",
    "naive_acc_drop": f"{(m('baseline', 'accuracy') - m('semantic-unguarded', 'accuracy')) * 100:.1f}",
    "guarded_false": f"{m('semantic-cache', 'false_hit_rate') * 100:.1f}%",
    "hit_rate": f"{m('semantic-cache', 'cache_hit_rate') * 100:.0f}%",
    "exact_hit": f"{m('exact-cache', 'cache_hit_rate') * 100:.0f}%",
    "router_acc": f"{m('router', 'router_accuracy') * 100:.0f}%",
    "router_cost_drop": f"{-pct_change('router', 'cost_per_1k_requests_usd'):.0f}%",
    "router_p95_drop": f"{-pct_change('router', 'p95_latency_ms'):.0f}%",
    "requests": f"{R['config']['requests']:,}",
    "seeds": str(R["config"]["seeds"]),
}

LABELS = {
    "baseline": "Always large model",
    "exact-cache": "Exact-match cache",
    "semantic-unguarded": "Semantic cache, similarity only",
    "semantic-cache": "Semantic cache, guarded",
    "router": "Router only",
    "cache+router": "Guarded cache + router",
}

CSS = """
:root{color-scheme:light;--bg:#f7f7f5;--surface:#fff;--border:#e4e3de;--text:#141413;
--muted:#5c5b56;--accent:#2a63c4;--accent-ink:#fff;--good:#0a7d33;--bad:#b42318;--code:#f0efea}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#121211;
--surface:#1b1b1a;--border:#2f2f2c;--text:#f1f0ec;--muted:#b4b3ab;--accent:#6ea2ff;--accent-ink:#0b1426;
--good:#4cc27a;--bad:#ff7a6e;--code:#252523}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#121211;--surface:#1b1b1a;--border:#2f2f2c;--text:#f1f0ec;
--muted:#b4b3ab;--accent:#6ea2ff;--accent-ink:#0b1426;--good:#4cc27a;--bad:#ff7a6e;--code:#252523}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--text);font:17px/1.65 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
a{color:var(--accent)}img{max-width:100%;height:auto;display:block}
.wrap{max-width:960px;margin:0 auto;padding:0 16px}
header.site{border-bottom:1px solid var(--border);background:var(--surface)}
header.site .wrap{display:flex;align-items:center;justify-content:space-between;gap:12px;min-height:60px;flex-wrap:wrap}
.brand{font-weight:700;text-decoration:none;color:var(--text);display:flex;align-items:center;gap:8px}
nav a{color:var(--muted);text-decoration:none;margin-left:16px;font-size:15px}
nav a:hover,nav a[aria-current]{color:var(--text)}
main{padding:40px 0 64px}
h1{font-size:clamp(30px,5vw,44px);line-height:1.15;margin:0 0 16px;letter-spacing:-.01em}
h2{font-size:26px;line-height:1.25;margin:48px 0 12px}h3{font-size:20px;margin:28px 0 8px}
.lede{font-size:20px;color:var(--muted);max-width:720px;margin:0 0 28px}
.cta{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:36px}
.btn{display:inline-block;padding:11px 18px;border-radius:10px;text-decoration:none;font-weight:600;border:1px solid var(--border);color:var(--text);background:var(--surface)}
.btn.primary{background:var(--accent);color:var(--accent-ink);border-color:var(--accent)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:8px 0 12px}
.stat{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:18px}
.stat b{display:block;font-size:34px;line-height:1.1;font-variant-numeric:tabular-nums}
.stat span{color:var(--muted);font-size:15px}
.note{color:var(--muted);font-size:15px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:20px 22px;margin:16px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
.table-wrap{overflow-x:auto;border:1px solid var(--border);border-radius:12px;background:var(--surface);margin:16px 0}
table{border-collapse:collapse;width:100%;font-size:15px;font-variant-numeric:tabular-nums}
th,td{padding:9px 10px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:600;white-space:normal;vertical-align:bottom}
table.text td,table.text th{white-space:normal;text-align:left;min-width:110px}
tr:last-child td{border-bottom:0}tr.hl td{font-weight:700}
.good{color:var(--good)}.bad{color:var(--bad)}
pre,code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:14px}
code{background:var(--code);padding:1px 5px;border-radius:5px}
pre{background:var(--code);padding:14px 16px;border-radius:10px;overflow-x:auto;line-height:1.5}
pre code{background:none;padding:0}
.shot{border:1px solid var(--border);border-radius:14px;overflow:hidden;margin:20px 0}
article{max-width:740px}article p,article li{max-width:70ch}
.meta{color:var(--muted);font-size:15px;margin-bottom:28px}
.crumbs{font-size:14px;color:var(--muted);margin-bottom:10px}.crumbs a{color:var(--muted)}
footer.site{border-top:1px solid var(--border);padding:28px 0;color:var(--muted);font-size:14px}
footer.site .wrap{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}
footer.site a{color:var(--muted)}
@media (max-width:560px){nav a{margin-left:0;margin-right:14px}header.site .wrap{padding-top:8px;padding-bottom:8px}}
""".strip()

FAVICON = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" '
           'rx="8" fill="#2a63c4"/><path d="M8 20l6-8 4 5 6-9" fill="none" stroke="#fff" '
           'stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>')

NAV = [("/", "Home"), ("/benchmark", "Benchmark"), ("/dashboard", "Dashboard"),
       ("/articles", "Articles"), (REPO_URL, "GitHub")]


def page(path: str, title: str, description: str, keywords: list[str], body: str,
         og_type: str = "website", jsonld: dict | list | None = None, crumbs: list | None = None,
         published: str | None = None) -> str:
    url = BASE_URL + path
    nav = "".join(
        f'<a href="{href}"{" aria-current=page" if href == path else ""}>{label}</a>'
        for href, label in NAV)
    ld = [{
        "@context": "https://schema.org", "@type": "WebSite", "name": SITE_NAME,
        "url": BASE_URL + "/",
    }] if path == "/" else []
    if crumbs:
        ld.append({"@context": "https://schema.org", "@type": "BreadcrumbList",
                   "itemListElement": [{"@type": "ListItem", "position": i + 1, "name": n,
                                        "item": BASE_URL + p} for i, (p, n) in enumerate(crumbs)]})
    if jsonld:
        ld += jsonld if isinstance(jsonld, list) else [jsonld]
    ld_html = "".join(f'<script type="application/ld+json">{json.dumps(x)}</script>' for x in ld)
    crumb_html = ""
    if crumbs:
        crumb_html = '<div class="crumbs">' + " / ".join(
            f'<a href="{p}">{html.escape(n)}</a>' for p, n in crumbs[:-1]) + "</div>"
    esc = html.escape
    pub = (f'<meta property="article:published_time" content="{published}">'
           f'<meta property="article:author" content="{AUTHOR}">') if published else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="{esc(description)}">
<meta name="keywords" content="{esc(', '.join(keywords))}">
<meta name="author" content="{AUTHOR}">
<link rel="canonical" href="{url}">
<meta name="robots" content="index, follow, max-image-preview:large">
<meta property="og:type" content="{og_type}">
<meta property="og:site_name" content="{SITE_NAME}">
<meta property="og:title" content="{esc(title)}">
<meta property="og:description" content="{esc(description)}">
<meta property="og:url" content="{url}">
<meta property="og:image" content="{BASE_URL}/og.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{esc(title)}">
<meta name="twitter:description" content="{esc(description)}">
<meta name="twitter:image" content="{BASE_URL}/og.png">
{pub}
<meta name="theme-color" content="#2a63c4">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="/styles.css">
{ld_html}
</head>
<body>
<header class="site"><div class="wrap">
<a class="brand" href="/"><img src="/favicon.svg" alt="" width="24" height="24">{SITE_NAME}</a>
<nav aria-label="Main">{nav}</nav>
</div></header>
<main><div class="wrap">
{crumb_html}
{body}
</div></main>
<footer class="site"><div class="wrap">
<span>Built by {AUTHOR}. MIT licensed. Code on <a href="{REPO_URL}">GitHub</a>.</span>
<span><a href="/benchmark">Benchmark</a> · <a href="/articles">Articles</a> · <a href="/sitemap.xml">Sitemap</a></span>
</div></footer>
</body>
</html>
"""


def results_table(rows: list[str], highlight: str = "cache+router") -> str:
    head = ("<tr><th>Strategy</th><th>Cost / 1k requests</th><th>vs baseline</th>"
            "<th>p95 latency</th><th>vs baseline</th><th>Accuracy</th><th>Cache hits</th>"
            "<th>False hits</th></tr>")
    out = []
    for name in rows:
        a = S[name]
        dc, dp = pct_change(name, "cost_per_1k_requests_usd"), pct_change(name, "p95_latency_ms")
        acc = a["accuracy"]["mean"] * 100
        fh = a.get("false_hit_rate")
        fh_txt = "–" if fh is None else f"{fh['mean'] * 100:.1f}%"
        fh_cls = ' class="bad"' if fh and fh["mean"] > 0.05 else ""
        acc_cls = ' class="bad"' if acc < m("baseline", "accuracy") * 100 - 5 else ""
        out.append(
            f'<tr{" class=hl" if name == highlight else ""}><td>{LABELS[name]}</td>'
            f'<td>${a["cost_per_1k_requests_usd"]["mean"]:.2f}</td>'
            f'<td>{"–" if name == "baseline" else f"{dc:+.0f}%"}</td>'
            f'<td>{a["p95_latency_ms"]["mean"] / 1000:.2f} s</td>'
            f'<td>{"–" if name == "baseline" else f"{dp:+.0f}%"}</td>'
            f'<td{acc_cls}>{acc:.1f}% ± {a["accuracy"]["std"] * 100:.1f}</td>'
            f'<td>{a["cache_hit_rate"]["mean"] * 100:.0f}%</td><td{fh_cls}>{fh_txt}</td></tr>')
    return f'<div class="table-wrap"><table>{head}{"".join(out)}</table></div>'


def sweep_table(key: str) -> str:
    rows = []
    for row in R[key]:
        a = row["metrics"]
        rows.append(
            f"<tr><td>{row['threshold']}</td><td>${a['cost_per_1k_requests_usd']['mean']:.2f}</td>"
            f"<td>{a['p95_latency_ms']['mean'] / 1000:.2f} s</td>"
            f"<td>{a['accuracy']['mean'] * 100:.1f}%</td>"
            f"<td>{a['cache_hit_rate']['mean'] * 100:.0f}%</td>"
            f"<td>{a['false_hit_rate']['mean'] * 100:.1f}%</td>"
            f"<td>{a['routed_to_large_share']['mean'] * 100:.0f}%</td></tr>")
    head = ("<tr><th>Threshold</th><th>Cost / 1k</th><th>p95</th><th>Accuracy</th>"
            "<th>Cache hits</th><th>False hits</th><th>Misses to large model</th></tr>")
    return f'<div class="table-wrap"><table>{head}{"".join(rows)}</table></div>'


# ------------------------------------------------------------------------- pages

ARTICLES = [
    {
        "slug": "semantic-caching-false-hits",
        "title": "Semantic Caching for LLMs: Avoiding Wrong Cached Answers",
        "short": "Why naive semantic caches return wrong answers",
        "description": (f"A similarity-only semantic cache served wrong answers on {NUM['naive_false']} of hits. "
                        f"Four checks cut false hits to {NUM['guarded_false']} and keep most of the savings."),
        "keywords": ["semantic caching", "LLM cache", "semantic cache false positives",
                     "embedding similarity cache", "RAG caching", "GPTCache alternative",
                     "LLM cost reduction"],
    },
    {
        "slug": "cost-aware-llm-routing",
        "title": "Cost-Aware LLM Routing: Send Easy Questions to Small Models",
        "short": "Cost-aware LLM routing with a small and a large model",
        "description": (f"A cheap complexity router sends easy questions to a small LLM and hard ones to a large "
                        f"one, cutting cost {NUM['router_cost_drop']} with {NUM['router_acc']} routing accuracy."),
        "keywords": ["LLM routing", "model routing", "cost-aware routing", "LLM router",
                     "small vs large model", "reduce LLM cost", "RAG cost optimization"],
    },
]


def home() -> str:
    body = f"""
<h1>Semantic caching and cost-aware model routing for RAG</h1>
<p class="lede">An open-source Python layer that sits in front of the LLMs in a
retrieval-augmented generation (RAG) service. It answers repeat questions from a semantic cache,
sends easy questions to a small model and hard ones to a large model, and tracks cost and
latency for every request.</p>
<div class="cta"><a class="btn primary" href="/benchmark">See the benchmark</a>
<a class="btn" href="/dashboard">Open the dashboard</a>
<a class="btn" href="{REPO_URL}">View on GitHub</a></div>

<div class="stats">
<div class="stat"><b>−{NUM['cost_drop']}</b><span>LLM cost per request ({NUM['base_cost']} → {NUM['best_cost']} per 1k)</span></div>
<div class="stat"><b>−{NUM['p95_drop']}</b><span>p95 latency ({NUM['base_p95']} → {NUM['best_p95']})</span></div>
<div class="stat"><b>{NUM['best_acc']}</b><span>accuracy vs {NUM['base_acc']} always using the large model (±{NUM['acc_std']} pts)</span></div>
</div>
<p class="note">{NUM['requests']} requests × {NUM['seeds']} seeds on a product-docs workload, with simulated
model tiers. See <a href="/benchmark#method">how the benchmark works</a>.</p>

<a href="/dashboard" class="shot"><img src="/dashboard-preview.webp" width="1200" height="1250"
alt="Dashboard showing cumulative LLM cost, rolling p95 latency, and the share of requests answered by the cache, small model and large model"
loading="lazy" decoding="async"></a>

<h2>Why RAG products get expensive</h2>
<p>Every question to a RAG service usually costs one large-model call, priced on the retrieved
context plus the answer. Real traffic is repetitive and mostly simple, so much of that spend goes
to questions that were already answered, or that a model costing a twentieth as much would
answer just as well. This project removes both kinds of waste and measures exactly what that
does to quality.</p>

<div class="grid">
<div class="card"><h3>Guarded semantic cache</h3><p>It reuses an answer when a new question means
the same thing. Embedding similarity alone is not enough: "storage on Pro" and "storage on
Business" look almost identical. So a hit also needs matching entities and numbers, overlapping
topic words, and the same retrieved documents.</p></div>
<div class="card"><h3>Cost-aware router</h3><p>A logistic complexity score sends simple
lookups to a small model. Comparisons, constraint problems and arithmetic go to a large model.
It uses only signals that are free before any LLM call: the query text, its entities and the
retrieval scores.</p></div>
<div class="card"><h3>Per-request accounting</h3><p>Every request logs its route, tokens,
cost and a latency breakdown (embed, retrieve, cache, LLM) to SQLite. A dashboard shows
cumulative cost, rolling p95 latency and the route mix.</p></div>
</div>

<h2>Results at a glance</h2>
{results_table(["baseline", "exact-cache", "semantic-unguarded", "semantic-cache", "router", "cache+router"])}
<p>The similarity-only cache looks cheapest, but it served the wrong answer on
{NUM['naive_false']} of hits and lost {NUM['naive_acc_drop']} accuracy points. The guards bring
false hits down to {NUM['guarded_false']}. The full analysis and threshold sweeps are on the
<a href="/benchmark">benchmark page</a>.</p>

<h2>How a request flows</h2>
<pre><code>query → embed → retrieve top-k → cache lookup ── hit ──→ cached answer (~1 ms, ~$0)
                                     │ miss
                                     ▼
                         router: complexity score
                          │ low              │ high
                          ▼                  ▼
                      small model        large model → answer → add to cache
every request → SQLite log → dashboard</code></pre>

<h2>Try it</h2>
<pre><code>git clone {REPO_URL}
cd semantic-caching-and-cost-aware-routing
pip install -e ".[dev]"
pytest
python benchmarks/run_benchmark.py
uvicorn ragroute.api:create_app --factory   # then open /dashboard</code></pre>
<p>To use real models, point it at any OpenAI-compatible endpoint with environment variables.
The <a href="{REPO_URL}#running-against-real-models">README</a> has the details.</p>

<h2>Articles</h2>
<ul>{''.join(f'<li><a href="/articles/{a["slug"]}">{html.escape(a["title"])}</a></li>' for a in ARTICLES)}</ul>
"""
    ld = {"@context": "https://schema.org", "@type": "SoftwareSourceCode",
          "name": SITE_NAME, "codeRepository": REPO_URL, "programmingLanguage": "Python",
          "license": "https://opensource.org/licenses/MIT", "author": {"@type": "Person", "name": AUTHOR},
          "description": "Semantic caching and cost-aware model routing for RAG services."}
    return page("/", "Semantic Caching & Cost-Aware LLM Routing for RAG",
                f"Open-source semantic cache and LLM router for RAG. It cuts LLM cost {NUM['cost_drop']} "
                f"and p95 latency {NUM['p95_drop']} at the same accuracy, with a per-request cost dashboard.",
                ["semantic caching", "LLM routing", "RAG", "LLM cost optimization",
                 "p95 latency", "model routing", "semantic cache", "Python"], body, jsonld=ld)


def benchmark() -> str:
    cfg = R["config"]["models"]
    body = f"""
<h1>Benchmark: semantic caching and model routing</h1>
<p class="lede">Six strategies replay the same {NUM['requests']}-request stream, repeated over
{NUM['seeds']} seeds. Results are mean ± standard deviation across seeds.</p>

{results_table(["baseline", "exact-cache", "semantic-unguarded", "semantic-cache", "router", "cache+router"])}

<h2>What the numbers say</h2>
<ul>
<li><strong>Naive semantic caching hurts quality.</strong> With similarity only,
{NUM['naive_false']} of cache hits returned another question's answer, and accuracy fell
{NUM['naive_acc_drop']} points. Look-alike questions such as "Pro vs Business on storage" and "Pro
vs Business on version history" embed very closely, especially when users add greetings and
sign-offs.</li>
<li><strong>Guards fix it cheaply.</strong> Requiring matching entities and numbers, overlapping topic
words and the same retrieved documents cuts false hits to {NUM['guarded_false']}. The hit rate stays
at {NUM['hit_rate']}, compared with {NUM['exact_hit']} for an exact-match cache.</li>
<li><strong>Routing alone cuts cost but not tail latency.</strong> The router saves
{NUM['router_cost_drop']} of cost, but p95 moves only {NUM['router_p95_drop']}%. The slowest
requests are the hard ones, and those correctly still go to the large model.</li>
<li><strong>Combined</strong>, cost per request falls {NUM['cost_drop']} and p95 latency
{NUM['p95_drop']}. Accuracy is {NUM['best_acc']} vs {NUM['base_acc']}, inside the seed-to-seed
noise.</li>
</ul>

<h2>Cache similarity threshold</h2>
<p>Lower thresholds give more hits and lower cost. Because the guards catch look-alikes, false
hits stay under 0.5% even at 0.6. Accuracy still becomes noisier at low thresholds, because a
wrong answer that gets cached is then reused more often.</p>
{sweep_table("cache_threshold_sweep")}

<h2>Router threshold</h2>
<p>The router threshold is the cost/quality dial. Above 0.5, hard questions start going to the
small model and accuracy drops.</p>
{sweep_table("route_threshold_sweep")}

<h2 id="method">How the benchmark works</h2>
<p><strong>Real:</strong> retrieval, embeddings, the cache and its guards, the router, token
counts for every prompt, local compute latency, and grading against reference answers.</p>
<p><strong>Simulated:</strong> the LLM calls. Each tier has an explicit price, latency profile and
accuracy per difficulty. The simulated model answers correctly with that probability and
otherwise gives a plausible wrong answer.</p>
<div class="table-wrap"><table class="text">
<tr><th></th><th>Small model</th><th>Large model</th></tr>
<tr><td>Price, USD per 1M tokens (in / out)</td><td>{cfg['small']['input_usd_per_mtok']:.2f} / {cfg['small']['output_usd_per_mtok']:.2f}</td><td>{cfg['large']['input_usd_per_mtok']:.2f} / {cfg['large']['output_usd_per_mtok']:.2f}</td></tr>
<tr><td>Accuracy, easy / hard questions</td><td>{cfg['small']['accuracy']['easy']:.0%} / {cfg['small']['accuracy']['hard']:.0%}</td><td>{cfg['large']['accuracy']['easy']:.0%} / {cfg['large']['accuracy']['hard']:.0%}</td></tr>
<tr><td>Time to first token / per output token</td><td>{cfg['small']['latency']['ttft_s']:.2f} s / {cfg['small']['latency']['s_per_output_token'] * 1000:.0f} ms</td><td>{cfg['large']['latency']['ttft_s']:.2f} s / {cfg['large']['latency']['s_per_output_token'] * 1000:.0f} ms</td></tr>
</table></div>
<p>The workload is a fictional SaaS product's documentation: {R['config']['documents']} documents
and {R['config']['intents']} question intents, each with 3–5 paraphrases. Requests are drawn with
Zipf popularity, and users add greetings, sign-offs, casing changes and word swaps. Hit rates
depend on how repetitive traffic is, so expect lower hit rates on long-tail traffic.
Accuracy differences under about 1 point are within noise.</p>
<p>Reproduce it with <code>python benchmarks/run_benchmark.py</code>. The raw results are in
<a href="{REPO_URL}/blob/main/benchmarks/results/results.json">results.json</a>.</p>
"""
    ld = {"@context": "https://schema.org", "@type": "Dataset",
          "name": "Semantic caching and LLM routing benchmark results",
          "description": "Cost, p95 latency and accuracy for six caching and routing strategies in a RAG service.",
          "url": BASE_URL + "/benchmark", "creator": {"@type": "Person", "name": AUTHOR},
          "license": "https://opensource.org/licenses/MIT"}
    return page("/benchmark", "LLM Cache & Routing Benchmark: Cost, p95, Accuracy",
                f"Six RAG caching and routing strategies compared. A guarded cache plus router cut LLM "
                f"cost {NUM['cost_drop']} and p95 latency {NUM['p95_drop']}; a naive cache had "
                f"{NUM['naive_false']} false hits.",
                ["LLM benchmark", "semantic cache benchmark", "LLM routing benchmark",
                 "p95 latency", "LLM cost per request", "RAG evaluation"], body, jsonld=ld,
                crumbs=[("/", "Home"), ("/benchmark", "Benchmark")])


def article_caching(a: dict) -> str:
    return f"""
<article>
<h1>{html.escape(a['title'])}</h1>
<p class="meta">By {AUTHOR} · {TODAY} · 7 min read</p>

<p>A semantic cache stores LLM answers keyed by the <em>embedding</em> of the question. When a new
question's embedding is close enough to a stored one, the stored answer is returned and no
model call is made. For a RAG product with repetitive traffic, this is the single largest cost
lever. In our benchmark the hit rate was {NUM['hit_rate']}, against {NUM['exact_hit']} for an
exact-match cache.</p>

<p>It is also the easiest way to quietly ship wrong answers. A cache that checks similarity
alone served another question's answer on <strong>{NUM['naive_false']} of its hits</strong>, and
overall accuracy dropped {NUM['naive_acc_drop']} points. This article explains why that happens
and the four checks that brought false hits down to {NUM['guarded_false']}.</p>

<h2>Why similarity alone is not enough</h2>
<p>Embeddings are built to put sentences with similar <em>wording and topic</em> close together.
Caching needs something stricter: questions that have <em>the same answer</em>. Those diverge in
three common ways.</p>
<ol>
<li><strong>Entity swaps.</strong> "How much storage does the Pro plan include?" and "How much
storage does the Business plan include?" differ by a single token. Their cosine similarity is
very high, but the answers are 1,000 GB and 3,000 GB.</li>
<li><strong>Number swaps.</strong> "What would 12 seats cost per year?" and "What would 30 seats
cost per year?" also differ by one token, and the answers differ.</li>
<li><strong>Filler inflation.</strong> Real users write "Hey team, quick question: … Thanks in
advance!" When two different short questions share the same greeting and sign-off, the shared
filler can push their similarity above any sensible threshold.</li>
</ol>
<p>Raising the threshold helps, but it gives up hits on real paraphrases and never fully
removes entity swaps, because those are often the <em>most</em> similar pairs of all.</p>

<h2>Four checks for a cache hit</h2>
<div class="table-wrap"><table class="text">
<tr><th>Check</th><th>What it compares</th><th>Catches</th></tr>
<tr><td>1. Similarity</td><td>cosine(query embeddings) ≥ threshold</td><td>unrelated questions</td></tr>
<tr><td>2. Key terms</td><td>numbers and entity names must match exactly</td><td>entity and number swaps</td></tr>
<tr><td>3. Topic overlap</td><td>Jaccard over query words found in the knowledge base</td><td>filler inflation, attribute swaps</td></tr>
<tr><td>4. Context overlap</td><td>Jaccard over retrieved document ids</td><td>answers grounded in different docs</td></tr>
</table></div>

<h3>Key terms, learned from your corpus</h3>
<p>You don't need a hand-maintained entity list. Scan the knowledge base for words that appear
capitalised mid-sentence (product names, plan names, feature names). Drop the ones that
appear in most documents, such as the product's own name, because they don't tell answers apart.
Add every number in the query. Two questions may share a cached answer only if these sets are
identical.</p>

<h3>Topic overlap ignores small talk</h3>
<p>Take the query words that also occur in the knowledge base, minus stopwords. "Thanks", "team"
and "proposal" never appear in product docs, so they drop out. "Storage", "history" and "rate
limit" do appear, so they count. Requiring a Jaccard overlap of at least 0.6 rejects "Pro vs
Business on <em>storage</em>" against "… on <em>version history</em>", however much filler the two
share.</p>

<h3>Context overlap keeps answers grounded</h3>
<p>Retrieval already runs before the cache lookup, because the router needs its scores too.
Storing the retrieved document ids with each cache entry is free. If a new query retrieves
different documents, the cached answer was grounded in different evidence and should not be
reused. This check also invalidates naturally when documents change.</p>

<h2>Results</h2>
{results_table(["baseline", "exact-cache", "semantic-unguarded", "semantic-cache"], highlight="semantic-cache")}
<p>The guarded cache keeps most of the naive cache's savings: cost falls
{-pct_change('semantic-cache', 'cost_per_1k_requests_usd'):.0f}% instead of
{-pct_change('semantic-unguarded', 'cost_per_1k_requests_usd'):.0f}%, with quality at the baseline level.
The guards cost microseconds, since they are set comparisons on a handful of tokens.</p>

<h2>Practical advice</h2>
<ul>
<li><strong>Measure false hits, not just hit rate.</strong> A hit-rate dashboard rewards exactly the
failure described here. Sample cache hits and check whether the matched question has the same answer.</li>
<li><strong>Start the threshold around 0.8 and sweep it.</strong> With guards in place, you can lower it
for more hits. Our <a href="/benchmark">threshold sweep</a> stays under 0.5% false hits even at 0.6.</li>
<li><strong>Remember the cache freezes answers.</strong> If a wrong answer gets cached, it is served
until it expires. Use a TTL, and consider caching only answers from your strongest model.</li>
<li><strong>Use a real sentence-embedding model in production.</strong> It catches true paraphrases
that lexical embedders miss. That raises similarity for look-alikes too, which makes the guards
even more important.</li>
</ul>

<p>The implementation is about 200 lines of Python in
<a href="{REPO_URL}/blob/main/src/ragroute/cache.py"><code>cache.py</code></a>. Also see
<a href="/articles/cost-aware-llm-routing">cost-aware LLM routing</a>, the other half of the cost story.</p>
</article>
"""


def article_routing(a: dict) -> str:
    small = R["config"]["models"]["small"]
    large = R["config"]["models"]["large"]
    ratio = large["output_usd_per_mtok"] / small["output_usd_per_mtok"]
    return f"""
<article>
<h1>{html.escape(a['title'])}</h1>
<p class="meta">By {AUTHOR} · {TODAY} · 6 min read</p>

<p>Most questions a RAG product receives are lookups: "What is the API rate limit on the Pro
plan?" A small model reads the retrieved paragraph and answers correctly. The large model also
answers correctly, at about {ratio:.0f}× the price per token and several times the latency.
A router's job is to tell the two kinds of question apart <em>before</em> paying for either.</p>

<h2>What makes a question hard</h2>
<p>In a product-docs workload, hard questions share a few traits:</p>
<ul>
<li><strong>Several entities:</strong> "Pro vs Business storage?" needs two facts and a comparison.</li>
<li><strong>Constraints:</strong> "We are 40 people and need SSO. What is the cheapest plan?" needs filtering and ranking.</li>
<li><strong>Arithmetic:</strong> "What would 30 seats on Pro cost per year with annual billing?"</li>
<li><strong>Reasoning words:</strong> why, explain, recommend, trade-offs.</li>
</ul>
<p>Length is a poor signal on its own. "Pro vs Business storage?" is four words and hard. "Our
designers keep a lot of large files around, so before we sign up I'd like to confirm how many
gigabytes each seat gets on Pro" is long and easy.</p>

<h2>A router that costs nothing to run</h2>
<p>The router is a logistic regression over eight features that are free to compute once
retrieval has run:</p>
<pre><code>multi_entity   two or more entity names in the query
compare        vs, compare, more, cheapest, difference, ...
aggregate      total, annual, yearly, per year, ...
constraint     need, must, if, requires, ...
reasoning      why, explain, should, recommend, ...
numbers        count of numbers in the query (capped)
log_len        log of query length
multi_doc      retrieved docs scoring close to the top one</code></pre>
<p>The score is a probability that the question is hard. At or above a threshold (default 0.5),
the question goes to the large model; below it, to the small one. The default weights are set
by hand. With labelled traffic, <code>ComplexityRouter.fit()</code> re-fits them in milliseconds.
On our workload, the default router sends {NUM['router_acc']} of requests to the right tier.</p>

<h2>Results</h2>
{results_table(["baseline", "router", "cache+router"], highlight="router")}
<p>Routing alone cuts cost {NUM['router_cost_drop']} for a 1-point accuracy cost. It barely
moves p95 latency ({NUM['router_p95_drop']}%), and that is expected. The slowest requests are
long answers to hard questions, and a good router keeps sending those to the large model. Tail
latency only falls once the cache removes repeat hard questions entirely. That's why the two
techniques complement each other: together they cut cost {NUM['cost_drop']} and p95
{NUM['p95_drop']}.</p>

<h2>Choosing the threshold</h2>
<p>The threshold is a business decision, so expose it as configuration. In our
<a href="/benchmark">router sweep</a>, raising the threshold from 0.5 to 0.8 saves another
{(1 - R['route_threshold_sweep'][4]['metrics']['cost_per_1k_requests_usd']['mean'] / R['route_threshold_sweep'][2]['metrics']['cost_per_1k_requests_usd']['mean']) * 100:.0f}%
of cost but loses about
{(R['route_threshold_sweep'][2]['metrics']['accuracy']['mean'] - R['route_threshold_sweep'][4]['metrics']['accuracy']['mean']) * 100:.0f}
accuracy points, because hard questions start landing on the small model. Lowering it below 0.5
spends more for no measurable gain.</p>

<h2>When to use something heavier</h2>
<ul>
<li><strong>Cascades:</strong> try the small model first and escalate when it is unsure. This works
when you can trust a confidence signal, but you pay for two calls on every escalation.</li>
<li><strong>Learned routers:</strong> a small classifier on query embeddings. Worth it once you have
thousands of labelled examples and a wide variety of question types.</li>
<li><strong>Per-tenant thresholds:</strong> premium customers get a lower threshold, so more of their
traffic goes to the large model.</li>
</ul>
<p>Start with the cheap feature router. Log the complexity score for every request; this
project's dashboard does. That gives you the data to tune it and, later, to train something better.</p>

<p>Read the <a href="{REPO_URL}/blob/main/src/ragroute/router.py">router source</a> or the companion
article on <a href="/articles/semantic-caching-false-hits">semantic caching without false hits</a>.</p>
</article>
"""


def articles_index() -> str:
    items = "".join(
        f'<div class="card"><h2 style="margin-top:0;font-size:22px"><a href="/articles/{a["slug"]}">'
        f'{html.escape(a["title"])}</a></h2><p>{html.escape(a["description"])}</p></div>'
        for a in ARTICLES)
    return page("/articles", "LLM Caching & Routing Articles | Semantic Cache Router",
                "Practical write-ups on semantic caching, cost-aware LLM routing and cutting RAG cost "
                "and latency, backed by a reproducible benchmark.",
                ["LLM articles", "semantic caching", "LLM routing", "RAG cost"],
                f"<h1>Articles</h1><p class=lede>What we learned building and benchmarking a semantic "
                f"cache and model router for RAG.</p>{items}",
                crumbs=[("/", "Home"), ("/articles", "Articles")])


def article_page(a: dict, body: str) -> str:
    path = f"/articles/{a['slug']}"
    ld = {"@context": "https://schema.org", "@type": "TechArticle", "headline": a["title"],
          "description": a["description"], "author": {"@type": "Person", "name": AUTHOR},
          "datePublished": TODAY, "dateModified": TODAY, "image": BASE_URL + "/og.png",
          "mainEntityOfPage": BASE_URL + path, "keywords": ", ".join(a["keywords"])}
    return page(path, a["title"], a["description"], a["keywords"], body, og_type="article",
                jsonld=ld, published=TODAY,
                crumbs=[("/", "Home"), ("/articles", "Articles"), (path, a["short"])])


def dashboard_page() -> str:
    src = (ROOT / "benchmarks/results/dashboard.html").read_text()
    title = "LLM Cost & Latency Dashboard | Semantic Cache Router"
    desc = ("Interactive per-request dashboard: cumulative LLM cost, rolling p95 latency and route mix "
            "for six semantic caching and model routing strategies.")
    head = (f'<meta name="description" content="{html.escape(desc)}">'
            f'<meta name="keywords" content="LLM cost dashboard, p95 latency, semantic cache, LLM routing">'
            f'<link rel="canonical" href="{BASE_URL}/dashboard">'
            f'<meta property="og:title" content="{html.escape(title)}">'
            f'<meta property="og:description" content="{html.escape(desc)}">'
            f'<meta property="og:url" content="{BASE_URL}/dashboard">'
            f'<meta property="og:image" content="{BASE_URL}/og.png">'
            f'<meta name="twitter:card" content="summary_large_image">'
            f'<link rel="icon" href="/favicon.svg" type="image/svg+xml">')
    src = re.sub(r"<title>.*?</title>", f"<title>{html.escape(title)}</title>{head}", src, count=1)
    src = src.replace("<h1>Benchmark run (seed 0)</h1>",
                      "<h1>LLM cost and latency dashboard</h1>", 1)
    src = src.replace("Per-request cost and latency from the semantic cache and model router.",
                      "Per-request cost and latency for six caching and routing strategies "
                      "(benchmark run, seed 0). Pick a run to compare.", 1)
    src = src.replace('<main>\n  <h1>', '<main>\n  <p style="margin:0 0 12px"><a href="/" '
                      'style="color:inherit">← Semantic Cache Router</a></p>\n  <h1>', 1)
    return src


def not_found() -> str:
    html_ = page("/404", "Page not found | Semantic Cache Router",
                 "This page does not exist. Go back to the Semantic Cache Router home page.", [],
                 '<h1>Page not found</h1><p class="lede">That page does not exist.</p>'
                 '<p><a class="btn primary" href="/">Go to the home page</a></p>')
    return html_.replace('content="index, follow, max-image-preview:large"', 'content="noindex"')


def main() -> None:
    if SITE.exists():
        shutil.rmtree(SITE)
    (SITE / "articles").mkdir(parents=True)
    files = {
        "index.html": home(),
        "benchmark.html": benchmark(),
        "dashboard.html": dashboard_page(),
        "articles/index.html": articles_index(),
        "404.html": not_found(),
        "styles.css": CSS + "\n",
        "favicon.svg": FAVICON + "\n",
    }
    files[f"articles/{ARTICLES[0]['slug']}.html"] = article_page(ARTICLES[0], article_caching(ARTICLES[0]))
    files[f"articles/{ARTICLES[1]['slug']}.html"] = article_page(ARTICLES[1], article_routing(ARTICLES[1]))
    for name, content in files.items():
        (SITE / name).write_text(content)

    urls = ["/", "/benchmark", "/dashboard", "/articles"] + [f"/articles/{a['slug']}" for a in ARTICLES]
    (SITE / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "".join(f"  <url><loc>{BASE_URL}{u}</loc><lastmod>{TODAY}</lastmod></url>\n" for u in urls)
        + "</urlset>\n")
    (SITE / "robots.txt").write_text(f"User-agent: *\nAllow: /\n\nSitemap: {BASE_URL}/sitemap.xml\n")
    (SITE / "vercel.json").write_text(json.dumps({
        "cleanUrls": True,
        "trailingSlash": False,
        "headers": [
            {"source": "/(.*)\\.(css|svg|png|webp)", "headers": [
                {"key": "Cache-Control", "value": "public, max-age=604800, stale-while-revalidate=86400"}]},
            {"source": "/(.*)", "headers": [
                {"key": "X-Content-Type-Options", "value": "nosniff"},
                {"key": "Referrer-Policy", "value": "strict-origin-when-cross-origin"}]},
        ],
    }, indent=2) + "\n")

    shot = Image.open(ROOT / "docs/dashboard.png").convert("RGB")
    shot.save(SITE / "dashboard-preview.webp", "WEBP", quality=80, method=6)
    og = shot.crop((0, 0, 1200, 630))
    og.save(SITE / "og.png", optimize=True)
    print("wrote", sorted(str(p.relative_to(SITE)) for p in SITE.rglob("*") if p.is_file()))


if __name__ == "__main__":
    main()
