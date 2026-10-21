"""Benchmark: cost, latency and accuracy of caching and routing strategies.

Each strategy replays the same request stream against a fresh service and cache,
and the whole comparison is repeated over several seeds (each seed draws a new
request stream and new LLM samples). Latency = measured local work (embedding,
retrieval, cache lookup) + the simulated LLM call. See src/ragroute/presets.py
for the model assumptions.

    python benchmarks/run_benchmark.py --requests 3000 --seeds 5 --out benchmarks/results
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
from dataclasses import asdict, dataclass, replace

from ragroute.cache import ExactMatchCache, SemanticCache
from ragroute.corpus import build_documents, build_intents, build_workload
from ragroute.dashboard import render_dashboard
from ragroute.embeddings import HashingEmbedder
from ragroute.metrics import MetricsStore, summarize
from ragroute.presets import (LARGE, LARGE_ACCURACY, LARGE_LATENCY, OUTPUT_TOKENS, SMALL,
                              SMALL_ACCURACY, SMALL_LATENCY, IntentOracle, grade,
                              simulated_models)
from ragroute.router import ComplexityRouter
from ragroute.service import RAGService


@dataclass(frozen=True)
class Strategy:
    name: str
    description: str
    cache: str | None = None  # None | "exact" | "semantic-unguarded" | "semantic"
    route: bool = False
    cache_threshold: float = 0.80
    route_threshold: float = 0.5


STRATEGIES = [
    Strategy("baseline", "Always the large model, no cache"),
    Strategy("exact-cache", "Exact-match cache + large model", cache="exact"),
    Strategy("semantic-unguarded", "Semantic cache, similarity check only + large model",
             cache="semantic-unguarded"),
    Strategy("semantic-cache", "Semantic cache with key-term, topic and context guards "
             "+ large model", cache="semantic"),
    Strategy("router", "Router (small vs large model), no cache", route=True),
    Strategy("cache+router", "Guarded semantic cache + router", cache="semantic", route=True),
]
HEADLINE = "cache+router"
CACHE_THRESHOLDS = [0.6, 0.7, 0.8, 0.9, 0.95]
ROUTE_THRESHOLDS = [0.2, 0.35, 0.5, 0.65, 0.8]

# metrics reported as mean ± std over seeds
METRICS = ["cost_per_1k_requests_usd", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms",
           "accuracy", "cache_hit_rate", "false_hit_rate", "router_accuracy",
           "routed_to_large_share"]


def make_cache(kind: str | None, dim: int, threshold: float):
    if kind is None:
        return None
    if kind == "exact":
        return ExactMatchCache()
    guarded = kind == "semantic"
    return SemanticCache(dim, threshold=threshold, check_key_terms=guarded,
                         check_topic=guarded, check_context=guarded)


def run_strategy(s: Strategy, workload, docs, embedder, metrics: MetricsStore,
                 seed: int, run: str | None = None) -> dict:
    oracle = IntentOracle()
    for item in workload:
        oracle.register(item.query, item.intent)
    router = (ComplexityRouter(SMALL.name, LARGE.name, threshold=s.route_threshold)
              if s.route else None)
    run = run or s.name
    svc = RAGService(docs, embedder, simulated_models(oracle, seed=seed), LARGE.name,
                     cache=make_cache(s.cache, embedder.dim, s.cache_threshold),
                     router=router, metrics=metrics, run=run)
    routed_large = routed_total = router_correct = 0
    false_hits = hits = 0
    for item in workload:
        ans, _ = svc.answer(item.query, grader=lambda a, i=item.intent: grade(a, i))
        if ans.cache_hit:
            hits += 1
            false_hits += oracle.intent(ans.matched_query).intent_id != item.intent.intent_id
        elif router:
            routed_total += 1
            to_large = ans.route == LARGE.name
            routed_large += to_large
            router_correct += to_large == (item.intent.difficulty == "hard")
    summary = summarize(metrics.records(run))
    summary["false_hit_rate"] = (false_hits / hits if hits else 0.0) if s.cache else None
    summary["router_accuracy"] = router_correct / routed_total if routed_total else None
    summary["routed_to_large_share"] = routed_large / routed_total if routed_total else None
    return summary


def aggregate(runs: list[dict]) -> dict:
    out = {}
    for m in METRICS:
        vals = [r[m] for r in runs if r.get(m) is not None]
        if vals:
            out[m] = {"mean": statistics.fmean(vals),
                      "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}
    return out


def _fmt(agg: dict, m: str, kind: str) -> str:
    if m not in agg:
        return "-"
    mean, std = agg[m]["mean"], agg[m]["std"]
    if kind == "pct":
        return f"{mean * 100:.1f}% ± {std * 100:.1f}"
    if kind == "ms":
        return f"{mean:,.0f} ± {std:,.0f}"
    return f"${mean:.3f} ± {std:.3f}"


def _delta(agg: dict, base: dict, m: str) -> str:
    return f"{(agg[m]['mean'] / base[m]['mean'] - 1) * 100:+.1f}%"


def write_markdown(path: str, results: dict) -> None:
    strat = results["strategies"]
    base = strat["baseline"]
    lines = [
        "# Benchmark results\n",
        f"{results['config']['requests']} requests × {results['config']['seeds']} seeds, "
        "mean ± std over seeds. Deltas are relative to `baseline`.\n",
        "| Strategy | Cost / 1k req | Δ cost | p50 latency (ms) | p95 latency (ms) | Δ p95 "
        "| Accuracy | Δ acc (pts) | Cache hit rate | False-hit rate |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, agg in strat.items():
        dacc = (agg["accuracy"]["mean"] - base["accuracy"]["mean"]) * 100
        lines.append(
            f"| `{name}` | {_fmt(agg, 'cost_per_1k_requests_usd', 'usd')} "
            f"| {_delta(agg, base, 'cost_per_1k_requests_usd')} "
            f"| {_fmt(agg, 'p50_latency_ms', 'ms')} | {_fmt(agg, 'p95_latency_ms', 'ms')} "
            f"| {_delta(agg, base, 'p95_latency_ms')} | {_fmt(agg, 'accuracy', 'pct')} "
            f"| {dacc:+.1f} | {_fmt(agg, 'cache_hit_rate', 'pct')} "
            f"| {_fmt(agg, 'false_hit_rate', 'pct')} |")
    for title, key in [("Cache similarity threshold sweep (guarded cache + router)",
                        "cache_threshold_sweep"),
                       ("Router threshold sweep (guarded cache + router)",
                        "route_threshold_sweep")]:
        lines += ["", f"## {title}\n",
                  "| Threshold | Cost / 1k req | p95 latency (ms) | Accuracy | Cache hit rate "
                  "| False-hit rate | Misses sent to large model |",
                  "|---|---|---|---|---|---|---|"]
        for row in results[key]:
            a = row["metrics"]
            lines.append(
                f"| {row['threshold']} | {_fmt(a, 'cost_per_1k_requests_usd', 'usd')} "
                f"| {_fmt(a, 'p95_latency_ms', 'ms')} | {_fmt(a, 'accuracy', 'pct')} "
                f"| {_fmt(a, 'cache_hit_rate', 'pct')} | {_fmt(a, 'false_hit_rate', 'pct')} "
                f"| {_fmt(a, 'routed_to_large_share', 'pct')} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="benchmarks/results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    db = os.path.join(args.out, "requests.sqlite")
    if os.path.exists(db):
        os.remove(db)

    docs = build_documents()
    embedder = HashingEmbedder().fit(d.text for d in docs)
    seeds = list(range(args.seeds))
    workloads = {s: build_workload(args.requests, seed=s) for s in seeds}
    logged = MetricsStore(db)  # full per-request log for seed 0, shown on the dashboard

    def over_seeds(strategy: Strategy, log: bool = False) -> dict:
        runs = []
        for s in seeds:
            store = logged if (s == 0 and log) else MetricsStore()
            runs.append(run_strategy(strategy, workloads[s], docs, embedder, store, seed=s))
        return aggregate(runs)

    results: dict = {
        "config": {
            "requests": args.requests, "seeds": args.seeds, "embedder": "HashingEmbedder",
            "intents": len(build_intents()), "documents": len(docs),
            "models": {
                "small": {**asdict(SMALL), "accuracy": SMALL_ACCURACY,
                          "latency": asdict(SMALL_LATENCY)},
                "large": {**asdict(LARGE), "accuracy": LARGE_ACCURACY,
                          "latency": asdict(LARGE_LATENCY)},
                "output_tokens": OUTPUT_TOKENS,
            },
            "python": platform.python_version(),
        },
        "strategies": {},
    }
    for strategy in STRATEGIES:
        results["strategies"][strategy.name] = over_seeds(strategy, log=True)
        print(f"done {strategy.name}", flush=True)

    head = next(s for s in STRATEGIES if s.name == HEADLINE)
    results["cache_threshold_sweep"] = [
        {"threshold": t, "metrics": over_seeds(replace(head, cache_threshold=t))}
        for t in CACHE_THRESHOLDS]
    results["route_threshold_sweep"] = [
        {"threshold": t, "metrics": over_seeds(replace(head, route_threshold=t))}
        for t in ROUTE_THRESHOLDS]

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    write_markdown(os.path.join(args.out, "results.md"), results)
    with open(os.path.join(args.out, "dashboard.html"), "w") as f:
        f.write(render_dashboard(logged, title="Benchmark run (seed 0)", benchmark=results))
    print(open(os.path.join(args.out, "results.md")).read())


if __name__ == "__main__":
    main()
