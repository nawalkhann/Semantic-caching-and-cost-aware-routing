"""HTTP API: answer questions through the cache and router, and serve the dashboard.

    uvicorn ragroute.api:create_app --factory
    curl -s localhost:8000/query -d '{"query": "How much storage does Pro include?"}' \
         -H 'content-type: application/json'
    open http://localhost:8000/dashboard

By default the service uses the simulated small/large models from presets.py.
Set RAGROUTE_BACKEND=openai-compatible with RAGROUTE_SMALL_MODEL,
RAGROUTE_LARGE_MODEL, RAGROUTE_BASE_URL and OPENAI_API_KEY to call real models
(prices are taken from presets.SMALL / presets.LARGE unless overridden with
RAGROUTE_{SMALL,LARGE}_PRICE="input,output" in USD per million tokens).
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .cache import SemanticCache
from .corpus import build_documents
from .dashboard import render_dashboard
from .embeddings import HashingEmbedder
from .llm import ModelSpec, OpenAICompatibleLLM
from .metrics import MetricsStore, summarize
from .presets import LARGE, SMALL, IntentOracle, simulated_models
from .router import ComplexityRouter
from .service import RAGService


def _spec(role: str, default: ModelSpec) -> ModelSpec:
    name = os.environ.get(f"RAGROUTE_{role}_MODEL", default.name)
    price = os.environ.get(f"RAGROUTE_{role}_PRICE")
    if price:
        inp, out = (float(x) for x in price.split(","))
        return ModelSpec(name, inp, out)
    return ModelSpec(name, default.input_usd_per_mtok, default.output_usd_per_mtok)


def build_service(db_path: str | None = None) -> RAGService:
    docs = build_documents()
    embedder = HashingEmbedder().fit(d.text for d in docs)
    if os.environ.get("RAGROUTE_BACKEND") == "openai-compatible":
        base_url = os.environ.get("RAGROUTE_BASE_URL", "https://api.openai.com/v1")
        small, large = _spec("SMALL", SMALL), _spec("LARGE", LARGE)
        models = {small.name: OpenAICompatibleLLM(small, base_url),
                  large.name: OpenAICompatibleLLM(large, base_url)}
    else:
        small, large = SMALL, LARGE
        models = simulated_models(IntentOracle())
    return RAGService(
        docs, embedder, models, default_model=large.name,
        cache=SemanticCache(embedder.dim,
                            threshold=float(os.environ.get("RAGROUTE_CACHE_THRESHOLD", 0.8))),
        router=ComplexityRouter(small.name, large.name,
                                threshold=float(os.environ.get("RAGROUTE_ROUTE_THRESHOLD", 0.5))),
        metrics=MetricsStore(db_path or os.environ.get("RAGROUTE_DB", "ragroute.sqlite")),
        run="live",
    )


class QueryIn(BaseModel):
    query: str


class QueryOut(BaseModel):
    answer: str
    route: str
    cache_hit: bool
    cost_usd: float
    latency_ms: float
    sources: list[str]
    similarity: float | None = None
    complexity: float | None = None


def create_app(service: RAGService | None = None) -> FastAPI:
    svc = service or build_service()
    api = FastAPI(title="ragroute", version="0.1.0")

    @api.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @api.post("/query", response_model=QueryOut)
    def query(body: QueryIn) -> QueryOut:
        ans, _ = svc.answer(body.query)
        return QueryOut(answer=ans.text, route=ans.route, cache_hit=ans.cache_hit,
                        cost_usd=ans.cost_usd, latency_ms=ans.latency_ms, sources=ans.sources,
                        similarity=ans.similarity, complexity=ans.complexity)

    @api.get("/api/stats")
    def stats(run: str = "live") -> dict:
        return summarize(svc.metrics.records(run))

    @api.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> str:
        return render_dashboard(svc.metrics)

    return api

