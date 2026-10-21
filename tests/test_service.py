from ragroute.cache import SemanticCache
from ragroute.corpus import build_intents, build_workload
from ragroute.metrics import MetricsStore, summarize
from ragroute.presets import LARGE, SMALL, IntentOracle, grade, simulated_models
from ragroute.router import ComplexityRouter
from ragroute.service import RAGService


def _service(docs, embedder, cache=True, router=True):
    return RAGService(
        docs, embedder, simulated_models(IntentOracle(), seed=1), LARGE.name,
        cache=SemanticCache(embedder.dim) if cache else None,
        router=ComplexityRouter(SMALL.name, LARGE.name) if router else None,
        metrics=MetricsStore())


def test_second_ask_is_served_from_cache(docs, embedder):
    svc = _service(docs, embedder)
    q = "How much storage does the Pro plan include?"
    first, _ = svc.answer(q)
    second, rec = svc.answer(q + " Thanks!")
    assert not first.cache_hit and second.cache_hit
    assert second.text == first.text
    assert second.cost_usd < first.cost_usd / 100
    assert rec.route == "cache" and rec.llm_ms == 0


def test_retrieval_finds_the_plan_doc(docs, embedder):
    svc = _service(docs, embedder)
    ans, _ = svc.answer("What is the API rate limit on the Business plan?")
    assert "plan-business" in ans.sources


def test_cost_is_priced_from_tokens(docs, embedder):
    svc = _service(docs, embedder, cache=False, router=False)
    _, rec = svc.answer("Does the Pro plan include SSO?")
    expected = LARGE.cost(rec.input_tokens, rec.output_tokens)
    assert abs(rec.cost_usd - expected) < 1e-6  # plus a negligible embedding cost


def test_metrics_log_and_summary(docs, embedder):
    svc = _service(docs, embedder)
    intents = {i.intent_id: i for i in build_intents()}
    for item in build_workload(200, seed=3):
        svc.answer(item.query, grader=lambda a, i=item.intent: grade(a, i))
    recs = svc.metrics.records("default")
    assert len(recs) == 200
    s = summarize(recs)
    assert s["requests"] == 200
    assert 0 < s["cache_hit_rate"] < 1
    assert s["p95_latency_ms"] >= s["p50_latency_ms"]
    assert set(s["by_route"]) <= {"cache", "small", "large"}
    assert intents  # workload intents come from the corpus


def test_api_end_to_end(tmp_path):
    from fastapi.testclient import TestClient

    from ragroute.api import build_service, create_app

    client = TestClient(create_app(build_service(str(tmp_path / "m.sqlite"))))
    q = {"query": "How much storage does the Pro plan include?"}
    r1 = client.post("/query", json=q).json()
    r2 = client.post("/query", json=q).json()
    assert r1["cache_hit"] is False and r2["cache_hit"] is True
    assert r1["route"] == "small"
    assert client.get("/api/stats").json()["requests"] == 2
    page = client.get("/dashboard")
    assert page.status_code == 200 and "Cumulative cost" in page.text
