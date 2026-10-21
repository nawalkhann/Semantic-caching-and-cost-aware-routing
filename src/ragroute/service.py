"""RAG service that puts the semantic cache and the router in front of the LLMs."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from .cache import ExactMatchCache, KeyTermExtractor, SemanticCache
from .corpus import Document
from .embeddings import Embedder, estimate_embed_tokens
from .llm import LLMClient
from .metrics import MetricsStore, RequestRecord
from .retriever import RetrievedDoc, Retriever
from .router import ComplexityRouter

SYSTEM_PROMPT = (
    "You answer questions about the Nimbus product using only the provided context. "
    "Be concise. Show the numbers you used for any calculation."
)


def build_prompt(query: str, docs: list[RetrievedDoc]) -> str:
    context = "\n\n".join(f"[{r.doc.doc_id}] {r.doc.title}\n{r.doc.text}" for r in docs)
    return f"Context:\n{context}\n\nQuestion: {query}"


def question_from_prompt(prompt: str) -> str:
    return prompt.rsplit("\nQuestion: ", 1)[-1]


@dataclass
class Answer:
    text: str
    route: str
    cache_hit: bool
    cost_usd: float
    latency_ms: float
    sources: list[str]
    similarity: float | None = None
    complexity: float | None = None
    matched_query: str | None = None  # the cached query this answer was reused from


class RAGService:
    """`router=None` always uses `default_model`; `cache=None` disables caching."""

    def __init__(self, docs: list[Document], embedder: Embedder,
                 models: dict[str, LLMClient], default_model: str,
                 cache: SemanticCache | ExactMatchCache | None = None,
                 router: ComplexityRouter | None = None,
                 analyzer: KeyTermExtractor | None = None, metrics: MetricsStore | None = None,
                 top_k: int = 3, embed_usd_per_mtok: float = 0.02, run: str = "default"):
        self.embedder = embedder
        self.retriever = Retriever(docs, embedder, top_k)
        self.models = models
        self.default_model = default_model
        self.cache = cache
        self.router = router
        self.analyzer = analyzer or KeyTermExtractor.from_corpus(d.text for d in docs)
        self.metrics = metrics
        self.embed_usd_per_mtok = embed_usd_per_mtok
        self.run = run

    def answer(self, query: str,
               grader: Callable[[str], bool] | None = None) -> tuple[Answer, RequestRecord]:
        """Answer a query and log one RequestRecord. `grader`, if given, marks the
        answer correct/incorrect in the log (used by the benchmark)."""
        t0 = time.perf_counter()
        vec = self.embedder.embed([query])[0]
        t1 = time.perf_counter()
        retrieved = self.retriever.search(vec)
        context_ids = frozenset(r.doc.doc_id for r in retrieved)
        t2 = time.perf_counter()
        sig = self.analyzer(query)
        embed_cost = estimate_embed_tokens(query) * self.embed_usd_per_mtok / 1e6

        hit = self.cache.lookup(query, vec, sig, context_ids) if self.cache is not None else None
        t3 = time.perf_counter()
        local_ms = (t3 - t0) * 1000
        sources = sorted(context_ids)

        if hit:
            ans = Answer(hit.entry.answer, "cache", True, embed_cost, local_ms, sources,
                         similarity=hit.similarity, matched_query=hit.entry.query)
            return ans, self._record(query, ans, 0, 0, t0, t1, t2, t3, 0.0, grader)

        decision = self.router.route(query, sig.key_terms, retrieved) if self.router else None
        model_name = decision.model if decision else self.default_model
        client = self.models[model_name]
        completion = client.complete(SYSTEM_PROMPT, build_prompt(query, retrieved))
        llm_ms = completion.latency_s * 1000
        cost = embed_cost + client.spec.cost(completion.input_tokens, completion.output_tokens)
        if self.cache is not None:
            self.cache.add(query, vec, completion.text, sig, context_ids, model_name)
        ans = Answer(completion.text, model_name, False, cost, local_ms + llm_ms, sources,
                     complexity=decision.complexity if decision else None)
        return ans, self._record(query, ans, completion.input_tokens, completion.output_tokens,
                                 t0, t1, t2, t3, llm_ms, grader)

    def _record(self, query, ans: Answer, in_tok, out_tok, t0, t1, t2, t3, llm_ms, grader):
        rec = RequestRecord(
            ts=time.time(), query=query, route=ans.route, cache_hit=ans.cache_hit,
            similarity=ans.similarity, complexity=ans.complexity, input_tokens=in_tok,
            output_tokens=out_tok, cost_usd=ans.cost_usd, latency_ms=ans.latency_ms,
            embed_ms=(t1 - t0) * 1000, retrieve_ms=(t2 - t1) * 1000,
            cache_ms=(t3 - t2) * 1000, llm_ms=llm_ms,
            correct=grader(ans.text) if grader else None, run=self.run)
        if self.metrics:
            self.metrics.log(rec)
        return rec
