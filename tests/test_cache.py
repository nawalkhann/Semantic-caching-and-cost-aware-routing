import numpy as np

from ragroute.cache import ExactMatchCache, SemanticCache

CTX = frozenset({"plan-pro"})


def _add(cache, embedder, analyzer, q, answer="a", ctx=CTX):
    cache.add(q, embedder.embed([q])[0], answer, analyzer(q), ctx, "large")


def _lookup(cache, embedder, analyzer, q, ctx=CTX):
    return cache.lookup(q, embedder.embed([q])[0], analyzer(q), ctx)


def test_near_duplicate_hits(embedder, analyzer):
    cache = SemanticCache(embedder.dim, threshold=0.8)
    _add(cache, embedder, analyzer, "How much storage does the Pro plan include?", "1000 GB")
    hit = _lookup(cache, embedder, analyzer, "hi, how much storage does the pro plan include? thanks!")
    assert hit is not None and hit.entry.answer == "1000 GB"
    assert cache.stats.hit_rate == 1.0


def test_entity_change_is_not_a_hit(embedder, analyzer):
    cache = SemanticCache(embedder.dim, threshold=0.5)
    _add(cache, embedder, analyzer, "How much storage does the Pro plan include?")
    q = "How much storage does the Business plan include?"
    v1, v2 = embedder.embed(["How much storage does the Pro plan include?", q])
    assert float(v1 @ v2) > 0.5  # embeddings alone would match
    assert _lookup(cache, embedder, analyzer, q) is None
    assert cache.stats.rejected_key_terms == 1


def test_number_change_is_not_a_hit(embedder, analyzer):
    cache = SemanticCache(embedder.dim, threshold=0.5)
    _add(cache, embedder, analyzer, "What would 12 seats on Pro cost per year?")
    assert _lookup(cache, embedder, analyzer, "What would 30 seats on Pro cost per year?") is None


def test_topic_change_is_not_a_hit(embedder, analyzer):
    cache = SemanticCache(embedder.dim, threshold=0.5)
    _add(cache, embedder, analyzer,
         "Hey team - which has more storage per seat, Pro or Business? Thanks in advance.")
    q = "Hey team - which has more version history, Pro or Business? Thanks in advance."
    assert _lookup(cache, embedder, analyzer, q) is None
    assert cache.stats.rejected_topic == 1


def test_context_change_is_not_a_hit(embedder, analyzer):
    cache = SemanticCache(embedder.dim, threshold=0.8)
    q = "How much storage does the Pro plan include?"
    _add(cache, embedder, analyzer, q, ctx=frozenset({"plan-pro", "plan-business"}))
    assert _lookup(cache, embedder, analyzer, q, ctx=frozenset({"policy-refunds"})) is None
    assert cache.stats.rejected_context == 1


def test_unguarded_cache_makes_the_false_hit(embedder, analyzer):
    cache = SemanticCache(embedder.dim, threshold=0.5, check_key_terms=False,
                          check_topic=False, check_context=False)
    _add(cache, embedder, analyzer, "How much storage does the Pro plan include?")
    assert _lookup(cache, embedder, analyzer, "How much storage does the Business plan include?")


def test_ttl_expiry():
    now = [0.0]
    cache = SemanticCache(4, threshold=0.5, ttl_s=10, clock=lambda: now[0],
                          check_topic=False, check_key_terms=False, check_context=False)
    v = np.array([1, 0, 0, 0], dtype=np.float32)
    cache.add("q", v, "a", None, frozenset(), "m")
    assert cache.lookup("q", v, None) is not None
    now[0] = 11
    assert cache.lookup("q", v, None) is None
    assert len(cache) == 0 and cache.stats.expirations == 1


def test_lru_eviction():
    now = [0.0]
    cache = SemanticCache(2, threshold=0.99, max_entries=2, ttl_s=None, clock=lambda: now[0],
                          check_topic=False, check_key_terms=False, check_context=False)
    a, b, c = (np.array(x, dtype=np.float32) for x in ([1, 0], [0, 1], [0.7071, 0.7071]))
    cache.add("a", a, "A", None, frozenset(), "m")
    now[0] = 1
    cache.add("b", b, "B", None, frozenset(), "m")
    now[0] = 2
    assert cache.lookup("a", a, None)  # touch a, so b is least recently used
    now[0] = 3
    cache.add("c", c, "C", None, frozenset(), "m")
    assert len(cache) == 2
    assert cache.lookup("b", b, None) is None
    assert cache.lookup("a", a, None).entry.answer == "A"


def test_exact_cache_normalises_text():
    cache = ExactMatchCache()
    cache.add("How much is Pro?", None, "15", None, frozenset(), "m")
    assert cache.lookup("how much is pro").entry.answer == "15"
    assert cache.lookup("how much is pro, thanks") is None


def test_key_terms_learned_from_corpus(analyzer):
    assert {"free", "starter", "pro", "business", "enterprise"} <= analyzer.entities
    assert "nimbus" not in analyzer.entities  # in most docs, so not discriminative
    sig = analyzer("Pro vs business storage for 12 seats?")
    assert sig.key_terms == frozenset({"pro", "business", "12"})
    assert "storage" in sig.topic_terms
