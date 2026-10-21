from ragroute.corpus import build_intents
from ragroute.router import FEATURES, ComplexityRouter, extract_features


def _router():
    return ComplexityRouter("small", "large")


def test_simple_lookup_goes_small(analyzer):
    r = _router()
    q = "How much storage does the Pro plan include?"
    assert r.route(q, analyzer(q).key_terms).model == "small"


def test_comparison_goes_large(analyzer):
    r = _router()
    q = "Which has more storage per seat, Pro or Business, and by how much?"
    assert r.route(q, analyzer(q).key_terms).model == "large"


def test_threshold_moves_traffic(analyzer):
    qs = [p for i in build_intents() for p in i.paraphrases]
    share = []
    for t in (0.2, 0.5, 0.8):
        r = ComplexityRouter("small", "large", threshold=t)
        share.append(sum(r.route(q, analyzer(q).key_terms).model == "large" for q in qs))
    assert share[0] >= share[1] >= share[2]


def test_default_router_accuracy_on_paraphrases(analyzer):
    r = _router()
    intents = build_intents()
    pairs = [(p, i.difficulty == "hard") for i in intents for p in i.paraphrases]
    correct = sum((r.route(q, analyzer(q).key_terms).model == "large") == hard for q, hard in pairs)
    assert correct / len(pairs) > 0.9


def test_fit_learns_separating_weights(analyzer):
    intents = build_intents()
    rows, labels = [], []
    for i in intents:
        for p in i.paraphrases:
            rows.append(extract_features(p, analyzer(p).key_terms))
            labels.append(i.difficulty == "hard")
    r = _router().fit(rows, labels)
    assert set(r.weights) == set(FEATURES)
    preds = [r.score(x) >= 0.5 for x in rows]
    assert sum(p == y for p, y in zip(preds, labels)) / len(labels) > 0.9
