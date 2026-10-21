"""Cost-aware model router.

Scores how hard a question is from cheap signals available before any LLM call
(the query text, its key terms and the retrieval result) and sends it to the
small model when the score is below `threshold`, otherwise to the large model.
The score is a logistic model over hand-built features. It ships with
hand-set default weights and can be re-fit on labelled traffic with `fit`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from .embeddings import tokenize
from .retriever import RetrievedDoc

_COMPARE = {"vs", "versus", "compare", "compared", "comparison", "more", "less", "difference",
            "gap", "better", "cheapest", "cheaper", "least", "minimise", "minimize", "stack",
            "between", "which"}
_AGGREGATE = {"total", "yearly", "annual", "annually", "sum", "bill", "altogether", "combined"}
_CONSTRAINT = {"need", "needs", "must", "if", "require", "requires", "required", "only"}
_REASONING = {"why", "explain", "should", "recommend", "tradeoff", "tradeoffs", "pros", "cons"}
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")

FEATURES = ["multi_entity", "compare", "aggregate", "constraint", "reasoning",
            "numbers", "log_len", "multi_doc"]

DEFAULT_WEIGHTS = {
    "multi_entity": 2.2, "compare": 1.4, "aggregate": 1.6, "constraint": 0.7,
    "reasoning": 1.5, "numbers": 1.1, "log_len": 0.2, "multi_doc": 0.6,
}
DEFAULT_BIAS = -2.6


def extract_features(query: str, key_terms: frozenset[str],
                     retrieved: list[RetrievedDoc] | None = None) -> dict[str, float]:
    toks = set(tokenize(query))
    entities = [t for t in key_terms if not _NUMBER.fullmatch(t)]
    multi_doc = 0.0
    if retrieved:
        top = retrieved[0].score
        # docs nearly as relevant as the best one suggest the answer spans several docs
        multi_doc = float(sum(1 for r in retrieved[1:] if r.score >= 0.85 * top))
    return {
        "multi_entity": float(len(entities) >= 2),
        "compare": float(bool(toks & _COMPARE)),
        "aggregate": float(bool(toks & _AGGREGATE) or "per year" in query.lower()),
        "constraint": float(bool(toks & _CONSTRAINT)),
        "reasoning": float(bool(toks & _REASONING)),
        "numbers": float(min(len(_NUMBER.findall(query)), 2)),
        "log_len": math.log1p(len(tokenize(query))),
        "multi_doc": multi_doc,
    }


@dataclass
class RouteDecision:
    model: str
    complexity: float
    features: dict[str, float]


@dataclass
class ComplexityRouter:
    small_model: str
    large_model: str
    threshold: float = 0.5
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    bias: float = DEFAULT_BIAS

    def score(self, features: dict[str, float]) -> float:
        z = self.bias + sum(self.weights[f] * features[f] for f in FEATURES)
        return 1.0 / (1.0 + math.exp(-z))

    def route(self, query: str, key_terms: frozenset[str],
              retrieved: list[RetrievedDoc] | None = None) -> RouteDecision:
        feats = extract_features(query, key_terms, retrieved)
        p = self.score(feats)
        return RouteDecision(self.large_model if p >= self.threshold else self.small_model,
                             p, feats)

    def fit(self, feature_rows: list[dict[str, float]], is_hard: list[bool],
            l2: float = 0.01, lr: float = 0.5, epochs: int = 2000) -> "ComplexityRouter":
        """Fit weights with L2-regularised logistic regression (batch gradient descent)."""
        X = np.array([[r[f] for f in FEATURES] for r in feature_rows], dtype=float)
        y = np.array(is_hard, dtype=float)
        w = np.zeros(X.shape[1])
        b = 0.0
        for _ in range(epochs):
            p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
            g = p - y
            w -= lr * (X.T @ g / len(y) + l2 * w)
            b -= lr * g.mean()
        self.weights = dict(zip(FEATURES, map(float, w)))
        self.bias = float(b)
        return self
