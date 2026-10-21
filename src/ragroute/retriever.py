"""Dense retriever over a small document set (brute-force cosine search)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .corpus import Document
from .embeddings import Embedder


@dataclass(frozen=True)
class RetrievedDoc:
    doc: Document
    score: float


class Retriever:
    def __init__(self, docs: list[Document], embedder: Embedder, top_k: int = 3):
        self.docs = docs
        self.embedder = embedder
        self.top_k = top_k
        self._matrix = embedder.embed([f"{d.title}. {d.text}" for d in docs])

    def search(self, query_vec: np.ndarray, k: int | None = None) -> list[RetrievedDoc]:
        k = k or self.top_k
        scores = self._matrix @ query_vec
        top = np.argsort(-scores)[:k]
        return [RetrievedDoc(self.docs[i], float(scores[i])) for i in top]
