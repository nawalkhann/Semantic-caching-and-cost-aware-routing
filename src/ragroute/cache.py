"""Semantic answer cache.

A lookup is a hit only when all four checks pass:

1. cosine similarity between query embeddings >= `threshold`;
2. the *key terms* match exactly (numbers and named entities such as plan names),
   because "storage on Pro" and "storage on Business" embed almost identically
   but need different answers;
3. the *topic terms* overlap (Jaccard >= `min_topic_overlap`). Topic terms are the
   query words that also occur in the knowledge base, so "Pro vs Business on
   storage" and "Pro vs Business on version history" differ even when filler
   words ("Hi team, ... thanks!") push their embeddings close together;
4. the retrieved context overlaps (Jaccard over document ids >= `min_context_overlap`),
   so an answer is never reused when retrieval would ground it in different docs.

Entries expire after `ttl_s` seconds and the cache evicts least-recently-used
entries beyond `max_entries`.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .embeddings import _STOP, tokenize

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


@dataclass(frozen=True)
class QuerySignature:
    key_terms: frozenset[str]  # numbers and entity names: must match exactly
    topic_terms: frozenset[str]  # knowledge-base words in the query: must mostly overlap


class KeyTermExtractor:
    """Pulls out the tokens whose change flips the answer.

    Entity vocabulary is learned from the corpus as words that appear capitalised
    mid-sentence (product, plan and feature names), and matched case-insensitively.
    Topic vocabulary is every other non-stopword in the corpus.
    """

    def __init__(self, entities: Iterable[str] = (), vocabulary: Iterable[str] = ()):
        self.entities = {e.lower() for e in entities}
        self.vocabulary = {v.lower() for v in vocabulary} - self.entities - _STOP

    @classmethod
    def from_corpus(cls, texts: Iterable[str], min_count: int = 2, max_df: float = 0.5,
                    ignore: Iterable[str] = ()) -> "KeyTermExtractor":
        """`max_df` drops names found in most documents (e.g. the product name),
        since they do not distinguish one answer from another."""
        texts = list(texts)
        counts: dict[str, int] = {}
        lower: dict[str, int] = {}
        df: dict[str, int] = {}
        for text in texts:
            for w in {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9-]*", text)}:
                df[w] = df.get(w, 0) + 1
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                words = re.findall(r"[A-Za-z][A-Za-z0-9-]*", sentence)
                for i, w in enumerate(words):
                    if i > 0 and w[0].isupper() and not w.isupper():
                        counts[w.lower()] = counts.get(w.lower(), 0) + 1
                    elif w.islower():
                        lower[w] = lower.get(w, 0) + 1
        skip = {w.lower() for w in ignore}
        # a word mostly written in lower case is an ordinary word, not a name
        ents = {w for w, c in counts.items()
                if c >= min_count and c > lower.get(w, 0) and w not in skip
                and df[w] <= max_df * len(texts)}
        vocab = {t for text in texts for t in tokenize(text) if not _NUMBER.fullmatch(t)}
        return cls(ents, vocab - {w for w in df if w not in ents and df[w] == len(texts)})

    def __call__(self, text: str) -> QuerySignature:
        toks = tokenize(text)
        terms = {t for t in toks if t in self.entities}
        terms.update(_NUMBER.findall(text))
        topics = {t for t in toks if t in self.vocabulary}
        return QuerySignature(frozenset(terms), frozenset(topics))


@dataclass
class CacheEntry:
    query: str
    answer: str
    signature: QuerySignature
    context_ids: frozenset[str]
    model: str
    created_at: float
    last_used: float
    hits: int = 0


@dataclass
class CacheHit:
    entry: CacheEntry
    similarity: float


@dataclass
class CacheStats:
    lookups: int = 0
    hits: int = 0
    rejected_key_terms: int = 0
    rejected_topic: int = 0
    rejected_context: int = 0
    evictions: int = 0
    expirations: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


@dataclass
class SemanticCache:
    dim: int
    threshold: float = 0.80
    min_context_overlap: float = 0.5
    min_topic_overlap: float = 0.6
    check_key_terms: bool = True
    check_topic: bool = True
    check_context: bool = True
    max_entries: int = 10_000
    ttl_s: float | None = 24 * 3600
    clock: callable = field(default=time.time, repr=False)

    def __post_init__(self):
        self._vecs = np.zeros((0, self.dim), dtype=np.float32)
        self._entries: list[CacheEntry] = []
        self.stats = CacheStats()

    def __len__(self) -> int:
        return len(self._entries)

    def lookup(self, query: str, vec: np.ndarray, signature: QuerySignature,
               context_ids: frozenset[str] | None = None) -> CacheHit | None:
        self.stats.lookups += 1
        self._expire()
        if not self._entries:
            return None
        sims = self._vecs @ vec
        rejected_kt = rejected_topic = rejected_ctx = False
        for idx in np.argsort(-sims):
            sim = float(sims[idx])
            if sim < self.threshold:
                break
            entry = self._entries[idx]
            if self.check_key_terms and entry.signature.key_terms != signature.key_terms:
                rejected_kt = True
                continue
            if (self.check_topic and _jaccard(entry.signature.topic_terms, signature.topic_terms)
                    < self.min_topic_overlap):
                rejected_topic = True
                continue
            if (self.check_context and context_ids is not None
                    and _jaccard(entry.context_ids, context_ids) < self.min_context_overlap):
                rejected_ctx = True
                continue
            entry.hits += 1
            entry.last_used = self.clock()
            self.stats.hits += 1
            return CacheHit(entry, sim)
        self.stats.rejected_key_terms += rejected_kt
        self.stats.rejected_topic += rejected_topic
        self.stats.rejected_context += rejected_ctx
        return None

    def add(self, query: str, vec: np.ndarray, answer: str, signature: QuerySignature,
            context_ids: frozenset[str], model: str) -> None:
        now = self.clock()
        self._entries.append(CacheEntry(query, answer, signature, context_ids, model, now, now))
        self._vecs = np.vstack([self._vecs, vec[None, :].astype(np.float32)])
        if len(self._entries) > self.max_entries:
            lru = min(range(len(self._entries)), key=lambda i: self._entries[i].last_used)
            self._drop([lru])
            self.stats.evictions += 1

    def _expire(self) -> None:
        if self.ttl_s is None:
            return
        cutoff = self.clock() - self.ttl_s
        stale = [i for i, e in enumerate(self._entries) if e.created_at < cutoff]
        if stale:
            self._drop(stale)
            self.stats.expirations += len(stale)

    def _drop(self, indices: list[int]) -> None:
        keep = np.ones(len(self._entries), dtype=bool)
        keep[indices] = False
        self._vecs = self._vecs[keep]
        self._entries = [e for e, k in zip(self._entries, keep) if k]


class ExactMatchCache:
    """Baseline: reuse an answer only for the same question text (case, whitespace
    and punctuation normalised)."""

    def __init__(self):
        self._store: dict[str, CacheEntry] = {}
        self.stats = CacheStats()

    def __len__(self) -> int:
        return len(self._store)

    @staticmethod
    def _key(query: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", query.lower()))

    def lookup(self, query: str, vec=None, signature=None, context_ids=None) -> CacheHit | None:
        self.stats.lookups += 1
        entry = self._store.get(self._key(query))
        if entry is None:
            return None
        entry.hits += 1
        self.stats.hits += 1
        return CacheHit(entry, 1.0)

    def add(self, query: str, vec, answer: str, signature: QuerySignature,
            context_ids: frozenset[str], model: str) -> None:
        now = time.time()
        self._store[self._key(query)] = CacheEntry(query, answer, signature, context_ids,
                                                   model, now, now)
