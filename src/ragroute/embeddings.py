"""Text embedders.

`HashingEmbedder` is dependency-free (numpy only) and deterministic: it hashes
word unigrams, word bigrams and character trigrams into a fixed-size vector,
weighted by IDF learned from the corpus. It captures lexical and sub-word
overlap but not synonyms, so it is a conservative choice for caching (it misses
some paraphrases rather than inventing matches).

`SentenceTransformerEmbedder` wraps any sentence-transformers model and is the
recommended choice in production.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Iterable, Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOP = frozenset(
    "a an the of on in for to and or is are was were be been being do does did i we our us my "
    "me you your it its this that these those what whats how which who whom when where why "
    "can could would should will shall may might must with per by from at as up all any each "
    "no not than then there here so if but about into over under also just only very more most "
    "some such own same too get got have has had am hi hello hey thanks thank thx please quick "
    "question time first".split()
)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower().replace("'", ""))


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised vectors."""


def _bucket(feature: str, dim: int) -> tuple[int, float]:
    h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
    return h % dim, (1.0 if (h >> 63) & 1 else -1.0)


class HashingEmbedder:
    def __init__(self, dim: int = 2048, char_ngram: int = 3, char_weight: float = 0.35):
        self.dim = dim
        self.char_ngram = char_ngram
        self.char_weight = char_weight
        self._idf: dict[str, float] = {}
        self._default_idf = 1.0

    def _features(self, text: str) -> Counter:
        toks = [t for t in tokenize(text) if t not in _STOP]
        feats: Counter = Counter()
        for t in toks:
            feats["w:" + t] += 1.0
        for a, b in zip(toks, toks[1:]):
            feats[f"b:{a}_{b}"] += 0.5
        n = self.char_ngram
        for t in toks:
            padded = f"<{t}>"
            for i in range(max(1, len(padded) - n + 1)):
                feats["c:" + padded[i:i + n]] += self.char_weight
        return feats

    def fit(self, corpus: Iterable[str]) -> "HashingEmbedder":
        """Learn IDF weights for word features from a corpus."""
        docs = [set(f for f in self._features(t) if f.startswith("w:")) for t in corpus]
        df: Counter = Counter()
        for d in docs:
            df.update(d)
        n = max(len(docs), 1)
        self._idf = {f: math.log((1 + n) / (1 + c)) + 1.0 for f, c in df.items()}
        self._default_idf = math.log(1 + n) + 1.0
        return self

    def _weight(self, feat: str) -> float:
        if feat.startswith("w:"):
            return self._idf.get(feat, self._default_idf)
        if feat.startswith("b:"):
            return self._default_idf
        return 1.0

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, tf in self._features(text).items():
                idx, sign = _bucket(feat, self.dim)
                out[row, idx] += sign * (1.0 + math.log(tf)) * self._weight(feat)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # optional dependency

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vecs.astype(np.float32)


def estimate_embed_tokens(text: str) -> int:
    return max(1, round(len(text) / 4))
