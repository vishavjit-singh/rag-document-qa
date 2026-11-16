"""Okapi BM25 (Robertson/Sparck Jones idf with the +1 smoothing used by Lucene)."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .text import tokens


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokens(d)) for d in docs]
        self.lens = np.array([sum(tf.values()) for tf in self.tfs], dtype=float)
        self.avgdl = float(self.lens.mean()) if len(self.lens) else 0.0
        self.n = len(docs)
        df: Counter = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        self.df = df
        self.postings: dict[str, list[int]] = {}
        for i, tf in enumerate(self.tfs):
            for t in tf:
                self.postings.setdefault(t, []).append(i)

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1 + (self.n - n + 0.5) / (n + 0.5))

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n)
        for t in set(tokens(query)):
            idf = self.idf(t)
            for i in self.postings.get(t, ()):
                f = self.tfs[i][t]
                out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl))
        return out

    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        s = self.scores(query)
        idx = [i for i in np.argsort(-s, kind="stable")[:k] if s[i] > 0]
        return [(int(i), float(s[i])) for i in idx]
