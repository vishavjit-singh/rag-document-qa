"""BM25, dense and hybrid retrieval with reciprocal rank fusion, version-aware ranking and optional MMR."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .store import Store

MODES = ("bm25", "dense", "hybrid")


@dataclass
class Hit:
    chunk: dict
    score: float
    rank: int
    ranks: dict = field(default_factory=dict)     # rank in each retriever (1-based) before fusion
    demoted: bool = False

    def to_dict(self) -> dict:
        c = self.chunk
        return {"rank": self.rank, "score": round(self.score, 5), "chunk_id": c["id"], "doc": c["doc"],
                "pages": c["pages"], "section": c["section"], "status": c["status"], "demoted": self.demoted,
                "ranks": self.ranks, "text": c["text"]}


def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k + rank)."""
    out: dict[int, float] = {}
    for ranking in rankings:
        for r, i in enumerate(ranking, 1):
            out[i] = out.get(i, 0.0) + 1.0 / (k + r)
    return out


def mmr(candidates: list[int], rel: np.ndarray, vectors: np.ndarray, k: int, lam: float = 0.7) -> list[int]:
    """Maximal marginal relevance: trade relevance against similarity to what is already selected."""
    rel = np.asarray(rel, dtype=float)
    if rel.max() > rel.min():
        rel = (rel - rel.min()) / (rel.max() - rel.min())
    chosen: list[int] = []
    left = list(range(len(candidates)))
    while left and len(chosen) < k:
        if chosen:
            sim = vectors[[candidates[i] for i in left]] @ vectors[[candidates[j] for j in chosen]].T
            red = sim.max(axis=1)
        else:
            red = np.zeros(len(left))
        scores = lam * rel[left] - (1 - lam) * red
        best = left[int(np.argmax(scores))]
        chosen.append(best)
        left.remove(best)
    return [candidates[i] for i in chosen]


VERSION_TOKEN = re.compile(r"\b(?:v(?:ersion)?\s*)?(\d{4}(?:\.\d+)?|\d+\.\d+)\b", re.I)


def version_tokens(chunk_or_doc: dict) -> set[str]:
    v, eff = chunk_or_doc.get("version", ""), chunk_or_doc.get("effective", "")
    out = {v} if v else set()
    if re.match(r"\d{4}\.", v):
        out.add(v.split(".")[0])
    out.update(re.findall(r"\b(\d{4})\b", eff))
    return out


class Retriever:
    def __init__(self, store: Store, candidates: int = 30, rrf_k: int = 60):
        self.store, self.candidates, self.rrf_k = store, candidates, rrf_k
        files = store.manifest.get("files") or {c["doc"]: c for c in store.chunks}
        self.doc_versions = {name: version_tokens(info) for name, info in files.items()}

    def mentioned_docs(self, query: str) -> set[str]:
        asked = {m.group(1) for m in VERSION_TOKEN.finditer(query)}
        return {d for d, toks in self.doc_versions.items() if asked & toks}

    def _demote(self, chunk: dict, mentioned: set[str]) -> bool:
        """Superseded versions rank after current ones, unless the question names a version or year: then the
        other versions of that document family are the ones that move down."""
        docs = [chunk["doc"]] + [x["doc"] for x in chunk["also_in"]]
        family = self.store.families.get(chunk["code"], set())
        if len(family) > 1 and family & mentioned:
            return not any(d in mentioned for d in docs)
        return chunk["status"] == "Superseded"

    def search(self, query: str, k: int = 5, mode: str = "hybrid", version_aware: bool = True,
               use_mmr: bool = False, mmr_lambda: float = 0.7, query_vector: np.ndarray | None = None) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        s, n = self.store, self.candidates
        mentioned = self.mentioned_docs(query) if version_aware else set()
        text = query
        if mentioned:            # "2024" or "version 1.0" picked the document version; don't also match it as text
            text = re.sub(r"\s{2,}", " ", VERSION_TOKEN.sub(" ", query)).strip()
        ranks: dict[str, list[int]] = {}
        raw: dict[str, dict[int, float]] = {}
        if mode in ("bm25", "hybrid"):
            top = s.bm25.top(text, n)
            ranks["bm25"] = [i for i, _ in top]
            raw["bm25"] = dict(top)
        if mode in ("dense", "hybrid") or use_mmr:
            qv = query_vector if query_vector is not None else s.embedder.encode([text])[0]
        if mode in ("dense", "hybrid"):
            sims, idx = s.index.search(qv[None, :], n)
            ranks["dense"] = [int(i) for i in idx[0] if i >= 0]
            raw["dense"] = {int(i): float(v) for i, v in zip(idx[0], sims[0]) if i >= 0}
        scores = rrf(list(ranks.values()), self.rrf_k) if mode == "hybrid" else raw[mode]
        order = sorted(scores, key=lambda i: -scores[i])
        demoted = set()
        if version_aware:
            demoted = {i for i in order if self._demote(s.chunks[i], mentioned)}
            order = [i for i in order if i not in demoted] + [i for i in order if i in demoted]
        if use_mmr and order:
            rel = np.linspace(1.0, 0.0, num=len(order))         # rank-based relevance after fusion/demotion
            order = mmr(order, rel, s.vectors, k, mmr_lambda) + order
            order = list(dict.fromkeys(order))
        hits = []
        for r, i in enumerate(order[:k], 1):
            hits.append(Hit(s.chunks[i], float(scores[i]), r,
                            {name: rk.index(i) + 1 for name, rk in ranks.items() if i in rk}, i in demoted))
        return hits
