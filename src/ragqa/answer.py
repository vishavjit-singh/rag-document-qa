"""Grounded extractive answering: pick the best-supported sentence from the retrieved chunks, cite it, or abstain.

Every sentence (or linearised table row) in the top chunks is scored by
  - lexical coverage: the idf-weighted share of the question's terms found in the sentence (terms found only in
    the section heading or document title count half), and
  - dense similarity between the question and the sentence (same embedder as retrieval),
then lightly discounted by the rank of the chunk it came from. The answer is the best sentence with a
``[file p.N]`` citation. If the best score is below the threshold the answerer says the answer was not found.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import numpy as np

from .retrieve import VERSION_TOKEN, Hit, version_tokens
from .store import Store
from .text import sentences, tokens

NOT_FOUND = "Not found in the documents."
W_LEX, W_DENSE, W_TYPE, RANK_DECAY = 0.65, 0.35, 0.10, 0.04
UNSEEN_IDF = None          # weight of question terms absent from the corpus (None = idf of an unseen term)
_QUANTITY_START = re.compile(r"^(how (many|much|long|often|soon|quickly|far|high|fast)|when|within (what|how))\b", re.I)
_QUANTITY_NOUN = re.compile(
    r"\b(rate|limit|length|time|range|cap|allowance|deadline|percentage|price|period|budget|weight|voltage|"
    r"temperature|interval|uptime|cost|hours|window|fee|credit|diem|amount|speed)s?\b", re.I)
_HAS_NUMBER = re.compile(r"\d")


def expects_quantity(question: str) -> bool:
    """Questions asking for an amount, duration, time or limit: answers that contain a number fit better."""
    q = re.sub(r"^(under|in|for|according to) [^,]*, ", "", question.strip(), flags=re.I)
    if _QUANTITY_START.match(q):
        return True
    head = " ".join(q.split()[:8])                  # "What uptime did ...", "Which ... rate ..."
    return bool(re.match(r"(what|which)\b", head, re.I) and _QUANTITY_NOUN.search(head))


def cite(doc: str, page: int) -> str:
    return f"[{doc} p.{page}]"


@dataclass
class Answer:
    question: str
    answer: str
    abstained: bool
    confidence: float | None
    citations: list[dict] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    backend: str = "extractive"

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def text_with_citations(self) -> str:
        if self.abstained:
            return self.answer
        return f"{self.answer} " + " ".join(cite(c["doc"], c["page"]) for c in self.citations)


class ExtractiveAnswerer:
    name = "extractive"

    def __init__(self, store: Store, threshold: float = 0.45):
        self.store, self.threshold = store, threshold

    def _units(self, hits: list[Hit]) -> list[dict]:
        seen, out = set(), []
        for h in hits:
            c = h.chunk
            versions = " ".join(version_tokens(c))
            for page, line, section, is_heading in c["lines"]:
                if is_heading:
                    continue
                for s in sentences(line):
                    if len(s.split()) < 3 or s in seen:
                        continue
                    seen.add(s)
                    out.append({"text": s, "doc": c["doc"], "page": page, "section": section,
                                "chunk_id": c["id"], "rank": h.rank,
                                "context": f"{c['title']} {c['chapter']} {section} {versions}"})
        return out

    def score(self, question: str, hits: list[Hit]) -> list[dict]:
        units = self._units(hits)
        if not units:
            return []
        # version numbers and years in the question choose the document version during retrieval; they are not
        # evidence for a particular sentence (otherwise "Effective date: 1 January 2024" wins every 2024 question)
        q = set(tokens(question)) - {m.group(1) for m in VERSION_TOKEN.finditer(question)}
        bm = self.store.bm25
        w = {t: bm.idf(t) if (t in bm.df or UNSEEN_IDF is None) else UNSEEN_IDF for t in q}
        total = sum(w.values()) or 1.0
        want_number = expects_quantity(question)
        qv = self.store.embedder.encode([question])[0]
        uv = self.store.embedder.encode([f"{u['section']}: {u['text']}" for u in units])
        dense = np.clip(uv @ qv, 0, 1)
        for u, d in zip(units, dense):
            in_text = set(tokens(u["text"]))
            in_ctx = set(tokens(u["context"])) - in_text
            lex = (sum(w[t] for t in q & in_text) + 0.5 * sum(w[t] for t in q & in_ctx)) / total
            fits = want_number and bool(_HAS_NUMBER.search(u["text"]))
            u["lexical"], u["dense"], u["type_match"] = round(lex, 4), round(float(d), 4), fits
            raw = W_LEX * lex + W_DENSE * float(d) + (W_TYPE if fits else 0.0)
            u["score"] = round(raw * (1 - RANK_DECAY * (u["rank"] - 1)), 4)
        return sorted(units, key=lambda u: -u["score"])

    def answer(self, question: str, hits: list[Hit], threshold: float | None = None) -> Answer:
        th = self.threshold if threshold is None else threshold
        ranked = self.score(question, hits)
        evidence = [{k: u[k] for k in ("text", "doc", "page", "section", "score", "lexical", "dense")}
                    for u in ranked[:3]]
        if not ranked or ranked[0]["score"] < th:
            conf = ranked[0]["score"] if ranked else 0.0
            return Answer(question, NOT_FOUND, True, conf, [], evidence)
        best = ranked[0]
        citation = {"doc": best["doc"], "page": best["page"], "section": best["section"],
                    "chunk_id": best["chunk_id"]}
        return Answer(question, best["text"], False, best["score"], [citation], evidence)
