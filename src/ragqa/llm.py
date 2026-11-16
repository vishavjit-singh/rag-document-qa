"""Optional generative answerer: OpenAI chat completions over plain urllib (no SDK needed).

Only used when the answer backend is ``openai`` *and* ``OPENAI_API_KEY`` is set. The demo and CI never call it;
the tests replace the HTTP call with a fake. The prompt restricts the model to the retrieved context and asks for
``[file p.N]`` citations; citations that do not point at a retrieved page are removed, and an answer left
without a valid citation is turned into an abstention.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

from .answer import NOT_FOUND, Answer
from .retrieve import Hit

SYSTEM = (
    "You answer questions about company documents using ONLY the numbered context passages. "
    "Every factual sentence must end with a citation copied exactly from the passage header, in the form "
    "[file.pdf p.N]. If the passages do not contain the answer, reply exactly: Not found in the documents. "
    "Do not use outside knowledge and do not guess.")
_CITATION = re.compile(r"\[([\w.\-]+\.pdf) p\.(\d+)\]")


def build_context(hits: list[Hit]) -> tuple[str, set[tuple[str, int]]]:
    parts, allowed = [], set()
    for h in hits:
        c = h.chunk
        for page, *_ in c["lines"]:
            allowed.add((c["doc"], page))
        body = "\n".join(f"[{c['doc']} p.{p}] {t}" for p, t, *_ in c["lines"])
        parts.append(f"Passage {h.rank} | {c['title']} v{c['version']} ({c['status']}) | {c['section']}\n{body}")
    return "\n\n".join(parts), allowed


class OpenAIAnswerer:
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None,
                 base_url: str = "https://api.openai.com/v1", timeout: float = 30.0, opener=None):
        self.model, self.timeout = model, timeout
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = os.environ.get("OPENAI_BASE_URL", base_url).rstrip("/")
        self.opener = opener or urllib.request.urlopen
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set; use the extractive answerer instead")

    def request(self, question: str, hits: list[Hit]) -> urllib.request.Request:
        context, _ = build_context(hits)
        payload = {"model": self.model, "temperature": 0,
                   "messages": [{"role": "system", "content": SYSTEM},
                                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]}
        return urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(payload).encode(),
                                      headers={"Authorization": f"Bearer {self.api_key}",
                                               "Content-Type": "application/json"}, method="POST")

    def answer(self, question: str, hits: list[Hit], threshold: float | None = None) -> Answer:
        _, allowed = build_context(hits)
        with self.opener(self.request(question, hits), timeout=self.timeout) as resp:
            content = json.loads(resp.read())["choices"][0]["message"]["content"].strip()
        cited = [(d, int(p)) for d, p in _CITATION.findall(content)]
        valid = [c for c in dict.fromkeys(cited) if c in allowed]
        text = re.sub(r"\s+([.,;])", r"\1", re.sub(r"\s{2,}", " ", _CITATION.sub("", content))).strip()
        if NOT_FOUND.lower().rstrip(".") in content.lower() or not valid:
            return Answer(question, NOT_FOUND, True, None, [], [], backend="openai")
        where = {(h.chunk["doc"], ln[0]): (ln[2], h.chunk["id"]) for h in hits for ln in h.chunk["lines"]}
        citations = [{"doc": d, "page": p, "section": where[(d, p)][0], "chunk_id": where[(d, p)][1]}
                     for d, p in valid]
        return Answer(question, text, False, None, citations, [], backend="openai")
