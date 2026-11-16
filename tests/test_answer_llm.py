import io
import json
import math

import pytest

from ragqa.answer import NOT_FOUND, ExtractiveAnswerer, expects_quantity
from ragqa.evaluate import calibrate, ranking_metrics
from ragqa.llm import SYSTEM, OpenAIAnswerer
from ragqa.retrieve import Retriever


def _ask(index, q, threshold=0.4):
    hits = Retriever(index).search(q, k=5)
    return ExtractiveAnswerer(index, threshold).answer(q, hits), hits


def test_extractive_answer_cites_the_gold_page(index, gold):
    q = next(g for g in gold if g["id"] == "op01")          # forklift speed limit indoors
    a, _ = _ask(index, q["question"])
    assert not a.abstained and q["answer"] in a.answer
    assert a.citations[0]["doc"] == q["doc"] and a.citations[0]["page"] == q["page"]
    assert a.text_with_citations.endswith(f"[{q['doc']} p.{q['page']}]")
    assert len(a.evidence) == 3 and a.evidence[0]["score"] >= a.evidence[1]["score"]


def test_abstains_below_the_threshold(index):
    a, _ = _ask(index, "Who is the CEO of Kestrelway Logistics?", threshold=0.6)
    assert a.abstained and a.answer == NOT_FOUND and a.citations == []
    b, hits = _ask(index, "Who is the CEO of Kestrelway Logistics?", threshold=0.0)
    assert not b.abstained and ExtractiveAnswerer(index).answer("x", []).abstained


def test_answer_type_detection():
    assert expects_quantity("How many paid sick days per year?")
    assert expects_quantity("Under the 2023 travel policy, what was the mileage rate for a personal car?")
    assert not expects_quantity("What does a steady red light indicate?")


def test_ranking_metrics_by_hand():
    rels = [[False, True, False], [True, False, False], [False, False, False]]
    m = ranking_metrics(rels, [1, 2, 1], ks=(1, 2))
    assert m["recall@1"] == pytest.approx(1 / 3, abs=1e-4) and m["recall@2"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["mrr@10"] == pytest.approx((0.5 + 1 + 0) / 3, abs=1e-4)
    q1 = 1 / math.log2(3)                      # hit at rank 2, one relevant chunk
    q2 = 1 / (1 + 1 / math.log2(3))            # hit at rank 1, two relevant chunks exist
    assert m["ndcg@10"] == pytest.approx((q1 + q2 + 0) / 3, abs=1e-4)


def test_calibration_separates_right_and_wrong():
    rows = [{"answerable": True, "span_hit": True, "confidence": c} for c in (0.7, 0.8, 0.9)] + \
           [{"answerable": False, "span_hit": False, "confidence": c} for c in (0.2, 0.3)]
    t = calibrate(rows)
    assert 0.3 < t < 0.7


class FakeHTTP:
    def __init__(self, content):
        self.content, self.requests = content, []

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": self.content}}]}).encode())


def _hits(index):
    return Retriever(index).search("What is the forklift speed limit indoors?", k=2)


def test_openai_request_carries_context_and_citation_rule(index):
    http = FakeHTTP("Not found in the documents.")
    OpenAIAnswerer(api_key="test", opener=http).answer("Forklift speed?", _hits(index))
    req = http.requests[0]
    body = json.loads(req.data)
    assert req.full_url.endswith("/chat/completions") and req.headers["Authorization"] == "Bearer test"
    assert body["messages"][0]["content"] == SYSTEM and body["temperature"] == 0
    assert "[warehouse-safety-procedures.pdf p.4]" in body["messages"][1]["content"]


def test_openai_citations_are_checked(index):
    hits = _hits(index)
    good = FakeHTTP("The limit is 8 km/h indoors [warehouse-safety-procedures.pdf p.4] [made-up.pdf p.9].")
    a = OpenAIAnswerer(api_key="k", opener=good).answer("Forklift speed?", hits)
    assert not a.abstained and a.citations == [{"doc": "warehouse-safety-procedures.pdf", "page": 4,
                                                 "section": a.citations[0]["section"],
                                                 "chunk_id": a.citations[0]["chunk_id"]}]
    assert "made-up" not in a.answer and a.answer == "The limit is 8 km/h indoors."
    uncited = OpenAIAnswerer(api_key="k", opener=FakeHTTP("It is 8 km/h.")).answer("Forklift speed?", hits)
    assert uncited.abstained and uncited.answer == NOT_FOUND


def test_openai_needs_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        OpenAIAnswerer()
