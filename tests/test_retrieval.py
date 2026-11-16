import math
import os

import numpy as np
import pytest

from ragqa.bm25 import BM25
from ragqa.embed import HashingEmbedder, LSAEmbedder
from ragqa.retrieve import Retriever, mmr, rrf
from ragqa.vindex import NumpyIndex, faiss_available, make_index

REQUIRE_FAISS = os.environ.get("RAGQA_REQUIRE_FAISS") == "1"
needs_faiss = pytest.mark.skipif(not faiss_available() and not REQUIRE_FAISS, reason="faiss not installed (CI job)")


def test_bm25_matches_the_formula():
    docs = ["meal allowance per day", "hotel cap per night", "meal meal allowance"]
    bm = BM25(docs, k1=1.2, b=0.75)
    idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    avgdl = (3 + 3 + 3) / 3            # "per" is a stopword: 3 tokens each
    tf, dl = 2, 3
    expected = idf * tf * 2.2 / (tf + 1.2 * (1 - 0.75 + 0.75 * dl / avgdl))
    assert bm.scores("meal")[2] == pytest.approx(expected)
    assert [i for i, _ in bm.top("meal allowance", 3)] == [2, 0]


def test_rrf_and_mmr():
    fused = rrf([[1, 2, 3], [3, 1]], k=60)
    assert fused[1] == pytest.approx(1 / 61 + 1 / 62) and fused[2] == pytest.approx(1 / 62)
    assert max(fused, key=fused.get) == 1
    v = np.array([[1, 0], [0.99, 0.141], [0, 1]], dtype=np.float32)
    assert mmr([0, 1, 2], np.array([1.0, 0.9, 0.5]), v, k=2, lam=0.5) == [0, 2]   # skips the near-duplicate


def test_numpy_index_is_exact():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 16)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    idx = NumpyIndex(16)
    idx.add(x)
    s, i = idx.search(x[:5], 3)
    assert (i[:, 0] == np.arange(5)).all() and np.allclose(s[:, 0], 1, atol=1e-5)
    assert make_index(16, "numpy").backend == "numpy"


@needs_faiss
def test_faiss_matches_numpy_on_the_demo_index(index, gold):
    flat = make_index(index.vectors.shape[1], "faiss")
    flat.add(index.vectors)
    ref = NumpyIndex(index.vectors.shape[1])
    ref.add(index.vectors)
    q = index.embedder.encode([g["question"] for g in gold])
    s1, i1 = flat.search(q, 10)
    s2, i2 = ref.search(q, 10)
    assert np.allclose(s1, s2, atol=1e-4)
    assert (i1[:, 0] == i2[:, 0]).mean() > 0.98      # identical apart from exact score ties


@needs_faiss
def test_faiss_ivf_path():
    from ragqa.vindex import FaissIndex

    rng = np.random.default_rng(1)
    x = rng.normal(size=(3000, 32)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    ivf = FaissIndex(32, ivf_min=1000)
    ivf.add(x)
    assert type(ivf.index).__name__ == "IndexIVFFlat"
    ivf.index.nprobe = ivf.index.nlist                # search every list -> exact
    _, i = ivf.search(x[:20], 1)
    assert (i[:, 0] == np.arange(20)).all()


def test_lsa_is_deterministic_and_normalised():
    texts = ["meal allowance per day", "hotel cap per night", "forklift speed limit", "password length rules"] * 3
    a = LSAEmbedder(dims=3).fit(texts).encode(texts[:4])
    b = LSAEmbedder(dims=3).fit(texts).encode(texts[:4])
    assert np.allclose(a, b) and np.allclose(np.linalg.norm(a, axis=1), 1, atol=1e-5)


def test_hashing_embedder_is_stateless():
    a = HashingEmbedder().encode(["USB storage is blocked"])
    b = HashingEmbedder().encode(["USB storage is blocked"])
    assert np.array_equal(a, b) and a.shape == (1, 4096)


@pytest.mark.skipif(os.environ.get("RAGQA_REQUIRE_ST") != "1", reason="sentence-transformers runs in a CI job")
def test_sentence_transformers_backend():
    from ragqa.embed import make_embedder

    e = make_embedder("sentence-transformers")
    v = e.encode(["How many sick days do I get?", "Employees receive 12 days of paid sick leave.", "Forklift speed"])
    assert v.shape == (3, 384) and v[0] @ v[1] > v[0] @ v[2]


def test_version_aware_ranking(index):
    r = Retriever(index)
    plain = r.search("What is the hotel cap per night in a Tier 1 city?", k=3, version_aware=True)
    assert plain[0].chunk["doc"] == "travel-expense-policy-2025.pdf" and not plain[0].demoted
    old = r.search("Under the 2023 travel policy, what was the hotel cap per night in a Tier 1 city?", k=3)
    assert old[0].chunk["doc"] == "travel-expense-policy-2023.pdf"
    assert r.mentioned_docs("the 2024 SLA (version 1.0)") >= {"track-sla-2024.pdf"}
    aware = r.search("What is the hotel cap per night in a Tier 1 city?", k=30, mode="bm25", version_aware=True)
    olds = [i for i, h in enumerate(aware) if h.chunk["status"] == "Superseded"]
    news = [i for i, h in enumerate(aware) if h.chunk["status"] == "Current"]
    assert olds and max(news) < min(olds)


def test_hits_carry_per_retriever_ranks(index):
    hits = Retriever(index).search("forklift speed limit indoors", k=3, mode="hybrid")
    assert hits[0].chunk["doc"] == "warehouse-safety-procedures.pdf"
    assert set(hits[0].ranks) <= {"bm25", "dense"} and hits[0].ranks
    with pytest.raises(ValueError):
        Retriever(index).search("x", mode="sparse")
