"""Evaluation against the gold QA set.

Retrieval: recall@k, MRR@10 and nDCG@10 for BM25, dense, hybrid and the version-aware variants. A chunk is
relevant when it covers the gold page of the gold document. Answers: the extractive answerer's abstention
threshold is chosen on a dev split and the answer metrics are reported on the held-out test split.
Also: retrieval latency, a chunk-size sweep and pdfplumber vs pypdf extraction.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np

from . import store as store_mod
from .answer import ExtractiveAnswerer
from .config import with_overrides
from .retrieve import Retriever
from .text import normalise

log = logging.getLogger(__name__)


def load_gold(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def split(gold: list[dict], dev_fraction: float = 0.5, seed: int = 7) -> tuple[set[str], set[str]]:
    """Stratified by answerable / unanswerable, deterministic."""
    rng = random.Random(seed)
    dev = set()
    for flag in (True, False):
        ids = sorted(q["id"] for q in gold if q["answerable"] is flag)
        rng.shuffle(ids)
        dev.update(ids[: round(len(ids) * dev_fraction)])
    return dev, {q["id"] for q in gold} - dev


def is_relevant(chunk: dict, q: dict) -> bool:
    locs = [{"doc": chunk["doc"], "pages": chunk["pages"]}] + chunk.get("also_in", [])
    return any(loc["doc"] == q["doc"] and q["page"] in loc["pages"] for loc in locs)


def ranking_metrics(rels: list[list[bool]], n_rel: list[int], ks=(1, 3, 5, 10)) -> dict:
    out = {f"recall@{k}": float(np.mean([any(r[:k]) for r in rels])) for k in ks}
    out["mrr@10"] = float(np.mean([next((1 / (i + 1) for i, x in enumerate(r[:10]) if x), 0.0) for r in rels]))
    nd = []
    for r, n in zip(rels, n_rel):
        dcg = sum(1 / math.log2(i + 2) for i, x in enumerate(r[:10]) if x)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(n, 10)))
        nd.append(dcg / idcg if idcg else 0.0)
    out["ndcg@10"] = float(np.mean(nd))
    return {k: round(v, 4) for k, v in out.items()}


def retrieval_configs(embedder_name: str) -> list[tuple[str, dict]]:
    dense = {"lsa": "Dense (LSA)", "hashing": "Dense (hashed n-grams)",
             "sentence-transformers": "Dense (MiniLM)"}.get(embedder_name, "Dense")
    return [("BM25", {"mode": "bm25", "version_aware": False}),
            (dense, {"mode": "dense", "version_aware": False}),
            ("Hybrid (RRF)", {"mode": "hybrid", "version_aware": False}),
            ("Hybrid + version-aware", {"mode": "hybrid", "version_aware": True}),
            ("Hybrid + version-aware + MMR", {"mode": "hybrid", "version_aware": True, "use_mmr": True})]


def eval_retrieval(store, gold: list[dict], cfg: dict, configs=None, verbose: bool = True) -> dict:
    r = cfg["retrieval"]
    ret = Retriever(store, r["candidates"], r["rrf_k"])
    qs = [q for q in gold if q["answerable"]]
    n_rel = [sum(is_relevant(c, q) for c in store.chunks) for q in qs]
    ks = cfg["eval"]["ks"]
    out = {}
    for name, kw in configs or retrieval_configs(store.embedder.name):
        rels, lat, by_q = [], [], {}
        for q in qs:
            t0 = time.perf_counter()
            hits = ret.search(q["question"], k=10, mmr_lambda=r["mmr_lambda"], **kw)
            lat.append(1000 * (time.perf_counter() - t0))
            rel = [is_relevant(h.chunk, q) for h in hits]
            rels.append(rel)
            by_q[q["id"]] = rel.index(True) + 1 if True in rel else None
        m = ranking_metrics(rels, n_rel, ks)
        ver = [i for i, q in enumerate(qs) if q["doc"] in _versioned_docs(store)]
        m["version_questions_recall@1"] = round(float(np.mean([any(rels[i][:1]) for i in ver])), 4) if ver else None
        m["latency_ms_p50"] = round(float(np.percentile(lat, 50)), 2)
        m["latency_ms_p95"] = round(float(np.percentile(lat, 95)), 2)
        m["first_relevant_rank"] = by_q
        out[name] = {"settings": kw, **m}
        if verbose:
            log.info("%-30s R@1 %.3f  R@5 %.3f  MRR %.3f  nDCG %.3f", name, m["recall@1"], m.get("recall@5", 0),
                     m["mrr@10"], m["ndcg@10"])
    return out


def _versioned_docs(store) -> set[str]:
    return {d for docs in store.families.values() if len(docs) > 1 for d in docs}


def _span_hit(answer: str, span: str) -> bool:
    return normalise(span).lower() in normalise(answer).lower()


def score_answers(store, gold: list[dict], cfg: dict) -> list[dict]:
    """Best candidate sentence and its confidence for every question (threshold applied later)."""
    r = cfg["retrieval"]
    ret = Retriever(store, r["candidates"], r["rrf_k"])
    ans = ExtractiveAnswerer(store)
    rows = []
    for q in gold:
        t0 = time.perf_counter()
        hits = ret.search(q["question"], r["k"], r["mode"], r["version_aware"], r["mmr"], r["mmr_lambda"])
        a = ans.answer(q["question"], hits, threshold=-1.0)
        ms = 1000 * (time.perf_counter() - t0)
        cite = a.citations[0] if a.citations else {}
        rows.append({"id": q["id"], "question": q["question"], "answerable": q["answerable"],
                     "gold_doc": q.get("doc"), "gold_page": q.get("page"), "gold_answer": q.get("answer"),
                     "answer": a.answer, "confidence": a.confidence, "doc": cite.get("doc"), "page": cite.get("page"),
                     "span_hit": bool(q["answerable"] and _span_hit(a.answer, q["answer"])),
                     "citation_hit": bool(q["answerable"] and cite.get("doc") == q["doc"]
                                          and cite.get("page") == q["page"]),
                     "latency_ms": round(ms, 2)})
    return rows


def decide(rows: list[dict], threshold: float) -> list[bool]:
    """True when the outcome is right: a correct span for answerable questions, an abstention otherwise."""
    return [(r["confidence"] >= threshold and r["span_hit"]) if r["answerable"] else r["confidence"] < threshold
            for r in rows]


def calibrate(rows: list[dict]) -> float:
    """Threshold with the most right outcomes on these rows (midpoint of the best range of cut points)."""
    conf = sorted({r["confidence"] for r in rows})
    cuts = [0.0] + [(a + b) / 2 for a, b in zip(conf, conf[1:])] + [conf[-1] + 1e-6]
    acc = [sum(decide(rows, t)) for t in cuts]
    best = [t for t, a in zip(cuts, acc) if a == max(acc)]
    return round(float(best[len(best) // 2]), 4)


def answer_metrics(rows: list[dict], threshold: float) -> dict:
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    abst = [r for r in rows if r["confidence"] < threshold]
    m = {
        "questions": len(rows), "answerable": len(ans), "unanswerable": len(una),
        "answered_rate": sum(r["confidence"] >= threshold for r in ans) / max(1, len(ans)),
        "span_accuracy": sum(r["confidence"] >= threshold and r["span_hit"] for r in ans) / max(1, len(ans)),
        "citation_accuracy": sum(r["confidence"] >= threshold and r["citation_hit"] for r in ans) / max(1, len(ans)),
        "span_accuracy_no_abstention": sum(r["span_hit"] for r in ans) / max(1, len(ans)),
        "abstention_recall": sum(r["confidence"] < threshold for r in una) / max(1, len(una)),
        "abstention_precision": (sum(not r["answerable"] for r in abst) / len(abst)) if abst else None,
        "decision_accuracy": sum(decide(rows, threshold)) / max(1, len(rows)),
    }
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}


def threshold_curve(rows: list[dict], points: int = 41) -> list[dict]:
    out = []
    for t in np.linspace(0.2, 0.8, points):
        m = answer_metrics(rows, float(t))
        out.append({"threshold": round(float(t), 3), "span_accuracy": m["span_accuracy"],
                    "abstention_recall": m["abstention_recall"], "decision_accuracy": m["decision_accuracy"]})
    return out


def chunk_sweep(docs, gold: list[dict], cfg: dict, sizes: list[int]) -> list[dict]:
    out = []
    va = [("Hybrid + version-aware", {"mode": "hybrid", "version_aware": True})]
    for size in sizes:
        c = with_overrides(cfg, chunking={"size": size, "overlap": max(5, size // 4)})
        st, stats = store_mod.build_store(docs, c)
        m = eval_retrieval(st, gold, c, va, verbose=False)["Hybrid + version-aware"]
        rows = score_answers(st, [q for q in gold if q["answerable"]], c)
        out.append({"size": size, "overlap": max(5, size // 4), "chunks": stats["chunks"],
                    "recall@1": m["recall@1"], "recall@5": m.get("recall@5"), "mrr@10": m["mrr@10"],
                    "span_accuracy_no_abstention": round(float(np.mean([r["span_hit"] for r in rows])), 4)})
        log.info("chunk size %d: %d chunks, R@1 %.3f, R@5 %.3f, answer span %.3f", size, stats["chunks"],
                 m["recall@1"], m.get("recall@5", 0), out[-1]["span_accuracy_no_abstention"])
    return out


def engine_comparison(paths: list[Path], index_dir: Path, gold: list[dict], cfg: dict) -> list[dict]:
    out = []
    va = [("Hybrid + version-aware", {"mode": "hybrid", "version_aware": True})]
    for engine in ("pdfplumber", "pypdf"):
        docs, _ = store_mod.load_extracted(paths, index_dir, engine)
        c = with_overrides(cfg, extract={"engine": engine})
        st, stats = store_mod.build_store(docs, c)
        m = eval_retrieval(st, gold, c, va, verbose=False)["Hybrid + version-aware"]
        rows = score_answers(st, [q for q in gold if q["answerable"]], c)
        out.append({"engine": engine, "chunks": stats["chunks"], "recall@1": m["recall@1"],
                    "recall@5": m.get("recall@5"), "mrr@10": m["mrr@10"],
                    "span_accuracy_no_abstention": round(float(np.mean([r["span_hit"] for r in rows])), 4)})
    return out


def run(cfg: dict, sweep: bool = True) -> dict:
    t0 = time.perf_counter()
    paths = cfg["paths"]
    gold = load_gold(paths["gold"])
    store = store_mod.load(paths["index"], cfg)
    dev, test = split(gold, cfg["eval"]["dev_fraction"], cfg["eval"]["seed"])

    retrieval = eval_retrieval(store, gold, cfg)
    rows = score_answers(store, gold, cfg)
    dev_rows = [r for r in rows if r["id"] in dev]
    test_rows = [r for r in rows if r["id"] in test]
    th = calibrate(dev_rows)
    (Path(paths["index"]) / "calibration.json").write_text(json.dumps(
        {"threshold": th, "dev_questions": len(dev_rows), "objective": "right outcomes on the dev split"}, indent=2))
    answers = {"threshold": th, "dev": answer_metrics(dev_rows, th), "test": answer_metrics(test_rows, th),
               "all": answer_metrics(rows, th), "threshold_curve_dev": threshold_curve(dev_rows),
               "ask_latency_ms_p50": round(float(np.percentile([r["latency_ms"] for r in rows], 50)), 2),
               "ask_latency_ms_p95": round(float(np.percentile([r["latency_ms"] for r in rows], 95)), 2)}
    log.info("threshold %.3f (dev) | test: span acc %.3f, citation acc %.3f, abstention P %s R %.3f", th,
             answers["test"]["span_accuracy"], answers["test"]["citation_accuracy"],
             answers["test"]["abstention_precision"], answers["test"]["abstention_recall"])

    reports = Path(paths["reports"])
    reports.mkdir(parents=True, exist_ok=True)
    for r in rows:
        r["split"] = "dev" if r["id"] in dev else "test"
        r["abstained"] = r["confidence"] < th
    with (reports / "answers.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    corpus_paths = sorted(Path(paths["corpus"]).glob("*.pdf"))
    result = {
        "corpus": {"documents": len(store.manifest["files"]),
                   "pages": sum(f["pages"] for f in store.manifest["files"].values()),
                   "chunks": len(store.chunks), "embedder": store.embedder.name, "index_backend": store.index.backend,
                   "dim": int(store.vectors.shape[1]), **{k: store.manifest.get("last_ingest", {}).get(k) for k in
                                                          ("chunks_before_dedup", "duplicates_dropped")}},
        "gold": {"questions": len(gold), "answerable": sum(q["answerable"] for q in gold),
                 "unanswerable": sum(not q["answerable"] for q in gold),
                 "version_questions": sum(1 for q in gold if q["answerable"] and q["doc"] in _versioned_docs(store)),
                 "dev": len(dev), "test": len(test)},
        "settings": {"chunking": cfg["chunking"], "retrieval": cfg["retrieval"]},
        "retrieval": retrieval, "answers": answers,
    }
    if sweep:
        docs, _ = store_mod.load_extracted(corpus_paths, Path(paths["index"]), cfg["extract"]["engine"])
        result["chunk_sweep"] = chunk_sweep(docs, gold, cfg, cfg["eval"]["chunk_sizes"])
        result["extraction"] = engine_comparison(corpus_paths, Path(paths["index"]), gold, cfg)
    result["seconds"] = round(time.perf_counter() - t0, 1)
    (reports / "eval.json").write_text(json.dumps(result, indent=2))
    return result
