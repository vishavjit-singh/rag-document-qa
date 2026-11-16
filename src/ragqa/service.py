"""The question-answering engine shared by the CLI and both API apps."""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from . import config as config_mod
from . import store as store_mod
from .answer import ExtractiveAnswerer
from .retrieve import Retriever

log = logging.getLogger(__name__)


def calibrated_threshold(index_dir: Path, default: float) -> tuple[float, str]:
    p = Path(index_dir) / "calibration.json"
    if p.exists():
        return float(json.loads(p.read_text())["threshold"]), "calibrated on the dev split"
    return float(default), "ragqa.yaml default"


class RagEngine:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or config_mod.load()
        self.lock = threading.RLock()
        self.store = None
        self.reload()

    @property
    def index_dir(self) -> Path:
        return Path(self.cfg["paths"]["index"])

    @property
    def corpus_dir(self) -> Path:
        return Path(self.cfg["paths"]["corpus"])

    def reload(self) -> None:
        with self.lock:
            self.store = store_mod.load(self.index_dir, self.cfg)
            r = self.cfg["retrieval"]
            self.retriever = Retriever(self.store, r["candidates"], r["rrf_k"])
            self.threshold, self.threshold_source = calibrated_threshold(self.index_dir,
                                                                         self.cfg["answer"]["threshold"])
            self.extractive = ExtractiveAnswerer(self.store, self.threshold)

    def answerer(self, backend: str | None = None):
        backend = backend or self.cfg["answer"]["backend"]
        if backend == "extractive":
            return self.extractive
        if backend == "openai":
            from .llm import OpenAIAnswerer

            return OpenAIAnswerer(self.cfg["answer"].get("openai_model", "gpt-4o-mini"))
        raise ValueError(f"unknown answer backend {backend!r}")

    def ask(self, question: str, k: int | None = None, mode: str | None = None, backend: str | None = None,
            version_aware: bool | None = None, mmr: bool | None = None) -> dict:
        r = self.cfg["retrieval"]
        t0 = time.perf_counter()
        with self.lock:
            hits = self.retriever.search(question, k or r["k"], mode or r["mode"],
                                         r["version_aware"] if version_aware is None else version_aware,
                                         r["mmr"] if mmr is None else mmr, r["mmr_lambda"])
            t1 = time.perf_counter()
            ans = self.answerer(backend).answer(question, hits)
        t2 = time.perf_counter()
        out = ans.to_dict()
        out.update(answer_with_citations=ans.text_with_citations, mode=mode or r["mode"],
                   threshold=self.threshold if ans.backend == "extractive" else None,
                   retrieved=[{k_: v for k_, v in h.to_dict().items() if k_ != "text"} for h in hits],
                   latency_ms={"retrieval": round(1000 * (t1 - t0), 2), "answer": round(1000 * (t2 - t1), 2)})
        return out

    def ingest(self, rebuild: bool = False) -> dict:
        with self.lock:
            stats = store_mod.ingest(self.corpus_dir, self.index_dir, self.cfg, rebuild=rebuild)
            if not stats.get("up_to_date"):
                self.reload()
            return stats

    def documents(self) -> list[dict]:
        return self.store.documents

    def health(self) -> dict:
        s = self.store
        return {"status": "ok", "documents": len(s.manifest.get("files", {})), "chunks": len(s.chunks),
                "embedder": s.embedder.name, "index_backend": s.index.backend,
                "answer_backend": self.cfg["answer"]["backend"], "threshold": self.threshold,
                "threshold_source": self.threshold_source}
