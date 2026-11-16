"""Ingestion and the on-disk index.

``ingest`` is incremental: every PDF is identified by its SHA-256. Extraction results are cached per file hash,
so only new or changed files are parsed again; deleted files drop out of the index. If no file and no setting
changed, nothing is rewritten. With a stateless embedder, chunks whose text did not change keep their vectors.

Layout of ``index_dir``::

    manifest.json          files (hash, pages, metadata, chunk count), settings, last ingest stats
    extracted/<sha>.json   cached extraction per file and engine
    chunks.jsonl           de-duplicated chunks with doc, section, pages and per-line page numbers
    vectors.npy            one L2-normalised embedding per chunk
    lsa.joblib             fitted embedder state (LSA only)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bm25 import BM25
from .chunking import chunk_document, dedupe, embed_text
from .embed import Embedder, make_embedder
from .extract import ExtractedDoc, extract
from .text import file_hash, sentences
from .vindex import make_index

log = logging.getLogger(__name__)


def settings_key(cfg: dict) -> dict:
    e = cfg["embedding"]
    return {"engine": cfg["extract"]["engine"], "chunk_size": cfg["chunking"]["size"],
            "overlap": cfg["chunking"]["overlap"], "embedder": e["backend"],
            "embedder_params": {"lsa": e.get("lsa_dims"), "hashing": e.get("hashing_features"),
                                "sentence-transformers": e.get("st_model")}.get(e["backend"])}


@dataclass
class Store:
    chunks: list[dict]
    vectors: np.ndarray
    embedder: Embedder
    index_backend: str = "auto"
    manifest: dict = field(default_factory=dict)

    def __post_init__(self):
        self.bm25 = BM25([embed_text(c) for c in self.chunks])
        self.index = make_index(self.vectors.shape[1], self.index_backend)
        if len(self.vectors):
            self.index.add(self.vectors)
        self.families: dict[str, set[str]] = {}          # document code -> files (versions of one document)
        files = self.manifest.get("files") or {c["doc"]: c for c in self.chunks}
        for name, info in files.items():
            self.families.setdefault(info.get("code") or name, set()).add(name)

    @property
    def documents(self) -> list[dict]:
        files = self.manifest.get("files", {})
        return [{"file": name, **{k: v for k, v in info.items() if k != "sha256"}, "sha256": info["sha256"][:12]}
                for name, info in sorted(files.items())]


def units_for_fit(chunks: list[dict]) -> list[str]:
    """LSA is fitted on sentence-level units plus whole chunks, which gives the SVD enough rows."""
    seen, out = set(), []
    for c in chunks:
        for _, line, section, is_heading in c["lines"]:
            for s in ([] if is_heading else sentences(line)):
                if s not in seen:
                    seen.add(s)
                    out.append(f"{section}: {s}")
        out.append(embed_text(c))
    return out


def build_store(docs: list[ExtractedDoc], cfg: dict, embedder: Embedder | None = None,
                reuse: dict[str, np.ndarray] | None = None, fit: bool = True) -> tuple[Store, dict]:
    """Chunk, de-duplicate and embed already-extracted documents (no disk writes)."""
    ch = cfg["chunking"]
    chunks: list[dict] = []
    for d in sorted(docs, key=lambda d: d.file):
        chunks += chunk_document(d, ch["size"], ch["overlap"])
    total = len(chunks)
    chunks, dropped = dedupe(chunks)
    emb = embedder or make_embedder(cfg["embedding"]["backend"], **cfg["embedding"])
    texts = [embed_text(c) for c in chunks]
    if emb.stateless and reuse:
        keys = [c["id"] + c["hash"] for c in chunks]
        missing = [i for i, k in enumerate(keys) if k not in reuse]
        new = dict(zip(missing, emb.encode([texts[i] for i in missing]))) if missing else {}
        vecs = np.stack([new[i] if i in new else reuse[k] for i, k in enumerate(keys)]).astype(np.float32)
        embedded = len(missing)
    else:
        if not emb.stateless and fit:
            emb.fit(units_for_fit(chunks))
        vecs = emb.encode(texts)
        embedded = len(chunks)
    files = {d.file: {"sha256": d.sha256, "pages": d.pages, "title": d.meta.get("title"),
                      "code": d.meta.get("code"), "version": d.meta.get("version"),
                      "status": d.meta.get("status"), "effective": d.meta.get("effective"),
                      "chunks": sum(1 for c in chunks if c["doc"] == d.file),
                      "header_footer_lines_removed": d.stripped_lines} for d in docs}
    manifest = {"files": files, "settings": settings_key(cfg)}
    stats = {"chunks_before_dedup": total, "duplicates_dropped": dropped, "chunks": len(chunks),
             "chunks_embedded": embedded, "dim": int(vecs.shape[1])}
    return Store(chunks, vecs, emb, cfg["index"]["backend"], manifest), stats


def _cache_path(index_dir: Path, sha: str, engine: str) -> Path:
    return index_dir / "extracted" / f"{sha[:24]}-{engine}.json"


def load_extracted(paths: list[Path], index_dir: Path, engine: str) -> tuple[list[ExtractedDoc], int]:
    docs, parsed = [], 0
    (index_dir / "extracted").mkdir(parents=True, exist_ok=True)
    for p in paths:
        cache = _cache_path(index_dir, file_hash(p), engine)
        if cache.exists():
            d = ExtractedDoc.from_dict(json.loads(cache.read_text()))
            d.file = p.name
        else:
            d = extract(p, engine)
            cache.write_text(json.dumps(d.to_dict()))
            parsed += 1
        docs.append(d)
    return docs, parsed


def ingest(corpus_dir: Path, index_dir: Path, cfg: dict, rebuild: bool = False) -> dict:
    t0 = time.perf_counter()
    corpus_dir, index_dir = Path(corpus_dir), Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(corpus_dir.glob("*.pdf"))
    old = json.loads((index_dir / "manifest.json").read_text()) if (index_dir / "manifest.json").exists() else {}
    old_files = old.get("files", {})
    hashes = {p.name: file_hash(p) for p in paths}
    added = sorted(n for n in hashes if n not in old_files)
    changed = sorted(n for n in hashes if n in old_files and old_files[n]["sha256"] != hashes[n])
    removed = sorted(n for n in old_files if n not in hashes)
    same_settings = old.get("settings") == settings_key(cfg)
    stats = {"files": len(paths), "added": added, "changed": changed, "removed": removed,
             "unchanged": len(paths) - len(added) - len(changed)}
    if not (added or changed or removed or rebuild) and same_settings and (index_dir / "chunks.jsonl").exists():
        stats.update(up_to_date=True, parsed=0, seconds=round(time.perf_counter() - t0, 2))
        log.info("index is up to date (%d files)", len(paths))
        return stats
    if rebuild:
        for f in (index_dir / "extracted").glob("*.json"):
            f.unlink()
    engine = cfg["extract"]["engine"]
    for name in removed + changed:                       # drop stale extraction caches
        _cache_path(index_dir, old_files[name]["sha256"], engine).unlink(missing_ok=True)
    docs, parsed = load_extracted(paths, index_dir, engine)

    reuse = None
    emb = make_embedder(cfg["embedding"]["backend"], **cfg["embedding"])
    if emb.stateless and same_settings and not rebuild and (index_dir / "vectors.npy").exists():
        prev = [json.loads(x) for x in (index_dir / "chunks.jsonl").read_text().splitlines()]
        vecs = np.load(index_dir / "vectors.npy")
        reuse = {c["id"] + c["hash"]: v for c, v in zip(prev, vecs)}
    store, bstats = build_store(docs, cfg, embedder=emb, reuse=reuse)
    save(store, index_dir)
    stats.update(up_to_date=False, parsed=parsed, **bstats, seconds=round(time.perf_counter() - t0, 2))
    store.manifest["last_ingest"] = stats
    (index_dir / "manifest.json").write_text(json.dumps(store.manifest, indent=2))
    log.info("ingested %d files (%d parsed): %d chunks (%d duplicates dropped), %d embedded", len(paths), parsed,
             bstats["chunks"], bstats["duplicates_dropped"], bstats["chunks_embedded"])
    return stats


def save(store: Store, index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    with (index_dir / "chunks.jsonl").open("w") as f:
        for c in store.chunks:
            f.write(json.dumps(c) + "\n")
    np.save(index_dir / "vectors.npy", store.vectors)
    store.embedder.save(index_dir)
    (index_dir / "manifest.json").write_text(json.dumps(store.manifest, indent=2))


def load(index_dir: Path, cfg: dict) -> Store:
    index_dir = Path(index_dir)
    if not (index_dir / "manifest.json").exists():
        raise FileNotFoundError(f"no index in {index_dir}; run `python -m ragqa ingest` first")
    manifest = json.loads((index_dir / "manifest.json").read_text())
    chunks = [json.loads(x) for x in (index_dir / "chunks.jsonl").read_text().splitlines()]
    vectors = np.load(index_dir / "vectors.npy")
    s = manifest["settings"]
    emb_cfg = {**cfg["embedding"], "backend": s["embedder"]}
    emb = make_embedder(s["embedder"], **emb_cfg).load(index_dir)
    return Store(chunks, vectors, emb, cfg["index"]["backend"], manifest)
