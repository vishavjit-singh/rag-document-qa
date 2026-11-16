"""Vector index behind one interface: FAISS when installed, exact NumPy inner product otherwise.

Vectors are L2-normalised, so inner product equals cosine similarity. FAISS uses ``IndexFlatIP`` (exact) for
small collections and ``IndexIVFFlat`` (approximate, ``nprobe`` lists searched) from ``ivf_min`` vectors up.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
    except ImportError:
        return False
    return True


class NumpyIndex:
    backend = "numpy"

    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        self.vectors = np.vstack([self.vectors, np.asarray(vectors, dtype=np.float32)])

    @property
    def size(self) -> int:
        return len(self.vectors)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.atleast_2d(np.asarray(queries, dtype=np.float32))
        k = min(k, self.size)
        sims = q @ self.vectors.T
        idx = np.argsort(-sims, axis=1, kind="stable")[:, :k]
        return np.take_along_axis(sims, idx, axis=1), idx


class FaissIndex:
    backend = "faiss"

    def __init__(self, dim: int, ivf_min: int = 4096, nprobe: int = 8):
        import faiss

        self.faiss, self.dim, self.ivf_min, self.nprobe = faiss, dim, ivf_min, nprobe
        self.index = None

    def add(self, vectors: np.ndarray) -> None:
        x = np.ascontiguousarray(vectors, dtype=np.float32)
        f = self.faiss
        if len(x) >= self.ivf_min:
            nlist = int(4 * np.sqrt(len(x)))
            quantizer = f.IndexFlatIP(self.dim)
            self.index = f.IndexIVFFlat(quantizer, self.dim, nlist, f.METRIC_INNER_PRODUCT)
            self.index.train(x)
            self.index.nprobe = self.nprobe
            log.info("FAISS IVF index: %d lists, nprobe=%d", nlist, self.nprobe)
        else:
            self.index = f.IndexFlatIP(self.dim)
        self.index.add(x)

    @property
    def size(self) -> int:
        return 0 if self.index is None else int(self.index.ntotal)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(np.atleast_2d(queries), dtype=np.float32)
        return self.index.search(q, min(k, self.size))


def make_index(dim: int, backend: str = "auto", **kw):
    if backend == "faiss" or (backend == "auto" and faiss_available()):
        if not faiss_available():
            raise RuntimeError("index backend 'faiss' requested but faiss is not installed (pip install faiss-cpu)")
        return FaissIndex(dim, **kw)
    if backend not in ("auto", "numpy"):
        raise ValueError(f"unknown index backend {backend!r} (auto, faiss, numpy)")
    return NumpyIndex(dim)
