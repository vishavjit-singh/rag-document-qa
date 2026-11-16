"""Text embedders behind one small interface.

- ``lsa`` (default): TF-IDF on words, bigrams and character n-grams, reduced with TruncatedSVD (latent semantic
  analysis). Deterministic, CPU only, fitted on the corpus, so the index is re-embedded when documents change.
- ``hashing``: hashed word and character n-grams. Stateless, so unchanged chunks keep their vectors on re-ingest.
- ``sentence-transformers``: a pretrained MiniLM model. Optional; installed and exercised in a CI job.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from .text import tokens

log = logging.getLogger(__name__)


def _word_ngrams(text: str) -> list[str]:
    t = tokens(text)
    return t + [f"{a}_{b}" for a, b in zip(t, t[1:])]


class Embedder:
    name = "base"
    stateless = False

    def fit(self, texts: list[str]) -> "Embedder":
        return self

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> "Embedder":
        return self


class LSAEmbedder(Embedder):
    name = "lsa"

    def __init__(self, dims: int = 160, seed: int = 0):
        self.dims, self.seed = dims, seed
        self.word = self.char = self.svd = None

    def _sparse(self, texts):
        return sparse.hstack([self.word.transform(texts), 0.6 * self.char.transform(texts)]).tocsr()

    def fit(self, texts):
        from sklearn.decomposition import TruncatedSVD

        self.word = TfidfVectorizer(analyzer=_word_ngrams, sublinear_tf=True, min_df=1).fit(texts)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2,
                                    lowercase=True).fit(texts)
        x = self._sparse(texts)
        k = max(2, min(self.dims, x.shape[0] - 1, x.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, algorithm="randomized", n_iter=8, random_state=self.seed).fit(x)
        log.info("LSA fitted on %d texts: %d word + %d char features -> %d dims (%.0f%% variance)", len(texts),
                 len(self.word.vocabulary_), len(self.char.vocabulary_), k,
                 100 * self.svd.explained_variance_ratio_.sum())
        return self

    def encode(self, texts):
        return normalize(self.svd.transform(self._sparse(list(texts)))).astype(np.float32)

    def save(self, path):
        import joblib

        joblib.dump({"word": self.word, "char": self.char, "svd": self.svd, "dims": self.dims}, path / "lsa.joblib")

    def load(self, path):
        import joblib

        d = joblib.load(path / "lsa.joblib")
        self.word, self.char, self.svd, self.dims = d["word"], d["char"], d["svd"], d["dims"]
        return self


class HashingEmbedder(Embedder):
    name = "hashing"
    stateless = True

    def __init__(self, n_features: int = 2048):
        self.word = HashingVectorizer(analyzer=_word_ngrams, n_features=n_features, alternate_sign=False,
                                      norm="l2")
        self.char = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=n_features,
                                      alternate_sign=False, norm="l2")

    def encode(self, texts):
        texts = list(texts)
        x = sparse.hstack([self.word.transform(texts), 0.6 * self.char.transform(texts)])
        return normalize(x).toarray().astype(np.float32)


class SentenceTransformerEmbedder(Embedder):
    name = "sentence-transformers"
    stateless = True

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.model_name = model
        self.model = SentenceTransformer(model, device="cpu")

    def encode(self, texts):
        return np.asarray(self.model.encode(list(texts), batch_size=32, normalize_embeddings=True,
                                            show_progress_bar=False), dtype=np.float32)


def make_embedder(name: str, **cfg) -> Embedder:
    if name == "lsa":
        return LSAEmbedder(dims=cfg.get("lsa_dims", 160))
    if name == "hashing":
        return HashingEmbedder(n_features=cfg.get("hashing_features", 2048))
    if name in ("sentence-transformers", "st"):
        return SentenceTransformerEmbedder(cfg.get("st_model", "sentence-transformers/all-MiniLM-L6-v2"))
    raise ValueError(f"unknown embedder {name!r} (lsa, hashing, sentence-transformers)")
