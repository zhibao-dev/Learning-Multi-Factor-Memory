"""
Text embedding utilities.

Two pluggable embedders, plus a cosine-similarity helper:

  - ``hash_embed`` — deterministic SHA1 bag-of-tokens embedding (zero deps,
    64-dim). Fast and reproducible; captures lexical-overlap similarity.
  - ``SBertEmbedder`` — lazy sentence-transformers wrapper for real semantic
    embeddings.

An Embedder is anything callable ``text -> list[float]``.
"""

from __future__ import annotations

import hashlib
import math
from typing import Callable


# ── Default hyperparameters ───────────────────────────────────────────────

DEFAULT_DIM = 64    # embedding dimension


# ── Embedding ─────────────────────────────────────────────────────────────

# An Embedder is anything callable text → list[float]. The two built-in
# implementations are `hash_embed` (default, dep-free) and `SBertEmbedder`
# (lazy sentence-transformers wrapper for real semantics).
Embedder = Callable[[str], list[float]]


def hash_embed(text: str, dim: int = DEFAULT_DIM) -> list[float]:
    """
    Deterministic bag-of-tokens hash embedding, L2-normalised.

    Each whitespace-split token is SHA1-hashed; each hash byte contributes
    a signed unit to one of `dim` dimensions. Sum and L2-normalise. This
    captures lexical-overlap similarity without external dependencies.
    """
    tokens = [t for t in (text or "").lower().split() if t]
    if not tokens:
        return [0.0] * dim
    vec = [0.0] * dim
    for tok in tokens:
        h = hashlib.sha1(tok.encode("utf-8")).digest()
        for i in range(dim):
            b = h[i % len(h)]
            vec[i] += (b / 255.0 - 0.5) * 2.0
    norm = math.sqrt(sum(x * x for x in vec))
    return vec if norm < 1e-9 else [x / norm for x in vec]


class SBertEmbedder:
    """
    Sentence-Transformers wrapper, lazy-loaded.

    Encodes text into a semantic embedding using a chosen SBERT model. The
    model is downloaded on first call and cached on the instance (and at the
    class level, so multiple instances share weights).

    Optional dependency: `pip install lmfm[sbert]` installs
    `sentence-transformers>=3.0`. We do NOT import sentence-transformers
    at module import time; the import happens only when the embedder is
    first invoked.
    """

    # Class-level cache so multiple SBertEmbedder() share weights.
    _model_cache: dict = {}

    # Default model is the standard small-fast SBERT (~22M params, 384-dim,
    # English). For richer semantics swap to "all-mpnet-base-v2".
    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None  # lazy

    def _ensure_loaded(self):
        if self._model is not None:
            return
        cached = type(self)._model_cache.get(self.model_name)
        if cached is not None:
            self._model = cached
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "SBertEmbedder requires sentence-transformers. "
                "Install via `pip install lmfm[sbert]`."
            ) from e
        self._model = SentenceTransformer(self.model_name)
        type(self)._model_cache[self.model_name] = self._model

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        assert self._model is not None
        return int(self._model.get_sentence_embedding_dimension() or DEFAULT_DIM)

    def __call__(self, text: str) -> list[float]:
        self._ensure_loaded()
        assert self._model is not None
        # sentence-transformers returns np.ndarray; convert to list[float]
        # for serialisation compatibility with the existing JSON store.
        vec = self._model.encode(text or "", normalize_embeddings=True)
        return [float(x) for x in vec.tolist()]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity ∈ [-1, 1]. Returns 0 for empty vectors."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot / (na * nb)
