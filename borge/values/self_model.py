"""
Self Model — Free-Energy formalization of "self" for memory modulation.

We treat the agent's "self" as a generative model `M_self = (μ_self, π_self)`
under the Free Energy Principle (Friston 2010; Limanowski & Blankenburg 2013;
Apps & Tsakiris 2014). It is *not* a body schema or a phenomenological
account — it is the running prior that drives:

  encoding depth :  depth ∝ |V|·A · (1 + λ_e · π_self · sr)
  forget score   :  score ← score × 1/(1 + λ_f · π_self · sr)
  retrieval      :  rank  += w_s · self_similarity(current μ_self, encoded μ_self)

where `sr` is the memory's self-relevance, the cosine similarity between
its embedding and the current μ_self, mapped into [0, 1].

Precision update follows the FEP-inspired rule

  π_self ∝ 1 / (1 + γ · Var[PE])

over a rolling window of prediction errors. Low PE-variance (the agent's
self-prior reliably predicts incoming self-relevant content) → high
precision → strong modulation of memory. High PE-variance (unstable
self-evidence) → low precision → memory falls back toward emotion-only
dynamics. This is the *operational* form of "self-evidencing".

The self model is initialised either:
  - from SOUL.md (`seed_text=` constructor) — bootstrap from value descriptors
  - or from the running stream of conversation (μ_self updated each turn)

Treat π_self ∈ [0.0, 1.0] as a single scalar. A two-precision (sensory vs
prior) extension is possible but unnecessary for the v1 hypothesis test.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field


# ── Default hyperparameters (paper-table reproducibility) ─────────────────

DEFAULT_DIM        = 64    # embedding dimension
DEFAULT_ALPHA      = 0.1   # EMA learning rate for μ_self
DEFAULT_GAMMA      = 10.0  # PE-variance scale in π update
DEFAULT_PI_INIT    = 0.5   # neutral prior on self-precision
PE_WINDOW          = 50    # rolling window for π_self update

# First-person markers (English + light zh). Updates to μ_self gate on the
# presence of one of these tokens so the self prior reflects identity-
# constitutive utterances, not arbitrary world talk.
SELF_TOKENS = {
    "i", "i'm", "i've", "i'd", "i'll",
    "me", "my", "mine", "myself",
    "we", "we're", "we've", "our", "ours", "ourselves",
    "我", "我的", "自己", "本人",
}


def has_self_reference(text: str) -> bool:
    """True if text contains any first-person token (case-insensitive)."""
    tokens = (text or "").lower().split()
    return any(t.strip(".,!?;:'\"()[]") in SELF_TOKENS for t in tokens)


# ── Embedding ─────────────────────────────────────────────────────────────

def embed(text: str, dim: int = DEFAULT_DIM) -> list[float]:
    """
    Deterministic bag-of-tokens hash embedding, L2-normalised.

    Each whitespace-split token is SHA1-hashed; each hash byte contributes
    a signed unit to one of `dim` dimensions. Sum and L2-normalise. This
    captures lexical-overlap similarity without external dependencies.

    For human-data benchmark comparisons (E5 vs CMR3), swap this for a
    sentence-transformers encoder via the `embedder=` argument on
    SelfModel.update() / self_relevance().
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


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity ∈ [-1, 1]. Returns 0 for empty vectors."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot / (na * nb)


# ── SelfModel ────────────────────────────────────────────────────────────

@dataclass
class SelfModel:
    """
    Running generative model of self: μ_self centroid + π_self precision.

    μ_self is an EMA-updated centroid in embedding space.
    π_self is updated from the variance of prediction errors over a rolling
    window of recent updates, following π ∝ 1/(1 + γ·Var[PE]).
    """

    mu_self: list[float]    = field(default_factory=list)
    pi_self: float          = DEFAULT_PI_INIT
    dim: int                = DEFAULT_DIM
    alpha: float            = DEFAULT_ALPHA
    gamma: float            = DEFAULT_GAMMA

    # Rolling prediction-error history for π_self update (not serialised)
    _pe_history: list[float] = field(default_factory=list, repr=False)
    _max_history: int        = field(default=PE_WINDOW, repr=False)

    # ── Constructors ──────────────────────────────────────────────────────

    @classmethod
    def from_seed(cls, seed_text: str, dim: int = DEFAULT_DIM) -> "SelfModel":
        """Bootstrap μ_self from a seed text (e.g., SOUL.md value descriptors)."""
        return cls(mu_self=embed(seed_text, dim=dim), dim=dim)

    @classmethod
    def empty(cls, dim: int = DEFAULT_DIM) -> "SelfModel":
        """Start with no μ_self; first observation becomes the seed."""
        return cls(mu_self=[], dim=dim)

    # ── Updates ───────────────────────────────────────────────────────────

    def update_from_text(self, text: str, weight: float = 1.0) -> float:
        """
        EMA-update μ_self toward the embedding of `text`.

        Returns the prediction error magnitude (used to update π_self).
        `weight` ∈ [0, 1] scales the learning rate — pass emotional_significance
        or similar gating term to bias updates toward emotionally vivid content.
        """
        return self.update(embed(text, dim=self.dim), weight)

    def update(self, embedding: list[float], weight: float = 1.0) -> float:
        if not embedding or all(abs(x) < 1e-12 for x in embedding):
            return 0.0
        if not self.mu_self:
            self.mu_self = list(embedding)
            return 0.0
        # EMA toward observation, scaled by weight
        a = max(0.0, min(1.0, self.alpha * weight))
        new_mu = [(1 - a) * m + a * e for m, e in zip(self.mu_self, embedding)]
        # Prediction error magnitude
        pe = math.sqrt(sum((a_ - b_) ** 2 for a_, b_ in zip(new_mu, self.mu_self)))
        self.mu_self = new_mu
        self._pe_history.append(pe)
        if len(self._pe_history) > self._max_history:
            self._pe_history.pop(0)
        self._refresh_precision()
        return pe

    def _refresh_precision(self) -> None:
        """π_self ∝ 1 / (1 + γ · Var[PE]) over rolling window."""
        if len(self._pe_history) < 5:
            return
        mean = sum(self._pe_history) / len(self._pe_history)
        var  = sum((p - mean) ** 2 for p in self._pe_history) / len(self._pe_history)
        self.pi_self = round(1.0 / (1.0 + self.gamma * var), 4)

    # ── Queries ───────────────────────────────────────────────────────────

    def self_relevance_of(self, text: str) -> float:
        """`sr ∈ [0, 1]` for arbitrary text. 0.5 when self model uninitialised."""
        return self.self_relevance(embed(text, dim=self.dim))

    def self_relevance(self, embedding: list[float]) -> float:
        """
        Self-relevance score for an embedding.

        Geometry:
          sim = cosine(embedding, μ_self) ∈ [-1, 1]
          raw_sr = (sim + 1) / 2 ∈ [0, 1]
          sr = π_self · raw_sr + (1 - π_self) · 0.5

        At π_self=1 the score equals the similarity (full self-prior trust).
        At π_self=0 the score is 0.5 (no self prior — neutral).
        """
        if not self.mu_self:
            return 0.5
        sim    = cosine(embedding, self.mu_self)
        raw_sr = 0.5 + 0.5 * sim
        return self.pi_self * raw_sr + (1 - self.pi_self) * 0.5

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "mu_self": [round(x, 6) for x in self.mu_self],
            "pi_self": round(self.pi_self, 4),
            "dim": self.dim,
            "alpha": self.alpha,
            "gamma": self.gamma,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SelfModel":
        return cls(
            mu_self=list(d.get("mu_self", [])),
            pi_self=float(d.get("pi_self", DEFAULT_PI_INIT)),
            dim=int(d.get("dim", DEFAULT_DIM)),
            alpha=float(d.get("alpha", DEFAULT_ALPHA)),
            gamma=float(d.get("gamma", DEFAULT_GAMMA)),
        )

    def __repr__(self) -> str:
        n = len(self.mu_self)
        return f"SelfModel(dim={self.dim}, π_self={self.pi_self:.3f}, μ_self_len={n})"
