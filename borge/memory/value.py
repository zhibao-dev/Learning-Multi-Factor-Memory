"""
Multi-Factor Memory Value (sister-paper substrate, no FEP).

A memory's value to future agent behaviour is a weighted combination of
interpretable factors:

    V(m) = Σ_i  w_i · factor_i(m)

Seven factors:
    emotion          |V|·A emotional intensity
    goal_relevance   cosine to the agent's active goal
    value_alignment  match to SOUL.md value priors
    self_relevance   cosine to μ_self / μ_user
    task_utility     marginal contribution to task success
    reliability      provenance / confidence
    usage            retrieval frequency + recency

This single scalar uniformly controls encoding depth, forget risk, and
retrieval rank, replacing the fixed product-of-resistances of the
self-FEP paper.

Theory base (NOT FEP): value-directed remembering (Castel), levels of
processing (Craik & Lockhart), emotional consolidation (McGaugh),
rational analysis of memory (Anderson & Schooler 1991), and RL /
expected-utility — value = marginal contribution to future task success.

Weights `w_i` are LEARNED (A2) from downstream task return by a
gradient-free optimiser (`learn_weights`), since the
encode→forget→retrieve→answer pipeline is non-differentiable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class MemoryValue:
    """Weighted linear value over the seven memory factors."""

    FACTORS = (
        "emotion",
        "goal_relevance",
        "value_alignment",
        "self_relevance",
        "task_utility",
        "reliability",
        "usage",
    )

    weights: dict[str, float] = field(default_factory=dict)

    @classmethod
    def uniform(cls) -> "MemoryValue":
        return cls(weights={f: 1.0 for f in cls.FACTORS})

    def value(self, factors: dict[str, float]) -> float:
        """V(m) = Σ w_i · factor_i. Missing factor or weight → 0 term."""
        return sum(
            self.weights.get(f, 0.0) * factors.get(f, 0.0)
            for f in self.FACTORS
        )

    def normalised(self) -> "MemoryValue":
        """Return a copy with weights L1-normalised to sum 1 (sign kept)."""
        s = sum(abs(self.weights.get(f, 0.0)) for f in self.FACTORS)
        if s < 1e-12:
            return MemoryValue(weights=dict(self.weights))
        return MemoryValue(weights={f: self.weights.get(f, 0.0) / s
                                    for f in self.FACTORS})


# Out-of-box weights: the full-479 LongMemEval blind fit
# (results/lme_blind_forgetting_full.json -> learned_weights_blind_mean),
# extended to all 7 factors. goal/value_alignment/task_utility were inert
# (0) in that fit; usage gets a small positive prior so frequent recall
# still resists forgetting. Override via config 'borge.memory.value.weights'.
SHIPPED_DEFAULT = {
    "emotion":         0.55,
    "goal_relevance":  0.00,
    "value_alignment": 0.00,
    "self_relevance":  0.23,
    "task_utility":    0.00,
    "reliability":     0.64,
    "usage":           0.10,
}


def default_memory_value(override: dict | None = None) -> "MemoryValue":
    """MemoryValue with shipped learned-default weights, merged over by `override`."""
    return MemoryValue(weights={**SHIPPED_DEFAULT, **(override or {})})


# ── Factor extraction + value-driven memory dynamics ─────────────────────

def memory_factors(row: dict[str, Any]) -> dict[str, float]:
    """
    Extract the 7-factor dict from a borge_memories row.

    emotion        = |V|·A
    self_relevance = self_relevance_score
    goal_relevance / value_alignment / task_utility / reliability =
        stored columns (default 0.0)
    usage          = retrieval-frequency saturating in [0,1]
    """
    def _f(key: str, default: float = 0.0) -> float:
        try:
            v = row.get(key)
            return default if v is None else float(v)
        except (TypeError, ValueError):
            return default

    valence = _f("emotional_valence")
    arousal = _f("emotional_arousal")
    rcount  = _f("retrieval_count")
    # usage saturates: 0 retrievals → 0, →1 as retrievals grow
    usage = rcount / (1.0 + rcount)

    return {
        "emotion":         abs(valence) * arousal,
        "goal_relevance":  _f("goal_relevance"),
        "value_alignment": _f("value_alignment"),
        "self_relevance":  _f("self_relevance_score"),
        "task_utility":    _f("task_utility"),
        "reliability":     _f("reliability"),
        "usage":           usage,
    }


def value_forget_score(
    row: dict[str, Any],
    mv: "MemoryValue",
    now: datetime | None = None,
    *,
    beta: float = 1.0,
) -> float:
    """
    Value-driven forget score (higher = more likely forgotten).

        score = recency_decay · usage_penalty · 1/(1 + β·V(m))

    High memory value V resists forgetting. Replaces the self-FEP
    product-of-fixed-resistances with one learned-weighted value term.
    Recency + usage are kept as the universal time/access dynamics.
    """
    if now is None:
        now = datetime.now()
    ts_str = row.get("last_retrieved") or row.get("timestamp")
    try:
        ts = datetime.fromisoformat(ts_str) if isinstance(ts_str, str) else now
    except ValueError:
        ts = now
    days_since = max(0.0, (now - ts).total_seconds() / 86400.0)

    rcount = float(row.get("retrieval_count") or 0)
    recency_decay = days_since ** 0.7
    usage_penalty = 1.0 / (1.0 + rcount)

    v = mv.value(memory_factors(row))
    value_resistance = 1.0 / (1.0 + beta * max(0.0, v))

    return recency_decay * usage_penalty * value_resistance


def value_encoding_depth(factors: dict[str, float], mv: "MemoryValue") -> int:
    """
    Map memory value → Craik-Lockhart encoding tier ∈ {1,2,3,4}.

    Thresholds on V normalised by the (uniform) max possible value so
    the tiers are scale-stable regardless of learned weight magnitude.
    """
    v = mv.value(factors)
    wsum = sum(abs(mv.weights.get(f, 0.0)) for f in MemoryValue.FACTORS)
    norm = v / wsum if wsum > 1e-12 else 0.0   # factors ∈ [0,1] → norm ∈ [0,1]
    if norm >= 0.70:
        return 4   # META
    if norm >= 0.45:
        return 3   # SCHEMATIC
    if norm >= 0.20:
        return 2   # SEMANTIC
    return 1       # SHALLOW


def learn_weights(
    task_return,
    factors: tuple[str, ...] | list[str],
    *,
    seed: int = 0,
    iters: int = 60,
    step: float = 0.5,
    decay: float = 0.92,
) -> tuple[dict[str, float], list[dict]]:
    """
    Gradient-free coordinate-ascent + random-restart-perturbation learner.

    Maximises `task_return(weights: dict) -> float`. Non-differentiable
    objective (the encode→forget→retrieve→answer pipeline), so we use a
    simple stochastic hill-climb: each iteration perturb the weight
    vector, keep the perturbation if it improves the return.

    Returns (best_weights, history) where history[i] = {iter, best_return}.
    Best-so-far return is monotone non-decreasing by construction.
    """
    rng = random.Random(seed)
    w = {f: 1.0 for f in factors}            # uniform init
    best_return = task_return(w)
    history = [{"iter": 0, "best_return": best_return}]
    sigma = step

    for it in range(1, iters + 1):
        cand = dict(w)
        # Perturb a random subset of coordinates
        for f in factors:
            if rng.random() < 0.5:
                cand[f] = max(0.0, cand[f] + rng.gauss(0.0, sigma))
        r = task_return(cand)
        if r > best_return:
            w = cand
            best_return = r
        history.append({"iter": it, "best_return": best_return})
        sigma *= decay                        # anneal perturbation scale

    return w, history
