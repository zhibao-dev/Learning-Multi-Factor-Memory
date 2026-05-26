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
