"""Gold-retention objective over an uploaded numeric factor matrix."""

from __future__ import annotations

from ..value import MemoryValue


def gold_retention(case: dict, weights: dict, *, keep_frac: float):
    mv = MemoryValue(weights=weights)
    scored = [(mv.value(m["factors"]), m) for m in case["memories"]]
    scored.sort(key=lambda x: -x[0])
    k = max(1, int(len(scored) * keep_frac))
    kept = [m for _, m in scored[:k]]
    total = sum(1 for m in case["memories"] if m["gold"])
    if total == 0:
        return None
    return sum(1 for m in kept if m["gold"]) / total


def matrix_objective(matrix: dict):
    """Return a task_return(weights)->float averaging gold retention."""
    kf = matrix["keep_frac"]
    cases = matrix["cases"]

    def obj(weights: dict) -> float:
        vals = [gold_retention(c, weights, keep_frac=kf) for c in cases]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    return obj
