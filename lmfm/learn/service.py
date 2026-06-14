"""Server-side: fit weights from an uploaded factor matrix."""

from __future__ import annotations

from ..value import learn_weights, MemoryValue
from .objective import matrix_objective

# Only learn over factors that actually vary in the upload; learning a
# weight for an all-zero factor would multiply out to nothing and just
# add noise.
def _live_factors(matrix: dict) -> tuple[str, ...]:
    order = matrix.get("factor_order") or list(MemoryValue.FACTORS)
    seen = {f: False for f in order}
    for c in matrix["cases"]:
        for m in c["memories"]:
            for f, v in m["factors"].items():
                if v not in (0, 0.0):
                    seen[f] = True
    live = tuple(f for f in order if seen.get(f))
    return live or tuple(order)


def learn_from_matrix(matrix: dict, *, iters: int = 120, seed: int = 1) -> dict:
    obj = matrix_objective(matrix)
    live = _live_factors(matrix)
    weights, hist = learn_weights(obj, live, seed=seed, iters=iters)
    return {
        "weights": {f: round(weights[f], 4) for f in live},
        "train_retention": round(hist[-1]["best_return"], 4),
        "model_version": "v1",
        "factors_learned": list(live),
    }
