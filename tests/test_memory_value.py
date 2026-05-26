"""
TDD tests for the multi-factor memory value model (sister paper, Stage A).

V(m) = Σ_i w_i · factor_i(m), with 7 factors. Weights are learnable
(A2): a gradient-free learner recovers the weighting that maximises a
task-return signal.

  A. MemoryValue.value is a linear combination of factors by weight.
  B. Unknown factors default to 0 contribution; unknown weights too.
  C. value() bounded behaviour: all-zero factors → 0.
  D. learn_weights recovers a planted optimal weighting on a synthetic
     task (proxy for LongMemEval QA return).
"""
from __future__ import annotations

import random


# ────────────────────────────────────────────────────────────────────
# A-C. MemoryValue linear combination
# ────────────────────────────────────────────────────────────────────

def test_value_is_weighted_sum():
    from borge.memory.value import MemoryValue
    mv = MemoryValue(weights={"emotion": 2.0, "goal_relevance": 1.0})
    v = mv.value({"emotion": 0.5, "goal_relevance": 0.4})
    assert abs(v - (2.0 * 0.5 + 1.0 * 0.4)) < 1e-9


def test_unknown_factor_zero_contribution():
    from borge.memory.value import MemoryValue
    mv = MemoryValue(weights={"emotion": 1.0})
    # factor present with no weight → 0 contribution
    v = mv.value({"emotion": 0.5, "reliability": 0.9})
    assert abs(v - 0.5) < 1e-9


def test_all_factors_known():
    from borge.memory.value import MemoryValue
    assert set(MemoryValue.FACTORS) == {
        "emotion", "goal_relevance", "value_alignment", "self_relevance",
        "task_utility", "reliability", "usage",
    }


def test_zero_factors_zero_value():
    from borge.memory.value import MemoryValue
    mv = MemoryValue.uniform()
    assert mv.value({k: 0.0 for k in MemoryValue.FACTORS}) == 0.0


# ────────────────────────────────────────────────────────────────────
# D. Weight learner recovers planted optimum (synthetic proxy task)
# ────────────────────────────────────────────────────────────────────

def test_learner_recovers_planted_weights():
    """
    Synthetic proxy for LongMemEval: a 'task' rewards keeping memories
    whose value (under the TRUE planted weights) is high. The learner
    should discover weights correlated with the planted ones, beating
    a uniform-weight init.
    """
    from borge.memory.value import MemoryValue, learn_weights

    rng = random.Random(0)
    factors = MemoryValue.FACTORS
    # Planted ground-truth: only emotion + goal_relevance + task_utility matter
    true_w = {f: (1.0 if f in ("emotion", "goal_relevance", "task_utility") else 0.0)
              for f in factors}

    # Generate a memory pool with random factor profiles
    pool = []
    for _ in range(200):
        fv = {f: rng.random() for f in factors}
        pool.append(fv)

    def true_value(fv):
        return sum(true_w[f] * fv[f] for f in factors)

    # Task return: given a weight vector, rank pool by predicted value,
    # keep top-k, return mean TRUE value of kept set. Maximised when
    # predicted ranking matches true ranking.
    K = 40
    def task_return(weights: dict[str, float]) -> float:
        mv = MemoryValue(weights=weights)
        ranked = sorted(pool, key=lambda fv: -mv.value(fv))
        kept = ranked[:K]
        return sum(true_value(fv) for fv in kept) / K

    uniform = {f: 1.0 for f in factors}
    base_return = task_return(uniform)

    learned, hist = learn_weights(task_return, factors, seed=1, iters=60)
    learned_return = task_return(learned)

    # Learned weights must beat uniform init on the task return
    assert learned_return > base_return + 0.02, (
        f"learner should improve task return; uniform={base_return:.4f}, "
        f"learned={learned_return:.4f}"
    )
    # And the learned weighting should rank the 3 true factors above the
    # 4 irrelevant ones (on average)
    true_factor_w = sum(learned[f] for f in ("emotion", "goal_relevance", "task_utility")) / 3
    junk_factor_w = sum(learned[f] for f in ("value_alignment", "self_relevance", "reliability", "usage")) / 4
    assert true_factor_w > junk_factor_w, (
        f"learner should upweight true factors; true={true_factor_w:.3f}, "
        f"junk={junk_factor_w:.3f}"
    )


def test_learn_weights_monotone_improvement():
    """Learner's best-so-far task return is non-decreasing across history."""
    from borge.memory.value import MemoryValue, learn_weights
    rng = random.Random(2)
    factors = MemoryValue.FACTORS
    pool = [{f: rng.random() for f in factors} for _ in range(100)]
    def task_return(weights):
        mv = MemoryValue(weights=weights)
        ranked = sorted(pool, key=lambda fv: -mv.value(fv))
        return sum(sum(fv.values()) for fv in ranked[:20]) / 20
    _, hist = learn_weights(task_return, factors, seed=3, iters=30)
    best = [h["best_return"] for h in hist]
    assert all(best[i] <= best[i+1] + 1e-9 for i in range(len(best)-1)), \
        "best-so-far return must be monotone non-decreasing"
