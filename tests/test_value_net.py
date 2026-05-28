"""
TDD tests for the neural interaction ablation (MLP g_θ).

`MemoryValueNet` is an INTERACTION ABLATION for paper2 — never the
headline. It exposes the SAME `.value(factors: dict) -> float` interface
as `MemoryValue` so every downstream consumer (encode/forget/retrieve)
is swap-compatible. It is trained with a pairwise logistic ranking loss
(gold turns ranked above non-gold within a case), because the top-κ keep
metric is non-differentiable.

  A. value() returns a Python float; a missing factor defaults to 0.
  B. The ranking loss actually separates gold from non-gold: on a toy
     set where gold turns have high reliability and non-gold low, after
     training the mean gold score exceeds the mean non-gold score by a
     clear margin.
"""
from __future__ import annotations

import random


# ────────────────────────────────────────────────────────────────────
# A. value() interface mirrors MemoryValue.value
# ────────────────────────────────────────────────────────────────────

def test_value_returns_float():
    from borge.memory.value_net import MemoryValueNet

    net = MemoryValueNet()
    v = net.value({"emotion": 0.5, "goal_relevance": 0.4,
                   "self_relevance": 0.3, "reliability": 0.9})
    assert isinstance(v, float)


def test_value_missing_factor_defaults_zero():
    from borge.memory.value_net import MemoryValueNet

    net = MemoryValueNet()
    # A dict missing every factor must not raise — all default to 0.0,
    # mirroring MemoryValue.value's `factors.get(f, 0.0)` contract.
    v_empty = net.value({})
    assert isinstance(v_empty, float)

    # A dict missing only `reliability` must score IDENTICALLY to one that
    # explicitly sets reliability=0.0 — proving the default is 0, not noise.
    base = {"emotion": 0.5, "goal_relevance": 0.4, "self_relevance": 0.3}
    v_missing = net.value(base)
    v_explicit_zero = net.value({**base, "reliability": 0.0})
    assert abs(v_missing - v_explicit_zero) < 1e-9


def test_value_is_deterministic():
    from borge.memory.value_net import MemoryValueNet

    net = MemoryValueNet()
    f = {"emotion": 0.2, "goal_relevance": 0.7,
         "self_relevance": 0.5, "reliability": 0.6}
    assert net.value(f) == net.value(f)


# ────────────────────────────────────────────────────────────────────
# B. Pairwise ranking loss separates gold from non-gold
# ────────────────────────────────────────────────────────────────────

def test_ranking_loss_separates_gold():
    """Toy data: gold turns have high reliability (0.9), non-gold low
    (0.1); all other factors are random noise. A working ranking loss
    must learn to score gold above non-gold by a clear margin."""
    from borge.memory.value_net import FACTORS, MemoryValueNet, train_value_net

    rng = random.Random(0)

    def _noise():
        return {f: rng.random() for f in FACTORS}

    train_cases = []
    for _ in range(12):
        case = []
        # 2 gold turns: reliability high, distinguishing signal
        for _ in range(2):
            t = _noise()
            t["reliability"] = 0.9
            t["has_answer"] = True
            case.append(t)
        # 6 non-gold turns: reliability low
        for _ in range(6):
            t = _noise()
            t["reliability"] = 0.1
            t["has_answer"] = False
            case.append(t)
        train_cases.append(case)

    net = train_value_net(train_cases, epochs=300, seed=1)
    assert isinstance(net, MemoryValueNet)

    gold = [t for case in train_cases for t in case if t["has_answer"]]
    nong = [t for case in train_cases for t in case if not t["has_answer"]]
    mean_gold = sum(net.value(t) for t in gold) / len(gold)
    mean_nong = sum(net.value(t) for t in nong) / len(nong)

    # A clear margin — the only consistent signal is reliability, so the
    # net must lean on it and rank gold well above non-gold.
    assert mean_gold > mean_nong + 0.3, (
        f"gold {mean_gold:.3f} not clearly above non-gold {mean_nong:.3f}")


def test_train_skips_degenerate_cases():
    """Cases with no positives or no negatives carry no pairwise signal
    and must be skipped silently (not crash)."""
    from borge.memory.value_net import FACTORS, train_value_net

    rng = random.Random(2)

    def _noise(ans):
        t = {f: rng.random() for f in FACTORS}
        t["has_answer"] = ans
        return t

    # One all-positive case, one all-negative case, one mixed case.
    train_cases = [
        [_noise(True), _noise(True)],          # no negatives → skipped
        [_noise(False), _noise(False)],        # no positives → skipped
        [_noise(True), _noise(False), _noise(False)],  # usable
    ]
    net = train_value_net(train_cases, epochs=10, seed=1)
    # Just needs to train without error and produce a usable scorer.
    assert isinstance(net.value(train_cases[-1][0]), float)


def test_train_no_usable_pairs_returns_net():
    """If NO case has both a positive and a negative, training has no
    pairs at all; it must still return a working (untrained) net."""
    from borge.memory.value_net import FACTORS, MemoryValueNet, train_value_net

    rng = random.Random(3)

    def _noise(ans):
        t = {f: rng.random() for f in FACTORS}
        t["has_answer"] = ans
        return t

    train_cases = [[_noise(True), _noise(True)]]  # no negatives anywhere
    net = train_value_net(train_cases, epochs=50, seed=1)
    assert isinstance(net, MemoryValueNet)
    assert isinstance(net.value(train_cases[0][0]), float)
