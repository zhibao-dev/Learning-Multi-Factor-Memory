"""
Neural interaction ablation for the multi-factor memory value (paper2).

paper2's headline is the INTERPRETABLE linear value V(m) = Σ_i w_i·f_i
(see `borge/memory/value.py::MemoryValue`). This module adds a small MLP
g_θ over the SAME factor vector PURELY as an interaction ablation: it
tests whether factor interactions (e.g. emotion mattering only when a
turn is self-relevant) buy extra gold-retention beyond the additive
linear model. It is NEVER the default scorer — the linear model stays the
headline. Two outcomes are both publishable:

  MLP ≈ linear → factors combine near-additively; the linear model
                 suffices (strengthens the interpretability story).
  MLP > linear → there is a named non-additive interaction worth keeping.

`MemoryValueNet` exposes the identical `.value(factors: dict) -> float`
interface as `MemoryValue`, so every downstream consumer (encode depth,
forget score, retrieval rank) is swap-compatible without code changes.

Training uses a PAIRWISE LOGISTIC RANKING LOSS rather than a regression
target: the downstream metric is "keep top-κ by value, retain gold",
which is non-differentiable. Within each case we rank gold (has_answer)
turns above non-gold via softplus(-(s_gold - s_nongold)). A small
weight_decay guards the tiny-data (few-hundred-case) regime.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Live factors only — the API-free annotator populates exactly these four
# (emotion, goal_relevance, self_relevance, reliability). The other three
# MemoryValue factors (value_alignment, task_utility, usage) are ×0 in the
# API-free regime, so feeding them to the MLP would only add dead inputs.
# This MUST match LIVE_FACTORS in experiments/lme_blind_forgetting.py.
FACTORS = ("emotion", "goal_relevance", "self_relevance", "reliability")


class MemoryValueNet(nn.Module):
    """MLP scorer over the factor vector, drop-in for `MemoryValue`.

    The network maps a factor vector → a scalar value. Mirrors
    `MemoryValue.value(factors: dict) -> float` so consumers are
    swap-compatible. A missing factor defaults to 0.0, exactly like the
    linear model's `factors.get(f, 0.0)`.
    """

    def __init__(self, factors: tuple[str, ...] = FACTORS,
                 hidden: tuple[int, ...] = (16, 16)):
        super().__init__()
        self.factors = tuple(factors)
        dims = [len(self.factors), *hidden, 1]
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers = layers[:-1]  # drop final ReLU → scalar output, unbounded
        self.net = nn.Sequential(*layers)

    def _vec(self, factors: dict) -> torch.Tensor:
        """Factor dict → [1, n_factors] tensor; missing factor → 0.0."""
        return torch.tensor(
            [[float(factors.get(f, 0.0)) for f in self.factors]],
            dtype=torch.float32,
        )

    @torch.no_grad()
    def value(self, factors: dict) -> float:
        """V(m) as a scalar — mirrors MemoryValue.value's signature."""
        self.eval()
        return float(self.net(self._vec(factors)).item())


def train_value_net(
    train_cases,
    *,
    factors: tuple[str, ...] = FACTORS,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 1e-3,
    seed: int = 1,
) -> MemoryValueNet:
    """Fit a `MemoryValueNet` with a pairwise logistic ranking loss.

    `train_cases` is a list of cases; each case is a list of per-turn
    dicts that carry the factor keys in `factors` ALONGSIDE a boolean
    `has_answer`. Within each case, every (gold, non-gold) pair
    contributes softplus(-(s_gold - s_nongold)); minimising this ranks
    gold turns above non-gold. Cases lacking either a gold or a non-gold
    turn carry no pairwise signal and are skipped.

    If no case yields a usable pair, training is a no-op and a fresh
    (untrained) net is returned — the caller still gets a valid scorer.
    """
    torch.manual_seed(seed)
    net = MemoryValueNet(factors)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    # Precompute (pos, neg) factor tensors per case once.
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for case in train_cases:
        pos = [t for t in case if t["has_answer"]]
        neg = [t for t in case if not t["has_answer"]]
        if not pos or not neg:
            continue
        P = torch.tensor([[float(t.get(f, 0.0)) for f in factors] for t in pos],
                         dtype=torch.float32)
        N = torch.tensor([[float(t.get(f, 0.0)) for f in factors] for t in neg],
                         dtype=torch.float32)
        pairs.append((P, N))

    if not pairs:
        return net  # no ranking signal — return untrained but valid scorer

    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.zeros((), dtype=torch.float32)
        for P, N in pairs:
            sp, sn = net.net(P), net.net(N)             # [|pos|,1], [|neg|,1]
            diff = sp.unsqueeze(1) - sn.unsqueeze(0)     # all pos-neg pairs
            loss = loss + F.softplus(-diff).mean()
        loss = loss / len(pairs)
        loss.backward()
        opt.step()

    return net
