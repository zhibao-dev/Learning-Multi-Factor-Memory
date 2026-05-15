"""
TDD test for L2: π_self as a variational posterior under a conjugate prior.

The v0.3 paper acknowledged that π_self = 1/(1 + γ·Var[PE]) was an
*engineered* form. v0.4 derives it as the variational posterior over
"self-prior reliability" under a Gamma prior on observation precision,
and the `gamma` hyperparameter is renamed (with alias) to
`prior_precision` so the connection is explicit at the API surface.

This test pins:
  - `SelfModel(prior_precision=g)` is equivalent to `SelfModel(gamma=g)`.
  - The variational interpretation: π_self equals the prior-precision
    weighted shrinkage of empirical PE variance toward zero.
"""
from __future__ import annotations


def test_prior_precision_alias_for_gamma():
    """`prior_precision=X` constructs the same SelfModel as `gamma=X`."""
    from borge.values.self_model import SelfModel
    a = SelfModel(gamma=7.5)
    b = SelfModel(prior_precision=7.5)
    assert a.gamma == b.gamma == 7.5


def test_pi_self_matches_variational_form_after_updates():
    """
    After populating PE history, π_self should exactly equal the
    variational posterior form:
        π_self = 1 / (1 + prior_precision · Var[PE])
    """
    from borge.values.self_model import SelfModel
    m = SelfModel.empty(prior_precision=5.0)

    # Force a known PE trajectory by manually populating the history.
    # The empirical variance of [0.1, 0.2, 0.1, 0.2, 0.1] is small.
    m._pe_history = [0.1, 0.2, 0.1, 0.2, 0.1]
    m._refresh_precision()

    pe = m._pe_history
    mean = sum(pe) / len(pe)
    var  = sum((p - mean) ** 2 for p in pe) / len(pe)
    expected = 1.0 / (1.0 + m.gamma * var)
    assert abs(m.pi_self - round(expected, 4)) < 1e-4, (
        f"π_self should match variational posterior; got {m.pi_self}, "
        f"expected {expected}"
    )


def test_pi_self_collapses_when_pe_variance_is_high():
    """High PE variance → unreliable self prior → π_self → 0."""
    from borge.values.self_model import SelfModel
    m = SelfModel.empty(prior_precision=10.0)
    m._pe_history = [0.0, 2.0, 0.0, 2.0, 0.0, 2.0]  # bimodal, high variance
    m._refresh_precision()
    assert m.pi_self < 0.2


def test_pi_self_approaches_unity_for_stable_predictions():
    """Near-zero PE variance → reliable self prior → π_self → 1."""
    from borge.values.self_model import SelfModel
    m = SelfModel.empty(prior_precision=10.0)
    m._pe_history = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]  # near-constant
    m._refresh_precision()
    assert m.pi_self > 0.99
