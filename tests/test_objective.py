# tests/test_objective.py
from lmfm.learn.objective import gold_retention, matrix_objective


def _case(*mems):
    return {"memories": [{"factors": f, "gold": g} for f, g in mems]}


def test_gold_retention_keeps_high_value_gold():
    # reliability-weighted: the gold item has high reliability
    weights = {"reliability": 1.0}
    case = _case(
        ({"reliability": 0.9}, True),    # gold, high value → kept
        ({"reliability": 0.1}, False),   # non-gold, low → dropped
    )
    r = gold_retention(case, weights, keep_frac=0.5)
    assert r == 1.0


def test_matrix_objective_averages_cases():
    weights = {"reliability": 1.0}
    mat = {"keep_frac": 0.5, "cases": [
        _case(({"reliability": 0.9}, True), ({"reliability": 0.1}, False)),
        _case(({"reliability": 0.8}, True), ({"reliability": 0.2}, False)),
    ]}
    obj = matrix_objective(mat)
    assert obj(weights) == 1.0
