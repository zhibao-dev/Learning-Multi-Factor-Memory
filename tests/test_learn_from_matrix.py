# tests/test_learn_from_matrix.py
from lmfm.learn.service import learn_from_matrix


def test_learn_from_matrix_returns_weights_and_score():
    mat = {"keep_frac": 0.5, "factor_order":
           ["emotion","goal_relevance","value_alignment","self_relevance",
            "task_utility","reliability","usage"],
           "cases": [
        {"memories": [
            {"factors": {"reliability": 0.9, "emotion": 0.1}, "gold": True},
            {"factors": {"reliability": 0.1, "emotion": 0.9}, "gold": False},
        ]},
    ]}
    out = learn_from_matrix(mat, iters=60, seed=1)
    assert set(out["weights"]) >= {"reliability", "emotion"}
    assert 0.0 <= out["train_retention"] <= 1.0
    # learner should prefer reliability for this gold structure
    assert out["weights"]["reliability"] >= out["weights"]["emotion"]
