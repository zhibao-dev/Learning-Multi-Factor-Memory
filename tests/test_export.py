import json
from lmfm.export import build_matrix


def test_build_matrix_has_no_text():
    annotated = [
        {"id": "m1", "ts": "t", "gold": True,
         "factors": {"emotion": 0.2, "goal_relevance": 0.5, "value_alignment": 0.0,
                     "self_relevance": 0.8, "task_utility": 0.0, "reliability": 0.7,
                     "usage": 0.0}},
    ]
    mat = build_matrix([annotated], keep_frac=0.3)
    blob = json.dumps(mat)
    assert "text" not in blob
    assert mat["keep_frac"] == 0.3
    assert mat["cases"][0]["memories"][0]["gold"] is True
    assert len(mat["cases"][0]["memories"][0]["factors"]) == 7
