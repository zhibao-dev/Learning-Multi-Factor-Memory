import json
from lmfm.weights import load_value_from_weights_file


def test_load_weights_file_into_value(tmp_path):
    wf = tmp_path / "weights.json"
    wf.write_text(json.dumps({"weights": {"reliability": 0.9, "emotion": 0.2}}))
    mv = load_value_from_weights_file(wf)
    # learned weights override defaults
    assert mv.weights["reliability"] == 0.9
    assert mv.weights["emotion"] == 0.2
    # untouched factors fall back to default
    assert mv.weights["self_relevance"] == 0.23
    # scoring works
    v = mv.value({"reliability": 1.0, "emotion": 0.0})
    assert round(v, 2) == 0.9
