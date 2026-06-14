from lmfm.value import MemoryValue, memory_factors, default_memory_value


def test_default_weights_match_paper():
    mv = default_memory_value()
    assert round(mv.weights["reliability"], 2) == 0.64
    assert round(mv.weights["emotion"], 2) == 0.55
    assert round(mv.weights["self_relevance"], 2) == 0.23
    assert mv.weights["goal_relevance"] == 0.0


def test_value_is_weighted_sum():
    mv = MemoryValue(weights={"emotion": 1.0, "reliability": 2.0})
    v = mv.value({"emotion": 0.5, "reliability": 0.25})
    assert v == 1.0


def test_memory_factors_extracts_emotion_as_absV_times_A():
    row = {"emotional_valence": -0.4, "emotional_arousal": 0.5,
           "reliability": 1.0, "retrieval_count": 0}
    f = memory_factors(row)
    assert round(f["emotion"], 3) == 0.2
    assert f["reliability"] == 1.0
    assert f["usage"] == 0.0


def test_override_merges_over_defaults():
    mv = default_memory_value({"goal_relevance": 0.9})
    assert mv.weights["goal_relevance"] == 0.9
    assert round(mv.weights["reliability"], 2) == 0.64
