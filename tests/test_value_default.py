def test_shipped_default_covers_all_factors():
    from borge.memory.value import SHIPPED_DEFAULT, MemoryValue
    assert set(SHIPPED_DEFAULT) == set(MemoryValue.FACTORS)  # all 7 keys


def test_default_factory_uses_shipped():
    from borge.memory.value import SHIPPED_DEFAULT, default_memory_value
    mv = default_memory_value()
    assert mv.weights["reliability"] == SHIPPED_DEFAULT["reliability"]
    assert mv.weights["emotion"] == SHIPPED_DEFAULT["emotion"]


def test_default_factory_override_merges():
    from borge.memory.value import default_memory_value, SHIPPED_DEFAULT
    mv = default_memory_value({"reliability": 2.0})
    assert mv.weights["reliability"] == 2.0          # override wins
    assert mv.weights["emotion"] == SHIPPED_DEFAULT["emotion"]  # others kept
