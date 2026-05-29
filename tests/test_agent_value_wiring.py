"""
Task 7 (paper2 runtime wiring): BorgeAgent builds ONE MemoryValue and
injects that SAME instance into all three memory engines, so the runtime
literally has one learned value driving encode + forget + retrieve.
"""

from borge.agent import BorgeAgent
from borge.memory.value import MemoryValue, SHIPPED_DEFAULT


def test_one_memory_value_drives_all_three_engines():
    a = BorgeAgent(None)
    assert isinstance(a._memory_value, MemoryValue)
    assert a._memory_value.weights["reliability"] == SHIPPED_DEFAULT["reliability"]
    # the SAME instance is shared by all three engines (identity, not equality)
    assert a._forgetting.memory_value is a._memory_value
    assert a._consolidation.memory_value is a._memory_value
    assert a._retrieval.memory_value is a._memory_value
