"""
TDD (paper2 runtime wiring, Task 5): ForgettingEngine deletion is driven by
the single MemoryValue.value_forget_score, not paper1's hand-tuned
product-of-resistances.

The discriminating signal here is `reliability` — a value factor the paper1
formula structurally cannot read. Both rows are equally old, equally unused,
shallow, and carry identical (neutral) emotion / self-relevance, so ONLY the
multi-factor memory value can explain why one survives and the other is pruned.

Under the old formula both rows score identically (reliability ignored) → both
above the prune threshold → both deleted, including the high-value one (RED).
Under value_forget_score the reliable row's value resists forgetting → it sits
below the prune threshold and survives, while the low-value row is pruned (GREEN).
"""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta

import pytest


@pytest.fixture()
def tmpdb():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def test_high_value_memory_survives_low_value_pruned(tmpdb):
    from borge.memory.store import MemoryStore
    from borge.memory.forgetting import ForgettingEngine

    store = MemoryStore(tmpdb)
    old_ts = (datetime.now() - timedelta(days=30)).isoformat()
    base = {
        "session_id":        "s",
        "role":              "user",
        "content":           "x",
        "timestamp":         old_ts,         # equally old
        "retrieval_count":   0,              # equally unused
        "encoding_depth":    1,              # SHALLOW → eligible for deletion
        # Neutral emotion + self-relevance, IDENTICAL across both rows, so
        # neither emotion nor self-relevance discriminate. Only `reliability`
        # (a value factor) differs → only the multi-factor value can decide.
        "emotional_valence": 0.0,
        "emotional_arousal": 0.1,
        "self_relevance_score": 0.1,
    }
    high = {**base, "id": "m-high", "reliability": 0.9}
    low  = {**base, "id": "m-low",  "reliability": 0.1}
    store.insert(high)
    store.insert(low)

    ForgettingEngine().run_forgetting_pass(tmpdb)

    high_row = store.get("m-high")
    low_row  = store.get("m-low")

    assert high_row is not None, "high-value (reliable) memory must survive the forgetting pass"
    assert low_row is None, "low-value memory must be pruned"
