"""Tests that the 4 multi-factor value columns round-trip through the store.

These columns are declared in the schema but were silently dropped by
`insert()` / `_normalize()`, so they always persisted as the default 0.0.
"""

from borge.memory.store import MemoryStore


def test_factor_columns_round_trip(tmp_path):
    """Inserted goal/value/task/reliability factors come back unchanged."""
    db = str(tmp_path / "borge.db")
    store = MemoryStore(db)
    store.insert({
        "id": "m1",
        "session_id": "s",
        "timestamp": "2026-01-01T00:00:00",
        "goal_relevance": 0.8,
        "value_alignment": 0.6,
        "task_utility": 0.3,
        "reliability": 0.7,
    })

    row = store.get("m1")
    assert row is not None
    assert row["goal_relevance"] == 0.8
    assert row["value_alignment"] == 0.6
    assert row["task_utility"] == 0.3
    assert row["reliability"] == 0.7


def test_factor_columns_default_zero_when_omitted(tmp_path):
    """Rows that omit the factor keys still insert and default to 0.0."""
    db = str(tmp_path / "borge.db")
    store = MemoryStore(db)
    store.insert({
        "id": "m2",
        "session_id": "s",
        "timestamp": "2026-01-01T00:00:00",
    })

    row = store.get("m2")
    assert row is not None
    assert row["goal_relevance"] == 0.0
    assert row["value_alignment"] == 0.0
    assert row["task_utility"] == 0.0
    assert row["reliability"] == 0.0
