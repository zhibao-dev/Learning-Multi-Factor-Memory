"""
TDD: unify MemoryValue into encode-depth / forget / retrieve (Stage B-1).

  A. borge_memories gains 4 factor columns: goal_relevance,
     value_alignment, task_utility, reliability.
  B. memory_factors(row) extracts the 7-factor dict from a stored row.
  C. value_forget_score: higher V → lower forget score (more retained),
     holding recency/usage constant.
  D. value_encoding_depth: higher V → deeper tier.
"""
from __future__ import annotations

import os
import sqlite3
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


# A. schema
def test_borge_memories_has_factor_columns(tmpdb):
    from borge.memory.store import MemoryStore
    MemoryStore(tmpdb).ensure_table()
    with sqlite3.connect(tmpdb) as c:
        cols = {r[1] for r in c.execute("PRAGMA table_info(borge_memories)")}
    for col in ("goal_relevance", "value_alignment", "task_utility", "reliability"):
        assert col in cols, f"missing column {col}"


# B. factor extraction
def test_memory_factors_extracts_seven():
    from borge.memory.value import memory_factors, MemoryValue
    row = {
        "emotional_valence": -0.8, "emotional_arousal": 0.9,
        "self_relevance_score": 0.7,
        "goal_relevance": 0.6, "value_alignment": 0.4,
        "task_utility": 0.5, "reliability": 0.9,
        "retrieval_count": 3,
        "timestamp": datetime.now().isoformat(),
    }
    f = memory_factors(row)
    assert set(f.keys()) == set(MemoryValue.FACTORS)
    # emotion = |V|·A
    assert abs(f["emotion"] - 0.8 * 0.9) < 1e-9
    assert f["self_relevance"] == 0.7
    assert f["goal_relevance"] == 0.6
    assert 0.0 <= f["usage"] <= 1.0


def test_memory_factors_defaults_missing():
    from borge.memory.value import memory_factors
    f = memory_factors({"timestamp": datetime.now().isoformat()})
    # all factors present, defaulting to neutral/zero
    assert f["goal_relevance"] == 0.0
    assert f["reliability"] == 0.0


# C. value-driven forgetting
def test_value_forget_score_higher_value_lower_score(tmpdb):
    from borge.memory.value import MemoryValue, value_forget_score
    now = datetime.now()
    ts = (now - timedelta(days=5)).isoformat()
    mv = MemoryValue.uniform()
    low_v = {"timestamp": ts, "retrieval_count": 0,
             "emotional_valence": 0.0, "emotional_arousal": 0.0,
             "self_relevance_score": 0.0, "goal_relevance": 0.0,
             "value_alignment": 0.0, "task_utility": 0.0, "reliability": 0.0}
    high_v = {**low_v,
              "emotional_valence": 0.9, "emotional_arousal": 0.9,
              "self_relevance_score": 0.9, "goal_relevance": 0.9,
              "value_alignment": 0.9, "task_utility": 0.9, "reliability": 0.9}
    s_low = value_forget_score(low_v, mv, now)
    s_high = value_forget_score(high_v, mv, now)
    assert s_high < s_low, f"high-value memory must resist forgetting: {s_high} vs {s_low}"


# D. value-driven encoding depth
def test_value_encoding_depth_monotone():
    from borge.memory.value import MemoryValue, value_encoding_depth
    mv = MemoryValue.uniform()
    lo = {f: 0.0 for f in MemoryValue.FACTORS}
    hi = {f: 1.0 for f in MemoryValue.FACTORS}
    assert value_encoding_depth(hi, mv) >= value_encoding_depth(lo, mv)
    assert 1 <= value_encoding_depth(lo, mv) <= 4
    assert 1 <= value_encoding_depth(hi, mv) <= 4
