"""
TDD tests for the sister paper (multi-factor value model): consolidation
Step 3 must compute the LIVE value factors at ENCODE time and persist them
to borge_memories, so the later forget/retrieve tasks can just read them.

The 6 live factors (design "D", all API-free):
  emotion         = |V|·A           (already persisted as emotional_significance)
  self_relevance  = sr              (already persisted as self_relevance_score)
  usage           = 0 at encode     (retrieval_count starts 0)
  reliability     = role heuristic  (0.7 user / 0.4 otherwise)
  value_alignment = 0.5 + 0.5·cos(embedding, value_centroid) | 0.0 if no centroid
  goal_relevance  = 0.5 + 0.5·cos(embedding, session_topic_centroid) | 0.5 fallback
  task_utility    = 0.0             (LLM-gated, out of scope)

These tests pin the four NEW factor columns Step 3 must now set.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from borge.memory.consolidation import MemoryConsolidationPipeline
from borge.memory.knowledge_graph import KnowledgeGraph
from borge.memory.store import MemoryStore
from borge.values.self_model import SelfModel


@pytest.fixture()
def tmpdb():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def test_step3_persists_live_value_factors(tmpdb):
    self_model = SelfModel.from_seed("research curiosity learning")
    # hash_embed default dim is 64; match it for the value centroid.
    assert len(self_model.mu_self) == 64
    value_centroid = [1.0] * 64

    store = MemoryStore(tmpdb)
    kg = KnowledgeGraph(tmpdb)
    pipeline = MemoryConsolidationPipeline(
        db_path=tmpdb,
        knowledge_graph=kg,
        memory_store=store,
        self_model=self_model,
        value_centroid=value_centroid,
    )

    messages = [
        {"role": "user", "content": "I want to learn about active inference"},
        {"role": "assistant", "content": "Active inference minimizes free energy"},
        {"role": "user", "content": "explain free energy in simple terms please"},
    ]
    # 2 user turns → 2 (V, A) snapshots
    emotional_history = [(0.5, 0.6), (0.2, 0.4)]

    pipeline.run("sess", messages, emotional_history)

    rows = store.by_session("sess")
    assert len(rows) == 3, f"expected 3 persisted rows, got {len(rows)}"

    for r in rows:
        role = r["role"]
        # reliability: role heuristic
        expected_reliability = 0.7 if role == "user" else 0.4
        assert r["reliability"] == pytest.approx(expected_reliability), (
            f"role={role} reliability={r['reliability']} "
            f"expected {expected_reliability}"
        )
        # goal_relevance: cos-based, in (0, 1]
        assert 0.0 < r["goal_relevance"] <= 1.0, (
            f"goal_relevance out of range: {r['goal_relevance']}"
        )
        # value_alignment: cos-based against a real centroid → in [0, 1]
        assert 0.0 <= r["value_alignment"] <= 1.0, (
            f"value_alignment out of range: {r['value_alignment']}"
        )
        # task_utility: LLM-gated, 0 at encode
        assert r["task_utility"] == pytest.approx(0.0), (
            f"task_utility should be 0.0, got {r['task_utility']}"
        )


def test_value_alignment_zero_when_no_centroid(tmpdb):
    """value_alignment degrades to 0.0 when no value_centroid is provided."""
    self_model = SelfModel.from_seed("research curiosity learning")
    store = MemoryStore(tmpdb)
    kg = KnowledgeGraph(tmpdb)
    pipeline = MemoryConsolidationPipeline(
        db_path=tmpdb,
        knowledge_graph=kg,
        memory_store=store,
        self_model=self_model,
        value_centroid=None,
    )

    messages = [{"role": "user", "content": "hello there"}]
    pipeline.run("noval", messages, [(0.3, 0.5)])

    rows = store.by_session("noval")
    assert len(rows) == 1
    assert rows[0]["value_alignment"] == pytest.approx(0.0)
    # goal_relevance still computed (single user turn vs its own centroid → ~1)
    assert 0.0 < rows[0]["goal_relevance"] <= 1.0


def test_goal_relevance_fallback_when_no_self_model(tmpdb):
    """No self_model → no embeddings → goal_relevance falls back to 0.5."""
    store = MemoryStore(tmpdb)
    kg = KnowledgeGraph(tmpdb)
    pipeline = MemoryConsolidationPipeline(
        db_path=tmpdb,
        knowledge_graph=kg,
        memory_store=store,
        self_model=None,
        value_centroid=[1.0] * 64,
    )

    messages = [{"role": "user", "content": "hello there"}]
    pipeline.run("noself", messages, [(0.3, 0.5)])

    rows = store.by_session("noself")
    assert len(rows) == 1
    assert rows[0]["goal_relevance"] == pytest.approx(0.5)
    # value_alignment also 0.0 (no embedding to compare)
    assert rows[0]["value_alignment"] == pytest.approx(0.0)
    # reliability heuristic still applies even without self_model
    assert rows[0]["reliability"] == pytest.approx(0.7)
