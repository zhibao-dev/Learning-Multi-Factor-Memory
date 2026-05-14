"""
Tests for the Self-FEP Memory feature (Idea #1 from idea-stage/IDEA_REPORT.md).

Six behaviors that pin the self ↔ memory wiring:

  A. SelfModel updates μ_self toward observed text and refreshes π_self
     based on PE variance.
  B. Bounded outputs: π_self ∈ (0, 1], self_relevance ∈ [0, 1].
  C. Consolidation Step 3 persists self_relevance_score + embedding for
     each message when a SelfModel is wired.
  D. forget_score includes a self_resistance factor — self-relevant
     memories score lower (resist forgetting more) under matched
     emotion / importance / recency.
  E. Retrieval ranks self-relevant memories higher when current
     μ_self matches their stored embedding (mood + recency held equal).
  F. BorgeAgent end-to-end: pre_turn updates the self model, recall
     returns rows whose self_relevance is non-default.
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


# ────────────────────────────────────────────────────────────────────────
# A.  SelfModel update + precision dynamics
# ────────────────────────────────────────────────────────────────────────

def test_self_model_starts_neutral_and_grows():
    from borge.values.self_model import SelfModel
    m = SelfModel.empty()
    assert m.self_relevance_of("anything") == 0.5  # uninitialised → neutral
    m.update_from_text("I love programming and learning new things")
    assert len(m.mu_self) > 0  # μ_self now set


def test_self_relevance_higher_for_self_consistent_text():
    from borge.values.self_model import SelfModel
    m = SelfModel.from_seed("intellectual honesty depth careful reasoning")
    # Multiple updates with consistent semantic context — μ_self stabilises
    for _ in range(20):
        m.update_from_text("I value intellectual honesty above shortcuts")
    # Text close to the seed should rank above clearly unrelated text
    high = m.self_relevance_of("intellectual honesty matters to me")
    low  = m.self_relevance_of("banana split sundae mountain lake")
    assert high > low


def test_self_model_precision_in_unit_range():
    from borge.values.self_model import SelfModel
    m = SelfModel.from_seed("hello world testing precision")
    for txt in [
        "hello world", "testing precision", "completely different topic",
        "another random phrase", "yet another distinct sentence",
        "rambling about something else entirely", "more noise here",
    ]:
        m.update_from_text(txt)
    assert 0.0 < m.pi_self <= 1.0


# ────────────────────────────────────────────────────────────────────────
# B. Consolidation persists self_relevance + embedding
# ────────────────────────────────────────────────────────────────────────

def test_consolidation_persists_self_relevance_and_embedding(tmpdb):
    from borge.agent import BorgeAgent

    a = BorgeAgent(agent_backend=None, db_path=tmpdb)
    a.on_session_start()

    # Inject deterministic emotional history
    a._emotional_history   = [(0.5, 0.7), (-0.3, 0.6), (0.1, 0.4)]
    a._session_f_history   = [0.6, 0.55, 0.50]

    messages = [
        {"role": "user",      "content": "I am genuinely excited about this work"},
        {"role": "assistant", "content": "glad to hear"},
        {"role": "user",      "content": "frustrated that the test keeps failing"},
        {"role": "assistant", "content": "let me check"},
        {"role": "user",      "content": "ok thanks"},
        {"role": "assistant", "content": "anytime"},
    ]
    a.on_session_end(session_id="s-self", messages=messages)

    from borge.memory.store import MemoryStore
    rows = MemoryStore(tmpdb).by_session("s-self")
    assert len(rows) >= 3

    # All rows must have a non-null self_relevance_score and a stored embedding
    for r in rows:
        assert r["self_relevance_score"] is not None
        assert isinstance(r["self_relevance_score"], float)
        assert 0.0 <= r["self_relevance_score"] <= 1.0
        assert r["embedding"] is not None
        # embedding is a JSON-encoded list
        import json
        emb = json.loads(r["embedding"])
        assert isinstance(emb, list) and len(emb) > 0


# ────────────────────────────────────────────────────────────────────────
# C. forget_score: self-relevant memories resist forgetting
# ────────────────────────────────────────────────────────────────────────

def test_forget_score_lower_for_self_relevant_memory(tmpdb):
    """Two identical memories, one with high self_relevance, one low.

    The self-relevant one should resist forgetting more (lower forget_score).
    """
    from borge.memory.store import MemoryStore
    from borge.memory.forgetting import ForgettingEngine

    store = MemoryStore(tmpdb)
    base = {
        "session_id":       "s",
        "role":             "user",
        "content":          "x",
        "timestamp":        (datetime.now() - timedelta(days=10)).isoformat(),
        "importance_score": 0.5,
        "retrieval_count":  0,
        # SCHEMATIC depth so neither row is deleted/compressed
        "encoding_depth":   3,
        # zero emotion to isolate the self effect
        "emotional_valence": 0.0,
        "emotional_arousal": 0.5,
    }
    distant = {**base, "id": "m-distant", "self_relevance_score": 0.1}
    central = {**base, "id": "m-central", "self_relevance_score": 0.9}
    store.insert(distant)
    store.insert(central)

    ForgettingEngine().run_forgetting_pass(tmpdb)

    d_row = store.get("m-distant")
    c_row = store.get("m-central")
    assert d_row is not None and c_row is not None
    assert c_row["forget_score"] < d_row["forget_score"], (
        f"self-relevant memory should resist forgetting more "
        f"(central={c_row['forget_score']}, distant={d_row['forget_score']})"
    )


# ────────────────────────────────────────────────────────────────────────
# D. Retrieval ranking uses self_similarity
# ────────────────────────────────────────────────────────────────────────

def test_retrieval_ranks_self_consistent_memory_higher(tmpdb):
    """With same mood + recency + content overlap, the row whose stored
    embedding matches the agent's current μ_self should rank higher."""
    from borge.memory.store import MemoryStore
    from borge.memory.retrieval import MemoryRetrieval
    from borge.values.self_model import SelfModel, embed

    store = MemoryStore(tmpdb)
    # Build a stable self model around "research learning curiosity"
    self_model = SelfModel.from_seed("research learning curiosity science")
    for _ in range(15):
        self_model.update_from_text("research learning and curiosity drive me")

    base = {
        "session_id": "s",
        "role": "user",
        "content": "filler",
        "timestamp": datetime.now().isoformat(),
        "emotional_valence": 0.0,
        "emotional_arousal": 0.5,
        "importance_score": 0.5,
        "self_relevance_score": 0.5,
    }
    # Self-aligned memory
    aligned = {**base, "id": "m-aligned",
               "embedding": embed("research and learning is what i live for")}
    # Self-distant memory
    distant = {**base, "id": "m-distant",
               "embedding": embed("pickles in the refrigerator")}
    store.insert(aligned)
    store.insert(distant)

    ret = MemoryRetrieval(tmpdb, store=store, self_model=self_model)
    results = ret.recall(
        query="",
        current_valence=0.0, current_arousal=0.5,
        k=2,
        mood_weight=0.0, recency_weight=0.0,
        relevance_weight=0.0, f_weight=0.0,
        self_weight=1.0,
    )
    assert results[0]["id"] == "m-aligned", (
        f"self-aligned memory should rank first; got {[r['id'] for r in results]}"
    )


# ────────────────────────────────────────────────────────────────────────
# E. BorgeAgent end-to-end
# ────────────────────────────────────────────────────────────────────────

def test_borge_agent_end_to_end_self_loop(tmpdb):
    from borge.agent import BorgeAgent
    from borge.memory.store import MemoryStore

    a = BorgeAgent(agent_backend=None, db_path=tmpdb)
    a.on_session_start(user_id="cli")

    a.pre_turn("I care deeply about getting the answer right", [])
    a.pre_turn("absolutely committed to learning the truth", [{"role":"user","content":"earlier"}])

    a.on_session_end(session_id="e2e-self", messages=[
        {"role": "user",      "content": "I care deeply about getting the answer right"},
        {"role": "assistant", "content": "noted"},
        {"role": "user",      "content": "absolutely committed to learning the truth"},
        {"role": "assistant", "content": "great"},
    ])

    rows = MemoryStore(tmpdb).by_session("e2e-self")
    assert len(rows) >= 2
    # Some rows must have non-default self_relevance (model updated through session)
    srs = [r["self_relevance_score"] for r in rows]
    assert any(abs(s - 0.5) > 1e-6 for s in srs), (
        f"self_relevance should diverge from 0.5 default; got {srs}"
    )

    # recall returns ranked dicts
    out = a.recall("learning")
    assert isinstance(out, list)
    assert len(out) >= 1
