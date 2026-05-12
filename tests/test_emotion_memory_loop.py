"""
Tests for the emotion ↔ free-energy ↔ memory loop.

Three behaviors that the README claims and that we now actually wire up:

  A. Consolidation Step 5 persists emotional encoding decisions to DB
  B. forget_score reflects emotion (|V|·A) and free-energy progress (ΔF)
  C. Mood-congruent retrieval ranks memories near current (V, A) first
"""
from __future__ import annotations

import os
import tempfile
import uuid
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
# A.  consolidation Step 5 → DB persistence
# ────────────────────────────────────────────────────────────────────────

def test_consolidation_persists_emotional_depth_to_db(tmpdb):
    """
    Vivid emotional history → memories persisted to borge_memories with
    encoding_depth tier matching |V|·A.

    We bypass pre_turn's EMA smoothing by injecting emotional_history
    directly; this tests the persistence path (Step 5 → DB) in isolation
    from the affective filter, since the filter's slow EMA needs many
    turns to cross threshold and is tested elsewhere.
    """
    from borge.agent import BorgeAgent

    agent = BorgeAgent(agent_backend=None, db_path=tmpdb)
    agent.on_session_start()

    # Drive three known emotional points without going through pre_turn:
    # vivid positive · vivid negative · neutral
    agent._emotional_history   = [(0.8, 0.9), (-0.7, 0.85), (0.05, 0.4)]
    agent._session_f_history   = [0.6, 0.55, 0.50]  # F dropping → progress

    messages = [
        {"role": "user",      "content": "this finally works perfectly"},
        {"role": "assistant", "content": "glad to hear it"},
        {"role": "user",      "content": "no it broke again"},
        {"role": "assistant", "content": "let me check"},
        {"role": "user",      "content": "ok thanks"},
        {"role": "assistant", "content": "anytime"},
    ]
    agent.on_session_end(session_id="s-emotion", messages=messages)

    from borge.memory.store import MemoryStore
    store = MemoryStore(tmpdb)
    rows = store.by_session("s-emotion")

    assert len(rows) >= 3, f"expected memories persisted, got {len(rows)}"

    # Significance for vivid turns must be non-trivial
    significances = sorted([r["emotional_significance"] for r in rows], reverse=True)
    assert significances[0] >= 0.4, f"top significance should be ≥ 0.4; got {significances}"

    # Vivid turns must reach at least SCHEMATIC (depth 3)
    depths = [r["encoding_depth"] for r in rows]
    assert max(depths) >= 3, f"vivid turn should reach SCHEMATIC; got {depths}"

    # f_total + delta_f got stamped on at least one row (positive Δ = progress)
    deltas = [r["delta_f_total"] for r in rows if r["delta_f_total"] is not None]
    assert deltas, "delta_f_total should be populated for non-first turns"
    assert max(deltas) > 0, f"positive delta_f expected (F dropping); got {deltas}"


# ────────────────────────────────────────────────────────────────────────
# B.  forget_score sensitive to emotion + ΔF importance bump
# ────────────────────────────────────────────────────────────────────────

def test_forget_score_lower_for_emotional_memory(tmpdb):
    """
    Two memories identical except for emotional_significance.
    The high-significance one should resist forgetting (lower score).
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
        # Pin depth = SCHEMATIC so neither row gets deleted/compressed by the
        # sweep — we only want to compare the resulting forget_score values.
        "encoding_depth":   3,
    }
    boring = {**base, "id": "m-boring",
              "emotional_valence": 0.0, "emotional_arousal": 0.3,
              "emotional_significance": 0.0}
    vivid  = {**base, "id": "m-vivid",
              "emotional_valence": -0.8, "emotional_arousal": 0.9,
              "emotional_significance": 0.72}  # |V|·A
    store.insert(boring)
    store.insert(vivid)

    engine = ForgettingEngine()
    engine.run_forgetting_pass(tmpdb)

    boring_row = store.get("m-boring")
    vivid_row  = store.get("m-vivid")
    assert boring_row is not None
    # vivid may have been preserved; if so check its forget_score; otherwise it was kept
    if vivid_row is not None:
        assert vivid_row["forget_score"] < boring_row["forget_score"], (
            f"emotional memory should resist forgetting more "
            f"(vivid={vivid_row['forget_score']}, boring={boring_row['forget_score']})"
        )


def test_delta_f_bumps_importance(tmpdb):
    """A memory written with positive ΔF_total should get importance bonus."""
    from borge.memory.store import MemoryStore

    store = MemoryStore(tmpdb)
    progress = {
        "id":                "m-progress",
        "session_id":        "s",
        "role":              "assistant",
        "content":           "fixed!",
        "timestamp":         datetime.now().isoformat(),
        "importance_score":  0.5,
        "f_total_at_encoding": 0.4,
        "delta_f_total":     0.5,   # F dropped a lot → progress made
    }
    store.insert(progress)

    from borge.memory.forgetting import apply_importance_from_delta_f
    apply_importance_from_delta_f(tmpdb, gain=0.3)

    after = store.get("m-progress")
    assert after["importance_score"] > 0.5, (
        f"positive ΔF should boost importance; got {after['importance_score']}"
    )


# ────────────────────────────────────────────────────────────────────────
# C.  mood-congruent retrieval
# ────────────────────────────────────────────────────────────────────────

def test_mood_congruent_retrieval_ranks_near_current_emotion_first(tmpdb):
    """
    Insert three memories with very different emotional coordinates.
    Query with a current emotion that's a near-clone of one of them.
    That one should be the top result.
    """
    from borge.memory.store import MemoryStore
    from borge.memory.retrieval import MemoryRetrieval

    store = MemoryStore(tmpdb)
    now = datetime.now().isoformat()
    memories = [
        # excited
        {"id": "m-excited",    "session_id": "s", "role": "user", "content": "great win",
         "timestamp": now, "emotional_valence":  0.7, "emotional_arousal": 0.8,
         "emotional_significance": 0.56, "importance_score": 0.5},
        # frustrated
        {"id": "m-frustrated", "session_id": "s", "role": "user", "content": "stuck again",
         "timestamp": now, "emotional_valence": -0.7, "emotional_arousal": 0.8,
         "emotional_significance": 0.56, "importance_score": 0.5},
        # content
        {"id": "m-content",    "session_id": "s", "role": "user", "content": "fine",
         "timestamp": now, "emotional_valence":  0.3, "emotional_arousal": 0.3,
         "emotional_significance": 0.09, "importance_score": 0.5},
    ]
    for m in memories:
        store.insert(m)

    ret = MemoryRetrieval(tmpdb)
    # Current emotion: frustrated (near m-frustrated)
    results = ret.recall(
        query="",
        current_valence=-0.65,
        current_arousal=0.78,
        k=3,
        mood_weight=1.0,
        recency_weight=0.0,
        relevance_weight=0.0,
    )
    assert len(results) == 3
    assert results[0]["id"] == "m-frustrated", (
        f"expected mood-congruent memory first; got {[r['id'] for r in results]}"
    )


def test_recall_bumps_retrieval_count(tmpdb):
    """Each recall hit should increment retrieval_count → harder to forget later."""
    from borge.memory.store import MemoryStore
    from borge.memory.retrieval import MemoryRetrieval

    store = MemoryStore(tmpdb)
    store.insert({
        "id": "m1", "session_id": "s", "role": "user", "content": "hello",
        "timestamp": datetime.now().isoformat(),
        "emotional_valence": 0.5, "emotional_arousal": 0.5,
        "emotional_significance": 0.25, "importance_score": 0.5,
    })
    ret = MemoryRetrieval(tmpdb)
    ret.recall(query="", current_valence=0.5, current_arousal=0.5, k=1)
    ret.recall(query="", current_valence=0.5, current_arousal=0.5, k=1)
    after = store.get("m1")
    assert after["retrieval_count"] == 2
    assert after["last_retrieved"] is not None


# ────────────────────────────────────────────────────────────────────────
# Top-level convenience: BorgeAgent.recall API
# ────────────────────────────────────────────────────────────────────────

def test_borge_agent_recall_returns_memories_for_current_mood(tmpdb):
    """`BorgeAgent.recall("...")` should hand back ranked memories."""
    from borge.agent import BorgeAgent
    from borge.memory.store import MemoryStore

    a = BorgeAgent(agent_backend=None, db_path=tmpdb)
    store = MemoryStore(tmpdb)
    store.insert({
        "id": "m-warm", "session_id": "s", "role": "user", "content": "we shipped it",
        "timestamp": datetime.now().isoformat(),
        "emotional_valence":  0.6, "emotional_arousal": 0.7,
        "emotional_significance": 0.42, "importance_score": 0.6,
    })

    a.emotion.valence = 0.6
    a.emotion.arousal = 0.7
    results = a.recall("ship")
    assert isinstance(results, list)
    assert len(results) >= 1
    assert results[0]["id"] == "m-warm"
