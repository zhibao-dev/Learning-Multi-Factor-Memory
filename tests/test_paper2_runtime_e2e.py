"""
Task 8 (paper2 runtime wiring) — capstone end-to-end proof.

One shared ``MemoryValue`` (``BorgeAgent._memory_value``) drives all three
live memory operations on the *agent's own* wired engines:

  ENCODE   consolidation Step 3 → ``value_encoding_depth``
  FORGET   ForgettingEngine     → ``value_forget_score``
  RETRIEVE MemoryRetrieval      → ``w_v · V + query terms + mood + recency``

We craft a HIGH-value turn H (vivid + self-referential) and a LOW-value
turn L (neutral chatter) and show each operation treats H and L
consistently, driven by that single value — API-free and deterministic
(hash_embed default, fixed contents/timestamps, no llm_caller).
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta

import pytest

from borge.agent import BorgeAgent
from borge.memory.value import memory_factors


# HIGH-value: vivid (V/A high) + self-referential ("I", "my").
# LOW-value: neutral chatter, flat affect. They share a distinctive token
# each ("project" / "weather") so the retrieval query can equalise
# token-overlap relevance and let V do the differentiating.
H_CONTENT = "I love my project and my family deeply"
L_CONTENT = "ok the weather is fine"


@pytest.fixture()
def agent(tmp_path, monkeypatch):
    # Isolate the agent: temp BORGE_HOME (SOUL/config resolution) + temp db,
    # so the test never touches ~/.borge and starts from a clean store.
    monkeypatch.setenv("BORGE_HOME", str(tmp_path))
    db = str(tmp_path / "borge.db")
    a = BorgeAgent(None, db_path=db)
    # Embeddings (self_relevance / goal_relevance factors) need a self model.
    assert a.self_model is not None
    return a


def _encode_session(a: BorgeAgent, session_id: str) -> dict[str, dict]:
    """
    Drive the REAL encode path through the agent's OWN consolidation engine.

    We call ``a._consolidation.run`` directly — the exact entry
    ``on_session_end`` uses — with an explicit ``emotional_history`` (one
    (V, A) per user turn). This is preferred over ``pre_turn`` +
    ``on_session_end`` because ``pre_turn`` runs the heuristic signal
    extractor + EMA affect dynamics, which would not yield the clean,
    deterministic (0.9, 0.9) vs (0.0, 0.1) pairing this proof needs.
    Returns {id: persisted_row}.
    """
    messages = [
        {"role": "user", "content": H_CONTENT, "id": "H"},
        {"role": "user", "content": L_CONTENT, "id": "L"},
    ]
    a._consolidation.run(
        session_id=session_id,
        messages=messages,
        emotional_history=[(0.9, 0.9), (0.0, 0.1)],  # aligned to H, L
        f_history=[1.0, 1.0],                          # equal F → no ΔF bias
    )
    return {r["id"]: r for r in a._consolidation.store.by_session(session_id)}


def test_one_memory_value_drives_encode_forget_retrieve(agent):
    a = agent

    # ── IDENTITY: one MemoryValue, three operations ─────────────────────
    assert (
        a._forgetting.memory_value
        is a._consolidation.memory_value
        is a._retrieval.memory_value
        is a._memory_value
    )

    # ── ENCODE: depth tracks value ──────────────────────────────────────
    rows = _encode_session(a, "S_encode")
    depth_h = rows["H"]["encoding_depth"]
    depth_l = rows["L"]["encoding_depth"]
    v_h = a._memory_value.value(memory_factors(rows["H"]))
    v_l = a._memory_value.value(memory_factors(rows["L"]))
    # The single value separates the two memories...
    assert v_h > v_l, f"H must out-value L: V_h={v_h:.4f} V_l={v_l:.4f}"
    # ...and that value yields a deeper Craik-Lockhart tier for H. Under the
    # shipped weights the crafted H/L are strictly separated (3 vs 2); >= is
    # the contractual guarantee, the strict check documents the real margin.
    assert depth_h >= depth_l
    assert depth_h > depth_l, (
        f"shipped weights should encode H deeper: "
        f"depth_h={depth_h} depth_l={depth_l}"
    )

    # ── FORGET: equally aged → low pruned, high survives (V-driven) ─────
    # Age BOTH rows identically (recency cancels out), force the prune path
    # (encoding_depth = 1 = SHALLOW, retrieval_count = 0) so the only thing
    # left to separate them is V. The real depths here are >1, so we set
    # depth=1 directly to exercise the deletion branch (depth<=1 & score>τ).
    # 45 days puts L's score well above prune_threshold (2.0) and H's well
    # below — a comfortable margin on both sides (see the e2e sweep).
    aged_ts = (datetime.now() - timedelta(days=45)).isoformat()
    with sqlite3.connect(a._db_path) as conn:
        conn.execute(
            "UPDATE borge_memories "
            "SET timestamp = ?, encoding_depth = 1, "
            "    retrieval_count = 0, last_retrieved = NULL "
            "WHERE session_id = 'S_encode'",
            (aged_ts,),
        )

    stats = a._forgetting.run_forgetting_pass(a._db_path)
    assert stats["deleted"] == 1, f"exactly the low-value row should prune: {stats}"
    survivors = {r["id"] for r in a._consolidation.store.by_session("S_encode")}
    assert survivors == {"H"}, (
        f"high-value H must survive, low-value L must be forgotten "
        f"at equal recency — got {survivors}"
    )

    # ── RETRIEVE: high ranks above low ──────────────────────────────────
    # Re-persist a fresh session (L was just pruned) so both rows are
    # present, then rank via the agent's own retrieval engine. The query
    # shares one token with each ("project" ∈ H, "weather" ∈ L) so
    # token-overlap relevance is equal and V is the differentiator.
    _encode_session(a, "S_retrieve")
    results = a._retrieval.recall(
        query="project weather",
        current_valence=0.0,
        current_arousal=0.5,
        k=10,
    )
    order = [r["id"] for r in results if r["id"] in ("H", "L")]
    assert order.index("H") < order.index("L"), (
        f"high-value H must rank above low-value L: order={order}"
    )
