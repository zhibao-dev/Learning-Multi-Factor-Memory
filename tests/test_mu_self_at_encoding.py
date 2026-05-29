"""
TDD tests for L5: persist μ_self_at_encoding per memory and use it in
retrieval ranking (encoding-specificity-faithful).

The v0.1 proposal sketch specified that retrieval should match
cos(current μ_self, memory.μ_self_at_encoding), not cos(current μ_self,
memory.embedding). v0.3 still used the latter (limitation L5).

These tests pin the new behaviour:
  A. borge_memories schema gains a `mu_self_at_encoding` column.
  B. Consolidation Step 3 persists the SelfModel's μ_self snapshot.

Note (paper2, multi-factor-eval branch): the original items C/D — retrieval
ranking via the snapshot/embedding↔μ_self cosine — are SUPERSEDED here.
Retrieval now ranks the self signal through the shared MemoryValue's
`self_relevance` factor (the stored `self_relevance_score`), so the
embedding/snapshot cosine is no longer consumed at retrieval. The snapshot
is still persisted (B) for provenance; value-driven self recall is covered
by tests/test_self_fep_memory.py and tests/test_retrieval_value.py.
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from datetime import datetime

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


# ────────────────────────────────────────────────────────────────────
# A. Schema has the new column
# ────────────────────────────────────────────────────────────────────

def test_borge_memories_has_mu_self_at_encoding_column(tmpdb):
    from borge.memory.store import MemoryStore
    store = MemoryStore(tmpdb)
    store.ensure_table()
    with sqlite3.connect(tmpdb) as c:
        cols = {row[1] for row in c.execute("PRAGMA table_info(borge_memories)")}
    assert "mu_self_at_encoding" in cols


# ────────────────────────────────────────────────────────────────────
# B. Consolidation Step 3 persists μ_self snapshot
# ────────────────────────────────────────────────────────────────────

def test_consolidation_persists_mu_self_snapshot(tmpdb):
    from borge.agent import BorgeAgent

    agent = BorgeAgent(agent_backend=None, db_path=tmpdb)
    agent.on_session_start()
    agent._emotional_history = [(0.7, 0.8)]
    agent._session_f_history = [0.5]

    messages = [
        {"role": "user", "content": "I really value careful learning"},
        {"role": "assistant", "content": "noted"},
    ]
    agent.on_session_end(session_id="snap", messages=messages)

    with sqlite3.connect(tmpdb) as c:
        c.row_factory = sqlite3.Row
        rows = list(c.execute(
            "SELECT mu_self_at_encoding FROM borge_memories WHERE session_id=?",
            ("snap",),
        ))
    assert len(rows) >= 2
    snaps = [r["mu_self_at_encoding"] for r in rows]
    # At least one row must carry the snapshot
    non_null = [s for s in snaps if s]
    assert non_null, f"expected μ_self snapshots on persisted rows, got {snaps}"

    # The snapshot must JSON-decode to a list of floats matching the agent's
    # current μ_self dimensionality.
    parsed = json.loads(non_null[0])
    assert isinstance(parsed, list)
    assert len(parsed) == len(agent.self_model.mu_self)


# ────────────────────────────────────────────────────────────────────
# C/D. (Superseded on multi-factor-eval) Retrieval ranking by the
# snapshot/embedding↔μ_self cosine is no longer used — the self signal is
# carried by MemoryValue's `self_relevance` factor. Value-driven self
# recall is covered by test_self_fep_memory.py and test_retrieval_value.py.
# ────────────────────────────────────────────────────────────────────
