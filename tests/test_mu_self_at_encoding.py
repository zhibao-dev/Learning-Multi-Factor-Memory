"""
TDD tests for L5: persist μ_self_at_encoding per memory and use it in
retrieval ranking (encoding-specificity-faithful).

The v0.1 proposal sketch specified that retrieval should match
cos(current μ_self, memory.μ_self_at_encoding), not cos(current μ_self,
memory.embedding). v0.3 still used the latter (limitation L5).

These tests pin the new behaviour:
  A. borge_memories schema gains a `mu_self_at_encoding` column.
  B. Consolidation Step 3 persists the SelfModel's μ_self snapshot.
  C. Retrieval ranking prefers the snapshot when available.
  D. Retrieval still works for legacy rows without a snapshot.
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
# C. Retrieval prefers the snapshot over memory.embedding
# ────────────────────────────────────────────────────────────────────

def test_self_similarity_uses_snapshot_when_present(tmpdb):
    """
    Direct test of the `_self_similarity` helper: when a memory row has
    `mu_self_at_encoding`, the score must be cos(current μ_self,
    snapshot) — NOT cos(current μ_self, memory.embedding).

    Setup: an embedding orthogonal to μ_self. A snapshot equal to μ_self.
    Snapshot-based score ≈ 1; embedding-based score ≈ 0.5 (neutralised).
    """
    from borge.memory.retrieval import MemoryRetrieval
    from borge.values.self_model import SelfModel

    self_model = SelfModel.from_seed("research curiosity learning")
    for _ in range(15):
        self_model.update_from_text("I love research curiosity learning")
    current_mu = list(self_model.mu_self)

    # An embedding that's orthogonal to μ_self (cos ≈ 0 → sr ≈ 0.5)
    orthogonal = list(current_mu)
    orthogonal[0] = -orthogonal[0]   # flip one component to drop similarity
    orthogonal = [x + 0.5 for x in orthogonal]  # arbitrary noise; far from μ_self

    row_with_snapshot = {
        "embedding":            json.dumps(orthogonal),
        "mu_self_at_encoding":  json.dumps(current_mu),
        "self_relevance_score": 0.5,
    }
    row_without_snapshot = {
        "embedding":            json.dumps(orthogonal),
        "mu_self_at_encoding":  None,
        "self_relevance_score": 0.5,
    }

    ret = MemoryRetrieval(tmpdb, self_model=self_model)
    sim_snap = ret._self_similarity(row_with_snapshot)
    sim_noembsnap = ret._self_similarity(row_without_snapshot)

    assert sim_snap > sim_noembsnap + 0.2, (
        f"Snapshot should dominate (sim_snap={sim_snap:.4f}, "
        f"sim_embedding={sim_noembsnap:.4f})"
    )
    # And specifically: snapshot equal to μ_self → similarity ≈ 1
    assert sim_snap > 0.9


def test_retrieval_falls_back_to_embedding_when_no_snapshot(tmpdb):
    """Legacy rows without a snapshot must still be ranked by embedding."""
    from borge.memory.store import MemoryStore
    from borge.memory.retrieval import MemoryRetrieval
    from borge.values.self_model import SelfModel, embed

    store = MemoryStore(tmpdb)
    self_model = SelfModel.from_seed("research curiosity learning")
    for _ in range(15):
        self_model.update_from_text("research curiosity learning")

    base = {
        "session_id": "s", "role": "user", "content": "x",
        "timestamp": datetime.now().isoformat(),
        "emotional_valence": 0.0, "emotional_arousal": 0.5,
        "importance_score": 0.5, "self_relevance_score": 0.5,
        # NB: no mu_self_at_encoding for either row (legacy data)
    }
    store.insert({**base, "id": "m-near",
                  "embedding": embed("research and learning drive me")})
    store.insert({**base, "id": "m-far",
                  "embedding": embed("pickles in the refrigerator")})

    ret = MemoryRetrieval(tmpdb, store=store, self_model=self_model)
    results = ret.recall(
        query="",
        current_valence=0.0, current_arousal=0.5,
        k=2,
        mood_weight=0.0, recency_weight=0.0,
        relevance_weight=0.0, f_weight=0.0,
        self_weight=1.0,
    )
    assert results[0]["id"] == "m-near"
