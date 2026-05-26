"""
Borge Memory Store — persistent storage for cognitive memory entries.

Standalone Borge needs its own table for memory persistence because in
standalone mode there is no Hermes `messages` table. This module owns the
`borge_memories` table where consolidation Step 5 writes emotional /
encoding-depth / free-energy decisions, and from which Forgetting and
Retrieval read.

Schema is intentionally a superset of the Hermes `messages` columns added
by `forgetting.BORGE_COLUMNS_SQL`, plus two new free-energy fields:
  - `f_total_at_encoding`  free energy at the turn this memory was formed
  - `delta_f_total`        F change vs. previous turn (positive = progress)

The two new columns let `forget_score` and mood-congruent retrieval factor
in cognitive surprise / progress, not just emotional valence × arousal.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from typing import Any, Optional

log = logging.getLogger(__name__)


BORGE_MEMORIES_SCHEMA = """
CREATE TABLE IF NOT EXISTS borge_memories (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT,
    content TEXT,
    timestamp TEXT NOT NULL,
    emotional_valence REAL DEFAULT 0.0,
    emotional_arousal REAL DEFAULT 0.5,
    emotional_significance REAL DEFAULT 0.0,
    encoding_depth INTEGER DEFAULT 1,
    importance_score REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    last_retrieved TEXT,
    forget_score REAL DEFAULT 0.0,
    f_total_at_encoding REAL,
    delta_f_total REAL,
    entity_tags TEXT DEFAULT '[]',
    graph_node_ids TEXT DEFAULT '[]',
    self_relevance_score REAL DEFAULT 0.5,
    embedding TEXT,
    mu_self_at_encoding TEXT,
    goal_relevance REAL DEFAULT 0.0,
    value_alignment REAL DEFAULT 0.0,
    task_utility REAL DEFAULT 0.0,
    reliability REAL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_borge_memories_session ON borge_memories(session_id);
CREATE INDEX IF NOT EXISTS idx_borge_memories_depth ON borge_memories(encoding_depth);
"""


class MemoryStore:
    """
    SQLite-backed persistent store for `borge_memories`.

    All operations open/close their own connection so the store is
    safe to share across threads without explicit pooling.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

    # ── Schema ────────────────────────────────────────────────────────────

    # Columns added after v0.1; ALTER each one to migrate pre-existing DBs.
    _LATE_COLUMNS = (
        ("self_relevance_score",  "REAL DEFAULT 0.5"),
        ("embedding",              "TEXT"),
        # v0.4 (L5): encoding-time μ_self snapshot for encoding-specificity-
        # faithful retrieval. Stored as JSON-encoded list of floats.
        ("mu_self_at_encoding",    "TEXT"),
        # Sister paper (multi-factor value): 4 extra value factors.
        ("goal_relevance",         "REAL DEFAULT 0.0"),
        ("value_alignment",        "REAL DEFAULT 0.0"),
        ("task_utility",           "REAL DEFAULT 0.0"),
        ("reliability",            "REAL DEFAULT 0.0"),
    )

    def ensure_table(self) -> None:
        """Create the table + indexes if they don't exist, and migrate older DBs."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for stmt in BORGE_MEMORIES_SCHEMA.strip().split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        conn.execute(stmt)
                for col, typ in self._LATE_COLUMNS:
                    try:
                        conn.execute(f"ALTER TABLE borge_memories ADD COLUMN {col} {typ}")
                    except sqlite3.OperationalError:
                        pass  # column already exists
        except sqlite3.OperationalError as e:
            log.warning(f"[MemoryStore] ensure_table failed: {e}")

    # ── Writes ────────────────────────────────────────────────────────────

    def insert(self, entry: dict[str, Any]) -> None:
        """
        Insert (or replace) a memory entry. The dict must contain at least
        `id`, `session_id`, `timestamp`. All other fields fall back to
        column defaults.
        """
        self.ensure_table()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO borge_memories (
                        id, session_id, role, content, timestamp,
                        emotional_valence, emotional_arousal, emotional_significance,
                        encoding_depth, importance_score, retrieval_count,
                        last_retrieved, forget_score, f_total_at_encoding,
                        delta_f_total, entity_tags, graph_node_ids,
                        self_relevance_score, embedding, mu_self_at_encoding
                    ) VALUES (
                        :id, :session_id, :role, :content, :timestamp,
                        :emotional_valence, :emotional_arousal, :emotional_significance,
                        :encoding_depth, :importance_score, :retrieval_count,
                        :last_retrieved, :forget_score, :f_total_at_encoding,
                        :delta_f_total, :entity_tags, :graph_node_ids,
                        :self_relevance_score, :embedding, :mu_self_at_encoding
                    )""",
                    self._normalize(entry),
                )
        except sqlite3.OperationalError as e:
            log.warning(f"[MemoryStore] insert failed: {e}")

    def update_importance(self, memory_id: str, importance: float) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE borge_memories SET importance_score = ? WHERE id = ?",
                    (max(0.0, min(1.0, importance)), memory_id),
                )
        except sqlite3.OperationalError as e:
            log.warning(f"[MemoryStore] update_importance failed: {e}")

    def update_score(self, memory_id: str, score: float) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE borge_memories SET forget_score = ? WHERE id = ?",
                    (round(score, 4), memory_id),
                )
        except sqlite3.OperationalError as e:
            log.warning(f"[MemoryStore] update_score failed: {e}")

    def record_retrieval(self, memory_id: str) -> None:
        """Bump retrieval_count and set last_retrieved = now."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """UPDATE borge_memories
                       SET retrieval_count = retrieval_count + 1,
                           last_retrieved  = ?
                       WHERE id = ?""",
                    (datetime.now().isoformat(), memory_id),
                )
        except sqlite3.OperationalError as e:
            log.warning(f"[MemoryStore] record_retrieval failed: {e}")

    def delete(self, memory_id: str) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM borge_memories WHERE id = ?", (memory_id,))
        except sqlite3.OperationalError as e:
            log.warning(f"[MemoryStore] delete failed: {e}")

    def compress(self, memory_id: str) -> None:
        """Replace content with a stub marker; keep metadata."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE borge_memories SET content = ? WHERE id = ?",
                    (f"[compressed:{memory_id[:8]}]", memory_id),
                )
        except sqlite3.OperationalError as e:
            log.warning(f"[MemoryStore] compress failed: {e}")

    # ── Reads ─────────────────────────────────────────────────────────────

    def all(self) -> list[dict]:
        """Return every row (small DBs only; for retrieval use `query`)."""
        self.ensure_table()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                return [dict(r) for r in conn.execute(
                    "SELECT * FROM borge_memories ORDER BY timestamp DESC"
                ).fetchall()]
        except sqlite3.OperationalError as e:
            log.warning(f"[MemoryStore] all() failed: {e}")
            return []

    def by_session(self, session_id: str) -> list[dict]:
        self.ensure_table()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                return [dict(r) for r in conn.execute(
                    "SELECT * FROM borge_memories WHERE session_id = ? ORDER BY timestamp",
                    (session_id,),
                ).fetchall()]
        except sqlite3.OperationalError as e:
            log.warning(f"[MemoryStore] by_session failed: {e}")
            return []

    def get(self, memory_id: str) -> Optional[dict]:
        self.ensure_table()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM borge_memories WHERE id = ?", (memory_id,)
                ).fetchone()
                return dict(row) if row else None
        except sqlite3.OperationalError as e:
            log.warning(f"[MemoryStore] get failed: {e}")
            return None

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(entry: dict[str, Any]) -> dict[str, Any]:
        """Fill in defaults for all schema columns from a partial dict."""
        def _j(v: Any) -> str:
            return v if isinstance(v, str) else json.dumps(v or [])

        emb = entry.get("embedding")
        emb_json = None if emb is None else (
            emb if isinstance(emb, str) else json.dumps(list(emb))
        )

        mu_snap = entry.get("mu_self_at_encoding")
        mu_snap_json = None if mu_snap is None else (
            mu_snap if isinstance(mu_snap, str) else json.dumps(list(mu_snap))
        )

        return {
            "id":                     entry["id"],
            "session_id":             entry["session_id"],
            "role":                   entry.get("role", ""),
            "content":                entry.get("content", "") or "",
            "timestamp":              entry.get("timestamp") or datetime.now().isoformat(),
            "emotional_valence":      float(entry.get("emotional_valence", 0.0)),
            "emotional_arousal":      float(entry.get("emotional_arousal", 0.5)),
            "emotional_significance": float(entry.get("emotional_significance", 0.0)),
            "encoding_depth":         int(entry.get("encoding_depth", 1)),
            "importance_score":       float(entry.get("importance_score", 0.5)),
            "retrieval_count":        int(entry.get("retrieval_count", 0)),
            "last_retrieved":         entry.get("last_retrieved"),
            "forget_score":           float(entry.get("forget_score", 0.0)),
            "f_total_at_encoding":    entry.get("f_total_at_encoding"),
            "delta_f_total":          entry.get("delta_f_total"),
            "entity_tags":            _j(entry.get("entity_tags", [])),
            "graph_node_ids":         _j(entry.get("graph_node_ids", [])),
            "self_relevance_score":   float(entry.get("self_relevance_score", 0.5)),
            "embedding":              emb_json,
            "mu_self_at_encoding":    mu_snap_json,
        }
