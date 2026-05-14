"""
Forgetting Engine — Active Memory Decay

Implements Ebbinghaus-inspired forgetting to prevent memory accumulation
from degrading retrieval quality.

Forgetting is tiered by encoding depth:
  SHALLOW  + forget_score > PRUNE_THRESHOLD  → delete
  SEMANTIC + forget_score > COMPRESS_THRESHOLD → compress to entity tag only
  SCHEMATIC / META                            → never delete, only compress

The score formula factors emotion AND self-relevance explicitly:
  score = recency_decay
        × usage_penalty
        × importance_resistance
        × emotion_resistance     ← vivid memories resist forgetting
        × self_resistance        ← self-relevant memories resist forgetting
                                   (gated by self-precision π_self at encoding)
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime

log = logging.getLogger(__name__)

PRUNE_THRESHOLD    = 2.0   # SHALLOW entries above this are deleted
COMPRESS_THRESHOLD = 3.0   # SEMANTIC entries above this are compressed

EMOTION_RESISTANCE_ALPHA = 2.0  # how strongly |V|·A resists forgetting
SELF_RESISTANCE_LAMBDA   = 2.0  # how strongly self-relevance resists forgetting

# Borge DB columns added to existing Hermes messages table
BORGE_COLUMNS_SQL = """
ALTER TABLE messages ADD COLUMN IF NOT EXISTS emotional_valence REAL DEFAULT 0.0;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS emotional_arousal REAL DEFAULT 0.5;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS emotional_significance REAL DEFAULT 0.0;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS encoding_depth INTEGER DEFAULT 1;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS entity_tags TEXT DEFAULT '[]';
ALTER TABLE messages ADD COLUMN IF NOT EXISTS graph_node_ids TEXT DEFAULT '[]';
ALTER TABLE messages ADD COLUMN IF NOT EXISTS retrieval_count INTEGER DEFAULT 0;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS last_retrieved TEXT;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS importance_score REAL DEFAULT 0.5;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS forget_score REAL DEFAULT 0.0;
"""


class ForgettingEngine:
    """
    Periodically recomputes forget_score for all messages and
    removes or compresses entries that exceed thresholds.
    """

    def __init__(
        self,
        prune_threshold: float = PRUNE_THRESHOLD,
        compress_threshold: float = COMPRESS_THRESHOLD,
    ):
        self.prune_threshold = prune_threshold
        self.compress_threshold = compress_threshold

    def run_forgetting_pass(self, db_path: str) -> dict:
        """
        Execute a forgetting pass over BOTH:
          - the Hermes `messages` table (when running as a Hermes plugin)
          - the standalone `borge_memories` table (when running via BorgeRunner)

        Returns {"deleted": N, "compressed": M} aggregating both.
        """
        stats = {"deleted": 0, "compressed": 0}
        for name, kind in (("messages", "hermes"), ("borge_memories", "borge")):
            sub = self._sweep_table(db_path, name, kind)
            stats["deleted"]    += sub["deleted"]
            stats["compressed"] += sub["compressed"]
        return stats

    def _sweep_table(self, db_path: str, table: str, kind: str) -> dict:
        deleted = 0
        compressed = 0

        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                if kind == "hermes":
                    self._ensure_columns(conn)
                    where_clause = "WHERE role IN ('user', 'assistant', 'tool')"
                else:
                    where_clause = ""

                # Probe whether self_relevance_score column exists on this
                # table (Hermes `messages` won't have it; standalone
                # `borge_memories` will).
                cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
                has_sr = "self_relevance_score" in cols
                sr_select = ", self_relevance_score" if has_sr else ""

                select_sql = f"""
                    SELECT id, timestamp, last_retrieved, retrieval_count,
                           importance_score, encoding_depth, content,
                           emotional_valence, emotional_arousal{sr_select}
                    FROM {table}
                    {where_clause}
                """
                rows = conn.execute(select_sql).fetchall()

                now = datetime.now()

                for row in rows:
                    score = self._compute_score(row, now)
                    depth = row["encoding_depth"] or 1

                    if depth <= 1 and score > self.prune_threshold:
                        conn.execute(f"DELETE FROM {table} WHERE id = ?", (row["id"],))
                        deleted += 1
                    elif depth == 2 and score > self.compress_threshold:
                        stub = f"[compressed:{row['id'][:8]}]"
                        conn.execute(
                            f"UPDATE {table} SET content = ? WHERE id = ?",
                            (stub, row["id"]),
                        )
                        compressed += 1
                    else:
                        conn.execute(
                            f"UPDATE {table} SET forget_score = ? WHERE id = ?",
                            (round(score, 4), row["id"]),
                        )

        except sqlite3.OperationalError as e:
            # Table doesn't exist (e.g. messages in standalone mode) — silently skip
            log.debug(f"[Forgetting] {table}: skipped ({e})")

        return {"deleted": deleted, "compressed": compressed}

    @staticmethod
    def _compute_score(row: sqlite3.Row, now: datetime) -> float:
        """
        Ebbinghaus-inspired forget score WITH emotion factor.

        Higher = more likely to be forgotten.
        Resisted by: recent retrieval, high importance, and emotional
        intensity (|V|·A).
        """
        ts_str = row["last_retrieved"] or row["timestamp"]
        try:
            ts = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            ts = now

        days_since    = max(0.0, (now - ts).total_seconds() / 86400.0)
        retrieval_cnt = row["retrieval_count"] or 0
        importance    = row["importance_score"] or 0.5

        # Emotion: |V| · A. Range [0, 1]. Vivid memories → strong resistance.
        try:
            valence = float(row["emotional_valence"] or 0.0)
            arousal = float(row["emotional_arousal"] or 0.5)
        except (TypeError, ValueError, KeyError):
            valence, arousal = 0.0, 0.5
        emotion_intensity  = abs(valence) * arousal
        emotion_resistance = 1.0 / (1.0 + EMOTION_RESISTANCE_ALPHA * emotion_intensity)

        # Self-relevance: sr ∈ [0, 1]. Self-relevant memories resist forgetting.
        # Hermes `messages` table has no such column → defaults to neutral 0.5.
        try:
            self_relevance = float(row["self_relevance_score"])
        except (IndexError, KeyError, TypeError, ValueError):
            self_relevance = 0.5
        self_resistance = 1.0 / (1.0 + SELF_RESISTANCE_LAMBDA * self_relevance)

        recency_decay    = days_since ** 0.7
        usage_penalty    = 1.0 / (1.0 + retrieval_cnt)
        importance_res   = 1.0 / (1.0 + importance)

        return (recency_decay
                * usage_penalty
                * importance_res
                * emotion_resistance
                * self_resistance)

    @staticmethod
    def _ensure_columns(conn: sqlite3.Connection) -> None:
        """Silently add Borge columns to existing Hermes messages table."""
        for stmt in BORGE_COLUMNS_SQL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError:
                    pass  # column already exists


def apply_importance_from_delta_f(db_path: str, gain: float = 0.3) -> int:
    """
    Free-energy progress → importance bonus.

    For each memory row with `delta_f_total > 0` (the agent made cognitive
    progress on that turn), bump `importance_score` by `delta_f * gain`,
    clipped to [0, 1]. Importance enters forget_score via 1/(1+importance),
    so progress-bearing memories become harder to forget.

    Returns the number of rows updated.
    """
    updated = 0
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT id, importance_score, delta_f_total
                   FROM borge_memories
                   WHERE delta_f_total > 0"""
            ).fetchall()
            for r in rows:
                bonus  = max(0.0, float(r["delta_f_total"] or 0.0)) * gain
                new_imp = min(1.0, float(r["importance_score"] or 0.5) + bonus)
                conn.execute(
                    "UPDATE borge_memories SET importance_score = ? WHERE id = ?",
                    (round(new_imp, 4), r["id"]),
                )
                updated += 1
    except sqlite3.OperationalError as e:
        log.debug(f"[apply_importance_from_delta_f] skipped: {e}")
    return updated
