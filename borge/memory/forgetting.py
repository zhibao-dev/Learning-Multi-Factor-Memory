"""
Forgetting Engine — Active Memory Decay

Implements Ebbinghaus-inspired forgetting to prevent memory accumulation
from degrading retrieval quality.

Forgetting is tiered by encoding depth:
  SHALLOW  + forget_score > PRUNE_THRESHOLD  → delete
  SEMANTIC + forget_score > COMPRESS_THRESHOLD → compress to entity tag only
  SCHEMATIC / META                            → never delete, only compress

The score is driven by the single multi-factor MemoryValue (paper2):
  score = recency_decay × usage_penalty × 1/(1 + β·V(m))
where V(m) = Σ wᵢ·factorᵢ over the seven memory factors (emotion,
goal/value/self relevance, task utility, reliability, usage). High-value
memories resist forgetting. This replaces paper1's hand-tuned
product-of-resistances; see borge/memory/value.py::value_forget_score.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime

from .value import MemoryValue, default_memory_value, value_forget_score

log = logging.getLogger(__name__)

PRUNE_THRESHOLD    = 2.0   # SHALLOW entries above this are deleted
COMPRESS_THRESHOLD = 3.0   # SEMANTIC entries above this are compressed

# Paper1 self-FEP resistance gains. Retained only for the paper1/self-FEP
# branch, where forget_score = product-of-resistances reads them. UNUSED on
# this (paper2/multi-factor-eval) branch — the value-driven forget score
# replaces the product entirely. (experiments/e3_pi_self_ablation.py is a
# paper1 ablation that belongs to the self-FEP branch and is expected-broken
# here; do not re-wire these constants into the value path to "fix" it.)
EMOTION_RESISTANCE_ALPHA = 2.0  # how strongly |V|·A resisted forgetting (paper1)
SELF_RESISTANCE_LAMBDA   = 2.0  # how strongly self-relevance resisted forgetting (paper1)

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
        memory_value: MemoryValue | None = None,
        beta: float = 8.0,
    ):
        self.prune_threshold = prune_threshold
        self.compress_threshold = compress_threshold
        self.memory_value = memory_value or default_memory_value()
        self.beta = beta

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

                # Probe which value-factor columns exist on this table.
                # Hermes `messages` has none of them; standalone
                # `borge_memories` has them all. value_forget_score reads
                # each via dict.get, so absent columns degrade gracefully.
                cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
                factor_cols = [
                    c for c in (
                        "self_relevance_score",
                        "goal_relevance",
                        "value_alignment",
                        "task_utility",
                        "reliability",
                    )
                    if c in cols
                ]
                factor_select = (", " + ", ".join(factor_cols)) if factor_cols else ""

                select_sql = f"""
                    SELECT id, timestamp, last_retrieved, retrieval_count,
                           importance_score, encoding_depth, content,
                           emotional_valence, emotional_arousal{factor_select}
                    FROM {table}
                    {where_clause}
                """
                rows = conn.execute(select_sql).fetchall()

                now = datetime.now()

                for row in rows:
                    score = self._compute_score(dict(row), now)
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

    def _compute_score(self, row: dict, now: datetime) -> float:
        """
        Value-driven forget score (higher = more likely forgotten).

            score = recency_decay × usage_penalty × 1/(1 + β·V(m))

        V(m) is the single multi-factor MemoryValue; high-value memories
        (vivid, self-relevant, reliable, frequently used, …) resist
        forgetting. Rows missing factor columns (Hermes `messages`)
        degrade gracefully — memory_factors reads each via dict.get.
        """
        return value_forget_score(row, self.memory_value, now, beta=self.beta)

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
    clipped to [0, 1].

    NOTE (paper2/multi-factor-eval branch): `importance_score` is retained
    as inspectable metadata but is NOT one of the seven MemoryValue factors,
    so it no longer feeds the value-driven forget score. The paper1
    self-FEP branch (where forget_score = product-of-resistances including
    1/(1+importance)) is where this ΔF→forgetting leg is live. On this
    branch, ΔF influences forgetting only if `importance`/progress is added
    as a MemoryValue factor.

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
