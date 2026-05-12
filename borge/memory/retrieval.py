"""
Memory Retrieval — mood-congruent + free-energy aware.

Implements the "feeling drives recall" half of the cognitive loop:
when the agent recalls past memories, it ranks them by

  • mood congruence — Gaussian similarity of (V, A) at encoding time
                      vs. the agent's current (V, A) state
  • recency         — exponential decay with a ~weekly half-life
  • text relevance  — token overlap with the optional query
  • F-progress      — small bonus for memories formed during cognitive
                      progress (positive delta_f_total at encoding)

Top-k results are returned, and each retrieved memory's
`retrieval_count` / `last_retrieved` is updated so that frequent recall
also makes a memory harder to forget (closes the retrieval ↔ forgetting
feedback loop).

Mood-congruent retrieval is grounded in encoding specificity
(Tulving 1973) and state-dependent learning literature: the cognitive
context at retrieval matters as much as the cue itself.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Optional

from .store import MemoryStore

log = logging.getLogger(__name__)


class MemoryRetrieval:
    """
    Ranks borge_memories rows by a weighted sum of four signals.

    The default weights are tuned so that:
      • In an "emotionally specific" query (mood far from baseline),
        the same memory carries a measurably higher rank.
      • In a "neutral" query (mood at baseline), recency and relevance
        dominate, matching standard episodic retrieval.
    """

    def __init__(self, db_path: str, store: Optional[MemoryStore] = None):
        self.db_path = db_path
        self.store   = store or MemoryStore(db_path)

    def recall(
        self,
        query: str = "",
        current_valence: float = 0.0,
        current_arousal: float = 0.5,
        current_f_total: Optional[float] = None,
        k: int = 5,
        mood_weight:      float = 0.4,
        recency_weight:   float = 0.2,
        relevance_weight: float = 0.3,
        f_weight:         float = 0.1,
    ) -> list[dict]:
        rows = self.store.all()
        if not rows:
            return []

        now      = datetime.now()
        q_lower  = (query or "").lower()
        q_tokens = {t for t in q_lower.split() if t}

        ranked: list[tuple[float, dict]] = []
        for r in rows:
            mood_sim  = self._mood_similarity(
                float(r.get("emotional_valence") or 0.0),
                float(r.get("emotional_arousal") or 0.5),
                current_valence,
                current_arousal,
            )
            recency   = self._recency(r.get("timestamp"), now)
            relevance = self._text_overlap(r.get("content", ""), q_tokens)
            f_bonus   = self._f_bonus(r.get("delta_f_total"))

            score = (mood_weight      * mood_sim
                   + recency_weight   * recency
                   + relevance_weight * relevance
                   + f_weight         * f_bonus)
            # Importance acts as a multiplier — well-consolidated memories
            # surface a bit more easily even when other signals are weak.
            score *= 1.0 + 0.5 * float(r.get("importance_score") or 0.5)

            ranked.append((score, r))

        ranked.sort(key=lambda kv: -kv[0])
        top = [r for _, r in ranked[:k]]

        # Closing the retrieval ↔ forgetting loop:
        # recalled memories accumulate retrieval_count → harder to forget.
        for m in top:
            self.store.record_retrieval(m["id"])

        return top

    # ── Signals ───────────────────────────────────────────────────────────

    @staticmethod
    def _mood_similarity(
        v_mem: float, a_mem: float,
        v_cur: float, a_cur: float,
    ) -> float:
        """
        Gaussian similarity on (V, A) plane.
        Range: 0 (far apart) to 1 (identical mood).

        The exponent uses raw squared distance — max V/A distance in the
        Russell space is sqrt(5) ≈ 2.24, so the Gaussian's effective tail
        is well-shaped without further normalization.
        """
        d_sq = (v_mem - v_cur) ** 2 + (a_mem - a_cur) ** 2
        return math.exp(-d_sq)

    @staticmethod
    def _recency(ts_iso: Optional[str], now: datetime) -> float:
        if not ts_iso:
            return 0.0
        try:
            ts = datetime.fromisoformat(ts_iso)
        except (ValueError, TypeError):
            return 0.0
        days = max(0.0, (now - ts).total_seconds() / 86400.0)
        return math.exp(-days / 7.0)  # ~weekly soft half-life

    @staticmethod
    def _text_overlap(content: str, q_tokens: set[str]) -> float:
        if not q_tokens:
            return 0.5  # neutral prior when no query supplied
        c_tokens = {t for t in (content or "").lower().split() if t}
        if not c_tokens:
            return 0.0
        return len(q_tokens & c_tokens) / max(1, len(q_tokens))

    @staticmethod
    def _f_bonus(delta_f: Optional[float]) -> float:
        if delta_f is None:
            return 0.0
        return max(0.0, min(1.0, float(delta_f)))
