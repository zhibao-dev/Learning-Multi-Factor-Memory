"""
Memory Retrieval — value-driven, mood-congruent recall.

Implements the "feeling drives recall" half of the cognitive loop:
when the agent recalls past memories, it ranks them by

  • value (V)       — the shared multi-factor MemoryValue over the row's
                      stored factors (self/emotion/usage/reliability/…);
                      query-agnostic durable worth of the memory
  • text relevance  — token overlap with the optional query (the only
                      query-DEPENDENT signal — keeps queries distinct)
  • mood congruence — Gaussian similarity of (V, A) at encoding time
                      vs. the agent's current (V, A) state
  • recency         — exponential decay with a ~weekly half-life

The same MemoryValue scalar drives encode-depth and forgetting (paper2),
so retrieval, encoding, and forgetting are governed by one learned value.

Top-k results are returned, and each retrieved memory's
`retrieval_count` / `last_retrieved` is updated so that frequent recall
also makes a memory harder to forget (closes the retrieval ↔ forgetting
feedback loop) — and feeds back into V's `usage` factor next time.

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
from .value import MemoryValue, default_memory_value, memory_factors

log = logging.getLogger(__name__)


class MemoryRetrieval:
    """
    Ranks borge_memories rows by

        rank = w_v·V(factors) + w_rel·relevance + w_mood·mood + w_rec·recency

    where `V` is the shared multi-factor `MemoryValue` that also drives
    encode-depth and forgetting. V is query-AGNOSTIC — it already folds in
    the durable worth of a memory (self/emotion/usage/reliability), so it
    subsumes the old self-similarity and importance terms. `relevance`
    (token overlap with the query) is the only genuinely query-DEPENDENT
    signal and stays live so a query still differentiates results; mood +
    recency are query-time context.
    """

    def __init__(
        self,
        db_path: str,
        store: Optional[MemoryStore] = None,
        self_model=None,
        memory_value: Optional[MemoryValue] = None,
    ):
        self.db_path = db_path
        self.store   = store or MemoryStore(db_path)
        # Retained for back-compat with callers that pass a self model; the
        # self signal now lives inside MemoryValue's self_relevance factor.
        self.self_model = self_model
        # Shared multi-factor value: one scalar drives encode/forget/retrieve.
        self.memory_value = memory_value or default_memory_value()

    def recall(
        self,
        query: str = "",
        current_valence: float = 0.0,
        current_arousal: float = 0.5,
        current_f_total: Optional[float] = None,
        k: int = 5,
        w_v:    float = 0.45,
        w_rel:  float = 0.25,
        w_mood: float = 0.20,
        w_rec:  float = 0.10,
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
            recency    = self._recency(r.get("timestamp"), now)
            relevance  = self._text_overlap(r.get("content", ""), q_tokens)
            v          = self.memory_value.value(memory_factors(r))

            score = (w_v    * v
                   + w_rel  * relevance
                   + w_mood * mood_sim
                   + w_rec  * recency)

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
