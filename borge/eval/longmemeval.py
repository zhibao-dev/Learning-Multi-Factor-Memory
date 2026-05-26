"""
LongMemEval loader + agent-memory adapter.

LongMemEval (Wu et al., ICLR 2025) benchmarks chat assistants on
long-term interactive memory: each item is a question over a long
multi-session chat history (the "haystack"), with a subset of sessions
containing the gold evidence.

This module:
  1. parses the dataset JSON into `LongMemEvalCase` objects,
  2. flattens haystack sessions into a BorgeAgent-ready message list
     with session-index + gold-answer tags, so a memory policy can be
     scored on whether it RETAINS the gold-evidence messages through
     consolidation/forgetting and RETRIEVES them at question time.

The dataset itself is NOT vendored (500 items, license + size). Point
`load_longmemeval` at a downloaded `longmemeval_*.json`. The LLM-based
answering step is intentionally left to the run harness (API-gated);
this module is the dependency-free parse + flatten layer.

Download: https://github.com/xiaowu0162/LongMemEval (or HF mirror).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class LongMemEvalCase:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    haystack_sessions: list           # list[ list[ {role, content, has_answer?} ] ]
    haystack_dates: list = field(default_factory=list)
    answer_session_ids: list = field(default_factory=list)

    @classmethod
    def from_record(cls, r: dict) -> "LongMemEvalCase":
        # answer_session_ids may be session indices or session-id strings;
        # normalise to a list (callers compare against session_idx).
        asid = r.get("answer_session_ids", [])
        norm_ids = []
        for x in asid:
            try:
                norm_ids.append(int(x))
            except (TypeError, ValueError):
                norm_ids.append(x)
        return cls(
            question_id=str(r.get("question_id", "")),
            question_type=str(r.get("question_type", "")),
            question=str(r.get("question", "")),
            answer=str(r.get("answer", "")),
            question_date=str(r.get("question_date", "")),
            haystack_sessions=r.get("haystack_sessions", []),
            haystack_dates=r.get("haystack_dates", []),
            answer_session_ids=norm_ids,
        )


def load_longmemeval(path: str | Path) -> Iterator[LongMemEvalCase]:
    """Yield LongMemEvalCase per record. Accepts a JSON list or JSONL."""
    p = Path(path)
    text = p.read_text()
    try:
        records = json.loads(text)
        if isinstance(records, dict):
            records = [records]
    except json.JSONDecodeError:
        # JSONL fallback
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    for r in records:
        yield LongMemEvalCase.from_record(r)


def flatten_to_messages(case: LongMemEvalCase) -> list[dict]:
    """
    Flatten haystack sessions into a BorgeAgent message list.

    Each message gains:
      session_idx : which haystack session it came from
      has_answer  : True if the turn is gold evidence (from the dataset)
      is_gold_session : True if session_idx ∈ answer_session_ids

    Order preserves session order then turn order, matching the
    chronological haystack so consolidation sees the real timeline.
    """
    msgs: list[dict] = []
    gold = set(case.answer_session_ids)
    for s_idx, session in enumerate(case.haystack_sessions):
        for turn in session:
            msgs.append({
                "role":            turn.get("role", "user"),
                "content":         turn.get("content", ""),
                "session_idx":     s_idx,
                "has_answer":      bool(turn.get("has_answer", False)),
                "is_gold_session": s_idx in gold,
            })
    return msgs


def gold_retention_rate(kept_messages: list[dict], case: LongMemEvalCase) -> float:
    """
    Fraction of gold-evidence turns retained after a forgetting pass.

    The headline memory-policy metric independent of the LLM answerer:
    a good value function keeps the gold evidence and forgets distractors.
    """
    gold = set(case.answer_session_ids)
    total_gold = sum(
        1
        for s_idx, session in enumerate(case.haystack_sessions)
        if s_idx in gold
        for _ in session
    )
    if total_gold == 0:
        return 0.0
    kept_gold = sum(1 for m in kept_messages if m.get("session_idx") in gold)
    return kept_gold / total_gold
