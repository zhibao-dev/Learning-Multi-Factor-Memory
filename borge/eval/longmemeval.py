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
    answer_session_ids: list = field(default_factory=list)   # session-id strings
    haystack_session_ids: list = field(default_factory=list)  # parallel to sessions

    @classmethod
    def from_record(cls, r: dict) -> "LongMemEvalCase":
        return cls(
            question_id=str(r.get("question_id", "")),
            question_type=str(r.get("question_type", "")),
            question=str(r.get("question", "")),
            answer=str(r.get("answer", "")),
            question_date=str(r.get("question_date", "")),
            haystack_sessions=r.get("haystack_sessions", []),
            haystack_dates=r.get("haystack_dates", []),
            # keep ids as-is (real LongMemEval uses session-id strings like
            # "answer_280352e9"; the synthetic fixture uses int indices)
            answer_session_ids=list(r.get("answer_session_ids", [])),
            haystack_session_ids=list(r.get("haystack_session_ids", [])),
        )

    def is_gold_session(self, s_idx: int) -> bool:
        """A session is gold if its id ∈ answer_session_ids, OR (fallback
        for fixtures using int indices) the index itself is listed."""
        gold = set(self.answer_session_ids)
        if s_idx < len(self.haystack_session_ids):
            if self.haystack_session_ids[s_idx] in gold:
                return True
        return s_idx in gold


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
    for s_idx, session in enumerate(case.haystack_sessions):
        sid = (case.haystack_session_ids[s_idx]
               if s_idx < len(case.haystack_session_ids) else s_idx)
        is_gold = case.is_gold_session(s_idx)
        for turn in session:
            msgs.append({
                "role":            turn.get("role", "user"),
                "content":         turn.get("content", ""),
                "session_idx":     s_idx,
                "session_id":      sid,
                "has_answer":      bool(turn.get("has_answer", False)),
                "is_gold_session": is_gold,
            })
    return msgs


def gold_retention_rate(kept_messages: list[dict], case: LongMemEvalCase) -> float:
    """
    Fraction of gold-evidence turns retained after a forgetting pass.

    Gold turns are those flagged `has_answer` by the dataset (the true
    needle). Independent of any LLM answerer: a good value function keeps
    the gold and forgets distractors.
    """
    total_gold = sum(
        1
        for session in case.haystack_sessions
        for t in session
        if t.get("has_answer")
    )
    if total_gold == 0:
        # fallback: count by gold-session membership
        total_gold = sum(
            1
            for s_idx, session in enumerate(case.haystack_sessions)
            if case.is_gold_session(s_idx)
            for _ in session
        )
        if total_gold == 0:
            return 0.0
        kept_gold = sum(1 for m in kept_messages if m.get("is_gold_session"))
        return kept_gold / total_gold
    kept_gold = sum(1 for m in kept_messages if m.get("has_answer"))
    return kept_gold / total_gold
