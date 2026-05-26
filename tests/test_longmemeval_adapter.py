"""
TDD: LongMemEval loader + agent-memory adapter (Stage B-2).

LongMemEval (Wu et al., ICLR 2025) record schema:
  question_id, question_type, question, answer, question_date,
  haystack_sessions: [[{role, content, has_answer?}, ...], ...],
  haystack_dates: [...], answer_session_ids: [...]

We can't ship the 500-item dataset, so tests use an inline fixture
matching the real schema. The actual run (LLM answering) is gated on
API budget; here we test parse + the session→memory flattening that
feeds BorgeAgent consolidation.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest


SAMPLE_RECORD = {
    "question_id": "q_demo_1",
    "question_type": "multi-session",
    "question": "What hobby did I say I picked up?",
    "answer": "rock climbing",
    "question_date": "2026-05-01",
    "haystack_sessions": [
        [
            {"role": "user", "content": "I started rock climbing last week.", "has_answer": True},
            {"role": "assistant", "content": "That's great, how is it going?"},
        ],
        [
            {"role": "user", "content": "The weather is nice today."},
            {"role": "assistant", "content": "Indeed."},
        ],
    ],
    "haystack_dates": ["2026-04-20", "2026-04-25"],
    "answer_session_ids": [0],
}


@pytest.fixture()
def sample_file():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump([SAMPLE_RECORD], f)
        path = f.name
    yield path
    os.unlink(path)


def test_loader_parses_record(sample_file):
    from borge.eval.longmemeval import load_longmemeval
    cases = list(load_longmemeval(sample_file))
    assert len(cases) == 1
    c = cases[0]
    assert c.question_id == "q_demo_1"
    assert c.question_type == "multi-session"
    assert c.answer == "rock climbing"
    assert len(c.haystack_sessions) == 2
    assert c.answer_session_ids == [0]


def test_flatten_sessions_to_messages(sample_file):
    """Adapter flattens haystack sessions into a message list with
    session-tagged metadata, ready for BorgeAgent consolidation."""
    from borge.eval.longmemeval import load_longmemeval, flatten_to_messages
    case = next(iter(load_longmemeval(sample_file)))
    msgs = flatten_to_messages(case)
    # 2 sessions × 2 turns = 4 messages
    assert len(msgs) == 4
    assert all("role" in m and "content" in m for m in msgs)
    assert all("session_idx" in m for m in msgs)
    # first message carries the gold-answer flag
    assert msgs[0]["content"].startswith("I started rock climbing")
    assert msgs[0].get("has_answer") is True


def test_gold_session_messages_identifiable(sample_file):
    """Messages from answer_session_ids are flagged gold so eval can
    measure whether value-driven forgetting RETAINS them."""
    from borge.eval.longmemeval import load_longmemeval, flatten_to_messages
    case = next(iter(load_longmemeval(sample_file)))
    msgs = flatten_to_messages(case)
    gold = [m for m in msgs if m["session_idx"] in case.answer_session_ids]
    distractor = [m for m in msgs if m["session_idx"] not in case.answer_session_ids]
    assert len(gold) == 2          # session 0 has 2 turns
    assert len(distractor) == 2    # session 1 has 2 turns
    assert any(m.get("has_answer") for m in gold)
