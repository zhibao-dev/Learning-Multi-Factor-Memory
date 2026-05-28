"""
TDD: per-turn factor cache builder (Task 1 of paper2 roadmap).

`experiments/build_factor_cache.py` lifts the SBert + factor-derivation
step out of `lme_blind_forgetting.py::annotate_dual` so the ~275k-turn
encoding pass runs ONCE. Every downstream evaluation (keep-frac sweep,
bootstrap CI, neural ablation) reads the JSONL cache in seconds.

This test pins the on-disk schema and confirms that:

  1. Each record has shape {"qid": str, "turns": [...]}.
  2. Each turn dict contains
       emotion, self, reliability, goal_oracle, goal_blind,
       has_answer, sidx
     with all numeric factors in [0, 1].
  3. Reliability is 0.7 for user / 0.4 for assistant turns (paper formula).
  4. Building twice with the same input + embedder produces identical
     records — the pipeline is deterministic (no hidden RNG).
  5. Filter: a case with no `has_answer` turn is dropped (matches the
     existing `lme_blind_forgetting.py` filter that gives 74 usable cases
     out of 100).

A fake fixed-vector embedder is monkey-patched in so the test runs
without sentence-transformers (no SBert model download, no torch).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ── Synthetic LongMemEval cases (inline fixture; same schema as the real one) ─

GOLD_CASE = {
    "question_id": "q_gold_1",
    "question_type": "multi-session",
    "question": "What hobby did I pick up last month?",
    "answer": "rock climbing",
    "question_date": "2026-05-01",
    "haystack_sessions": [
        [
            {"role": "user", "content": "I started rock climbing last week.",
             "has_answer": True},
            {"role": "assistant", "content": "How is it going?"},
        ],
        [
            {"role": "user", "content": "Weather is nice today."},
            {"role": "assistant", "content": "Indeed it is."},
        ],
    ],
    "haystack_dates": ["2026-04-20", "2026-04-25"],
    "answer_session_ids": [0],
}

# This case has NO `has_answer` turn — the cache builder must drop it
# (matches the `lme_blind_forgetting.py` filter).
NO_GOLD_CASE = {
    "question_id": "q_no_gold",
    "question_type": "multi-session",
    "question": "Anything?",
    "answer": "nothing",
    "question_date": "2026-05-01",
    "haystack_sessions": [
        [
            {"role": "user", "content": "Just chatting."},
            {"role": "assistant", "content": "Ok."},
        ],
    ],
    "haystack_dates": ["2026-04-20"],
    "answer_session_ids": [],
}


@pytest.fixture()
def dataset_file(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic_lme.json"
    path.write_text(json.dumps([GOLD_CASE, NO_GOLD_CASE]))
    return path


class _FakeEmbedder:
    """
    Deterministic fixed-vector embedder: hashes text → a tiny vector.

    Distinct strings get distinct vectors (so cosines vary), identical
    strings get the same vector (so determinism is testable). No external
    deps, no model download.
    """

    def __init__(self) -> None:
        self.dim = 8

    def __call__(self, text: str) -> list[float]:
        import hashlib
        h = hashlib.sha1((text or "").encode("utf-8")).digest()
        # 8 signed unit slots — non-zero so cosine is well defined.
        return [(b / 255.0 - 0.5) * 2.0 for b in h[: self.dim]]


def test_cache_record_schema(dataset_file: Path, tmp_path: Path):
    """Schema + value-range + determinism + has-answer filter."""
    from experiments import build_factor_cache

    out_path = tmp_path / "factor_cache.jsonl"
    embedder = _FakeEmbedder()

    n_written = build_factor_cache.build(
        data_path=dataset_file,
        out_path=out_path,
        n_cases=10,
        embedder=embedder,
    )

    # only 1 of the 2 input cases has a `has_answer` turn → 1 written.
    assert n_written == 1, "must drop cases with zero has_answer turns"

    lines = out_path.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["qid"] == "q_gold_1"
    assert isinstance(rec["turns"], list)
    assert len(rec["turns"]) == 4   # 2 sessions × 2 turns

    expected_keys = {
        "emotion", "self", "reliability",
        "goal_oracle", "goal_blind",
        "has_answer", "sidx",
    }
    for t in rec["turns"]:
        assert expected_keys.issubset(t.keys()), \
            f"turn missing keys: {expected_keys - t.keys()}"
        for k in ("emotion", "self", "reliability", "goal_oracle", "goal_blind"):
            v = t[k]
            assert isinstance(v, (int, float))
            assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"
        assert isinstance(t["has_answer"], bool)
        assert isinstance(t["sidx"], int)

    # Reliability follows the paper formula: 0.7 user / 0.4 assistant.
    assert rec["turns"][0]["reliability"] == pytest.approx(0.7)  # user
    assert rec["turns"][1]["reliability"] == pytest.approx(0.4)  # assistant

    # Determinism: same input, same embedder → identical bytes.
    out_path2 = tmp_path / "factor_cache_b.jsonl"
    build_factor_cache.build(
        data_path=dataset_file,
        out_path=out_path2,
        n_cases=10,
        embedder=_FakeEmbedder(),
    )
    assert out_path.read_bytes() == out_path2.read_bytes(), \
        "cache must be deterministic across runs with the same embedder"
