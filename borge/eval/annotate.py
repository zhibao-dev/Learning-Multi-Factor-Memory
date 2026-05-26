"""
API-free factor annotation for LongMemEval turns.

Six of the seven memory factors can be scored without any LLM call:

  emotion         signal_extractor rules on the turn text  → |ΔV|·arousal
  goal_relevance  SBert cos(turn, question)  (question = goal proxy)
  self_relevance  SBert cos(turn, μ_user)    (μ_user = centroid of user turns)
  reliability     role heuristic: user-stated facts > assistant text
  usage           0 in a static eval (no access history)
  value_alignment 0 (no SOUL profile in LongMemEval)
  task_utility    0 here (needs an LLM judge — the API-gated factor)

Embeddings are batched in one SBert call per case for speed.
"""

from __future__ import annotations

from ..affective.signal_extractor import EmotionalSignalExtractor
from ..values.self_model import SBertEmbedder, cosine
from .longmemeval import LongMemEvalCase, flatten_to_messages

_EXTRACTOR = EmotionalSignalExtractor()


def _sim01(a: list[float], b: list[float]) -> float:
    """cosine mapped to [0,1]."""
    return 0.5 + 0.5 * cosine(a, b)


def annotate_case(case: LongMemEvalCase, embedder: SBertEmbedder) -> list[dict]:
    """
    Return one factor dict per flattened message (same order as
    flatten_to_messages), plus is_gold/has_answer tags for scoring.

    Free factors only; value_alignment + task_utility left at 0.
    """
    msgs = flatten_to_messages(case)
    contents = [m["content"] for m in msgs]

    # Batch-encode all turn texts + the question in one SBert call.
    q_emb = embedder(case.question)
    turn_embs = [embedder(c) for c in contents] if contents else []

    # μ_user = centroid of user-turn embeddings
    user_embs = [e for e, m in zip(turn_embs, msgs) if m["role"] == "user"]
    if user_embs:
        dim = len(user_embs[0])
        mu_user = [sum(e[i] for e in user_embs) / len(user_embs) for i in range(dim)]
    else:
        mu_user = []

    out = []
    for m, emb in zip(msgs, turn_embs):
        dv, da = _EXTRACTOR.extract(m["content"], [])
        emotion = abs(dv) * (0.5 + da)
        goal_rel = _sim01(emb, q_emb)
        self_rel = _sim01(emb, mu_user) if mu_user else 0.5
        reliability = 0.7 if m["role"] == "user" else 0.4
        out.append({
            "factors": {
                "emotion":         max(0.0, min(1.0, emotion)),
                "goal_relevance":  goal_rel,
                "value_alignment": 0.0,
                "self_relevance":  self_rel,
                "task_utility":    0.0,
                "reliability":     reliability,
                "usage":           0.0,
            },
            "has_answer":      m["has_answer"],
            "is_gold_session": m["is_gold_session"],
            "session_idx":     m["session_idx"],
            "timestamp_idx":   m["session_idx"],   # session order ≈ time order
        })
    return out
