"""Annotate memories into numeric factor vectors + gold flags, locally.

This is the privacy boundary: raw text is reduced to seven scalar
factors here, on the user's machine. Only the resulting numbers (never
the text or embeddings) need ever leave for cloud weight learning.

Both relevance factors use query-agnostic anchors, matching the blind
consolidation regime: the goal anchor is the centroid of ALL turn
embeddings (the session topic, "what this memory set is about"), while
the self anchor is the centroid of user-turn embeddings only.
"""

from __future__ import annotations

from .emotion import EmotionSignalExtractor
from .embedder import cosine

_EXTRACTOR = EmotionSignalExtractor()


def _sim01(a, b) -> float:
    if not a or not b:
        return 0.5
    return 0.5 * (1.0 + cosine(a, b))


def annotate_memories(records, *, embedder, gold_ids=None) -> list[dict]:
    """Return one dict per record: {factors: {7}, gold: bool, id, ts}.

    `gold_ids` marks which record ids are the must-retain "gold" set
    (the learning target). `embedder(text) -> list[float]`.
    """
    gold_ids = set(gold_ids or [])
    texts = [r.text for r in records]
    embs = [embedder(t) for t in texts] if texts else []

    user_embs = [e for e, r in zip(embs, records) if r.role == "user"]
    if user_embs:
        dim = len(user_embs[0])
        mu_user = [sum(e[i] for e in user_embs) / len(user_embs) for i in range(dim)]
    else:
        mu_user = []

    if embs:
        dim = len(embs[0])
        mu_all = [sum(e[i] for e in embs) / len(embs) for i in range(dim)]
    else:
        mu_all = []

    out = []
    for r, emb in zip(records, embs):
        dv, da = _EXTRACTOR.extract(r.text, [])
        emotion = abs(dv) * (0.5 + da)
        self_rel = _sim01(emb, mu_user) if mu_user else 0.5
        goal_rel = _sim01(emb, mu_all) if mu_all else 0.5
        reliability = 0.7 if r.role == "user" else 0.4
        rcount = float((r.metadata or {}).get("retrieval_count", 0) or 0)
        out.append({
            "id": r.id,
            "ts": r.timestamp,
            "gold": r.id in gold_ids,
            "factors": {
                "emotion":         max(0.0, min(1.0, emotion)),
                "goal_relevance":  goal_rel,
                "value_alignment": 0.0,
                "self_relevance":  self_rel,
                "task_utility":    0.0,
                "reliability":     reliability,
                "usage":           rcount / (1.0 + rcount),
            },
        })
    return out
