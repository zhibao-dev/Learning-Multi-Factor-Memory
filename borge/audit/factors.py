"""Blind factor annotation for static memory dumps (``borge audit``).

A dumped memory has no future query, so we score the same seven
``MemoryValue`` factors as consolidation Step 3 / ``borge/eval/annotate.py``
using *blind* anchors derived from the dump itself:

  emotion         signal_extractor rules on the text       → |ΔV|·(0.5+arousal)
  self_relevance  SBert cos(text, μ_user)  (μ_user = centroid of user rows)
  goal_relevance  SBert cos(text, topic centroid)  (centroid of ALL rows)
  value_alignment SBert cos(text, soul_centroid) if given, else 0
  reliability     role heuristic: user-stated facts > assistant text
  usage           metadata retrieval_count, saturating in [0,1]
  task_utility    0 (the LLM-gated factor — no judge in a blind pass)

Pure-local: one ``SBertEmbedder`` call per record, no API.
"""

from __future__ import annotations

from ..affective.signal_extractor import EmotionalSignalExtractor
from ..memory.value import MemoryValue
from ..values.self_model import SBertEmbedder, cosine
from .ingest import MemoryRecord

_EXTRACTOR = EmotionalSignalExtractor()


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _sim01(a: list[float], b: list[float]) -> float:
    """cosine mapped to [0,1]."""
    return 0.5 + 0.5 * cosine(a, b)


def _centroid(embs: list[list[float]]) -> list[float]:
    """Element-wise mean of embeddings; [] for an empty list."""
    if not embs:
        return []
    dim = len(embs[0])
    return [sum(e[i] for e in embs) / len(embs) for i in range(dim)]


def annotate_dump(
    records: list[MemoryRecord],
    *,
    embedder: SBertEmbedder | None = None,
    soul_centroid: list[float] | None = None,
) -> list[dict]:
    """Score the 7 ``MemoryValue`` factors for each record (blind anchors).

    Returns one factor dict per record, aligned by index. Each dict has
    exactly ``MemoryValue.FACTORS`` keys, all floats in [0, 1].
    """
    if embedder is None:
        embedder = SBertEmbedder()

    embs = [embedder(r.text) for r in records]

    user_embs = [e for e, r in zip(embs, records) if r.role == "user"]
    mu_user = _centroid(user_embs) or _centroid(embs)
    topic_centroid = _centroid(embs)

    out: list[dict] = []
    for rec, emb in zip(records, embs):
        dv, da = _EXTRACTOR.extract(rec.text, [])
        emotion = abs(dv) * (0.5 + da)
        self_rel = _sim01(emb, mu_user) if mu_user else 0.5
        goal_rel = _sim01(emb, topic_centroid) if topic_centroid else 0.5
        value_align = _sim01(emb, soul_centroid) if soul_centroid else 0.0
        reliability = 0.7 if rec.role == "user" else 0.4
        rcount = float(rec.metadata.get("retrieval_count", 0) or 0)
        usage = rcount / (1.0 + rcount)

        factors = {
            "emotion":         emotion,
            "goal_relevance":  goal_rel,
            "value_alignment": value_align,
            "self_relevance":  self_rel,
            "task_utility":    0.0,
            "reliability":     reliability,
            "usage":           usage,
        }
        out.append({k: _clamp01(factors[k]) for k in MemoryValue.FACTORS})
    return out
