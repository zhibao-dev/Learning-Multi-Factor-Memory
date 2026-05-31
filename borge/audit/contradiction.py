"""Candidate-contradiction detection for ``borge audit``.

Flags **candidate contradictions** between dumped memories — pairs that
assert mutually incompatible things (e.g. "I'm a strict vegetarian" vs
"I had an amazing steak last night"). This is the audit feature competitors
(Mem0/Zep) don't surface: stale/conflicting facts that silently pollute an
agent's memory.

Pipeline (100% local — customer memory data never leaves their infra; there
is **no external API call anywhere** in this module):

  1. Embed every record once (``SBertEmbedder``).
  2. Prefilter: keep only same-topic ``i < j`` pairs (cosine ≥ ``sim_threshold``),
     sorted by similarity desc and capped at ``max_pairs``. This bounds the
     cross-encoder cost and avoids running NLI over the full O(N²) pair set.
  3. NLI: a local cross-encoder (``cross-encoder/nli-deberta-v3-small``,
     auto-downloaded to the HF cache on first use, local thereafter) scores
     each surviving pair in **both directions**; the contradiction-label
     probability is read off the model's own ``id2label``.
  4. ``contradiction_score`` = max contradiction probability over both
     directions; keep pairs ≥ ``nli_threshold``.

**These are CANDIDATES for human review, not verdicts.** This is a *product*
feature, NOT a paper2-backed result. Nothing here auto-resolves, merges, or
deletes a memory — ``likely_stale_id`` is only a hint (the older of the two
timestamps, since a newer memory contradicting an older one suggests the
older is the superseded one). A reviewer decides.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..values.self_model import SBertEmbedder, cosine
from .ingest import MemoryRecord

_NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

# Module-level cache so the cross-encoder loads its weights only once across
# repeated find_contradictions calls (e.g. per-tier audit runs, tests).
_nli_model = None


def _get_nli_model():
    """Lazily load and cache the local NLI cross-encoder."""
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder

        _nli_model = CrossEncoder(_NLI_MODEL_NAME)
    return _nli_model


def _contradiction_label_index(model) -> int:
    """Read the 'contradiction' class index off the model's own id2label.

    We do NOT hardcode the index — different NLI checkpoints order the
    three classes differently. Raise a clear error if no contradiction
    label is present so a wrong model fails loudly rather than silently
    scoring the wrong logit.
    """
    id2label = model.config.id2label
    for idx, label in id2label.items():
        if str(label).lower() == "contradiction":
            return int(idx)
    raise ValueError(
        f"NLI model {_NLI_MODEL_NAME!r} has no 'contradiction' label in "
        f"id2label={id2label!r}; cannot score contradictions."
    )


def _softmax_contradiction_prob(logits, contra_idx: int) -> float:
    """Softmax a 3-logit row and return the contradiction-class probability."""
    import math

    m = max(logits)
    exps = [math.exp(float(x) - m) for x in logits]
    total = sum(exps)
    return exps[contra_idx] / total if total > 0 else 0.0


@dataclass
class CandidatePair:
    a_id: str
    b_id: str
    contradiction_score: float
    likely_stale_id: str


def find_contradictions(
    records: list[MemoryRecord],
    *,
    embedder: SBertEmbedder | None = None,
    sim_threshold: float = 0.3,
    nli_threshold: float = 0.5,
    max_pairs: int = 200,
) -> list[CandidatePair]:
    """Flag candidate contradictions between memories for human review.

    Returns ``CandidatePair`` records (sorted by ``contradiction_score`` desc),
    each scored in [0, 1] with a ``likely_stale_id`` hint = the id of the
    *older* timestamp. These are CANDIDATES for human review — a product
    feature, not a paper2-backed result; nothing is auto-resolved or deleted.

    ``embedder`` defaults to a fresh ``SBertEmbedder``. ``sim_threshold`` gates
    the same-topic embedding prefilter; ``nli_threshold`` gates the local NLI
    contradiction probability; ``max_pairs`` caps how many prefiltered pairs
    reach the cross-encoder (bounds cost / avoids unbounded N² NLI).
    """
    if embedder is None:
        embedder = SBertEmbedder()
    if len(records) < 2:
        return []

    embs = [embedder(r.text) for r in records]

    # ── Prefilter: same-topic i<j pairs by cosine, sorted desc, capped ──
    prefiltered: list[tuple[float, int, int]] = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            sim = cosine(embs[i], embs[j])
            if sim >= sim_threshold:
                prefiltered.append((sim, i, j))
    prefiltered.sort(key=lambda t: t[0], reverse=True)
    prefiltered = prefiltered[:max_pairs]
    if not prefiltered:
        return []

    # ── NLI: both directions per pair, contradiction prob off id2label ──
    model = _get_nli_model()
    contra_idx = _contradiction_label_index(model)

    inputs: list[tuple[str, str]] = []
    for _sim, i, j in prefiltered:
        inputs.append((records[i].text, records[j].text))
        inputs.append((records[j].text, records[i].text))
    logits = model.predict(inputs)

    results: list[CandidatePair] = []
    for k, (_sim, i, j) in enumerate(prefiltered):
        p_ij = _softmax_contradiction_prob(logits[2 * k], contra_idx)
        p_ji = _softmax_contradiction_prob(logits[2 * k + 1], contra_idx)
        score = max(p_ij, p_ji)
        if score < nli_threshold:
            continue
        # Newer memory contradicting an older one → the older is the stale one.
        if records[i].timestamp <= records[j].timestamp:
            stale_id = records[i].id
        else:
            stale_id = records[j].id
        results.append(
            CandidatePair(
                a_id=records[i].id,
                b_id=records[j].id,
                contradiction_score=float(score),
                likely_stale_id=stale_id,
            )
        )

    results.sort(key=lambda p: p.contradiction_score, reverse=True)
    return results
