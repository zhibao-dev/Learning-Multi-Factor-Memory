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
from typing import Callable

from ..values.self_model import SBertEmbedder, cosine
from .ingest import MemoryRecord

_NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

# Records with fewer whitespace tokens than this are dropped before pairing:
# ultra-short chatter ("ok", "haha") can't carry a contradiction and only
# floods the candidate list. Raw token count, NOT a value score — the value
# score does not separate bloat ("haha" scores higher than real content).
MIN_TOKENS = 4

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


def _body(text: str) -> str:
    """Return the record body with leading markdown heading lines dropped.

    Strips only the LEADING ``#``-prefixed heading line(s); keeps the rest of
    the text. Records with no heading are returned unchanged. Used to count
    body tokens (not the heading) for the ultra-short chatter filter, so a
    one-word reply like ``"## Reply 1\nok"`` is correctly skipped.
    """
    import re

    lines = text.splitlines()
    idx = 0
    while idx < len(lines) and re.match(r"^#{1,6}\s", lines[idx]):
        idx += 1
    return "\n".join(lines[idx:]).strip()


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
    skip_ids: set[str] | None = None,
    judge: "Callable | None" = None,
) -> list[CandidatePair]:
    """Flag candidate contradictions between memories for human review.

    Returns ``CandidatePair`` records (sorted by ``contradiction_score`` desc),
    each scored in [0, 1] with a ``likely_stale_id`` hint = the id of the
    *older* timestamp. These are CANDIDATES for human review — a product
    feature, not a paper2-backed result; nothing is auto-resolved or deleted.

    Only same-role pairs are compared (assertions, not responses); ultra-short
    memories (<MIN_TOKENS tokens) are skipped.

    ``embedder`` defaults to a fresh ``SBertEmbedder``. ``sim_threshold`` gates
    the same-topic embedding prefilter; ``nli_threshold`` gates the local NLI
    contradiction probability; ``max_pairs`` caps how many prefiltered pairs
    reach the cross-encoder (bounds cost / avoids unbounded N² NLI).
    ``skip_ids`` excludes known low-value / bloat records by id from checking.

    ``judge`` is an OPTIONAL precision filter. When ``None`` (default) the
    behavior is byte-identical to before — no external call, NLI survivors are
    returned as-is. When provided, ``judge(text_a, text_b)`` is called on each
    NLI survivor (after ``nli_threshold``) and must return a dict; the pair is
    kept only if ``v.get("contradict")`` is truthy. This lets an LLM strip
    same-topic-but-not-conflicting false positives the local NLI conflates.
    ``v.get("stale_id")`` (``"a"``/``"b"`` mapping to the respective record id)
    overrides ``likely_stale_id`` when given; anything else (incl. ``None``)
    falls back to the older-timestamp hint. A judge returning ``None`` (abstain)
    drops the pair — precision-first.
    """
    if embedder is None:
        embedder = SBertEmbedder()

    # ── Drop ultra-short records before pairing (body token count) ──
    # Count BODY tokens (heading lines dropped) so heading + 1-word chatter
    # ("## Reply 1\nok") is correctly skipped.
    records = [r for r in records if len(_body(r.text).split()) >= MIN_TOKENS]
    # ── Caller-supplied exclusions (known low-value / bloat records) ──
    if skip_ids:
        records = [r for r in records if r.id not in skip_ids]
    if len(records) < 2:
        return []

    embs = [embedder(r.text) for r in records]

    # ── Prefilter: same-topic i<j pairs by cosine, sorted desc, capped ──
    # Same-role only: a contradiction is conflicting *assertions*, so an
    # assistant turn responding to a topic is not a counter-assertion.
    prefiltered: list[tuple[float, int, int]] = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if records[i].role != records[j].role:
                continue
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
        # ── Optional LLM-judge precision filter (survivors only) ──
        if judge is not None:
            v = judge(records[i].text, records[j].text)
            if not (v and v.get("contradict")):
                continue  # abstain / not-confirmed → drop (precision-first)
            judged_stale = v.get("stale_id")
            if judged_stale == "a":
                stale_id = records[i].id
            elif judged_stale == "b":
                stale_id = records[j].id
            # else (None / anything else) → keep older-timestamp fallback
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
