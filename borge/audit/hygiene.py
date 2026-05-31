"""Hygiene checks for ``borge audit``: near-duplicate clusters + stale-by-age.

Two pure, deterministic passes over a dumped memory list. Neither deletes
anything — both only *flag* ids for a reviewer.

  find_duplicates  greedy single-pass clustering on SBert cosine; near-identical
                   memories (e.g. "project deadline March 15" said twice) land
                   in one cluster.
  find_stale       flag ids whose age (relative to an injected ``now_iso``)
                   exceeds ``age_days``.

100% local: the only model is the shared ``SBertEmbedder`` (no external API).

Duplicate threshold
-------------------
The default ``threshold=0.80`` is tuned to the planted fixture geometry and is
deliberately a *near*-duplicate gate, not exact match. On
``tests/fixtures/audit_dump.json`` the two planted paraphrases sit at
cosine 0.809 ("My project deadline is March 15." vs "The project deadline is
March 15th."), while the tightest *non*-duplicate pair — a fact and the
assistant's reply about it (``keep-4`` ↔ ``asst-1``) — sits at 0.781. 0.80
falls in that gap, so it clusters the paraphrase pair without merging a
question/answer pair. Distinct user facts sit far lower (≤ ~0.15).
"""

from __future__ import annotations

from datetime import datetime

from ..values.self_model import SBertEmbedder, cosine
from .ingest import MemoryRecord


def find_duplicates(
    records: list[MemoryRecord],
    *,
    embedder: SBertEmbedder | None = None,
    threshold: float = 0.80,
) -> list[list[str]]:
    """Greedily cluster near-identical memories; return ≥2-member clusters.

    Each record is embedded once (reusing the shared ``SBertEmbedder``). A
    record joins the first existing cluster whose first member it matches at
    cosine ≥ ``threshold``, else it seeds a new cluster. Only clusters with two
    or more members are returned, each as a list of record ids in input order.
    Nothing is deleted — these are duplicate *candidates* for review.
    """
    if embedder is None:
        embedder = SBertEmbedder()

    embs = [embedder(r.text) for r in records]

    # Greedy single-pass clustering, keyed on each cluster's first member.
    cluster_ids: list[list[str]] = []
    cluster_heads: list[list[float]] = []
    for rec, emb in zip(records, embs):
        for ids, head in zip(cluster_ids, cluster_heads):
            if cosine(emb, head) >= threshold:
                ids.append(rec.id)
                break
        else:
            cluster_ids.append([rec.id])
            cluster_heads.append(emb)

    return [ids for ids in cluster_ids if len(ids) >= 2]


def find_stale(
    records: list[MemoryRecord],
    now_iso: str,
    *,
    age_days: int = 180,
) -> list[str]:
    """Flag ids of memories older than ``age_days`` relative to ``now_iso``.

    ``now_iso`` is required and injected for determinism — this function never
    calls ``datetime.now()``. Both ``now_iso`` and each ``record.timestamp``
    are parsed with ``datetime.fromisoformat``; records whose timestamp can't
    be parsed are skipped (not flagged). Nothing is deleted.
    """
    now = _parse_iso(now_iso)
    if now is None:
        raise ValueError(f"now_iso is not a parseable ISO-8601 timestamp: {now_iso!r}")

    stale: list[str] = []
    for rec in records:
        ts = _parse_iso(rec.timestamp)
        if ts is None:
            continue
        if (now - ts).days > age_days:
            stale.append(rec.id)
    return stale


def _parse_iso(value: str) -> datetime | None:
    """Parse an ISO-8601 string to a naive UTC datetime; None on failure.

    A trailing 'Z' is normalised to '+00:00' so UTC dump timestamps parse, then
    tzinfo is dropped so aware (dump, 'Z') and naive (injected ``now_iso``)
    values compare without a mixed-awareness ``TypeError``. The dump is UTC
    throughout, so this is a no-op shift, not a silent timezone bug.
    """
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError, AttributeError):
        return None
    return dt.replace(tzinfo=None)
