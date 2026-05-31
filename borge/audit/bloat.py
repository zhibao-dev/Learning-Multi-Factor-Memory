"""Bloat / forget-list ranking for ``borge audit``.

Ranks dumped memories by the paper2 :class:`~borge.memory.value.MemoryValue`
(one scalar over the 7 annotated factors) and partitions them into tiered
forget-lists. Lowest-value memories are the first to forget.

The tiers are *keep-fractions*: ``safe`` keeps the top 70% by value,
``aggressive`` only the top 30%. No deletion happens here and no new score
is introduced — the ranking is purely ``MemoryValue.value(factors)``.
"""

from __future__ import annotations

from .factors import MemoryRecord
from ..memory.value import MemoryValue

_DEFAULT_TIERS = {"safe": 0.7, "moderate": 0.5, "aggressive": 0.3}


def forget_ranking(
    records: list[MemoryRecord],
    factors: list[dict],
    mv: MemoryValue,
    *,
    tiers: dict[str, float] | None = None,
) -> dict:
    """Rank ``records`` by ``MemoryValue`` and build tiered forget-lists.

    ``factors`` is aligned to ``records`` by index (as returned by
    :func:`borge.audit.factors.annotate_dump`). Each tier maps a name to a
    keep-fraction in [0, 1]; for that tier the lowest-value memories beyond
    the kept fraction go to ``forget_ids``, the rest to ``keep_ids``.
    """
    if tiers is None:
        tiers = _DEFAULT_TIERS

    value_by_id = {rec.id: mv.value(factors[i]) for i, rec in enumerate(records)}

    # Lowest value first → forgotten first.
    ranked_ids = sorted(value_by_id, key=lambda i: value_by_id[i])
    n = len(records)

    tier_out: dict[str, dict] = {}
    for name, keep_frac in tiers.items():
        k_keep = round(n * keep_frac)
        n_forget = n - k_keep
        forget_ids = ranked_ids[:n_forget]
        keep_ids = ranked_ids[n_forget:]
        tier_out[name] = {
            "keep_frac": keep_frac,
            "forget_ids": forget_ids,
            "keep_ids": keep_ids,
        }

    return {"value_by_id": value_by_id, "tiers": tier_out}
