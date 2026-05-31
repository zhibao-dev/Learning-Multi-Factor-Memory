"""Token/$ savings estimate for ``borge audit``.

Turns a forget-list (the bloat memories ``borge audit`` recommends dropping)
into the headline number: tokens and dollars saved per month. The estimate
rests on **one assumed retrieval frequency** — how often that forgotten
context would otherwise be re-injected into the prompt — so every assumption
is returned in the ``assumptions`` dict rather than baked into a silent
constant (the design's honesty rule).

Token counting is ``tiktoken`` (``cl100k_base``) when available, else a
deterministic ``char/4`` fallback; the path actually used is recorded in the
assumptions.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _texts(forget_texts: list) -> list[str]:
    """Accept a list of strings or of objects with a ``.text`` attribute."""
    return [getattr(t, "text", t) for t in forget_texts]


def estimate_savings(
    forget_texts: list,
    *,
    retrieval_freq: float = 1.0,
    price_per_1k: float = 0.003,
    tokenizer: str = "char4",
) -> dict:
    """Estimate tokens/$ saved per month by forgetting ``forget_texts``.

    ``forget_texts`` may be a list of strings or of ``MemoryRecord``-like
    objects (anything with a ``.text`` attribute). ``tokenizer`` is
    ``"char4"`` (``total_chars // 4``) or ``"tiktoken"`` (``cl100k_base``);
    a failed tiktoken import falls back to char4 and says so in the
    returned assumptions.

    The estimate assumes the forgotten memories would otherwise be
    re-injected ``retrieval_freq`` times/month — this is the load-bearing
    assumption and is returned in ``assumptions`` for transparency.
    """
    texts = _texts(forget_texts)
    used_tokenizer = tokenizer

    if tokenizer == "tiktoken":
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            tokens_per_retrieval = sum(len(enc.encode(t)) for t in texts)
        except Exception as exc:  # tiktoken missing or encoding unavailable
            logger.warning("tiktoken unavailable (%s); falling back to char/4", exc)
            used_tokenizer = "char4 (tiktoken unavailable)"
            tokens_per_retrieval = sum(len(t) for t in texts) // 4
    else:
        tokens_per_retrieval = sum(len(t) for t in texts) // 4

    tokens_saved_per_month = tokens_per_retrieval * retrieval_freq
    usd_per_month = tokens_saved_per_month / 1000 * price_per_1k

    return {
        "tokens_per_retrieval": tokens_per_retrieval,
        "tokens_saved_per_month": tokens_saved_per_month,
        "usd_per_month": usd_per_month,
        "assumptions": {
            "retrieval_freq": retrieval_freq,
            "price_per_1k": price_per_1k,
            "tokenizer": used_tokenizer,
            "note": (
                "Assumes the forgotten memories would otherwise be re-injected "
                f"{retrieval_freq} times/month; savings scale linearly with "
                "this assumed retrieval frequency."
            ),
        },
    }
