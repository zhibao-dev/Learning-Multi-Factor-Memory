"""Build the numeric upload payload from locally-annotated memories.

A "case" is one annotated memory set. The payload contains ONLY factor
scalars + gold flags + a keep fraction — no text, no embeddings, no ids
of the original content beyond an opaque local id.
"""

from __future__ import annotations

FACTOR_ORDER = ("emotion", "goal_relevance", "value_alignment",
                "self_relevance", "task_utility", "reliability", "usage")


def build_matrix(cases, *, keep_frac: float) -> dict:
    """`cases` = list of annotated-memory lists (from annotate_memories)."""
    return {
        "schema": "lmfm.factors.v1",
        "keep_frac": keep_frac,
        "factor_order": list(FACTOR_ORDER),
        "cases": [
            {"memories": [
                {"factors": {k: a["factors"][k] for k in FACTOR_ORDER},
                 "gold": bool(a["gold"])}
                for a in case
            ]}
            for case in cases
        ],
    }
