"""
Task 6 (paper2): retrieval ranks by the shared MemoryValue.

One `MemoryValue` scalar drives encode-depth, forget, and now retrieval.
`recall` scores each row as

    rank = w_v·V(factors) + w_rel·relevance + w_mood·mood + w_rec·recency

V is query-AGNOSTIC (durable worth: it already folds in self/emotion/
usage/reliability). `relevance` is the only genuinely query-DEPENDENT
term and must stay live, else every query returns the same high-V set.

These tests pin:
  A. Two rows that are equally recent and equally relevant to the query,
     differing ONLY in value factors, order by V (high-V first).
  B. The query-relevance term is live: a query matching a low-V row's
     content lifts that row above where it sat on a neutral query.
"""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta

import pytest

from borge.memory.store import MemoryStore
from borge.memory.retrieval import MemoryRetrieval


@pytest.fixture()
def tmpdb():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def test_recall_orders_equally_recent_relevant_rows_by_value(tmpdb):
    """Two rows equal in everything the OLD 5-term formula could see (mood,
    recency, relevance, self_relevance_score, importance) and differing only
    in `reliability` — a factor the old formula entirely ignored but the
    unified V weights heavily. Only V can put the high-reliability row first.

    To make this a genuine RED against the old formula (whose `self_sim` term
    read `self_relevance_score`), the low-V row is given the HIGHER
    self_relevance_score: under the old formula that pulls the low-V row to the
    top (wrong); under V, reliability dominates and the high-V row wins.

    To keep V *load-bearing* (so the test FAILS if V were ever dropped, e.g.
    w_v=0), the high-V row is also made the OLDER of the two: recency now
    actively FAVORS the low-V row, so only the V term can overcome that and
    rank the high-V row first. (With w_v=0 the order flips to low-first.)
    """
    store = MemoryStore(tmpdb)
    now = datetime.now()
    shared = "shared overlap tokens"  # both rows match the query equally
    # Equal (V,A) → equal mood term AND equal emotion factor; differ in
    # reliability (new in V) and an inverted self_relevance to force the RED.
    base = {
        "session_id": "s", "role": "user", "content": shared,
        "emotional_valence": 0.3, "emotional_arousal": 0.5,
    }
    # High-V row is OLDER → recency opposes V (low-V row is the more recent).
    high = {**base, "id": "m-high", "reliability": 0.9, "self_relevance_score": 0.1,
            "timestamp": (now - timedelta(days=2)).isoformat()}
    low  = {**base, "id": "m-low",  "reliability": 0.1, "self_relevance_score": 0.8,
            "timestamp": now.isoformat()}
    store.insert(high)
    store.insert(low)

    ret = MemoryRetrieval(tmpdb, store=store)
    # current mood == both rows' (V,A) → identical mood term; relevance also
    # identical. Recency now favors the low-V row, so ONLY V can rank high first.
    results = ret.recall(
        query=shared,
        current_valence=0.3, current_arousal=0.5,
        k=2,
    )
    assert [r["id"] for r in results] == ["m-high", "m-low"], (
        f"high-value (high-reliability) row must rank first; "
        f"got {[r['id'] for r in results]}"
    )


def test_query_relevance_term_is_live(tmpdb):
    """Two rows with EQUAL V differ only in content; the query selects which
    ranks first — proving the query-relevance term is live, not swamped by V.
    (If V were the only term, the two rows would tie regardless of the query.)"""
    store = MemoryStore(tmpdb)
    ts = datetime.now().isoformat()

    # Identical value factors → identical V; only content differs.
    base = {
        "session_id": "s", "role": "user", "timestamp": ts,
        "emotional_valence": 0.5, "emotional_arousal": 0.5,
        "self_relevance_score": 0.5, "reliability": 0.5,
    }
    store.insert({**base, "id": "m-apples",  "content": "apples oranges pears"})
    store.insert({**base, "id": "m-pickles", "content": "pickles cucumbers brine"})

    ret = MemoryRetrieval(tmpdb, store=store)
    by_apples = ret.recall(
        query="apples oranges pears",
        current_valence=0.5, current_arousal=0.5, k=2,
    )
    assert by_apples[0]["id"] == "m-apples", (
        f"query should select the matching row; got {[r['id'] for r in by_apples]}"
    )

    by_pickles = ret.recall(
        query="pickles cucumbers brine",
        current_valence=0.5, current_arousal=0.5, k=2,
    )
    assert by_pickles[0]["id"] == "m-pickles", (
        "query-relevance term must be live; "
        f"got {[r['id'] for r in by_pickles]}"
    )
