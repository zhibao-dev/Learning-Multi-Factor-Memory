from borge.audit.ingest import MemoryRecord, load_dump
from borge.memory.value import MemoryValue


def test_annotate_dump_blind_factors():
    from borge.audit.factors import annotate_dump

    recs = load_dump("tests/fixtures/audit_dump.json")
    factors = annotate_dump(recs)

    # aligned, one dict per record
    assert len(factors) == len(recs)

    # every dict carries EXACTLY the 7 MemoryValue factor keys
    want = set(MemoryValue.FACTORS)
    for f in factors:
        assert set(f.keys()) == want

    by_id = {r.id: f for r, f in zip(recs, factors)}

    # A substantive self-referential keep-* scores above a content-free
    # bloat-* on both blind anchors. NB: with an all-user dump of mostly
    # short utterances, μ_user is pulled toward the short text, so the
    # *self_relevance* margin is geometry-dependent — the robust separator
    # on this fixture is keep-5 (a full factual sentence) over bloat-1
    # ("ok"). keep-2 ("...Mia.") wins decisively on emotion.
    assert by_id["keep-2"]["emotion"] > by_id["bloat-1"]["emotion"]
    assert by_id["keep-5"]["self_relevance"] > by_id["bloat-1"]["self_relevance"]
    assert by_id["keep-5"]["emotion"] > by_id["bloat-1"]["emotion"]

    # task_utility is the LLM-gated factor → 0 in a static blind pass
    assert all(f["task_utility"] == 0.0 for f in factors)

    # reliability is the role heuristic: 0.7 for user (all fixture rows)
    assert all(f["reliability"] == 0.7 for f in factors)

    # every factor value is a float in [0, 1]
    for f in factors:
        for k in MemoryValue.FACTORS:
            v = f[k]
            assert isinstance(v, float)
            assert 0.0 <= v <= 1.0


def test_annotate_dump_reliability_non_user():
    from borge.audit.factors import annotate_dump

    recs = [
        MemoryRecord(id="u", text="I'm allergic to penicillin.",
                     timestamp="2026-01-01T00:00:00Z", role="user"),
        MemoryRecord(id="a", text="Noted, I'll remember that.",
                     timestamp="2026-01-01T00:01:00Z", role="assistant"),
    ]
    factors = annotate_dump(recs)
    by_id = {r.id: f for r, f in zip(recs, factors)}
    assert by_id["u"]["reliability"] == 0.7
    assert by_id["a"]["reliability"] == 0.4
