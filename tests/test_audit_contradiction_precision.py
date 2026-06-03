def test_body_only_skip_drops_heading_chatter():
    from borge.audit.contradiction import find_contradictions
    from borge.audit.ingest import MemoryRecord
    recs = [
        MemoryRecord("r1", "## Reply 1\nok", "2026-05-01", "user", {}),
        MemoryRecord("r2", "## Reply 2\nthanks", "2026-05-01", "user", {}),
        MemoryRecord("d1", "## Diet\nI am a strict vegetarian, I never eat meat.", "2026-03-01", "user", {}),
        MemoryRecord("d2", "## Dinner\nI had an amazing steak last night, so good.", "2026-04-01", "user", {}),
    ]
    pairs = find_contradictions(recs)
    flagged = {frozenset((p.a_id, p.b_id)) for p in pairs}
    assert frozenset(("r1", "r2")) not in flagged   # chatter skipped (body "ok"/"thanks" <4 tokens)
    assert frozenset(("d1", "d2")) in flagged        # real contradiction kept


def test_skip_ids_excludes_records():
    from borge.audit.contradiction import find_contradictions
    from borge.audit.ingest import MemoryRecord
    recs = [
        MemoryRecord("d1", "## Diet\nI am a strict vegetarian, I never eat meat.", "2026-03-01", "user", {}),
        MemoryRecord("d2", "## Dinner\nI had an amazing steak last night, so good.", "2026-04-01", "user", {}),
    ]
    assert find_contradictions(recs) != []                     # found without skip
    assert find_contradictions(recs, skip_ids={"d2"}) == []     # d2 excluded → no pair


def test_judge_filters_topical_false_positives():
    from borge.audit.contradiction import find_contradictions
    from borge.audit.ingest import MemoryRecord
    recs = [
        MemoryRecord("d1", "## Diet\nStrict vegetarian, I never eat meat.", "2026-03-01", "user", {}),
        MemoryRecord("d2", "## Dinner\nHad an amazing steak last night.", "2026-04-01", "user", {}),
        MemoryRecord("c1", "## CI\nCI runs on GitHub Actions for every PR.", "2026-03-01", "user", {}),
        MemoryRecord("c2", "## Testing\nIntegration tests hit a throwaway database.", "2026-03-01", "user", {}),
    ]
    calls = []
    def fake_judge(a, b):
        calls.append((a, b))
        real = "vegetarian" in (a + b).lower() and "steak" in (a + b).lower()
        return {"contradict": real, "stale_id": None}
    pairs = find_contradictions(recs, judge=fake_judge, nli_threshold=0.0)  # nli passes many; judge decides
    flagged = {frozenset((p.a_id, p.b_id)) for p in pairs}
    assert frozenset(("d1", "d2")) in flagged              # judge-confirmed contradiction kept
    assert all(frozenset(("c1", "c2")) != f for f in flagged)   # topical FP filtered out by judge
    assert calls                                            # judge actually invoked on survivors


def test_judge_none_unchanged():
    # judge=None keeps the API-free behavior: real contradiction still found
    from borge.audit.contradiction import find_contradictions
    from borge.audit.ingest import MemoryRecord
    recs = [
        MemoryRecord("d1", "## Diet\nStrict vegetarian, I never eat meat.", "2026-03-01", "user", {}),
        MemoryRecord("d2", "## Dinner\nHad an amazing steak last night.", "2026-04-01", "user", {}),
    ]
    assert find_contradictions(recs) != []
