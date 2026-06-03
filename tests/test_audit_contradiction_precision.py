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
