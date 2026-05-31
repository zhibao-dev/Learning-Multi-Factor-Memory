def test_finds_planted_contradiction_not_unrelated():
    from borge.audit.ingest import load_dump
    from borge.audit.contradiction import find_contradictions
    recs = load_dump("tests/fixtures/audit_dump.json")
    pairs = find_contradictions(recs)   # default thresholds
    flagged = {(p.a_id, p.b_id) for p in pairs} | {(p.b_id, p.a_id) for p in pairs}
    assert ("contra-A", "contra-B") in flagged       # planted contradiction caught
    # an unrelated benign pair is NOT flagged
    assert ("keep-1", "bloat-1") not in flagged
    # each pair carries a score and a likely-stale id
    for p in pairs:
        assert 0.0 <= p.contradiction_score <= 1.0
        assert p.likely_stale_id in (p.a_id, p.b_id)
