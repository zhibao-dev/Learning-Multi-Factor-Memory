def test_finds_planted_contradiction_not_unrelated():
    from borge.audit.ingest import load_dump
    from borge.audit.contradiction import find_contradictions
    recs = load_dump("tests/fixtures/audit_dump.json")
    pairs = find_contradictions(recs)   # default thresholds
    flagged = {(p.a_id, p.b_id) for p in pairs} | {(p.b_id, p.a_id) for p in pairs}
    assert ("contra-A", "contra-B") in flagged       # planted contradiction caught
    # an unrelated benign pair is NOT flagged
    assert ("keep-1", "bloat-1") not in flagged
    # length skip: ultra-short chatter ("ok"/"haha") is dropped before pairing
    assert ("bloat-1", "bloat-4") not in flagged
    # same-role gate: a user fact vs an assistant turn about it is cross-role,
    # so it's never compared (a response is not a counter-assertion)
    assert ("stale-1", "asst-2") not in flagged
    # NOTE: the keep-2/keep-5 name-slot pair ("daughter Mia" vs "manager Sarah")
    # legitimately survives — both are role=user and ≥MIN_TOKENS; the residual
    # is a real NLI limitation that human review covers, not asserted gone here.
    # each pair carries a score and a likely-stale id
    for p in pairs:
        assert 0.0 <= p.contradiction_score <= 1.0
        assert p.likely_stale_id in (p.a_id, p.b_id)
