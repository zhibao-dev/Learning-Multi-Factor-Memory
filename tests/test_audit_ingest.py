def test_load_dump_parses_and_skips_bad():
    from borge.audit.ingest import load_dump

    recs = load_dump("tests/fixtures/audit_dump.json")
    ids = {r.id for r in recs}
    assert {"contra-A", "contra-B", "dup-A", "dup-B", "stale-1", "bloat-1"} <= ids
    assert all(r.text and r.timestamp for r in recs)  # none empty
    assert all(r.id for r in recs)  # no empty id
    # the malformed (missing text) row is skipped
    assert len(recs) == 16
