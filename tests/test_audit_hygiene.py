def test_find_duplicates_clusters_near_identical():
    from borge.audit.ingest import load_dump
    from borge.audit.hygiene import find_duplicates
    recs = load_dump("tests/fixtures/audit_dump.json")
    # 0.80 sits in the gap between the planted paraphrase pair (cos 0.809)
    # and the tightest non-duplicate pair (keep-4 fact vs asst-1 reply, 0.781).
    clusters = find_duplicates(recs, threshold=0.80)
    # dup-A and dup-B land in the same cluster
    assert any({"dup-A", "dup-B"} <= set(c) for c in clusters)
    # distinct keep facts are NOT clustered together
    assert not any({"keep-1", "keep-2"} <= set(c) for c in clusters)


def test_find_stale_flags_old_memories():
    from borge.audit.ingest import load_dump
    from borge.audit.hygiene import find_stale
    recs = load_dump("tests/fixtures/audit_dump.json")
    stale = find_stale(recs, now_iso="2026-05-29T00:00:00", age_days=180)
    assert {"stale-1", "stale-2"} <= set(stale)
    # recent memories not flagged
    assert "keep-1" not in stale and "bloat-1" not in stale
