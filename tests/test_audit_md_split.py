def test_bullet_split_one_record_per_item():
    from borge.audit.ingest_md import load_markdown_dump
    recs = load_markdown_dump("tests/fixtures/audit_memory_bullets.md", split="bullet")
    assert len(recs) >= 5
    assert all(r.text.strip() for r in recs)
    assert len({r.id for r in recs}) == len(recs)   # unique ids


def test_dated_split_recovers_per_entry_timestamp():
    from borge.audit.ingest_md import load_markdown_dump
    recs = load_markdown_dump("tests/fixtures/audit_memory_log.md", split="dated")
    assert any(r.timestamp.startswith("2025-") for r in recs)   # old entry, its OWN date
    assert any(r.timestamp.startswith("2026-") for r in recs)
    # per-entry dates differ → not all the same (not just file mtime)
    assert len({r.timestamp[:10] for r in recs}) >= 2


def test_auto_detects_dated_and_bullet():
    from borge.audit.ingest_md import load_markdown_dump
    log = load_markdown_dump("tests/fixtures/audit_memory_log.md", split="auto")
    assert any(r.timestamp.startswith("2025-") for r in log)     # auto→dated kept per-entry dates
    bul = load_markdown_dump("tests/fixtures/audit_memory_bullets.md", split="auto")
    assert len(bul) >= 5                                          # auto→bullet split the list
