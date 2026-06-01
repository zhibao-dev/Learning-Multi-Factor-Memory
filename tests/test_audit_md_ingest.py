def test_markdown_heading_split_with_frontmatter_and_timestamps():
    from borge.audit.ingest_md import load_markdown_dump
    from borge.audit.ingest import MemoryRecord
    recs = load_markdown_dump("tests/fixtures/audit_memory.md", split="heading")
    assert recs and all(isinstance(r, MemoryRecord) for r in recs)
    joined = " ".join(r.text for r in recs).lower()
    assert "vegetarian" in joined and "steak" in joined
    # frontmatter `modified` recovered as timestamp for sections lacking an inline date
    assert any(r.timestamp.startswith("2026-05-20") for r in recs)
    # ids unique + stable (slug from heading)
    assert len({r.id for r in recs}) == len(recs)
    assert all(r.id for r in recs)
