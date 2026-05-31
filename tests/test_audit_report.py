def test_build_report_has_sections_and_is_honest():
    from borge.audit.ingest import load_dump
    from borge.audit.report import build_audit
    result = build_audit("tests/fixtures/audit_dump.json", now_iso="2026-05-29T00:00:00")
    md = result["markdown"]
    for header in ["Executive Summary", "Bloat", "Pollution", "Safety", "Methodology"]:
        assert header in md
    # honesty: never claims deletion
    assert "deleted" not in md.lower()
    # contradiction caveat present
    assert "human review" in md.lower()
    # the planted contradiction surfaces in the Pollution section
    assert "contra-A" in md or "contra-B" in md
    # reversible forget-id script produced (dry-run), nothing deleted
    assert isinstance(result["forget_script"], dict)
    assert "forget_ids" in result["forget_script"]
