"""build_audit routes ingest by file type: .md → markdown, .json → JSON."""

from __future__ import annotations


def test_build_audit_routes_markdown():
    from borge.audit.report import build_audit

    out = build_audit("tests/fixtures/audit_memory.md", now_iso="2026-06-01T00:00:00")
    md = out["markdown"]
    assert "vegetarian" in md.lower() or "steak" in md.lower()  # pipeline ran on the .md
    assert "Pollution" in md  # full report rendered
    assert isinstance(out["forget_script"], dict)


def test_build_audit_still_routes_json():
    from borge.audit.report import build_audit

    out = build_audit("tests/fixtures/audit_dump.json", now_iso="2026-06-01T00:00:00")
    assert "Executive Summary" in out["markdown"]  # JSON path unbroken
