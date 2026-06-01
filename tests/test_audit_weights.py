"""Per-customer ``--weights`` override for ``borge audit``.

Weights are 千人千面 — a companion app weights factors differently than a coding
agent. These tests prove a customer-supplied weight dict (1) really flows through
into the value model that drives the forget ranking, and (2) is surfaced in the
report (honesty + "tunable per your business" upsell), while the no-override path
still shows the shipped LongMemEval default.
"""


def test_custom_weights_flow_through_and_show_in_report():
    from borge.audit.report import build_audit
    base = build_audit("tests/fixtures/audit_dump.json", now_iso="2026-06-01T00:00:00")
    custom = build_audit("tests/fixtures/audit_dump.json", now_iso="2026-06-01T00:00:00",
                         weights={"reliability": 0.0, "emotion": 0.0})
    # weights surfaced in the report (Methodology / header)
    assert "reliability" in custom["markdown"].lower()
    # zeroing the two heavy factors changes the chosen forget set → weights really flow through
    assert custom["forget_script"]["forget_ids"] != base["forget_script"]["forget_ids"]


def test_default_weights_unchanged_when_none():
    from borge.audit.report import build_audit
    from borge.memory.value import SHIPPED_DEFAULT
    out = build_audit("tests/fixtures/audit_dump.json", now_iso="2026-06-01T00:00:00")
    # the report shows the shipped default reliability weight when no override
    assert str(SHIPPED_DEFAULT["reliability"]) in out["markdown"]
