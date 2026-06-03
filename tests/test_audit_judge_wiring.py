"""Wiring of the optional LLM judge into build_audit + report label flip."""


def test_judge_used_labels_llm_verified():
    from borge.audit.report import build_audit

    def fake_judge(a, b):  # confirm everything → flip the label + exercise the path
        return {"contradict": True, "stale_id": None}

    out = build_audit(
        "tests/fixtures/audit_dump.json",
        now_iso="2026-06-01T00:00:00",
        judge=fake_judge,
    )
    assert "LLM-verified" in out["markdown"]


def test_no_judge_keeps_candidate_label():
    from borge.audit.report import build_audit

    out = build_audit(
        "tests/fixtures/audit_dump.json",
        now_iso="2026-06-01T00:00:00",
    )
    md = out["markdown"].lower()
    assert "human review" in md or "candidate" in md  # current NLI-only wording
    assert "llm-verified" not in md
