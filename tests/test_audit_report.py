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


def test_methodology_discloses_role_driven_ranking():
    """Methodology must disclose that authorship/reliability drives forget order."""
    from borge.audit.report import build_audit
    md = build_audit("tests/fixtures/audit_dump.json", now_iso="2026-05-29T00:00:00")[
        "markdown"
    ]
    lower = md.lower()
    assert "reliability" in lower
    assert "assistant" in lower or "authorship" in lower


def test_forget_ids_disjoint_from_contradiction_pairs():
    """No id may be in BOTH the forget script AND a contradiction candidate pair.

    Forgetting one member while the Pollution section flags the other as
    likely-stale is contradictory advice on the same pair — those ids are
    deferred to human review instead.
    """
    from borge.audit.contradiction import find_contradictions
    from borge.audit.ingest import load_dump
    from borge.audit.report import build_audit

    fixture = "tests/fixtures/audit_dump.json"
    now = "2026-05-29T00:00:00"
    forget_ids = set(build_audit(fixture, now_iso=now)["forget_script"]["forget_ids"])

    records = load_dump(fixture)
    contradiction_ids = {
        cid for p in find_contradictions(records) for cid in (p.a_id, p.b_id)
    }
    assert forget_ids.isdisjoint(contradiction_ids), (
        f"ids in both forget list and a contradiction pair: "
        f"{sorted(forget_ids & contradiction_ids)}"
    )


def test_name_slot_caveat_present():
    """The NLI name-slot false-positive caveat must accompany the contradictions."""
    from borge.audit.report import build_audit
    md = build_audit("tests/fixtures/audit_dump.json", now_iso="2026-05-29T00:00:00")[
        "markdown"
    ]
    assert "name slots" in md or "name slot" in md or "entity/name slots" in md
