"""End-to-end capstone for ``borge audit``.

One deterministic run of the real orchestrator (``build_audit``) over the
synthetic fixture ``tests/fixtures/audit_dump.json``, proving the whole
pipeline surfaces every *planted* problem (bloat, contradiction, duplicate,
stale) plus a savings figure across all five report sections — and that it
stays a read-only dry-run (deletes nothing, mutates no input).

The SBert embedder + NLI cross-encoder load locally (first run may download to
the HF cache; local thereafter), same pattern as the other audit tests — no
skip guard.
"""

import hashlib

from borge.audit.report import build_audit

_FIXTURE = "tests/fixtures/audit_dump.json"
_NOW = "2026-05-29T00:00:00"  # fixed → deterministic stale cutoff


def _exec_summary_region(md: str) -> str:
    """Slice the Executive Summary section (its header to the next ``## ``)."""
    start = md.index("## Executive Summary")
    nxt = md.index("## ", start + len("## Executive Summary"))
    return md[start:nxt]


def test_audit_e2e_surfaces_all_planted_problems_and_is_dry_run():
    # ── capture input bytes BEFORE the run (dry-run proof) ──
    with open(_FIXTURE, "rb") as f:
        before = f.read()
    before_sha = hashlib.sha256(before).hexdigest()

    result = build_audit(_FIXTURE, now_iso=_NOW)
    md = result["markdown"]
    forget_ids = result["forget_script"]["forget_ids"]

    # ── 1. Bloat: chosen-tier forget set catches the throwaway chatter ──
    # ≥3 of the 4 planted bloat-* ids are flagged to forget...
    bloat_flagged = {i for i in forget_ids if i.startswith("bloat-")}
    assert len(bloat_flagged) >= 3, (
        f"expected >=3 bloat-* ids in chosen forget set, got {sorted(bloat_flagged)}"
    )
    # ...and no substantive high-retrieval keep (keep-2..keep-6) is sacrificed.
    # keep-1 is the weakest keep and MAY drop at aggressive budgets, so it is
    # deliberately excluded from the must-not-forget set.
    must_keep = {"keep-2", "keep-3", "keep-4", "keep-5", "keep-6"}
    wrongly_forgotten = must_keep & set(forget_ids)
    assert not wrongly_forgotten, (
        f"substantive keep-* memories wrongly flagged to forget: {sorted(wrongly_forgotten)}"
    )

    # ── 2. Contradiction: the planted vegetarian/steak pair is named ──
    assert "contra-A" in md or "contra-B" in md

    # ── 3. Duplicate: the planted near-duplicate deadline pair is named ──
    assert "dup-A" in md or "dup-B" in md

    # ── 4. Stale: at least one planted year-old memory is named ──
    assert "stale-1" in md or "stale-2" in md

    # ── 5. Savings: a tokens/month or $/month figure in the Exec Summary ──
    region = _exec_summary_region(md)
    assert "$" in region, "no $/month figure in Executive Summary"
    assert any(ch.isdigit() for ch in region), "no numeric figure in Executive Summary"

    # ── 6. Honesty / dry-run ──
    assert "deleted" not in md.lower()  # never claims deletion
    fs = result["forget_script"]
    assert "forget_ids" in fs and "note" in fs  # reversible candidate list + provenance
    assert isinstance(fs["forget_ids"], list)
    # the input fixture is byte-for-byte unchanged (nothing was written back)
    with open(_FIXTURE, "rb") as f:
        after = f.read()
    assert after == before
    assert hashlib.sha256(after).hexdigest() == before_sha

    # ── 7. All five report sections present ──
    for header in ["Executive Summary", "Bloat", "Pollution", "Safety", "Methodology"]:
        assert header in md, f"missing report section: {header}"
