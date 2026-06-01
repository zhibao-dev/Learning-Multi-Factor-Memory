"""End-to-end capstone for ``borge audit`` over a **Markdown** memory dump.

One deterministic run of the real orchestrator (``build_audit``) over the
Markdown fixture ``tests/fixtures/audit_memory.md`` with a per-customer
``--weights`` override, proving the whole pipeline runs end-to-end on a
markdown source: it surfaces every *planted* problem (contradiction,
duplicate, stale) across all five report sections, shows the custom weight,
honestly acknowledges the markdown bloat-ranking limitation, and stays a
read-only dry-run (deletes nothing, mutates the input file by not a byte).

The SBert embedder + NLI cross-encoder load locally (first run may download to
the HF cache; local thereafter), same pattern as the other audit tests — no
skip guard.
"""

import hashlib

from borge.audit.report import build_audit

_FIXTURE = "tests/fixtures/audit_memory.md"
_NOW = "2026-06-01T00:00:00"  # fixed → deterministic stale cutoff


def _section(md: str, header: str) -> str:
    """Slice one ``## <header>`` section (its header to the next ``## ``)."""
    start = md.index(f"## {header}")
    nxt = md.find("\n## ", start + 1)
    return md[start:] if nxt == -1 else md[start:nxt]


def test_audit_md_e2e_surfaces_planted_problems_with_custom_weights_dry_run():
    # ── capture input bytes BEFORE the run (dry-run proof) ──
    with open(_FIXTURE, "rb") as f:
        before = f.read()
    before_sha = hashlib.sha256(before).hexdigest()

    result = build_audit(_FIXTURE, now_iso=_NOW, weights={"reliability": 0.9})
    md = result["markdown"]

    # ── 1. Markdown ran end-to-end: all five report sections present ──
    for header in ["Executive Summary", "Bloat", "Pollution", "Safety", "Methodology"]:
        assert header in md, f"missing report section: {header}"

    pollution = _section(md, "Pollution (Contradictions, Duplicates, Stale)")

    # ── 2. Contradiction: the planted vegetarian/steak pair surfaces ──
    # Both the heading-slug ids AND the conflicting nouns appear.
    assert "audit_memory#diet" in pollution and "audit_memory#dinner" in pollution
    assert "vegetarian" in md and "steak" in md

    # ── 3. Duplicate: the planted near-duplicate deadline pair surfaces ──
    dup_section = pollution[pollution.index("Near-duplicate clusters"):]
    assert "audit_memory#deadline-a" in dup_section and "audit_memory#deadline-b" in dup_section
    assert "deadline" in dup_section.lower()

    # ── 4. Stale: the planted old-dated note is flagged stale ──
    stale_section = pollution[pollution.index("Stale memories"):]
    assert "audit_memory#old-note" in stale_section
    assert "2025-03-01" in stale_section  # its recovered inline date

    # ── 5. Custom weight is surfaced in the report ──
    assert "reliability=0.9" in md

    # ── 6. Honesty / dry-run safety ──
    assert "deleted" not in md.lower()  # never claims deletion
    fs = result["forget_script"]
    assert isinstance(fs, dict) and "forget_ids" in fs
    assert isinstance(fs["forget_ids"], list)
    # the input markdown file is byte-for-byte unchanged (nothing written back)
    with open(_FIXTURE, "rb") as f:
        after = f.read()
    assert after == before
    assert hashlib.sha256(after).hexdigest() == before_sha

    # ── 7. Markdown honesty: the bloat-ranking limitation is acknowledged ──
    methodology = _section(md, "Methodology")
    assert "best-effort" in methodology
