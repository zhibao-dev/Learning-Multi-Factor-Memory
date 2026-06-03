"""End-to-end test: the LLM-judge path yields a CLEAN contradiction report.

Capstone for the contradiction-precision plan. Runs the full ``build_audit``
pipeline on the realistic demo dump with a FAKE judge (no real LLM) that
confirms ONLY the two genuine contradictions (TypeScript↔Go, Vercel↔Kubernetes)
and rejects every topical NLI false-positive (ci↔testing, auth↔secrets,
database↔testing, *↔region, …) and all chatter (## Reply 1..5).

Without a judge the NLI prefilter surfaces 8 candidate pairs (2 real + 6
topical FPs); with this judge only the 2 real pairs survive into the
"LLM-verified contradictions" section. The judge matches purely on text
content, so the separation is genuine, not a hard-coded id allowlist.
"""

import hashlib

DEMO = "packaging/borge-audit/demo/sample_agent_memory.md"

# Heading-slug ids the report renders for the two REAL contradiction pairs.
REAL_TS_GO = ("sample_agent_memory#tech-stack",
              "sample_agent_memory#stack-migration-2026-04-12")
REAL_VERCEL_K8 = ("sample_agent_memory#deployment-target",
                  "sample_agent_memory#infra-change-2026-05-02")

# Topical NLI false-positives the judge must reject (id pairs the NLI-only run
# would otherwise flag at score 1.00 / 0.68 / 0.51).
FP_IDS = [
    "sample_agent_memory#ci",
    "sample_agent_memory#region",
    "sample_agent_memory#postgres-version-note",
    "sample_agent_memory#working-hours-note",
]


def fake_judge(a, b):
    """Confirm contradiction ONLY for the two real pairs, by text content."""
    t = (a + " " + b).lower()
    ts_go = "typescript" in t and (
        "migrate" in t and " go" in t
        or "scaffolded in go" in t
        or "to go" in t
    )
    verc_k8 = "vercel" in t and ("kubernetes" in t or "self-hosted" in t)
    return {"contradict": bool(ts_go or verc_k8), "stale_id": None}


def test_llm_judge_yields_clean_contradiction_report():
    from borge.audit.report import build_audit

    before = hashlib.sha256(open(DEMO, "rb").read()).hexdigest()

    out = build_audit(
        DEMO,
        now_iso="2026-06-01T00:00:00",
        md_split="heading",
        judge=fake_judge,
    )
    md = out["markdown"]

    # The judge path is labelled "LLM-verified".
    assert "LLM-verified" in md

    # Both REAL contradiction pairs surface, by their rendered heading-slug ids.
    for pair in (REAL_TS_GO, REAL_VERCEL_K8):
        for mem_id in pair:
            assert mem_id in md, f"expected real contradiction id {mem_id!r} in report"

    # Isolate the candidate-contradiction section to assert FPs/chatter absent
    # from the contradiction list specifically (ids may appear elsewhere, e.g.
    # the stale section or forget table).
    start = md.index("LLM-verified contradictions")
    end = md.index("### Near-duplicate clusters", start)
    contra_section = md[start:end]

    # The judge rejected every topical false-positive: none appear as a
    # contradiction row. (region/ci/etc. legitimately appear elsewhere in the
    # report, so we only scope this to the contradiction section.)
    for fp_id in FP_IDS:
        assert fp_id not in contra_section, (
            f"topical false-positive {fp_id!r} leaked into contradiction section"
        )

    # Chatter (## Reply 1..5: ok/thanks/lgtm/…) never appears as a contradiction.
    low = contra_section.lower()
    for chatter in ("reply 1", "reply 2", "lgtm", "thanks, that works"):
        assert chatter not in low, f"chatter {chatter!r} leaked into contradiction section"

    # Exactly the two real pairs survived: two contradiction data rows (each
    # rendered as a markdown table line starting with "| `sample_agent_memory#").
    row_count = sum(
        1 for ln in contra_section.splitlines()
        if ln.startswith("| `sample_agent_memory#")
    )
    assert row_count == 2, f"expected exactly 2 contradiction rows, got {row_count}"

    # Dry-run safety: forget script is a dict with forget_ids, input untouched.
    assert isinstance(out["forget_script"], dict)
    assert "forget_ids" in out["forget_script"]

    after = hashlib.sha256(open(DEMO, "rb").read()).hexdigest()
    assert before == after, "input .md must be byte-unchanged after the audit"
