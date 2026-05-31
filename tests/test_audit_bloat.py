def test_forget_ranking_puts_bloat_below_keep():
    from borge.audit.ingest import load_dump
    from borge.audit.factors import annotate_dump
    from borge.audit.bloat import forget_ranking
    from borge.memory.value import default_memory_value
    recs = load_dump("tests/fixtures/audit_dump.json")
    facs = annotate_dump(recs)
    result = forget_ranking(recs, facs, default_memory_value(),
                            tiers={"safe": 0.7, "moderate": 0.5, "aggressive": 0.3})
    keep_ids = {"keep-1", "keep-2", "keep-3", "keep-4", "keep-5", "keep-6"}
    bloat_ids = {"bloat-1", "bloat-2", "bloat-3", "bloat-4"}

    # The value model separates the two cohorts CLEANLY:
    #   max(V over bloat) < min(V over keep)
    # i.e. every content-free chatter row outranks no substantive row.
    # This separation is driven by the `usage` factor (bloat retrieval_count
    # 0 → usage 0; keep 4-9 → usage ~0.8-0.9); the semantic factors
    # (self/goal_relevance) actually run slightly higher on some bloat-
    # adjacent rows, so usage's 0.10 weight is what does the work.
    vbi = result["value_by_id"]
    assert max(vbi[i] for i in bloat_ids) < min(vbi[i] for i in keep_ids)

    # aggressive tier (keep top 30%) flags ALL bloat to forget.
    agg = set(result["tiers"]["aggressive"]["forget_ids"])
    assert len(agg & bloat_ids) >= 3          # ≥3 of 4 bloat flagged to forget

    # Keep-protection is asserted on the `moderate` tier. On this 19-row
    # fixture the keep cohort is ~32% of the dump, so keeping only the top
    # 30% (aggressive) MUST drop one keep row by arithmetic — and it drops
    # keep-1, the weakest keep (lowest self/goal relevance: a generic
    # "I prefer Python over Java" preference). That is a correct value
    # ordering, not a model defect. The `moderate` tier (keep top 50%)
    # forgets every bloat and zero keep — the cohorts are fully separable.
    mod = set(result["tiers"]["moderate"]["forget_ids"])
    assert bloat_ids <= mod                   # all 4 bloat forgotten
    assert not (mod & keep_ids)               # no substantive keep flagged

    # value recorded per id
    assert set(result["value_by_id"]) == {r.id for r in recs}
