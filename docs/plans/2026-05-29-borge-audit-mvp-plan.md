# borge audit MVP — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Branch:** `multi-factor-eval-business` ONLY. Do NOT merge into main / self-FEP / multi-factor-eval (business layer is its own line).

**Goal:** A self-hosted CLI `borge-audit <dump.json>` that ingests a canonical-JSON agent memory dump and emits a markdown audit report: (1) bloat/forget-list, (2) contradiction/pollution candidates, (3) dedup+stale, (4) token/$ savings + retention-under-budget — API-free, data never leaves infra, never auto-deletes.

**Architecture:** New package `borge/audit/` (ingest → factors → bloat → contradiction → hygiene → savings → report → CLI). Reuses paper2 `MemoryValue` for bloat ranking and `SBertEmbedder` for embeddings. Contradiction is a NEW module (embedding-prefilter → local NLI CrossEncoder), kept as a SEPARATE artifact (not folded into MemoryValue). Everything deterministic + local.

**Tech Stack:** Python 3.11 (`/Users/max_abel/opt/anaconda3/bin/python` — `python` NOT on PATH). Reuse: `borge/memory/value.py` (`MemoryValue`, `memory_factors`, `default_memory_value`), `borge/values/self_model.py` (`SBertEmbedder`, `cosine`), `borge/affective/signal_extractor.py` (`EmotionalSignalExtractor.extract(text, history)->(dv,da)`). Deps already installed (NO pyproject change): `sentence-transformers` (gives both SBert + `CrossEncoder`), `tiktoken`. NLI model `cross-encoder/nli-deberta-v3-small` auto-downloads/caches on first use (local, no API).

**Test convention:** flat `tests/test_audit_*.py` (match existing `tests/test_*.py`); fixture under `tests/fixtures/`. Run with `/Users/max_abel/opt/anaconda3/bin/python -m pytest`.

**Honest-scope guardrails (enforce in every task):** contradiction detection is product-driven, NOT paper2-backed — never claim it's "paper-proven"; keep it OUT of `MemoryValue`. The audit NEVER deletes — it emits recommendations + a reversible forget-id script only.

---

### Task 1: Canonical ingest + synthetic fixture

**Files:**
- Create: `tests/fixtures/audit_dump.json`
- Create: `borge/audit/__init__.py` (empty), `borge/audit/ingest.py`
- Test: `tests/test_audit_ingest.py`

**Step 1 — fixture:** hand-author `tests/fixtures/audit_dump.json` — a list of ~18 memory records `{id, text, timestamp(ISO), role, metadata}` with PLANTED structure (document each in a top comment is impossible in JSON, so track ids by convention):
- `bloat-1..4`: low-value chatter ("ok", "thanks", "sounds good", "haha"), recent.
- `contra-A` text "I'm a strict vegetarian, no meat ever", `contra-B` text "I had an amazing steak last night", both role=user, weeks apart.
- `dup-A` / `dup-B`: near-identical ("My project deadline is March 15" / "The project deadline is March 15th").
- `stale-1` / `stale-2`: timestamp ~400 days ago, role=user.
- `keep-1..6`: substantive user facts/preferences, varied dates.
- 1 malformed row (missing `text`) to test skip.

**Step 2 — failing test:** `tests/test_audit_ingest.py`:
```python
def test_load_dump_parses_and_skips_bad(tmp_path=None):
    from borge.audit.ingest import load_dump
    recs = load_dump("tests/fixtures/audit_dump.json")
    ids = {r.id for r in recs}
    assert "contra-A" in ids and "contra-B" in ids
    assert all(r.text and r.timestamp for r in recs)   # no empty
    assert not any(r.id == "" for r in recs)
    # malformed row (missing text) skipped → count is len(file)-1
```
Run `/Users/max_abel/opt/anaconda3/bin/python -m pytest tests/test_audit_ingest.py -v` → FAIL (module absent).

**Step 3 — implement** `borge/audit/ingest.py`: `@dataclass MemoryRecord(id:str, text:str, timestamp:str, role:str="user", metadata:dict=field(default_factory=dict))`; `load_dump(path)->list[MemoryRecord]` — read JSON list, skip rows missing `id`/`text`/`timestamp` (log a warning), parse the rest. No DB, no embedding here.

**Step 4:** rerun → PASS.

**Step 5 — commit:** `feat(audit): canonical-JSON ingest + synthetic memory-dump fixture`.

---

### Task 2: Blind factor annotation for a dump

**Files:**
- Create: `borge/audit/factors.py`
- Test: `tests/test_audit_factors.py`

**Context:** A static dump has NO future query → use BLIND anchors (as in `borge/eval/annotate.py` + consolidation Step 3): emotion via signal_extractor; self via cos to the μ_user centroid (mean of user-turn embeddings); reliability via role (0.7 user / 0.4 else); goal via cos to a global topic centroid (mean of ALL embeddings); value_alignment via cos to `soul_centroid` if provided else 0.0; usage from `metadata.get("retrieval_count")` saturating else 0.0; task_utility 0.0. READ `borge/eval/annotate.py` and `borge/memory/value.py::MemoryValue.FACTORS` to match factor keys EXACTLY.

**Step 1 — failing test:** build records from the fixture; `annotate_dump(recs, embedder=SBertEmbedder())` → list of factor dicts aligned with recs. Assert: a vivid/self-referential `keep-*` user fact has higher `emotion`+`self_relevance` than a `bloat-*` ("ok"); every dict has all 7 `MemoryValue.FACTORS` keys; `task_utility==0.0`; `reliability` 0.7 for user / 0.4 else.

**Step 2:** run → FAIL.

**Step 3 — implement** `factors.py`: `annotate_dump(records, *, embedder=None, soul_centroid=None) -> list[dict]`. Embed all texts once (reuse `SBertEmbedder`; default-construct if None). Compute μ_user centroid + global topic centroid. Per record build the 7-factor dict (clamp cos terms to [0,1] via `0.5+0.5*cosine`). Reuse `EmotionalSignalExtractor().extract(text, [])`. Pure-local.

**Step 4:** run → PASS.

**Step 5 — commit:** `feat(audit): blind factor annotation for static memory dumps`.

---

### Task 3: Bloat / forget-list (reuse MemoryValue)

**Files:**
- Create: `borge/audit/bloat.py`
- Test: `tests/test_audit_bloat.py`

**Step 1 — failing test:** given fixture records + their factor dicts + `default_memory_value()`, `forget_ranking(recs, factors, mv, keep_fracs=(0.3,0.5,0.7))` returns `{value_by_id, tiers}` where `tiers["aggressive"].forget_ids` (keep 0.3) contains the `bloat-*` ids and NOT the substantive `keep-*` ids. Assert bloat ranked below keep by V.

**Step 2:** run → FAIL.

**Step 3 — implement** `bloat.py`: compute `V = mv.value(factors[i])` per record; sort asc (low V = forget first); for each keep_frac produce a tier = the lowest-(1-frac) by value as `forget_ids`, rest `keep_ids`; return per-id values + tiers (safe=keep 0.7, moderate=0.5, aggressive=0.3, or map names→fracs). Reuse `MemoryValue` only; no new scoring.

**Step 4:** run → PASS.

**Step 5 — commit:** `feat(audit): bloat/forget-list ranking via paper2 MemoryValue`.

---

### Task 4: Contradiction module (embedding prefilter → local NLI)

**Files:**
- Create: `borge/audit/contradiction.py`
- Test: `tests/test_audit_contradiction.py`

**Step 1 — failing test:** `find_contradictions(recs, embedder=SBertEmbedder(), sim_threshold=0.3, nli_threshold=0.5, max_pairs=200)` → list of `CandidatePair(a_id,b_id,contradiction_score,likely_stale_id)` sorted desc. Assert the (`contra-A`,`contra-B`) pair appears with score above threshold; an unrelated (`keep-1`,`bloat-1`) pair does NOT. (Mark: first run downloads the NLI model — like the SBert test, allow it; keep the fixture tiny so it's fast.)

**Step 2:** run → FAIL.

**Step 3 — implement** `contradiction.py`:
- Embed all texts (reuse embedder). **Prefilter:** for each pair with `cosine >= sim_threshold` (same-topic), collect candidate pairs; cap at `max_pairs` (highest-similarity first) to bound cost (avoids N² NLI).
- **NLI:** `from sentence_transformers import CrossEncoder`; lazy-load `CrossEncoder("cross-encoder/nli-deberta-v3-small")` (module-level cache). For each candidate pair score BOTH directions `model.predict([(a,b),(b,a)])`; map the model's contradiction-label logit→prob (read the model's `config.id2label` to find the `contradiction` index; do NOT hardcode index blindly — assert label found). `contradiction_score = max(prob_ab, prob_ba)`.
- Keep pairs with `contradiction_score >= nli_threshold`; `likely_stale_id` = the OLDER timestamp's id (newer contradicts older → older is stale). Sort desc. Return list.
- Honest: docstring says "candidate contradictions for human review, product feature, not paper2-backed."

**Step 4:** run → PASS.

**Step 5 — commit:** `feat(audit): contradiction candidates via embedding-prefilter + local NLI`.

---

### Task 5: Hygiene — dedup + stale

**Files:**
- Create: `borge/audit/hygiene.py`
- Test: `tests/test_audit_hygiene.py`

**Step 1 — failing test:** `find_duplicates(recs, embedder, threshold=0.92)` → clusters; assert (`dup-A`,`dup-B`) land in one cluster, `keep-*` do not. `find_stale(recs, now_iso, age_days=180)` → ids; assert `stale-1`/`stale-2` flagged, recent ones not. (`now_iso` passed in for determinism — do NOT call datetime.now() inside.)

**Step 2:** run → FAIL.

**Step 3 — implement** `hygiene.py`: `find_duplicates` — embed, greedy cluster by cosine ≥ threshold (union-find or simple groups). `find_stale(recs, now_iso, age_days)` — parse timestamps, flag age > age_days. Both pure + deterministic (time injected).

**Step 4:** run → PASS.

**Step 5 — commit:** `feat(audit): dedup (near-duplicate clusters) + stale-by-age detection`.

---

### Task 6: Savings estimate

**Files:**
- Create: `borge/audit/savings.py`
- Test: `tests/test_audit_savings.py`

**Step 1 — failing test:** `estimate_savings(forget_records, *, retrieval_freq=1.0, price_per_1k=0.003, tokenizer="char4")` → `{tokens_saved, usd_per_month, assumptions}`. Assert deterministic token count for a known text (char/4) and that `assumptions` lists `retrieval_freq` + `price_per_1k` + tokenizer. Add a second case `tokenizer="tiktoken"` (cl100k) returns a positive int.

**Step 2:** run → FAIL.

**Step 3 — implement** `savings.py`: token count per forgotten record (char/4 default; optional tiktoken `cl100k_base`). `tokens_saved = sum(tokens) * retrieval_freq` (context re-sent per retrieval/month assumption). `usd_per_month = tokens_saved/1000 * price_per_1k`. Return numbers + an explicit `assumptions` dict. No hidden magic.

**Step 4:** run → PASS.

**Step 5 — commit:** `feat(audit): token/$ savings estimate with explicit assumptions`.

---

### Task 7: Report generator + CLI

**Files:**
- Create: `borge/audit/report.py`, `borge/audit/__main__.py`
- Modify: `pyproject.toml` ([project.scripts] add `borge-audit = "borge.audit.__main__:main"`)
- Test: `tests/test_audit_report.py`

**Step 1 — failing test:** `build_report(recs, factors, bloat, contradictions, dup_clusters, stale_ids, savings) -> str` returns markdown containing the 5 sections (Executive Summary, Bloat/Forget, Pollution, Safety/dry-run, Methodology appendix) and a fenced JSON forget-id script. Assert all 5 section headers present; assert the report TEXT never says "deleted" (only "recommend"/"candidate"); assert a planted `contra-A` id appears in the Pollution section.

**Step 2:** run → FAIL.

**Step 3 — implement** `report.py`: assemble markdown per design Section 3 (headline numbers from savings + bloat tier sizes + counts; tables for forget-list/contradictions/dups/stale; explicit assumptions + honest caveats incl. "contradiction = product feature, review required; forgetting informed by paper2, not guaranteed on your data"; a reversible forget-id JSON block). `__main__.py`: argparse `borge-audit <dump.json> [--soul soul.md] [--budget 0.3] [-o report.md]` → run ingest→factors→bloat→contradiction→hygiene→savings→report, write report, write `<out>.forget.json`. NEVER delete anything.

**Step 4:** run → PASS; also smoke `/Users/max_abel/opt/anaconda3/bin/python -m borge.audit tests/fixtures/audit_dump.json -o /tmp/r.md` → file written.

**Step 5 — commit:** `feat(audit): markdown report generator + borge-audit CLI (dry-run only)`.

---

### Task 8: End-to-end audit test

**Files:**
- Test: `tests/test_audit_e2e.py`

**Step 1 — write the test (the deliverable):** run the full pipeline on `tests/fixtures/audit_dump.json` via the same entry the CLI uses. Assert:
- forget-list (aggressive tier) contains ≥3 of the 4 `bloat-*` ids, none of `keep-*`;
- Pollution section flags the (`contra-A`,`contra-B`) contradiction;
- dedup flags (`dup-A`,`dup-B`); stale flags `stale-1`/`stale-2`;
- savings `usd_per_month` ≥ 0 and assumptions present;
- report is markdown with all 5 sections; a `*.forget.json` reversible script is produced; the input fixture file is UNCHANGED (no deletion/mutation).

**Step 2:** run → iterate until PASS. If the NLI threshold makes the contradiction miss/over-fire on the fixture, tune the THRESHOLD or the fixture texts (report which) — do not weaken the assertion to cheat.

**Step 3 — full suite:** `/Users/max_abel/opt/anaconda3/bin/python -m pytest -q` → green (existing 84 + new audit tests).

**Step 4 — commit:** `test(audit): end-to-end audit on synthetic dump (bloat+contradiction+dup+stale)`.

---

## Done criteria

- `borge-audit tests/fixtures/audit_dump.json -o report.md` produces a 5-section report + reversible `report.forget.json`, deletes nothing.
- Full `pytest -q` green.
- `borge/audit/` is self-contained; reuses `MemoryValue`/`SBertEmbedder` unchanged; contradiction kept OUT of `MemoryValue`.
- NOT merged to main/self-FEP/multi-factor-eval.
- Honest caveats present in the report template (contradiction = product feature; forgetting = paper2-informed not guaranteed).

## Notes for the implementer

- First run downloads the NLI model (~400MB) to the HF cache — same pattern as the existing SBert test; allow network on first run, cached after.
- Keep the fixture SMALL (~18 records) so NLI pairwise stays fast in CI.
- Determinism: inject `now_iso` for stale; fixed thresholds; no `datetime.now()` inside scoring.
- DRY: do not re-implement value scoring or embedding — import from `borge/memory/value.py` and `borge/values/self_model.py`.
