# borge-audit markdown + weights — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Branch:** `multi-factor-eval-business` ONLY.

**Goal:** Let `borge-audit` ingest **markdown** agent-memory files (frontmatter + timestamp recovery + configurable splitting) and run with **per-customer weights** (`--weights`), so the tool is usable on real customers' memory, not just canonical JSON with default weights.

**Architecture:** New `borge/audit/ingest_md.py` produces the SAME `MemoryRecord` list as the JSON ingest, so the value pipeline is unchanged. `build_audit` + the CLI route by file extension and accept a `weights` override (threaded into the existing `default_memory_value(override)`); the report shows the weights used. Design: `docs/plans/2026-06-01-audit-markdown-and-weights-design.md`.

**Tech Stack:** Python 3.11 (`/Users/max_abel/opt/anaconda3/bin/python` — `python` NOT on PATH). Reuse `borge/audit/ingest.py::MemoryRecord`, `borge/memory/value.py::default_memory_value`. YAML frontmatter parse: hand-roll a tiny `key: value` parser (NO new pyyaml dep — keep it dependency-light; frontmatter is simple key:value). Tests: flat `tests/test_audit_md_*.py`; markdown fixtures in `tests/fixtures/`.

**Honest guardrails:** markdown lacks `usage` → bloat is best-effort there (report says so); recency ≠ bloat signal. Do NOT add a triviality factor (out of scope; would touch the vendored value model). Never delete input. Keep `MemoryRecord` shape identical so the pipeline is untouched.

---

### Task 1: Markdown fixture + heading-split ingest with frontmatter + timestamp recovery

**Files:**
- Create: `tests/fixtures/audit_memory.md`
- Create: `borge/audit/ingest_md.py`
- Test: `tests/test_audit_md_ingest.py`

**Step 1 — fixture** `tests/fixtures/audit_memory.md`: a CLAUDE.md-style file —
- a YAML frontmatter block: `---\nmodified: 2026-05-20T09:00:00Z\ntags: [profile]\n---`
- `##`/`###` heading sections, each a memory, with PLANTED ids encodable via heading text:
  - section "Diet" body "I am a strict vegetarian, never eat meat."
  - section "Dinner" body "Had an amazing steak last night." (← contradiction pair with Diet)
  - section "Deadline A" "Project deadline is March 15." and "Deadline B" "The project deadline is March 15th." (← duplicate)
  - a couple substantive `## ` sections (allergy, manager) as keep
  - one section whose body carries an inline old date `(2025-03-01)` for stale.

**Step 2 — failing test** `tests/test_audit_md_ingest.py`:
```python
def test_markdown_heading_split_with_frontmatter_and_timestamps():
    from borge.audit.ingest_md import load_markdown_dump
    from borge.audit.ingest import MemoryRecord
    recs = load_markdown_dump("tests/fixtures/audit_memory.md", split="heading")
    assert all(isinstance(r, MemoryRecord) for r in recs)
    # one record per heading section; vegetarian + steak both present
    texts = " ".join(r.text for r in recs).lower()
    assert "vegetarian" in texts and "steak" in texts
    # frontmatter `modified` recovered as the timestamp for records lacking an inline date
    assert any(r.timestamp.startswith("2026-05-20") for r in recs)
    # ids are stable + unique
    assert len({r.id for r in recs}) == len(recs)
```
Run `/Users/max_abel/opt/anaconda3/bin/python -m pytest tests/test_audit_md_ingest.py -v` → FAIL.

**Step 3 — implement** `borge/audit/ingest_md.py`:
- `load_markdown_dump(path, *, split="auto") -> list[MemoryRecord]` (this task: implement `split="heading"` + the `auto`→heading fallback; bullet/dated come in Task 2).
- `_parse_frontmatter(text) -> (meta: dict, body: str)`: if text starts with `---\n`, read until the next `---`, parse simple `key: value` lines (strip `[]` list brackets into a list for tags); return meta + remaining body.
- timestamp recovery helper `_recover_ts(record_inline_date, frontmatter_meta, path)`: inline date (if found in the section, ISO or `(YYYY-MM-DD)`) → `frontmatter_meta` `modified`/`created`/`timestamp` → file mtime (`datetime.fromtimestamp(os.path.getmtime(path), tz=utc).isoformat()`). Return ISO string (or "" if all fail — but mtime always exists, so effectively always set).
- heading split: split body on lines matching `^#{2,6}\s` ; each section = heading line + following lines until next heading. `text` = heading + body (or body only — include heading for context). `id` = `<filestem>#<slugified-heading>` (lowercase, non-alnum→`-`); dedupe ids by appending index. `role` = frontmatter `role` else `"user"`. `metadata` = `{"source_file": path, "heading_path": heading, "tags": meta.get("tags", [])}`. Content before the first heading (preamble) → one record id `<filestem>#_preamble` if non-empty.

**Step 4** → PASS.

**Step 5 — commit** `feat(audit): markdown ingest — heading split + frontmatter + timestamp recovery`.

---

### Task 2: Bullet + dated split modes + auto-detect

**Files:**
- Modify: `borge/audit/ingest_md.py`
- Test: `tests/test_audit_md_split.py` (+ fixtures `tests/fixtures/audit_memory_bullets.md`, `audit_memory_log.md`)

**Step 1 — fixtures:**
- `audit_memory_bullets.md`: a flat `- fact` list (preferences/facts), some duplicates.
- `audit_memory_log.md`: dated append log — entries like `## 2026-05-30` or `- [2026-05-28] used Django 3.2` with several dates incl. an old one (2025-...).

**Step 2 — failing test** `tests/test_audit_md_split.py`:
```python
def test_bullet_split_one_record_per_item():
    from borge.audit.ingest_md import load_markdown_dump
    recs = load_markdown_dump("tests/fixtures/audit_memory_bullets.md", split="bullet")
    assert len(recs) >= 4            # one per top-level bullet
    assert all(r.text.strip() for r in recs)

def test_dated_split_recovers_per_entry_timestamp():
    from borge.audit.ingest_md import load_markdown_dump
    recs = load_markdown_dump("tests/fixtures/audit_memory_log.md", split="dated")
    # each dated entry gets its OWN timestamp (not the file mtime)
    assert any(r.timestamp.startswith("2025-") for r in recs)   # the old entry
    assert any(r.timestamp.startswith("2026-") for r in recs)

def test_auto_detects_structure():
    from borge.audit.ingest_md import load_markdown_dump
    # log file auto-detects dated; bullet file auto-detects bullet
    log = load_markdown_dump("tests/fixtures/audit_memory_log.md", split="auto")
    assert any(r.timestamp.startswith("2025-") for r in log)
```
Run → FAIL.

**Step 3 — implement:** add `split="bullet"` (split on top-level `^[-*]\s` lines; each = a record; text = item incl. nested lines), `split="dated"` (split on a date heading/prefix regex `^#{1,6}\s*\d{4}-\d{2}-\d{2}` or `^[-*]\s*\[?\d{4}-\d{2}-\d{2}`; per-entry inline date → that record's timestamp via `_recover_ts`), and `auto` detection: count dated-entry matches → if ≥2 use `dated`; else count top-level bullets vs headings → bullet if bullets dominate, else heading.

**Step 4** → PASS; `/Users/max_abel/opt/anaconda3/bin/python -m pytest tests/test_audit_md_ingest.py tests/test_audit_md_split.py -q`.

**Step 5 — commit** `feat(audit): markdown bullet + dated split modes + auto-detect`.

---

### Task 3: CLI + build_audit format routing (.md vs .json)

**Files:**
- Modify: `borge/audit/report.py` (`build_audit` — route ingest by extension), `borge/audit/__main__.py` (add `--md-split`)
- Test: `tests/test_audit_md_routing.py`

**Step 1 — failing test:**
```python
def test_build_audit_routes_markdown():
    from borge.audit.report import build_audit
    out = build_audit("tests/fixtures/audit_memory.md", now_iso="2026-06-01T00:00:00")
    md = out["markdown"]
    assert "vegetarian" in md.lower() or "steak" in md.lower()   # pipeline ran on the .md
    assert "Pollution" in md
```
Run → FAIL (build_audit calls load_dump → JSON-only → breaks on .md).

**Step 2 — implement:**
- `build_audit(dump_path, *, ..., md_split="auto")`: choose ingest by extension — `str(dump_path).lower().endswith(".md")` → `load_markdown_dump(dump_path, split=md_split)`, else `load_dump(dump_path)`. Import `load_markdown_dump`.
- `__main__`: add `--md-split` (choices heading/bullet/dated/auto, default auto); pass to `build_audit`. Extension routing is automatic.

**Step 3** → PASS; smoke `/Users/max_abel/opt/anaconda3/bin/python -m borge.audit tests/fixtures/audit_memory.md -o /tmp/md.audit.md` writes a report; input `.md` unchanged.

**Step 4** full suite green.

**Step 5 — commit** `feat(audit): route ingest by file type (.md → markdown), --md-split flag`.

---

### Task 4: Per-customer weights (`--weights`) + report shows weights used

**Files:**
- Modify: `borge/audit/report.py` (`build_audit` weights param; `_render_markdown` show weights), `borge/audit/__main__.py` (`--weights`)
- Test: `tests/test_audit_weights.py`

**Step 1 — failing test:**
```python
def test_custom_weights_change_ranking_and_show_in_report(tmp_path):
    import json
    from borge.audit.report import build_audit
    # weight reliability to 0 → forget order changes vs default; report shows weights
    w = tmp_path / "w.json"; w.write_text(json.dumps({"reliability": 0.0}))
    out = build_audit("tests/fixtures/audit_dump.json", now_iso="2026-06-01T00:00:00",
                      weights=json.loads(w.read_text()))
    assert "reliability" in out["markdown"].lower()    # weights surfaced in report
    base = build_audit("tests/fixtures/audit_dump.json", now_iso="2026-06-01T00:00:00")
    # overriding a heavy weight changes the chosen forget set (sanity that weights flow through)
    assert out["forget_script"]["forget_ids"] != base["forget_script"]["forget_ids"]
```
Run → FAIL (build_audit has no `weights` param).

**Step 2 — implement:**
- `build_audit(..., weights: dict | None = None)` → `mv = default_memory_value(weights)` (replaces `default_memory_value()`).
- `_render_markdown(..., weights_used: dict)`: in Methodology (and a one-line header note) render the exact weights used + the sentence: *"Weights are tunable per your business — defaults are fit to a general benchmark and may not match your scenario."* Pass `mv.weights` in.
- `__main__`: `--weights path.json` → `json.loads(Path(...).read_text())` → pass to `build_audit`. If absent, weights=None (shipped default).

**Step 3** → PASS.

**Step 4** full suite green.

**Step 5 — commit** `feat(audit): --weights per-customer override + report shows weights used`.

---

### Task 5: End-to-end — markdown audit (+ weights), dry-run

**Files:**
- Test: `tests/test_audit_md_e2e.py`

**Step 1 — write the test:** `build_audit("tests/fixtures/audit_memory.md", now_iso=fixed, weights={"reliability":0.9})` →
- markdown ran: report has 5 sections;
- the planted contradiction (vegetarian/steak) appears in Pollution;
- the planted duplicate (deadline) appears;
- the inline-old-dated section flagged stale;
- report shows the custom weights;
- `forget_script` is a dry-run dict; the `.md` fixture file is byte-unchanged after the run;
- markdown-honesty note present (bloat best-effort on markdown / usage absent) — assert a string like "best-effort" or "usage" appears in Methodology.

**Step 2** → iterate to PASS (tune the markdown fixture's contradiction wording if NLI misses it — report, don't weaken).

**Step 3** full `/Users/max_abel/opt/anaconda3/bin/python -m pytest -q` green.

**Step 4 — commit** `test(audit): e2e markdown audit with custom weights (dry-run)`.

---

## Done criteria

- `borge-audit tests/fixtures/audit_memory.md --md-split auto --weights w.json` produces a 5-section report + dry-run forget.json, never touches the input.
- Markdown + JSON both route correctly; `MemoryRecord` shape unchanged → value pipeline untouched.
- Report shows the weights used + markdown-honesty note.
- Full `pytest -q` green; only `borge/audit/` + tests touched (NO `borge/memory/`, NO `borge/values/` changes — keeps the vendored thin-slice clean).
- Rebuild the private wheel after merge (`bash packaging/borge-audit/build_wheel.sh`) so the new ingest ships.

## Notes for the implementer

- Keep frontmatter parsing dependency-free (simple `key: value`); do NOT add pyyaml.
- `MemoryRecord` MUST stay identical — markdown is just another producer of it.
- `usage`/retrieval_count absent in markdown is EXPECTED; don't fabricate it.
- Determinism: `now_iso` injected; for mtime-based timestamps in tests, assert the frontmatter/inline path (deterministic), not mtime (varies).
