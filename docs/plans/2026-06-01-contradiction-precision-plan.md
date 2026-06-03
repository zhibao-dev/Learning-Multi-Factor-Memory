# borge-audit contradiction precision — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Branch:** `multi-factor-eval-business` ONLY.

**Goal:** Cut contradiction false positives. Two free-tier cheap wins (body-only chatter skip + exclude bloat records) plus an OPTIONAL LLM-judge复审 pass (NLI pre-filters → a caller-provided OpenAI-compatible LLM judges each survivor) that, when configured, filters same-topic FPs the local NLI can't.

**Architecture:** `find_contradictions` gains `skip_ids` + `judge` params (both optional; `judge=None` keeps the current API-free behaviour). A new `borge/audit/judge.py` builds an OpenAI-compatible judge callable via a tiny `urllib` POST (covers customer cloud LLM AND local Ollama; never you-hosted). `build_audit`/CLI wire endpoint config + report labels ("LLM-verified" vs "NLI candidates for review"). Design: `docs/plans/2026-06-01-contradiction-precision-design.md`.

**Tech Stack:** Python 3.11 (`/Users/max_abel/opt/anaconda3/bin/python` — `python` NOT on PATH). NO new deps (urllib is stdlib; SBert/NLI already present). Tests inject a FAKE judge / FAKE HTTP — never hit a real LLM.

**Guardrails:** `judge=None` MUST preserve the current API-free pipeline. Memory text only ever goes to the caller-configured endpoint (never you-hosted). Don't touch `borge/memory` or `borge/values`. Keep the honest report framing.

---

### Task 1: Free-tier cheap wins — body-only chatter skip + skip_ids

**Files:**
- Modify: `borge/audit/contradiction.py`
- Test: `tests/test_audit_contradiction_precision.py`

**Step 1 — failing test:**
```python
def test_body_only_skip_drops_heading_chatter():
    from borge.audit.contradiction import find_contradictions
    from borge.audit.ingest import MemoryRecord
    recs = [
        MemoryRecord("r1", "## Reply 1\nok", "2026-05-01", "user", {}),
        MemoryRecord("r2", "## Reply 2\nthanks", "2026-05-01", "user", {}),
        MemoryRecord("d1", "## Diet\nI am a strict vegetarian, never eat meat.", "2026-03-01", "user", {}),
        MemoryRecord("d2", "## Dinner\nI had an amazing steak last night, so good.", "2026-04-01", "user", {}),
    ]
    pairs = find_contradictions(recs)
    flagged = {frozenset((p.a_id, p.b_id)) for p in pairs}
    assert frozenset(("r1", "r2")) not in flagged       # chatter skipped (body "ok"/"thanks" <4 tokens)
    assert frozenset(("d1", "d2")) in flagged            # real contradiction kept

def test_skip_ids_excludes_records():
    from borge.audit.contradiction import find_contradictions
    from borge.audit.ingest import MemoryRecord
    recs = [
        MemoryRecord("d1", "## Diet\nI am a strict vegetarian, never eat meat.", "2026-03-01", "user", {}),
        MemoryRecord("d2", "## Dinner\nI had an amazing steak last night, so good.", "2026-04-01", "user", {}),
    ]
    assert find_contradictions(recs) != []                       # found w/o skip
    assert find_contradictions(recs, skip_ids={"d2"}) == []       # excluded → no pair
```
Run `/Users/max_abel/opt/anaconda3/bin/python -m pytest tests/test_audit_contradiction_precision.py -v` → FAIL.

**Step 2 — implement** in `contradiction.py`:
- Add `_body(text: str) -> str`: drop leading markdown heading lines (`^#{1,6}\s`) — return the text after the first heading line (or the whole text if no heading). Use it in the `MIN_TOKENS` filter: `records = [r for r in records if len(_body(r.text).split()) >= MIN_TOKENS]`.
- Add param `skip_ids: set[str] | None = None`; after the length filter, `records = [r for r in records if not skip_ids or r.id not in skip_ids]`.
- Everything else (prefilter, NLI, same-role gate) unchanged.

**Step 3** → PASS.

**Step 4 — no regressions:** `/Users/max_abel/opt/anaconda3/bin/python -m pytest tests/test_audit_contradiction.py tests/test_audit_contradiction_precision.py -q`.

**Step 5 — commit** `feat(audit): contradiction free-tier precision — body-only chatter skip + skip_ids`.

---

### Task 2: Optional LLM-judge hook in find_contradictions (fake-judge tested)

**Files:**
- Modify: `borge/audit/contradiction.py` (add `judge` param)
- Test: extend `tests/test_audit_contradiction_precision.py`

**Step 1 — failing test:**
```python
def test_judge_filters_topical_false_positives():
    from borge.audit.contradiction import find_contradictions
    from borge.audit.ingest import MemoryRecord
    recs = [
        MemoryRecord("d1", "## Diet\nStrict vegetarian, never eat meat.", "2026-03-01", "user", {}),
        MemoryRecord("d2", "## Dinner\nHad an amazing steak last night.", "2026-04-01", "user", {}),
        MemoryRecord("c1", "## CI\nCI runs on GitHub Actions.", "2026-03-01", "user", {}),
        MemoryRecord("c2", "## Testing\nIntegration tests hit a throwaway DB.", "2026-03-01", "user", {}),
    ]
    # fake judge: only the diet/dinner pair is a real contradiction
    def fake_judge(a, b):
        real = ("vegetarian" in (a+b).lower() and "steak" in (a+b).lower())
        return {"contradict": real, "stale_id": None}
    pairs = find_contradictions(recs, judge=fake_judge, nli_threshold=0.0)  # let NLI pass many; judge filters
    flagged = {frozenset((p.a_id, p.b_id)) for p in pairs}
    assert frozenset(("d1", "d2")) in flagged
    assert all(frozenset(("c1","c2")) != f for f in flagged)   # topical FP filtered by judge
```
Run → FAIL.

**Step 2 — implement:** add `judge: callable | None = None`. After NLI produces the survivor list (pairs ≥ nli_threshold), if `judge` is set: for each survivor call `v = judge(text_a, text_b)`; keep only `v and v.get("contradict")`; set `likely_stale_id = v.get("stale_id") or <older-timestamp id>`. If `judge is None`, behaviour unchanged. (The judge runs on the already-narrowed survivors, not all pairs.)

**Step 3** → PASS.

**Step 4** suite green.

**Step 5 — commit** `feat(audit): optional LLM-judge filter in find_contradictions (judge=None → unchanged)`.

---

### Task 3: OpenAI-compatible judge client (urllib, fake-HTTP tested)

**Files:**
- Create: `borge/audit/judge.py`
- Test: `tests/test_audit_judge.py`

**Step 1 — failing test:**
```python
def test_make_openai_judge_parses_verdict(monkeypatch):
    import borge.audit.judge as J
    # fake the HTTP call → return a canned chat-completions JSON
    def fake_post(url, payload, headers):
        return {"choices": [{"message": {"content": '{"contradict": true, "stale_id": "a"}'}}]}
    monkeypatch.setattr(J, "_post_json", fake_post)
    judge = J.make_openai_judge(base_url="http://x/v1", model="m", api_key="k")
    v = judge("memory A text", "memory B text")
    assert v["contradict"] is True and v["stale_id"] == "a"

def test_make_openai_judge_handles_bad_response(monkeypatch):
    import borge.audit.judge as J
    monkeypatch.setattr(J, "_post_json", lambda *a, **k: {"choices": [{"message": {"content": "not json"}}]})
    judge = J.make_openai_judge(base_url="http://x/v1", model="m")
    assert judge("a", "b") is None     # unparseable → None (pair kept? no — None means "judge abstains" → drop)
```
Run → FAIL.

**Step 2 — implement** `borge/audit/judge.py`:
- `_post_json(url, payload, headers) -> dict`: a thin `urllib.request` POST (JSON in/out). Isolated so tests monkeypatch it.
- `make_openai_judge(base_url, model, api_key=None, timeout=30) -> callable`: returns `judge(text_a, text_b)` that builds a chat-completions request to `{base_url}/chat/completions` with a strict prompt — *"You compare two agent-memory entries. Do they assert CONTRADICTORY FACTS about the SAME subject (not merely the same topic)? Reply ONLY JSON: {\"contradict\": bool, \"stale_id\": \"a\"|\"b\"|null} where stale_id is the entry more likely outdated."* — passes the two texts labelled A/B; parses the JSON from the response `content`; maps `stale_id` "a"/"b" → the caller resolves to the real id (the judge callable used by find_contradictions gets raw texts, so return `{"contradict":..., "stale_id": None}` here and let find_contradictions set the id by older-timestamp, OR have find_contradictions pass ids — keep judge text-only + stale by timestamp to stay simple). On any parse/HTTP error → return `None` (abstain).
- Use `response_format`/low temperature if supported; keep it OpenAI-vanilla so Ollama works.

**Step 3** → PASS (no real network — monkeypatched).

**Step 4** suite green.

**Step 5 — commit** `feat(audit): OpenAI-compatible LLM judge client (cloud or local Ollama; urllib, no new dep)`.

---

### Task 4: Wire into build_audit + CLI + report labels

**Files:**
- Modify: `borge/audit/report.py` (build_audit: endpoint params → make judge → pass judge + skip_ids; `_render_markdown` label), `borge/audit/__main__.py` (`--llm-endpoint/--llm-model/--llm-key`)
- Test: `tests/test_audit_judge_wiring.py`

**Step 1 — failing test:** `build_audit(..., judge=<fake stub>)` (add a `judge` param for testability) → the Pollution section labels contradictions "LLM-verified"; without judge → "candidate"/"for human review". Also assert `skip_ids` (bottom-value records) are passed so chatter doesn't reach NLI. (Test with the JSON fixture + a fake judge; do NOT require an endpoint.)

**Step 2 — implement:**
- `build_audit(..., judge=None, llm_endpoint=None, llm_model=None, llm_key=None)`: if `judge` is None but `llm_endpoint` set → `judge = make_openai_judge(llm_endpoint, llm_model, llm_key)`. Compute `skip_ids` = the lowest-value records (e.g. the aggressive-tier forget_ids, or value below a small floor) and pass both into `find_contradictions(records, embedder=embedder, judge=judge, skip_ids=skip_ids)`. Thread a `judge_used: bool` into `_render_markdown`.
- `_render_markdown`: when `judge_used` → header/section says "LLM-verified contradictions"; else current "candidates for human review". Keep all other honest framing.
- `__main__`: add `--llm-endpoint`, `--llm-model`, `--llm-key` (key may also come from env); pass to `build_audit`.

**Step 3** → PASS.

**Step 4** full suite green; smoke `borge-audit tests/fixtures/audit_dump.json -o /tmp/x.md` (no endpoint → NLI path, unchanged).

**Step 5 — commit** `feat(audit): wire optional LLM judge + skip_ids into build_audit/CLI + report labels`.

---

### Task 5: e2e — demo dump with fake judge → clean report

**Files:**
- Test: `tests/test_audit_judge_e2e.py`

**Step 1 — write the test:** run `build_audit("packaging/borge-audit/demo/sample_agent_memory.md", now_iso="2026-06-01T00:00:00", md_split="heading", judge=<fake stub that confirms only TS↔Go and Vercel↔k8s>)`. Assert:
- the two REAL contradictions (tech-stack↔stack-migration, deployment↔infra-change) appear in Pollution;
- the topical FPs (ci↔testing, auth↔secrets) do NOT;
- the chatter pairs (reply-*) do NOT (body-only skip);
- report labels them "LLM-verified";
- dry-run: forget_script dict, input `.md` byte-unchanged.
The fake stub keys on the texts (e.g. `contradict = ("typescript" in t and " go" in t.lower()) or ("vercel" in t and "kubernetes" in t.lower())`).

**Step 2** → PASS.

**Step 3 — full suite** `/Users/max_abel/opt/anaconda3/bin/python -m pytest -q` green.

**Step 4 — regenerate the teaser** (now clean) and commit it: with the fake-judge path proven, the controller will re-run the demo with a real/fake judge to refresh `packaging/borge-audit/demo/sample_audit_report.md`. Commit the e2e test. Message: `test(audit): e2e LLM-judge contradiction filtering on demo dump (fake judge)`.

---

## Done criteria

- Free tier (no endpoint): chatter pairs gone (body-only skip), bloat excluded; NLI candidates labelled "for review".
- Pro path (endpoint/judge set): topical FPs filtered, contradictions "LLM-verified"; demo dump yields the 2 real contradictions only.
- `judge=None`/no endpoint → API-free pipeline unchanged; full `pytest -q` green; no real LLM in tests.
- No `borge/memory` or `borge/values` edits. Rebuild the private wheel after (`build_wheel.sh`) so `judge.py` ships.
- NOT merged to research branches.

## Notes for the implementer

- Tests must NEVER hit a real LLM — inject a fake judge callable / monkeypatch `_post_json`.
- `judge` returns `None` on any error → treat as abstain (drop the pair, since the whole point is precision).
- Keep the judge text-only; resolve `likely_stale_id` by older timestamp inside `find_contradictions` for simplicity.
- After merge, refresh `packaging/borge-audit/demo/sample_audit_report.md` so the teaser reflects the clean output.
