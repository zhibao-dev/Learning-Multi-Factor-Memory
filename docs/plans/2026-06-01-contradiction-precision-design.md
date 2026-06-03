# borge-audit: contradiction precision — Design

> Branch: `multi-factor-eval-business`. Validated via brainstorm 2026-06-01.
> Motivated by a teaser run: on a realistic 23-section markdown agent-memory,
> `find_contradictions` returned **20 candidates, only 2 real** (the rest chatter
> pairs + same-topic non-contradictions, all tied ~1.00 → threshold can't separate).

## Problem (proven)

`borge/audit/contradiction.py` uses a local NLI cross-encoder. On markdown memory it floods false positives because:
1. **Chatter leaks** — the `<MIN_TOKENS` skip counts the whole record text incl. the heading (`"## Reply 1\nok"` = 4+ tokens), so 1-word replies aren't skipped → "ok"/"thanks" pairs score ~1.0.
2. **Same-role gate inert** — markdown records all default `role=user`, so the cross-role filter does nothing.
3. **NLI conflates topic with contradiction** — `ci↔testing`, `auth↔secrets` score ~1.0 though not conflicting. Threshold can't fix (real + garbage both ~1.0).

## Two cheap free-tier wins (no LLM)

- **body-only token skip**: strip leading heading line(s) before the `<MIN_TOKENS` check → drops the chatter flood (~10 of the 18 FPs).
- **exclude low-value/bloat records**: `find_contradictions(..., skip_ids=...)`; `build_audit` passes the bottom-value records' ids (don't bother NLI-comparing memories already flagged as bloat).

These cut the chatter, but the **same-topic FPs** (`ci↔testing`) remain (not short) → need the LLM judge.

## The precision layer: optional LLM-judge复审

The audit is otherwise **LLM-free** (SBert + NLI are small *local encoders/classifiers*, not generative LLMs — API-free, data-in-infra, zero per-call cost). The judge is the FIRST and ONLY generative-LLM step, deliberately **optional**:

- `find_contradictions(..., judge=None)`. `judge` is a callable `(text_a, text_b) -> {"contradict": bool, "stale_id": str|None} | None`.
- Runs ONLY on the NLI survivors (NLI already narrowed N²→few; the judge is cheap). Keeps pairs the judge confirms; `likely_stale` from the judge (it reads dates/semantics) else older-timestamp.
- **`judge=None` → skip → return NLI candidates (current behaviour).** The API-free tool is unchanged.

### LLM source (decided): bring-your-own, never you-hosted

One **OpenAI-compatible judge client** (`base_url + model + api_key`) covers BOTH the customer's cloud LLM AND a local Ollama (Ollama exposes an OpenAI-compatible endpoint) — one code path, two deployments. The memory text goes only to **the LLM the customer already trusts with their agent's data** → zero new data exposure, zero new trust ask, works for finance if their LLM is already approved. A you-hosted API is explicitly rejected (would break data-in-infra). Dependency-light: a tiny `urllib` POST to `{base_url}/chat/completions`, no new SDK.

### Free vs pro (pricing label, NOT code gating)

- **Free** = NLI candidates, honestly labelled "for human review" (noisy).
- **Pro** = LLM-judge-filtered, labelled "LLM-verified" (high precision).
- The noise becomes the upsell. **No license gating built now** (0 customers) — code is just an optional pass; "pro" is how it's priced later.

## Report honesty

When the judge ran → confirmed contradictions are "LLM-verified" (drop the "candidate" hedge for those). When not → "NLI candidates for review" (current wording). The label reflects which path ran. Doc states the judge uses the customer's own configured LLM.

## Testing (no real LLM)

- cheap wins: body-only skip drops `"## Reply\nok"`; `skip_ids` excludes bloat.
- judge hook: inject a **fake judge stub** (deterministic, no API) — returns contradict=True for the planted real pair, False for topical FPs → assert only real survive.
- judge client: test prompt-build + response-parse with a **fake HTTP response** (monkeypatch the POST); never hit a real endpoint.
- e2e: demo dump + fake judge → clean 2-contradiction report.

## Anchor points (verified 2026-06-01)

- `borge/audit/contradiction.py`: `MIN_TOKENS=4` (line ~43, used line ~123 over whole text); same-role gate line ~135; `find_contradictions(records, *, embedder=None, sim_threshold=0.3, nli_threshold=0.5, max_pairs=200)`; `CandidatePair(a_id,b_id,contradiction_score,likely_stale_id)`.
- `borge/audit/report.py::build_audit(...)` calls `find_contradictions(records, embedder=embedder)` (~line 63) + computes per-record value (for `skip_ids`).

## Out of scope (YAGNI)

license gating · scenario weight presets · entity/NER contradiction model · multi-judge voting · the deferred triviality bloat factor. All later, customer-pulled.
