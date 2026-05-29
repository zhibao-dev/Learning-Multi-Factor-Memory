# Paper2 Runtime Wiring — Design

**Branch:** `multi-factor-eval` ONLY. The `self-FEP` branch keeps paper1's
self-FEP product-of-resistances forget/retrieve; `main` stays self-FEP core.
Core changes here MUST NOT be merged into `main`/`self-FEP`.

**Goal:** make paper2's claim real in the runtime — one learned multi-factor
value `V(m)=Σ wᵢ fᵢ` uniformly controls the three live memory operations
(encoding depth, forget score, retrieval rank). Today they run paper1's
hand-tuned formulas and nothing in the live path calls `MemoryValue`.

## Locked decisions (brainstorm 2026-05-28)

- **Q1 live factors = D (6 free-proxy):** emotion (signal_extractor),
  self_relevance (cos μ_self), usage (retrieval_count), reliability
  (role/provenance heuristic), value_alignment (cos to SOUL value centroid —
  free, no LLM), goal_relevance (cos to session-topic centroid — the paper's
  blind anchor, free). `task_utility` stays 0 (LLM future). The value object
  stays 7-wide so task_utility slots in later.
- **Q2 weights = shipped default ⊕ config override.** Out-of-box weights =
  the full-479 fit; `borge.memory.value.weights` in config overrides. SOUL /
  online-learning are future.
- **Q3 timing = encode-time annotation, persisted.** Factors computed at
  consolidation (session context present), written to existing
  `borge_memories` factor columns; forget/retrieve READ the columns. No
  schema change. Hermes `messages` table (no factor columns) degrades to
  emotion+usage (derivable from valence/arousal/retrieval_count); never crash.
- **Q4 retrieval = V + query terms.** `rank = w_v·V(factors) +
  w_q·query_relevance + w_mood·mood`. encode/forget are pure-V; retrieval
  adds a query-relevance term V cannot represent (V is query-agnostic).
- **Q5 verification = thick.** Per-op unit tests + shared-instance assertion
  + end-to-end golden scenario + harness-consistency (live forget ordering ==
  offline `lme_blind_forgetting`).

## Architecture

One `MemoryValue` instance on `BorgeAgent` (`self._memory_value`). The SAME
instance is injected into the consolidation/encoding step, `ForgettingEngine`,
and `MemoryRetrieval`. Factors are computed once at encode time (where session
context — μ_self, SOUL centroid, session-topic centroid — exists) and
persisted; the other two ops read stored factors. The four lifecycle hooks
(`on_session_start`, `pre_turn`, `post_tool`, `on_session_end`) are unchanged;
all wiring is internal to `BorgeAgent`.

## Components / changes

1. **`borge/memory/value.py`** — add
   `runtime_factors(row, *, mu_self, soul_centroid, session_centroid, role)`:
   derives the 6 free-proxy factors from a memory + encode-time context,
   missing input → neutral (0.5 for similarities, 0 for the rest). Distinct
   from the existing `memory_factors(row)` which only reads already-stored
   columns (used by forget/retrieve).

2. **`borge/agent.py::__init__`** — build
   `self._memory_value = MemoryValue(weights=_load_value_weights(cfg))`.
   `_load_value_weights` = `SHIPPED_DEFAULT` merged with config
   `borge.memory.value.weights`. `SHIPPED_DEFAULT` = full-479 blind fit
   `{emotion:0.55, goal_relevance:0.0, self_relevance:0.23, reliability:0.64,
   value_alignment:0.0, task_utility:0.0, usage:0.1}` (small usage prior; the
   four that carried signal dominate). Inject `self._memory_value` into the
   ForgettingEngine, MemoryRetrieval, and the consolidation pipeline.

3. **encode** (consolidation persist step) — for each message compute
   `runtime_factors(...)` with this session's μ_self / SOUL centroid /
   session-topic centroid, store into the existing factor columns
   (`goal_relevance`, `value_alignment`, `self_relevance_score`, `reliability`,
   …), and set `encoding_depth = value_encoding_depth(factors, self._mv)`.

4. **forget** (`borge/memory/forgetting.py`) — replace the `_compute_score`
   body with `value_forget_score(row, mv, beta)` (recency × usage × value
   resistance; already written + tested). Engine holds the injected `mv`.
   Degrade: a row missing factor columns → `memory_factors` returns 0 for
   them → forget falls back to emotion+usage. **This file diverges from
   self-FEP and must never merge back.**

5. **encoding depth** — see (3); tier set at encode via `value_encoding_depth`.

6. **retrieve** (`borge/memory/retrieval.py`) —
   `rank = w_v·V(stored factors) + w_q·query_relevance + w_mood·mood`.
   The single V replaces the importance/self/recency-ish terms; query_relevance
   (token/embedding overlap with the actual query) and mood-congruence stay.
   Weights configurable (`borge.memory.retrieval.*`), defaults preserve current
   behavior shape. `recall` still bumps `retrieval_count` → usage feeds back
   into V (closes the retrieval↔forgetting loop).

## Data flow

```
user turn → pre_turn (emotion signal) → … →
on_session_end consolidation:
   for each msg: runtime_factors(session ctx) → persist factors
                 + encoding_depth = value_encoding_depth(factors, V)
   forgetting pass: read factors → value_forget_score(row, V) → prune/keep
later recall(query):
   rank = w_v·V(stored factors) + w_q·query_relevance + w_mood·mood
   → bump retrieval_count (usage → V next time)
```

## Verification (thick) — `tests/test_runtime_value_wiring.py`

- **shared-instance**: ForgettingEngine, MemoryRetrieval, and the
  consolidation step all reference the same `BorgeAgent._memory_value`
  (assert by object identity) — the hard proof of "one value, three ops".
- **encode**: two memories, higher-V factor vector → deeper encoding tier.
- **forget golden**: store a high-V and a low-V row; after the forgetting
  pass the high-V survives, the low-V is pruned.
- **retrieve golden**: high-V ranked above low-V on a neutral query; a
  query-relevant low-V row jumps when the query matches (proves the
  query term is live).
- **harness consistency**: feed fixed factor vectors through the live
  `ForgettingEngine` path and assert the ranking equals
  `experiments/lme_blind_forgetting` ordering on the same input — ties the
  runtime to the paper's 0.770 evidence.

All deterministic: `hash_embed` default embedder, fixed weights, no LLM.

## Constraints honored

- 4 lifecycle hooks unchanged; all cognitive state funneled through
  `BorgeAgent`; plugin hooks still `try/except` → degrade (zero-invasion).
- Zero schema change: `borge_memories` factor columns already exist; Hermes
  `messages` table untouched → graceful degradation.
- `self-FEP` branch untouched; `forgetting.py`/`retrieval.py` here diverge and
  must not be cross-merged.

## Out of scope

- `task_utility` factor (needs LLM; stays 0 with the 7-wide slot reserved).
- Online weight learning (no runtime gold signal; future).
- New paper experiments (the offline harness already provides the evidence).
