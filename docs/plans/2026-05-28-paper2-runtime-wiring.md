# Paper2 Runtime Wiring — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Branch:** `multi-factor-eval` ONLY. Do NOT merge core `borge/memory/` changes into `main` or `self-FEP` (those keep paper1's self-FEP formulas). The branch IS the paper separation.

**Goal:** Make paper2's claim real in the live `BorgeAgent`: one learned multi-factor value `V(m)=Σ wᵢ·fᵢ` uniformly drives all three live memory operations — encoding depth, forget score, retrieval rank — replacing paper1's three hand-tuned formulas.

**Architecture:** A single `MemoryValue` instance lives in `BorgeAgent`, built from a shipped learned-default weight vector (overridable via `config.yaml`). It is injected into the three engines (`MemoryConsolidationPipeline`, `ForgettingEngine`, `MemoryRetrieval`). Factors are computed **at encode time** (Step 3 of consolidation, where session + SOUL context exists) and **persisted** to `borge_memories` columns; forget/retrieve read the stored factors. Tables lacking factor columns (Hermes `messages`) degrade gracefully to the factors derivable from existing columns.

**Tech Stack:** Python 3.11, SQLite, existing `borge/memory/value.py` (`MemoryValue`, `value_forget_score`, `value_encoding_depth`), pytest. Interpreter: `/Users/max_abel/opt/anaconda3/bin/python` (`python` not on PATH).

---

## Design decisions (from brainstorm, LOCKED)

- **D — 6 free-proxy factors.** Live factors = `emotion`, `self_relevance`, `usage`, `reliability`, `value_alignment` (cos to SOUL value centroid), `goal_relevance` (cos to session-topic centroid). `task_utility` = 0 (LLM-gated, future). All computed API-free.
- **Weights: shipped default + config override.** `SHIPPED_DEFAULT` = full-479 fit (emotion 0.55, goal_relevance 0.0, self_relevance 0.23, reliability 0.64; value_alignment 0.0, task_utility 0.0, usage small). Override via `config.yaml` key `borge.memory.value.weights`. With the shipped default, goal/value_alignment ≈0 → runtime behaves like the 4-factor "A" set, but the 6-factor machinery is real and re-weightable.
- **Encode-time annotation + persist.** Factors computed in consolidation Step 3, written to `borge_memories`. No schema change to Hermes `messages` (zero-invasion). Missing columns → `MemoryValue.value` `get(f,0)` → graceful degrade.
- **Retrieval = V + query terms.** `rank = w_v·V(factors) + w_rel·relevance + w_mood·mood + w_rec·recency`. V (query-agnostic) supplies durable worth, absorbing the old self/emotion/f/importance terms; `relevance` is the only genuinely query-dependent signal kept; mood + recency are query-time context.

## Existing-code anchor points (verified 2026-05-28)

- `borge/memory/value.py`: `MemoryValue(weights).value(factors)->float`; `value_forget_score(row, mv, now=None, *, beta)`; `value_encoding_depth(factors, mv)->int{1..4}`; `memory_factors(row)` (reads borge_memories row → factor dict); `FACTORS` 7-tuple.
- `borge/memory/store.py`: `BORGE_MEMORIES_SCHEMA` + `_LATE_COLUMNS` ALREADY define columns `goal_relevance/value_alignment/task_utility/reliability` (lines 52–55, 83–86). BUT `insert()` SQL (117–131) and `_normalize()` (248–269) DO NOT write them. **Bug to fix in Task 1.**
- `borge/memory/consolidation.py`: Step 3 `_step3_emotional_significance` (220–332) is the encode/persist path; computes significance, self_relevance, depth (254–305), calls `self.store.insert({...})` (308–323). `self.self_model._embed(content)` available; `has_self_reference` imported.
- `borge/memory/forgetting.py`: `_compute_score` (136–180) = paper1 product-of-resistances (REPLACE); `_sweep_table` SELECT (99–105) needs factor columns added; `ForgettingEngine.__init__` (56–62).
- `borge/memory/retrieval.py`: `recall()` (61–114) 5-term weighted sum (REPLACE scoring); `MemoryRetrieval.__init__` (49–59).
- `borge/agent.py`: engines built ~104–116 (`_forgetting`, `_consolidation`, `_retrieval`); `_cfg(key, default)` helper; holds `self._self_model`, `self._value_system` (SOUL). `recall()` ~239–257.

## Test-replacement note

This branch **intentionally replaces** the self-FEP forget/retrieve formulas. Tests that pin the OLD product-of-resistances or the OLD 5-term recall (e.g. parts of `tests/test_*forgetting*`, self-FEP `experiments/e1*`) will break HERE — that is expected and correct (those live intact on `self-FEP`). Each task updates/replaces the tests it invalidates, asserting the NEW value-driven behavior. Do NOT weaken a test just to make it pass; rewrite it to assert paper2 intent.

---

## Task 1: Store persists the 4 factor columns

**Files:**
- Modify: `borge/memory/store.py` (`insert()` SQL 117–131, `_normalize()` 248–269)
- Test: `tests/test_store_factors.py` (create)

**Step 1: Write failing test** — insert a row with `goal_relevance=0.8, value_alignment=0.6, task_utility=0.3, reliability=0.7`, read back via `get(id)`, assert all four round-trip (not 0.0).

**Step 2: Run** `…/python -m pytest tests/test_store_factors.py -v` → FAIL (columns written as default 0).

**Step 3: Implement** — add the 4 columns to the INSERT column list + `VALUES` `:named` params, and to `_normalize()`'s returned dict (`float(entry.get(col, 0.0))`).

**Step 4: Run** → PASS. Also run `…/python -m pytest tests/test_store*.py tests/test_value*.py -q` → no regression.

**Step 5: Commit** `feat(store): persist goal/value_alignment/task_utility/reliability factor columns`.

---

## Task 2: SHIPPED_DEFAULT weights + config plumbing

**Files:**
- Modify: `borge/memory/value.py` (add `SHIPPED_DEFAULT` dict + a `default_memory_value()` factory)
- Test: `tests/test_value_default.py` (create)

**Step 1: Failing test** — `from borge.memory.value import SHIPPED_DEFAULT, default_memory_value`; assert `SHIPPED_DEFAULT` has all 7 `MemoryValue.FACTORS` keys; assert `default_memory_value().weights["reliability"] == SHIPPED_DEFAULT["reliability"]`; assert `default_memory_value({"reliability": 2.0}).weights["reliability"] == 2.0` (override merges over default).

**Step 2: Run** → FAIL (symbols absent).

**Step 3: Implement** — `SHIPPED_DEFAULT = {"emotion":0.55,"goal_relevance":0.0,"value_alignment":0.0,"self_relevance":0.23,"task_utility":0.0,"reliability":0.64,"usage":0.10}` (full-479 fit; usage small nonzero so frequent recall still resists forgetting). `default_memory_value(override: dict|None=None)` returns `MemoryValue(weights={**SHIPPED_DEFAULT, **(override or {})})`. Docstring cites the fit + `results/lme_blind_forgetting_full.json`.

**Step 4: Run** → PASS.

**Step 5: Commit** `feat(value): shipped learned-default weights + default_memory_value factory`.

---

## Task 3: Encode-time factor computation + persist (consolidation Step 3)

**Files:**
- Modify: `borge/memory/consolidation.py` (`_step3_emotional_significance` 220–332; `__init__` to accept `value_centroid`)
- Test: `tests/test_consolidation_factors.py` (create)

**Step 1: Failing test** — build a pipeline with a `SelfModel` (hash_embed) + a `value_centroid` (a fixed vector). Run `run(session_id, messages, emotional_history)` on 2 user + 1 assistant message. Read `store.by_session` → assert each row has nonzero `reliability` (user 0.7 / assistant 0.4), a `goal_relevance` in [0,1], a `value_alignment` in [0,1]; assert `task_utility == 0.0`.

**Step 2: Run** → FAIL.

**Step 3: Implement** in Step 3, per message (reuse the embedding already computed at line 288):
- `reliability = 0.7 if role == "user" else 0.4`
- session-topic centroid: before the per-message loop, embed all user-turn contents once → `sess_centroid` (mean); `goal_relevance = 0.5 + 0.5*cosine(embedding, sess_centroid)` (fallback 0.5 if no centroid)
- `value_alignment = 0.5 + 0.5*cosine(embedding, value_centroid)` if `value_centroid` else 0.0
- add `goal_relevance / value_alignment / task_utility=0.0 / reliability` to the `store.insert({...})` dict.
- `__init__(..., value_centroid: list[float] | None = None)`; store as `self.value_centroid`.

**Step 4: Run** → PASS; `…/python -m pytest tests/test_consolidation*.py -q`.

**Step 5: Commit** `feat(consolidation): compute+persist 6 live value factors at encode time`.

---

## Task 4: Encoding depth via value_encoding_depth

**Files:**
- Modify: `borge/memory/consolidation.py` (depth block 254–305), accept shared `memory_value`
- Test: extend `tests/test_consolidation_factors.py`

**Step 1: Failing test** — two crafted turns: one high-value (high emotion+self), one low; after `run`, assert `encoding_depth` of the high-value row > the low-value row, AND that depth equals `value_encoding_depth(factors, mv)` for the persisted factors.

**Step 2: Run** → FAIL (depth still significance-threshold based).

**Step 3: Implement** — replace the significance→depth ladder (254–259) and the self-bump (304–305) with `depth = value_encoding_depth(factors, self.memory_value)` where `factors` is the 6-factor dict just computed. `__init__(..., memory_value: MemoryValue | None = None)` → `self.memory_value = memory_value or default_memory_value()`.

**Step 4: Run** → PASS. Expect some self-FEP depth tests to break → rewrite them to assert value-driven depth (or move assertion to the new test); document in commit.

**Step 5: Commit** `feat(consolidation): encoding depth driven by MemoryValue (paper2)`.

---

## Task 5: Forget via value_forget_score + inject MemoryValue

**Files:**
- Modify: `borge/memory/forgetting.py` (`__init__` 56–62 add `memory_value`; `_sweep_table` SELECT 99–105; replace `_compute_score` 136–180 with a call to `value_forget_score`)
- Test: `tests/test_forgetting_value.py` (create); update/retire old `_compute_score` assertions

**Step 1: Failing test** — seed a temp DB `borge_memories` (via `MemoryStore.insert`) with a high-reliability+emotion row (old) and a low-everything row (old). Run `run_forgetting_pass`. Assert the low-value SHALLOW row is deleted and the high-value row survives — driven by `MemoryValue`, NOT recency (make both equally old).

**Step 2: Run** → FAIL.

**Step 3: Implement** — `ForgettingEngine.__init__(..., memory_value: MemoryValue | None = None)`; `self.memory_value = memory_value or default_memory_value()`. Extend `_sweep_table` SELECT to include `goal_relevance, value_alignment, task_utility, reliability` when those columns exist (probe like the existing `has_sr` check; build the dynamic select). Replace `_compute_score(row, now)` body with `return value_forget_score(dict(row), self.memory_value, now, beta=self._cfg_beta)` (beta default 8.0; `memory_factors` inside handles missing keys → graceful degrade for the Hermes `messages` table). Keep the depth-tiered prune/compress thresholds unchanged.

**Step 4: Run** → PASS; rewrite the old product-of-resistances test to assert value-driven ordering (or delete if fully superseded — note in commit).

**Step 5: Commit** `feat(forgetting): forget score driven by MemoryValue.value_forget_score (paper2)`.

---

## Task 6: Retrieval = V + query terms + inject MemoryValue

**Files:**
- Modify: `borge/memory/retrieval.py` (`__init__` add `memory_value`; `recall()` scoring 82–104)
- Test: `tests/test_retrieval_value.py` (create); update old 5-term recall tests

**Step 1: Failing test** — store two rows equally recent + equal query relevance, one high-V (reliability+self) one low-V; `recall(query=...)` → assert the high-V row ranks first, driven by V.

**Step 2: Run** → FAIL.

**Step 3: Implement** — `__init__(..., memory_value: MemoryValue | None = None)`. In `recall`, compute `v = self.memory_value.value(memory_factors(r))`; `score = w_v*v + w_rel*relevance + w_mood*mood_sim + w_rec*recency` with defaults `w_v=0.45, w_rel=0.25, w_mood=0.20, w_rec=0.10`. Drop the separate `self_weight`/`f_weight` terms (V subsumes them) and the importance multiplier. Keep `record_retrieval` loop. Keep param names back-compatible where cheap (deprecate unused weights with a comment).

**Step 4: Run** → PASS; update any test asserting the old 5-term sum.

**Step 5: Commit** `feat(retrieval): rank by MemoryValue + query relevance/mood/recency (paper2)`.

---

## Task 7: Wire the single shared MemoryValue in BorgeAgent

**Files:**
- Modify: `borge/agent.py` (engine construction ~104–116; add value-centroid computation)
- Test: `tests/test_agent_value_wiring.py` (create)

**Step 1: Failing test** — `BorgeAgent(None)`; assert `a._memory_value` is a `MemoryValue` with `SHIPPED_DEFAULT` weights; assert the SAME instance is referenced by `a._forgetting.memory_value`, `a._consolidation.memory_value`, `a._retrieval.memory_value` (identity check — "one value, three ops").

**Step 2: Run** → FAIL.

**Step 3: Implement** — in `__init__`: `self._memory_value = default_memory_value(self._cfg("memory.value.weights", None))`. Compute `value_centroid` once: if `self._value_system` has value descriptors and `self._self_model`, embed each → mean; else `None`. Pass `memory_value=self._memory_value` to all three engine constructors, and `value_centroid=...` to the pipeline. Keep the 4-hook contract + plugin exception-swallowing untouched.

**Step 4: Run** → PASS; `…/python -m pytest -q` (full suite); fix any remaining fallout.

**Step 5: Commit** `feat(agent): inject one shared MemoryValue into all three memory engines (paper2)`.

---

## Task 8: End-to-end "one value, three ops" verification (Q5)

**Files:**
- Test: `tests/test_paper2_runtime_e2e.py` (create)

**Step 1: Write the test** (this IS the deliverable — proves the paper claim in the live runtime, API-free, deterministic):
- Build `BorgeAgent(None)` with a temp DB + hash_embed SelfModel.
- Drive the encode path: craft a session of messages where turn H has a high-value profile (vivid emotion + self-referential user fact) and turn L is low-value chatter; call `on_session_end(session_id, messages)` (or the consolidation entry) with matching `emotional_history`.
- Assert (one MemoryValue drives all three):
  1. **encode:** persisted `encoding_depth[H] > encoding_depth[L]`.
  2. **forget:** after `run_forgetting_pass` with both aged equally, L is pruned / H survives.
  3. **retrieve:** `recall(query matching both)` ranks H above L.
- Assert all three used `a._memory_value` (identity).

**Step 2: Run** `…/python -m pytest tests/test_paper2_runtime_e2e.py -v` → PASS.

**Step 3:** Full suite `…/python -m pytest -q` green (with self-FEP-formula tests rewritten per Tasks 4–6).

**Step 4: Commit** `test(paper2): e2e proof that one MemoryValue drives encode+forget+retrieve`.

---

## Done criteria

- `…/python -m pytest -q` green on `multi-factor-eval`.
- `borge/memory/{forgetting,retrieval,consolidation}.py` route through `MemoryValue`; `grep value_forget_score|value_encoding_depth|memory_value borge/agent.py` non-empty.
- Smoke: `…/python -c "from borge.agent import BorgeAgent; a=BorgeAgent(None); print(a._memory_value.weights)"` prints shipped weights.
- NOT merged to `main`/`self-FEP`.
- A short `docs/` note or CLAUDE.md update: paper2 branch runs value-driven memory; paper1/self-FEP branch runs product-of-resistances.
