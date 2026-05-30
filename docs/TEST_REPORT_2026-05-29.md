# Test Report — BorgeAgent (paper2 runtime)

**Date:** 2026-05-29
**Branch:** `multi-factor-eval`  **Commit:** `21b36fa`  **Tree:** clean
**Interpreter:** `/Users/max_abel/opt/anaconda3/bin/python` (CPython 3.11)
**Runtime deps present:** torch 2.10.0, sentence-transformers (all-MiniLM-L6-v2), numpy, sqlite3
**Command:** `python -m pytest -p no:cacheprovider`

---

## Summary

| Metric | Result |
|--------|--------|
| Total tests | **84** |
| Passed | **84** |
| Failed | **0** |
| Errors | 0 |
| Test files | 18 |
| Wall-clock | ~20 s (single CPU, no GPU, no network) |
| Smoke (agent instantiation) | ✅ pass |

**Verdict: PASS.** All 84 tests green. The paper2 runtime claim ("one `MemoryValue` drives encode + forget + retrieve") is verified end-to-end. No API calls required.

---

## Suite breakdown

### A. paper2 runtime-wiring tests (the new work — 14 tests, all pass)

| File | Tests | Verifies |
|------|-------|----------|
| `test_store_factors.py` | 2 | store persists goal/value_alignment/task_utility/reliability columns (round-trip + default-0 back-compat) |
| `test_value_default.py` | 3 | `SHIPPED_DEFAULT` covers all 7 factors; `default_memory_value` factory + config override merge |
| `test_consolidation_factors.py` | 4 | encode-time 6-factor compute+persist; value_alignment→0 w/o centroid; goal fallback w/o self_model; **encoding depth = `value_encoding_depth`** |
| `test_forgetting_value.py` | 1 | **forget = `value_forget_score`**; equally-aged high-value survives, low-value pruned (value-driven, not recency) |
| `test_retrieval_value.py` | 2 | **rank = w_v·V + query terms**; high-V ranks first (V load-bearing — fails under w_v=0); query relevance still differentiates |
| `test_agent_value_wiring.py` | 1 | **one shared `MemoryValue` instance** in all three engines (identity) |
| `test_paper2_runtime_e2e.py` | 1 | **capstone**: one value drives encode+forget+retrieve on the agent's own wired engines |

### B. pre-existing cognitive-layer tests (regression — all pass)

`test_signal_extractor_rules.py` (14), `test_self_fep_memory.py` (7, retrieval/forget assertions rewritten to value-driven intent), `test_value_net.py` (6, MLP ablation), `test_pluggable_embedder.py` (6, SBert), `test_memory_value.py` (6), `test_emotion_memory_loop.py` (6, mood-recall rewritten), `test_value_unified.py` (5), `test_pi_self_variational.py` (4), `test_longmemeval_adapter.py` (3), `test_mu_self_at_encoding.py` (2, snapshot persistence retained; dropped-ranking-term tests removed), `test_factor_cache.py` (1).

---

## Runtime verification — "one value, three operations"

Deterministic, API-free. High-value turn H ("I love my project and my family deeply", V/A=0.9/0.9) vs low-value turn L ("ok the weather is fine", V/A=0.0/0.1), scored by the single shipped-weight `MemoryValue`:

| Operation | H | L | Driver |
|-----------|---|---|--------|
| **Value** `V(m)` | **1.004** | 0.536 | emotion 0.55 + self 0.23 + reliability 0.64 + usage 0.10 |
| **ENCODE** depth | **3** (SCHEMATIC) | 2 (SEMANTIC) | `value_encoding_depth(V)` |
| **FORGET** score (aged 45 d) | **1.59 → survives** | 2.71 → pruned | `value_forget_score`, prune τ=2.0 |
| **RETRIEVE** rank | **above L** | below | `w_v·V + relevance + mood + recency` |

All three operations consume the *same* `MemoryValue` instance (`a._forgetting.memory_value is a._consolidation.memory_value is a._retrieval.memory_value is a._memory_value` → True). Fresh (un-aged) rows score forget=0 (recency floor) — correct; the value separation drives forgetting only once memories age, as the e2e test exercises.

**Smoke:** `BorgeAgent(None)` instantiates; `pre_turn("hello", [])` returns a context-injection string; shipped weights live = `{emotion 0.55, goal_relevance 0.0, value_alignment 0.0, self_relevance 0.23, task_utility 0.0, reliability 0.64, usage 0.10}`.

---

## Notes / transparency

- **SBert test flake (non-blocking):** during one full-suite run `test_sbert_embedder_produces_384_dim_vector` failed once on a first-time model-load/resource contention; it passes 6/6 in isolation and on re-run. Not a code regression — a one-time model-download timing artifact.
- **Graceful degradation verified** (in `test_forgetting_value.py` + review): on a Hermes `messages` table lacking factor columns, forgetting runs without crash, degrading to emotion+usage factors (missing factors → 0 via `memory_factors.get`).
- **Honest scope:** under `SHIPPED_DEFAULT`, `goal_relevance`/`value_alignment`/`task_utility` carry weight 0 — live separation rests on emotion+self+reliability+usage (faithful to the paper's full-479 fit). The 6-factor machinery is present and re-weightable via config, not dead.
- **Not covered by this run (separate, by design):** the offline eval harness (`experiments/lme_*`) reproduces the paper numbers but requires the 41 MB factor cache (gitignored; regenerate via `build_factor_cache.py`). The paper's headline (full-479 blind: learned 0.770±0.011) lives in `results/lme_blind_forgetting_full.json`, validated in the earlier auto-review loop — out of scope for this code/runtime test report.

---

## Coverage gaps (acknowledged, non-blocking)

- `task_utility` factor has no live computation (LLM-gated; weight 0 by default) → no runtime test of a non-zero task_utility path.
- QA-accuracy is a 5-case Codex pilot, not a scaled test.
- `current_f_total` retrieval param is accepted-but-unused (back-compat); no test asserts its no-op.
