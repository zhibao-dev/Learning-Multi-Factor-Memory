# Paper2 Runtime Wiring — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Route the live `BorgeAgent` encode / forget / retrieve through one shared `MemoryValue` instance, so paper2's "one learned multi-factor value controls all three memory operations" is true in the runtime, not just the offline harness.

**Architecture:** One `MemoryValue` on `BorgeAgent._memory_value` (shipped default ⊕ config weights) is injected into `ForgettingEngine`, `MemoryRetrieval`, and the consolidation pipeline. Factors are computed at encode time (session context present), persisted to existing `borge_memories` columns, and read at forget/retrieve. Forget uses `value_forget_score`; encode tier uses `value_encoding_depth`; retrieval rank = `w_v·V + w_q·query_relevance + w_mood·mood`.

**Tech Stack:** Python 3.11, SQLite, existing `borge/memory/value.py` (`MemoryValue`, `memory_factors`, `value_forget_score`, `value_encoding_depth`), pytest.

---

## ⚠️ Branch & environment invariants (read before every task)

- **Branch `multi-factor-eval` ONLY.** `borge/memory/forgetting.py` and `borge/memory/retrieval.py` edits here DIVERGE from `self-FEP` and **must never be cross-merged to `main`/`self-FEP`** (those keep paper1's self-FEP product-of-resistances). Verify before starting: `git branch --show-current` → `multi-factor-eval`.
- **`python` is NOT on PATH.** Use `/Users/max_abel/opt/anaconda3/bin/python` for ALL python/pytest. Sanity: `/Users/max_abel/opt/anaconda3/bin/python -c "import torch; print(torch.__version__)"` → `2.10.0`.
- **Zero schema change.** `borge_memories` factor columns already exist (`goal_relevance`, `value_alignment`, `self_relevance_score`, `reliability`, `task_utility`). NEVER `ALTER` the Hermes `messages` table beyond the existing `BORGE_COLUMNS_SQL`.
- **Hooks unchanged.** The 4 lifecycle hooks (`on_session_start`, `pre_turn`, `post_tool`, `on_session_end`) keep their signatures. All new state funnels through `BorgeAgent`; nothing leaks into `runner.py`/`plugins/`. `BorgeAgent(agent_backend=None)` MUST stay valid (smoke test at the end).
- **Each task commits separately.** Use `/Users/max_abel/opt/anaconda3/bin/python -m pytest`.
- Design reference: `docs/plans/2026-05-28-paper2-runtime-wiring-design.md`.

---

## Task 1: `runtime_factors()` — encode-time factor derivation

Derives the 6 free-proxy factors from a memory + encode-time context. Distinct from existing `memory_factors(row)` (which only reads stored columns). `task_utility` stays 0.0.

**Files:**
- Modify: `borge/memory/value.py` (add function after `memory_factors`, ~line 109)
- Test: `tests/test_runtime_factors.py` (create)

**Step 1 — failing test:**
```python
# tests/test_runtime_factors.py
import math
from borge.memory.value import runtime_factors

def _vec(*xs): return list(xs)

def test_runtime_factors_full_context():
    # memory embedding identical to all anchors → similarity factors = 1.0
    e = _vec(1.0, 0.0)
    f = runtime_factors(
        embedding=e, valence=-0.8, arousal=0.9, retrieval_count=3,
        role="user", mu_self=e, soul_centroid=e, session_centroid=e,
    )
    assert math.isclose(f["emotion"], 0.8 * 0.9, rel_tol=1e-6)   # |V|·A
    assert math.isclose(f["self_relevance"], 1.0, abs_tol=1e-6)  # cos→[0,1], identical→1
    assert math.isclose(f["value_alignment"], 1.0, abs_tol=1e-6)
    assert math.isclose(f["goal_relevance"], 1.0, abs_tol=1e-6)
    assert f["reliability"] == 0.7           # user role
    assert math.isclose(f["usage"], 3/4, rel_tol=1e-6)
    assert f["task_utility"] == 0.0          # LLM future

def test_runtime_factors_missing_context_neutral():
    # no embedding / no anchors → similarity factors fall to neutral 0.5, never crash
    f = runtime_factors(
        embedding=None, valence=0.0, arousal=0.5, retrieval_count=0,
        role="assistant", mu_self=None, soul_centroid=None, session_centroid=None,
    )
    assert f["self_relevance"] == 0.5
    assert f["value_alignment"] == 0.5
    assert f["goal_relevance"] == 0.5
    assert f["reliability"] == 0.4           # assistant role
    assert f["usage"] == 0.0
    assert set(f) == {"emotion","goal_relevance","value_alignment",
                      "self_relevance","task_utility","reliability","usage"}
```

**Step 2 — run, expect fail:** `/Users/max_abel/opt/anaconda3/bin/python -m pytest tests/test_runtime_factors.py -v` → `ImportError: cannot import name 'runtime_factors'`.

**Step 3 — implement** (in `borge/memory/value.py`; reuse `cosine` from `..values.self_model`):
```python
from ..values.self_model import cosine  # add near top with other imports

def _sim01(a, b) -> float:
    """cosine mapped to [0,1]; neutral 0.5 when either side is missing."""
    if not a or not b:
        return 0.5
    return 0.5 + 0.5 * cosine(a, b)

def runtime_factors(
    *,
    embedding: list[float] | None,
    valence: float, arousal: float, retrieval_count: float,
    role: str,
    mu_self: list[float] | None,
    soul_centroid: list[float] | None,
    session_centroid: list[float] | None,
) -> dict[str, float]:
    """
    Derive the 6 free-proxy factors at ENCODE time from a memory + context.
    Missing context degrades to neutral (0.5 for similarities, 0 elsewhere);
    never raises. task_utility stays 0 (LLM future); the dict is 7-wide so it
    slots straight into MemoryValue.value / value_encoding_depth.
    """
    rc = float(retrieval_count or 0)
    return {
        "emotion":         abs(float(valence)) * float(arousal),
        "goal_relevance":  _sim01(embedding, session_centroid),
        "value_alignment": _sim01(embedding, soul_centroid),
        "self_relevance":  _sim01(embedding, mu_self),
        "task_utility":    0.0,
        "reliability":     0.7 if role == "user" else 0.4,
        "usage":           rc / (1.0 + rc),
    }
```

**Step 4 — run, expect pass.**

**Step 5 — commit:** `git add borge/memory/value.py tests/test_runtime_factors.py && git commit -m "feat(value): runtime_factors — encode-time 6-factor derivation with neutral degradation"`

---

## Task 2: SHIPPED_DEFAULT weights + `_load_value_weights`

**Files:**
- Modify: `borge/memory/value.py` (add `SHIPPED_DEFAULT_WEIGHTS` constant near `MemoryValue`)
- Modify: `borge/agent.py` (`__init__`, after `self._afe` ~line 100; add helper method)
- Test: `tests/test_value_weights_loading.py` (create)

**Step 1 — failing test:**
```python
# tests/test_value_weights_loading.py
from borge.memory.value import SHIPPED_DEFAULT_WEIGHTS, MemoryValue
from borge.agent import BorgeAgent

def test_shipped_default_matches_full479_fit():
    w = SHIPPED_DEFAULT_WEIGHTS
    assert w["reliability"] == 0.64
    assert w["emotion"] == 0.55
    assert w["self_relevance"] == 0.23
    assert w["goal_relevance"] == 0.0
    assert set(w) == set(MemoryValue.FACTORS)   # 7-wide

def test_agent_builds_memory_value_default():
    a = BorgeAgent(None)
    assert isinstance(a._memory_value, MemoryValue)
    assert a._memory_value.weights["reliability"] == 0.64

def test_config_overrides_default(tmp_path, monkeypatch):
    # config borge.memory.value.weights.emotion overrides the shipped default
    monkeypatch.setenv("BORGE_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "borge:\n  memory:\n    value:\n      weights:\n        emotion: 1.5\n")
    a = BorgeAgent(None)
    assert a._memory_value.weights["emotion"] == 1.5      # overridden
    assert a._memory_value.weights["reliability"] == 0.64 # default kept
```

**Step 2 — run, expect fail** (`ImportError` / `AttributeError: _memory_value`).

**Step 3 — implement.**
In `borge/memory/value.py` (after the `MemoryValue` class):
```python
# Out-of-box weights = the full-479 LongMemEval-S blind fit (paper2). The four
# factors that carried signal dominate; goal/value_alignment/task_utility ~0.
SHIPPED_DEFAULT_WEIGHTS = {
    "emotion":         0.55,
    "goal_relevance":  0.0,
    "value_alignment": 0.0,
    "self_relevance":  0.23,
    "task_utility":    0.0,
    "reliability":     0.64,
    "usage":           0.1,
}
```
In `borge/agent.py` — add a helper method and build the instance in `__init__` BEFORE the memory infrastructure block (~line 102, so it can be injected):
```python
from .memory.value import MemoryValue, SHIPPED_DEFAULT_WEIGHTS  # add to imports

def _load_value_weights(self) -> dict:
    """Shipped default ⊕ config borge.memory.value.weights (config wins per key)."""
    w = dict(SHIPPED_DEFAULT_WEIGHTS)
    override = self._cfg("memory.value.weights", {}) or {}
    if isinstance(override, dict):
        for k, v in override.items():
            if k in w:
                try:
                    w[k] = float(v)
                except (TypeError, ValueError):
                    pass
    return w
```
In `__init__`, just before `# ── Memory infrastructure`:
```python
self._memory_value = MemoryValue(weights=self._load_value_weights())
```

**Step 4 — run, expect pass.**

**Step 5 — commit:** `git add borge/memory/value.py borge/agent.py tests/test_value_weights_loading.py && git commit -m "feat(agent): shipped-default value weights + config override; build BorgeAgent._memory_value"`

---

## Task 3: Inject the shared `_memory_value` into the three engines

The hard "one value, three ops" proof = all three engines reference the SAME object.

**Files:**
- Modify: `borge/memory/forgetting.py` (`ForgettingEngine.__init__` — add `memory_value=None` param, store `self._mv`, add `beta`)
- Modify: `borge/memory/retrieval.py` (`MemoryRetrieval.__init__` — add `memory_value=None` param, store `self._mv`)
- Modify: `borge/memory/consolidation.py` (`MemoryConsolidationPipeline.__init__` — add `memory_value=None` param, store `self._mv`)
- Modify: `borge/agent.py` (pass `self._memory_value` into all three constructors, ~lines 104-120)
- Test: `tests/test_runtime_value_wiring.py` (create — shared-instance assertion only for this task)

**Step 1 — failing test:**
```python
# tests/test_runtime_value_wiring.py
from borge.agent import BorgeAgent

def test_three_engines_share_one_memory_value():
    a = BorgeAgent(None)
    mv = a._memory_value
    assert a._forgetting._mv is mv
    assert a._retrieval._mv is mv
    assert a._consolidation._mv is mv
```

**Step 2 — run, expect fail** (`AttributeError: _mv`).

**Step 3 — implement.** Add `memory_value=None` kwarg to each `__init__`, store `self._mv = memory_value`. In `borge/agent.py` pass `memory_value=self._memory_value` into `ForgettingEngine(...)`, `MemoryConsolidationPipeline(...)`, `MemoryRetrieval(...)`. Keep all params optional/defaulted so `BorgeAgent(None)` and any direct constructions in tests still work.

**Step 4 — run, expect pass.**

**Step 5 — commit:** `git add borge/memory/forgetting.py borge/memory/retrieval.py borge/memory/consolidation.py borge/agent.py tests/test_runtime_value_wiring.py && git commit -m "feat(memory): inject shared MemoryValue into forgetting/retrieval/consolidation"`

---

## Task 4: Forget via `value_forget_score` (+ graceful degradation)

Replace the self-FEP product in `_compute_score` with the value-driven score, reading whatever factor columns exist.

**Files:**
- Modify: `borge/memory/forgetting.py` (`_sweep_table` SELECT widening + `_compute_score` → use `self._mv`)
- Test: `tests/test_runtime_value_wiring.py` (add forget-golden + degradation tests)

**Step 1 — failing tests** (append):
```python
import sqlite3, os
from datetime import datetime, timedelta
from borge.memory.forgetting import ForgettingEngine
from borge.memory.value import MemoryValue, SHIPPED_DEFAULT_WEIGHTS

def _mk_borge_db(path):
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE borge_memories(
        id TEXT PRIMARY KEY, timestamp TEXT, last_retrieved TEXT,
        retrieval_count INTEGER DEFAULT 0, importance_score REAL DEFAULT 0.5,
        encoding_depth INTEGER DEFAULT 1, content TEXT,
        emotional_valence REAL DEFAULT 0.0, emotional_arousal REAL DEFAULT 0.5,
        self_relevance_score REAL DEFAULT 0.0, goal_relevance REAL DEFAULT 0.0,
        value_alignment REAL DEFAULT 0.0, reliability REAL DEFAULT 0.0,
        task_utility REAL DEFAULT 0.0, forget_score REAL DEFAULT 0.0)""")
    return conn

def test_forget_keeps_high_value_prunes_low_value(tmp_path):
    db = str(tmp_path / "b.db"); conn = _mk_borge_db(db)
    old = (datetime.now() - timedelta(days=30)).isoformat()
    # both old + shallow; HIGH reliability/self (high V) vs LOW (low V)
    conn.execute("INSERT INTO borge_memories VALUES('hi',?,?,0,0.5,1,'x',0.1,0.5,0.9,0,0,0.9,0,0)", (old, old))
    conn.execute("INSERT INTO borge_memories VALUES('lo',?,?,0,0.5,1,'y',0.1,0.5,0.0,0,0,0.0,0,0)", (old, old))
    conn.commit(); conn.close()
    eng = ForgettingEngine(memory_value=MemoryValue(weights=SHIPPED_DEFAULT_WEIGHTS))
    eng.run_forgetting_pass(db)
    conn = sqlite3.connect(db)
    ids = {r[0] for r in conn.execute("SELECT id FROM borge_memories")}
    assert "hi" in ids and "lo" not in ids   # high-V survived, low-V pruned

def test_forget_degrades_without_factor_columns(tmp_path):
    # a messages-like table lacking factor columns must not crash; emotion+usage drive it
    db = str(tmp_path / "m.db"); conn = sqlite3.connect(db)
    conn.execute("""CREATE TABLE messages(
        id TEXT PRIMARY KEY, role TEXT, timestamp TEXT, last_retrieved TEXT,
        retrieval_count INTEGER DEFAULT 0, importance_score REAL DEFAULT 0.5,
        encoding_depth INTEGER DEFAULT 1, content TEXT,
        emotional_valence REAL DEFAULT 0.0, emotional_arousal REAL DEFAULT 0.5,
        forget_score REAL DEFAULT 0.0)""")
    old = (datetime.now() - timedelta(days=30)).isoformat()
    conn.execute("INSERT INTO messages VALUES('m1','user',?,?,0,0.5,1,'x',0.9,0.9,0)", (old, old))
    conn.commit(); conn.close()
    eng = ForgettingEngine(memory_value=MemoryValue(weights=SHIPPED_DEFAULT_WEIGHTS))
    stats = eng.run_forgetting_pass(db)        # must not raise
    assert isinstance(stats, dict)
```

**Step 2 — run, expect fail** (low-V not pruned / `_mv` not used yet).

**Step 3 — implement.**
- In `_sweep_table`, widen the column probe + SELECT to pull factor columns when present:
```python
cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
factor_cols = [c for c in ("self_relevance_score","goal_relevance",
                           "value_alignment","reliability","task_utility")
               if c in cols]
extra = ("," + ",".join(factor_cols)) if factor_cols else ""
select_sql = f"""SELECT id, timestamp, last_retrieved, retrieval_count,
                        importance_score, encoding_depth, content,
                        emotional_valence, emotional_arousal{extra}
                 FROM {table} {where_clause}"""
```
- Replace the `self._compute_score(row, now)` call with the value path:
```python
score = self._forget_score(dict(row), now)
```
- Replace `_compute_score` static method with an instance method that delegates to `value_forget_score` when a `MemoryValue` is present, else falls back to the existing self-FEP formula (so a `ForgettingEngine()` built with no `memory_value` is unchanged — back-compat for any direct callers):
```python
def _forget_score(self, row: dict, now: datetime) -> float:
    if self._mv is not None:
        from .value import value_forget_score
        return value_forget_score(row, self._mv, now, beta=self._beta)
    return self._compute_score_self_fep(row, now)   # rename of old _compute_score
```
- Add `beta` to `__init__` (`self._beta = beta`, default e.g. 8.0 — same as the experiment harness) and keep the old formula as `_compute_score_self_fep` (NOTE in a comment: retained only for the no-`mv` fallback; the paper1 self-FEP behavior lives on the `self-FEP` branch).
- `dict(row)` works on `sqlite3.Row`; `memory_factors`/`value_forget_score` use `.get`, so missing factor keys → 0 (degradation). Confirm `value_forget_score` reads `last_retrieved`/`timestamp` (it does).

**Step 4 — run, expect pass.**

**Step 5 — commit:** `git add borge/memory/forgetting.py tests/test_runtime_value_wiring.py && git commit -m "feat(forget): value_forget_score drives live forgetting; degrades w/o factor columns"`

---

## Task 5: Encode — persist factors + value-driven encoding depth

At consolidation persist time, compute `runtime_factors` from session context and store them + set `encoding_depth = value_encoding_depth`.

**Files:**
- Modify: `borge/memory/consolidation.py` (the persist-to-`borge_memories` step — locate where each message row is written / `encoding_depth` set)
- Test: `tests/test_runtime_value_wiring.py` (add encode test)

**Step 1 — failing test:** drive a consolidation persist on a tiny borge db and assert (a) a high-free-factor message gets a deeper `encoding_depth` than a low one, and (b) factor columns are populated (non-default). *Implementer: read `consolidation.py` to find the exact persist entrypoint and the available per-message context (μ_self via `self._self_model`, session centroid from the session's user-turn embeddings, SOUL centroid from `self_model` seed). Build the test against that real entrypoint.*

**Step 2 — run, expect fail.**

**Step 3 — implement.** In the persist step, for each message:
```python
from .value import runtime_factors, value_encoding_depth
factors = runtime_factors(
    embedding=<msg embedding or None>,
    valence=<v>, arousal=<a>, retrieval_count=0,
    role=<role>,
    mu_self=<self._self_model.mu_self or None>,
    soul_centroid=<self._self_model soul/seed centroid or None>,
    session_centroid=<centroid of this session's user-turn embeddings or None>,
)
depth = value_encoding_depth(factors, self._mv) if self._mv else <existing depth>
# write factors into goal_relevance/value_alignment/self_relevance_score/reliability
# columns + encoding_depth=depth
```
Keep the existing depth logic as the fallback when `self._mv is None`. Use the embedder already available to the pipeline / self_model; if no embedding is obtainable, `runtime_factors` degrades to neutral (Task 1 guarantees no crash).

**Step 4 — run, expect pass.**

**Step 5 — commit:** `git add borge/memory/consolidation.py tests/test_runtime_value_wiring.py && git commit -m "feat(encode): persist runtime factors + value_encoding_depth at consolidation"`

---

## Task 6: Retrieve — rank = w_v·V + w_q·query_relevance + w_mood·mood

**Files:**
- Modify: `borge/memory/retrieval.py` (`recall` ranking — fold a `V(stored factors)` term in; keep a query-relevance term and mood)
- Test: `tests/test_runtime_value_wiring.py` (add retrieve-golden tests)

**Step 1 — failing tests:** on a small borge db with stored factor columns: (a) on a NEUTRAL/empty query, the high-V row ranks above the low-V row; (b) on a query whose tokens match the low-V row's content, that row's rank jumps above where it sat on the neutral query (proves the query term is live, not swamped by V). *Implementer: match the real `recall` return shape (list of dicts) and the existing weight kwargs.*

**Step 2 — run, expect fail.**

**Step 3 — implement.** In `recall`, compute `v = self._mv.value(memory_factors(row))` per candidate (when `self._mv` present) and form:
```
rank = w_v * v + w_q * query_relevance + w_mood * mood_congruence
```
Add `value_weight` (w_v) kwarg; keep `relevance_weight` (w_q) and `mood_weight`. Subsume the old importance/self/recency terms into V (V already carries self + usage; recency stays as a small standalone or via the existing recency term — implementer's call, but document it). When `self._mv is None`, keep the current 5-term formula (back-compat). Defaults: `value_weight=0.45, relevance_weight=0.35, mood_weight=0.20` (tune so the golden tests pass; record final defaults in the commit message).

**Step 4 — run, expect pass.**

**Step 5 — commit:** `git add borge/memory/retrieval.py tests/test_runtime_value_wiring.py && git commit -m "feat(retrieve): rank = V + query_relevance + mood (one value drives retrieval too)"`

---

## Task 7: Harness-consistency test — runtime == offline ordering

Ties the live forget path to the paper's evidence: the same factor vectors, ranked by the live `ForgettingEngine`, produce the SAME order as the offline `experiments/lme_blind_forgetting` value ranking.

**Files:**
- Test: `tests/test_runtime_harness_consistency.py` (create)

**Step 1 — write the test:**
```python
# tests/test_runtime_harness_consistency.py
import sqlite3
from datetime import datetime, timedelta
from borge.memory.value import MemoryValue, value_forget_score, memory_factors

def test_live_forget_order_matches_offline_value_order():
    """The live forget engine's keep-order over a fixed factor set equals the
    offline harness's V-descending order — runtime IS the benchmarked thing."""
    mv = MemoryValue(weights={"emotion":0.55,"goal_relevance":0.0,
        "self_relevance":0.23,"reliability":0.64,"value_alignment":0.0,
        "task_utility":0.0,"usage":0.1})
    rows = [
        {"id":"a","emotional_valence":0.1,"emotional_arousal":0.5,
         "self_relevance_score":0.9,"reliability":0.9,"goal_relevance":0.0,
         "value_alignment":0.0,"task_utility":0.0,"retrieval_count":0,
         "timestamp":(datetime.now()-timedelta(days=10)).isoformat()},
        {"id":"b","emotional_valence":0.1,"emotional_arousal":0.5,
         "self_relevance_score":0.1,"reliability":0.1,"goal_relevance":0.0,
         "value_alignment":0.0,"task_utility":0.0,"retrieval_count":0,
         "timestamp":(datetime.now()-timedelta(days=10)).isoformat()},
    ]
    # offline: rank by V descending (what lme_blind_forgetting does → keep high V)
    offline_keep_order = [r["id"] for r in sorted(rows, key=lambda r:-mv.value(memory_factors(r)))]
    # live: lower forget_score = more likely kept → ascending forget == descending V
    live_keep_order = [r["id"] for r in sorted(rows, key=lambda r: value_forget_score(r, mv, beta=8.0))]
    assert live_keep_order == offline_keep_order == ["a","b"]
```

**Step 2 — run, expect pass** (both functions exist by now; this is a tie-down/regression test, not driving new code). If it fails, the forget direction is inverted — fix the sign, not the test.

**Step 3 — commit:** `git add tests/test_runtime_harness_consistency.py && git commit -m "test: live forget ordering matches offline lme_blind_forgetting value ranking"`

---

## Task 8: Full-suite + smoke + invariant check

**Step 1 — full suite:** `/Users/max_abel/opt/anaconda3/bin/python -m pytest -q` → all green (existing 72 + new). Fix any regression (esp. tests that construct `ForgettingEngine()`/`MemoryRetrieval()`/`MemoryConsolidationPipeline()` directly — the new kwargs are optional so they should still pass; if any asserted the old self-FEP forget score, that's expected divergence on this branch — update the test and note it).

**Step 2 — smoke (no API key, no torch needed):** `/Users/max_abel/opt/anaconda3/bin/python -c "from borge.agent import BorgeAgent; a=BorgeAgent(None); print(a.pre_turn('hello', [])); print('mv', a._memory_value.weights['reliability'])"` → prints a context string (or empty) and `mv 0.64`; no exception. Confirms `BorgeAgent(agent_backend=None)` still valid.

**Step 3 — branch invariant:** `git branch --show-current` → `multi-factor-eval`. Confirm `forgetting.py`/`retrieval.py` changes are committed only here:
`git log --oneline self-FEP..multi-factor-eval -- borge/memory/forgetting.py borge/memory/retrieval.py` should list the new commits (proving they are NOT on self-FEP). **Do not merge this branch into `main`/`self-FEP`.**

**Step 4 — final commit (if any test fixups):** `git add -A && git commit -m "test: align suite with value-driven runtime on multi-factor-eval"`

---

## Done criteria

- One `MemoryValue` on `BorgeAgent`, shared by forgetting + retrieval + consolidation (Task 3 assertion green).
- Live forget = `value_forget_score`; encode tier = `value_encoding_depth`; retrieve = V + query + mood.
- Factors computed at encode, persisted to `borge_memories`; Hermes `messages` degrades to emotion+usage, no schema change.
- Harness-consistency test green (runtime order == offline order).
- Full suite green; `BorgeAgent(None)` smoke passes; 4 hooks unchanged.
- All commits on `multi-factor-eval`; never merged to `main`/`self-FEP`.
