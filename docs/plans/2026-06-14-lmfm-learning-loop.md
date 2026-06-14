# LMFM Learning-Loop MVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a clean, `borge`-free `lmfm` package that lets a customer extract memory factors locally (privacy-preserving), upload only a numeric matrix to a cloud `/learn` endpoint, and get back learned weights they load into the value model.

**Architecture:** Open-core. The open-source `lmfm` package does local factor extraction + scoring (no cloud, no text leaves the machine). A paid cloud service runs the gradient-free weight learner over uploaded *numeric factor matrices* (never raw text) behind an API key. The client round-trips the returned `weights.json` back into `default_memory_value(override=...)`.

**Tech Stack:** Python 3.11+, pytest (TDD), sentence-transformers (local SBert, lazy), FastAPI + uvicorn (cloud `/learn`), stdlib `urllib` (client upload), no `borge` naming anywhere.

**Naming rule (HARD):** No `borge` string in any package, folder, file, class, function, CLI, or import — not even in comments/docstrings carried over from ported code. Every ported file must be scrubbed.

**Source-of-truth for ported logic** (research repo, to be re-implemented + renamed, NOT imported):
- `borge/memory/value.py` → value function, `learn_weights`, factors
- `borge/affective/signal_extractor.py` → `EmotionalSignalExtractor`
- `borge/values/self_model.py` → `SBertEmbedder`, `cosine`
- `borge/audit/ingest_md.py` → markdown loader, `MemoryRecord`
- `borge/eval/annotate.py` → per-turn factor annotation
- `experiments/lme_real_retention.py::gold_retention` → learning objective

---

## Naming Map (apply everywhere)

| Old (research repo) | New (`lmfm`) |
|---|---|
| `borge/memory/value.py` | `lmfm/value.py` |
| `MemoryValue`, `memory_factors`, `default_memory_value`, `learn_weights`, `value_forget_score`, `value_encoding_depth` | same names (no `borge` substring) — keep |
| `SHIPPED_DEFAULT` | `DEFAULT_WEIGHTS` |
| `borge/affective/signal_extractor.py` | `lmfm/factors/emotion.py` |
| `EmotionalSignalExtractor` | `EmotionSignalExtractor` |
| `borge/values/self_model.py` (SBert part only) | `lmfm/factors/embedder.py` |
| `SBertEmbedder`, `cosine` | same |
| `borge/audit/ingest_md.py` | `lmfm/io/markdown.py` |
| `MemoryRecord`, `load_markdown_dump` | `MemoryRecord`, `load_markdown` |
| `borge/eval/annotate.py` | `lmfm/factors/annotate.py` |
| `annotate_case` | `annotate_memories` |
| `gold_retention` (experiments) | `lmfm/learn/objective.py::gold_retention` |
| install extra `borge-agent[sbert]` | `lmfm[sbert]` |

---

## Phase 0 — Clean repo + package skeleton

### Task 0.1: Create the package skeleton

**Files:**
- Create: `lmfm/__init__.py`
- Create: `pyproject.toml`
- Create: `tests/__init__.py` (empty)
- Create: `README.md` (one-line stub; real README is a later task)

**Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "lmfm"
version = "0.1.0"
description = "Learning Multi-Factor Memory — a cognitively grounded value model for agentic memory"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
dependencies = []

[project.optional-dependencies]
sbert = ["sentence-transformers>=3.0"]
cloud = ["fastapi>=0.110", "uvicorn>=0.29"]
dev = ["pytest>=8.0", "sentence-transformers>=3.0", "fastapi>=0.110", "httpx>=0.27"]

[project.scripts]
lmfm = "lmfm.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

**Step 2: Write `lmfm/__init__.py`**

```python
"""Learning Multi-Factor Memory (lmfm).

Local, privacy-preserving multi-factor memory value model. Factor
extraction and scoring run entirely on the user's machine; only numeric
factor matrices (never raw text) are ever sent to the optional cloud
learner.
"""

__version__ = "0.1.0"
```

**Step 3: Commit**

```bash
git add pyproject.toml lmfm/__init__.py tests/__init__.py README.md
git commit -m "chore: lmfm package skeleton"
```

---

## Phase 1 — Core value model (ported, scrubbed)

### Task 1.1: Port the value function with a failing test first

**Files:**
- Test: `tests/test_value.py`
- Create: `lmfm/value.py`

**Step 1: Write the failing test**

```python
# tests/test_value.py
from lmfm.value import MemoryValue, memory_factors, default_memory_value


def test_default_weights_match_paper():
    mv = default_memory_value()
    assert round(mv.weights["reliability"], 2) == 0.64
    assert round(mv.weights["emotion"], 2) == 0.55
    assert round(mv.weights["self_relevance"], 2) == 0.23
    assert mv.weights["goal_relevance"] == 0.0


def test_value_is_weighted_sum():
    mv = MemoryValue(weights={"emotion": 1.0, "reliability": 2.0})
    v = mv.value({"emotion": 0.5, "reliability": 0.25})
    assert v == 1.0  # 1.0*0.5 + 2.0*0.25


def test_memory_factors_extracts_emotion_as_absV_times_A():
    row = {"emotional_valence": -0.4, "emotional_arousal": 0.5,
           "reliability": 1.0, "retrieval_count": 0}
    f = memory_factors(row)
    assert round(f["emotion"], 3) == 0.2   # |−0.4|·0.5
    assert f["reliability"] == 1.0
    assert f["usage"] == 0.0


def test_override_merges_over_defaults():
    mv = default_memory_value({"goal_relevance": 0.9})
    assert mv.weights["goal_relevance"] == 0.9
    assert round(mv.weights["reliability"], 2) == 0.64
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_value.py -v`
Expected: FAIL with "No module named 'lmfm.value'"

**Step 3: Write `lmfm/value.py`**

Port `borge/memory/value.py` verbatim EXCEPT:
- Rename `SHIPPED_DEFAULT` → `DEFAULT_WEIGHTS`.
- Scrub the module docstring of any `borge`/`self-FEP`/`FEP` wording; replace with a neutral description of the multi-factor value.
- Keep class `MemoryValue`, functions `memory_factors`, `value_forget_score`, `value_encoding_depth`, `learn_weights`, `default_memory_value`.
- `default_memory_value` reads `DEFAULT_WEIGHTS`.

Keep `DEFAULT_WEIGHTS` exactly:

```python
DEFAULT_WEIGHTS = {
    "emotion":         0.55,
    "goal_relevance":  0.00,
    "value_alignment": 0.00,
    "self_relevance":  0.23,
    "task_utility":    0.00,
    "reliability":     0.64,
    "usage":           0.10,
}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_value.py -v`
Expected: PASS (4 passed)

**Step 5: Grep for forbidden naming**

Run: `! grep -rin "borge\|self-fep\|\bfep\b" lmfm/value.py`
Expected: no matches (command exits 0)

**Step 6: Commit**

```bash
git add lmfm/value.py tests/test_value.py
git commit -m "feat: core multi-factor value model (scrubbed)"
```

### Task 1.2: Port the learner with a test

**Files:**
- Test: `tests/test_learn_weights.py`
- Modify: `lmfm/value.py` (already contains `learn_weights` from 1.1; this task only tests it)

**Step 1: Write the failing test**

```python
# tests/test_learn_weights.py
from lmfm.value import learn_weights


def test_learner_improves_a_simple_objective():
    # Objective maximised by putting all mass on 'a'.
    def obj(w):
        return w["a"] - w["b"]
    best, hist = learn_weights(obj, ("a", "b"), seed=1, iters=80)
    assert best["a"] >= best["b"]
    # best-so-far return is monotone non-decreasing
    rets = [h["best_return"] for h in hist]
    assert rets == sorted(rets)
```

**Step 2: Run** `pytest tests/test_learn_weights.py -v` → Expected: PASS (already implemented in 1.1).

**Step 3: Commit**

```bash
git add tests/test_learn_weights.py
git commit -m "test: weight learner monotonicity + direction"
```

---

## Phase 2 — Factor extractors (ported, scrubbed)

### Task 2.1: Port the SBert embedder

**Files:**
- Test: `tests/test_embedder.py`
- Create: `lmfm/factors/__init__.py` (empty)
- Create: `lmfm/factors/embedder.py`

**Step 1: Write the failing test** (skips if sentence-transformers absent)

```python
# tests/test_embedder.py
import pytest
from lmfm.factors.embedder import cosine


def test_cosine_basic():
    assert round(cosine([1, 0], [1, 0]), 5) == 1.0
    assert round(cosine([1, 0], [0, 1]), 5) == 0.0


def test_embedder_importable_without_st():
    # constructing must NOT import sentence-transformers (lazy)
    from lmfm.factors.embedder import SBertEmbedder
    SBertEmbedder()  # no exception at construction time
```

**Step 2: Run** `pytest tests/test_embedder.py -v` → Expected: FAIL "No module named 'lmfm.factors.embedder'"

**Step 3: Port `borge/values/self_model.py`** into `lmfm/factors/embedder.py`, keeping ONLY `SBertEmbedder`, `cosine`, `DEFAULT_DIM`, `hash_embed` (drop `SelfModel`, `has_self_reference`, `SelfModel`-specific code). Scrub the docstring: change the install hint from `borge-agent[sbert]` to `lmfm[sbert]`, remove the `SelfModel`/`borge` example.

**Step 4: Run** `pytest tests/test_embedder.py -v` → Expected: PASS

**Step 5: Grep** `! grep -rin "borge" lmfm/factors/embedder.py` → Expected: clean

**Step 6: Commit**

```bash
git add lmfm/factors/embedder.py lmfm/factors/__init__.py tests/test_embedder.py
git commit -m "feat: local SBert embedder (lazy, scrubbed)"
```

### Task 2.2: Port the emotion signal extractor

**Files:**
- Test: `tests/test_emotion.py`
- Create: `lmfm/factors/emotion.py`

**Step 1: Write the failing test**

```python
# tests/test_emotion.py
from lmfm.factors.emotion import EmotionSignalExtractor


def test_extract_returns_capped_deltas():
    ex = EmotionSignalExtractor()
    dv, da = ex.extract("This is absolutely terrible and broken!", [])
    assert -0.40 <= dv <= 0.40
    assert -0.30 <= da <= 0.30


def test_neutral_text_low_signal():
    ex = EmotionSignalExtractor()
    dv, da = ex.extract("The file is at path x.", [])
    assert abs(dv) < 0.2
```

**Step 2: Run** → Expected: FAIL.

**Step 3: Port `borge/affective/signal_extractor.py`** → `lmfm/factors/emotion.py`, rename class `EmotionalSignalExtractor` → `EmotionSignalExtractor`. Scrub docstrings of `borge`. Keep all rules/logic identical.

**Step 4: Run** → Expected: PASS.

**Step 5: Grep** `! grep -rin "borge" lmfm/factors/emotion.py` → clean.

**Step 6: Commit**

```bash
git add lmfm/factors/emotion.py tests/test_emotion.py
git commit -m "feat: emotion signal extractor (renamed, scrubbed)"
```

### Task 2.3: Port the markdown loader

**Files:**
- Test: `tests/test_markdown.py`
- Create: `lmfm/io/__init__.py` (empty)
- Create: `lmfm/io/markdown.py`

**Step 1: Write the failing test**

```python
# tests/test_markdown.py
from pathlib import Path
from lmfm.io.markdown import load_markdown, MemoryRecord


def test_heading_split(tmp_path: Path):
    p = tmp_path / "mem.md"
    p.write_text("## Allergy\nUser is allergic to penicillin.\n\n## Pref\nLikes dark mode.\n")
    recs = load_markdown(p, split="heading")
    assert all(isinstance(r, MemoryRecord) for r in recs)
    texts = " ".join(r.text for r in recs)
    assert "penicillin" in texts and "dark mode" in texts
    assert len(recs) == 2
```

**Step 2: Run** → Expected: FAIL.

**Step 3: Port `borge/audit/ingest_md.py`** → `lmfm/io/markdown.py`. Rename public `load_markdown_dump` → `load_markdown`. Define `MemoryRecord` locally in this module (port the dataclass from `borge/audit/ingest.py`; fields: `id, text, timestamp, role="user", metadata=None`). Remove the `from borge.audit.ingest import MemoryRecord` import. Scrub docstrings of `borge`/agent-product references.

**Step 4: Run** → Expected: PASS.

**Step 5: Grep** `! grep -rin "borge" lmfm/io/markdown.py` → clean.

**Step 6: Commit**

```bash
git add lmfm/io/markdown.py lmfm/io/__init__.py tests/test_markdown.py
git commit -m "feat: markdown memory loader (renamed, scrubbed)"
```

---

## Phase 3 — Local factor annotation (the privacy foundation)

### Task 3.1: Annotate MemoryRecords → factor vectors + gold flags

This is the core privacy step: turn raw memories into a numeric matrix locally.

**Files:**
- Test: `tests/test_annotate.py`
- Create: `lmfm/factors/annotate.py`

**Step 1: Write the failing test** (uses a fake embedder, no SBert download)

```python
# tests/test_annotate.py
from lmfm.io.markdown import MemoryRecord
from lmfm.factors.annotate import annotate_memories


class FakeEmbedder:
    # deterministic 2-d embedding by text length parity
    def __call__(self, text):
        return [1.0, 0.0] if len(text) % 2 == 0 else [0.0, 1.0]


def test_annotate_emits_seven_factor_dict_and_gold():
    recs = [
        MemoryRecord(id="m1", text="User is allergic to penicillin.",
                     timestamp="2026-01-10T09:00:00Z", role="user"),
        MemoryRecord(id="m2", text="ok", timestamp="2026-01-10T09:00:01Z",
                     role="assistant"),
    ]
    ann = annotate_memories(recs, embedder=FakeEmbedder(), gold_ids={"m1"})
    assert len(ann) == 2
    f0 = ann[0]["factors"]
    assert set(f0) == {"emotion", "goal_relevance", "value_alignment",
                       "self_relevance", "task_utility", "reliability", "usage"}
    assert ann[0]["gold"] is True
    assert ann[1]["gold"] is False
    # reliability prior: user > assistant
    assert ann[0]["factors"]["reliability"] > ann[1]["factors"]["reliability"]
    # all factors in [0,1]
    for a in ann:
        for v in a["factors"].values():
            assert 0.0 <= v <= 1.0
```

**Step 2: Run** → Expected: FAIL.

**Step 3: Write `lmfm/factors/annotate.py`**

```python
"""Annotate memories into numeric factor vectors + gold flags, locally.

This is the privacy boundary: raw text is reduced to seven scalar
factors here, on the user's machine. Only the resulting numbers (never
the text or embeddings) need ever leave for cloud weight learning.

Goal relevance uses a query-agnostic anchor (the centroid of user-turn
embeddings, "what this memory set is about"), matching the blind
consolidation regime.
"""

from __future__ import annotations

from .emotion import EmotionSignalExtractor
from .embedder import cosine

_EXTRACTOR = EmotionSignalExtractor()


def _sim01(a, b) -> float:
    if not a or not b:
        return 0.5
    return 0.5 * (1.0 + cosine(a, b))


def annotate_memories(records, *, embedder, gold_ids=None) -> list[dict]:
    """Return one dict per record: {factors: {7}, gold: bool, id, ts}.

    `gold_ids` marks which record ids are the must-retain "gold" set
    (the learning target). `embedder(text) -> list[float]`.
    """
    gold_ids = set(gold_ids or [])
    texts = [r.text for r in records]
    embs = [embedder(t) for t in texts] if texts else []

    user_embs = [e for e, r in zip(embs, records) if r.role == "user"]
    if user_embs:
        dim = len(user_embs[0])
        mu = [sum(e[i] for e in user_embs) / len(user_embs) for i in range(dim)]
    else:
        mu = []

    out = []
    for r, emb in zip(records, embs):
        dv, da = _EXTRACTOR.extract(r.text, [])
        emotion = abs(dv) * (0.5 + da)
        self_rel = _sim01(emb, mu) if mu else 0.5
        reliability = 0.7 if r.role == "user" else 0.4
        rcount = float((r.metadata or {}).get("retrieval_count", 0) or 0)
        out.append({
            "id": r.id,
            "ts": r.timestamp,
            "gold": r.id in gold_ids,
            "factors": {
                "emotion":         max(0.0, min(1.0, emotion)),
                "goal_relevance":  _sim01(emb, mu) if mu else 0.5,
                "value_alignment": 0.0,
                "self_relevance":  self_rel,
                "task_utility":    0.0,
                "reliability":     reliability,
                "usage":           rcount / (1.0 + rcount),
            },
        })
    return out
```

**Step 4: Run** → Expected: PASS.

**Step 5: Grep** `! grep -rin "borge" lmfm/factors/annotate.py` → clean.

**Step 6: Commit**

```bash
git add lmfm/factors/annotate.py tests/test_annotate.py
git commit -m "feat: local factor annotation (privacy boundary)"
```

---

## Phase 4 — `lmfm export-factors` CLI (client side, open source)

### Task 4.1: Factor-matrix export format

**Files:**
- Test: `tests/test_export.py`
- Create: `lmfm/export.py`

**Step 1: Write the failing test**

```python
# tests/test_export.py
import json
from lmfm.export import build_matrix


def test_build_matrix_has_no_text():
    annotated = [
        {"id": "m1", "ts": "t", "gold": True,
         "factors": {"emotion": 0.2, "goal_relevance": 0.5, "value_alignment": 0.0,
                     "self_relevance": 0.8, "task_utility": 0.0, "reliability": 0.7,
                     "usage": 0.0}},
    ]
    mat = build_matrix([annotated], keep_frac=0.3)
    blob = json.dumps(mat)
    # privacy: no raw text fields leak
    assert "text" not in blob
    assert mat["keep_frac"] == 0.3
    assert mat["cases"][0]["memories"][0]["gold"] is True
    assert len(mat["cases"][0]["memories"][0]["factors"]) == 7
```

**Step 2: Run** → Expected: FAIL.

**Step 3: Write `lmfm/export.py`**

```python
"""Build the numeric upload payload from locally-annotated memories.

A "case" is one annotated memory set. The payload contains ONLY factor
scalars + gold flags + a keep fraction — no text, no embeddings, no ids
of the original content beyond an opaque local id.
"""

from __future__ import annotations

FACTOR_ORDER = ("emotion", "goal_relevance", "value_alignment",
                "self_relevance", "task_utility", "reliability", "usage")


def build_matrix(cases, *, keep_frac: float) -> dict:
    """`cases` = list of annotated-memory lists (from annotate_memories)."""
    return {
        "schema": "lmfm.factors.v1",
        "keep_frac": keep_frac,
        "factor_order": list(FACTOR_ORDER),
        "cases": [
            {"memories": [
                {"factors": {k: a["factors"][k] for k in FACTOR_ORDER},
                 "gold": bool(a["gold"])}
                for a in case
            ]}
            for case in cases
        ],
    }
```

**Step 4: Run** → Expected: PASS.

**Step 5: Commit**

```bash
git add lmfm/export.py tests/test_export.py
git commit -m "feat: numeric factor-matrix export format (no text)"
```

### Task 4.2: Wire the `lmfm export-factors` CLI command

**Files:**
- Test: `tests/test_cli_export.py`
- Create: `lmfm/cli.py`

**Step 1: Write the failing test** (uses fake embedder via monkeypatch-free path: a tiny .md + `--no-embed` hash fallback)

```python
# tests/test_cli_export.py
import json
from pathlib import Path
from lmfm.cli import main


def test_export_factors_writes_numeric_matrix(tmp_path: Path):
    md = tmp_path / "mem.md"
    md.write_text("## A\nUser allergic to penicillin.\n\n## B\nLikes dark mode.\n")
    gold = tmp_path / "gold.txt"
    gold.write_text("mem#a\n")          # id = <stem>#<slug>
    out = tmp_path / "factors.json"
    rc = main(["export-factors", str(md), "--gold", str(gold),
               "-o", str(out), "--hash-embed"])
    assert rc == 0
    mat = json.loads(out.read_text())
    assert mat["schema"] == "lmfm.factors.v1"
    assert "text" not in out.read_text()
```

**Step 2: Run** → Expected: FAIL.

**Step 3: Write `lmfm/cli.py`** with an `export-factors` subcommand. Use `hash_embed` (from `embedder.py`) when `--hash-embed` is passed (so tests need no model download); otherwise `SBertEmbedder()`. Read gold ids from the `--gold` file (one id per line). Pipeline: `load_markdown` → `annotate_memories` → `build_matrix` → write JSON.

```python
"""lmfm command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io.markdown import load_markdown
from .factors.annotate import annotate_memories
from .export import build_matrix


def _embedder(use_hash: bool):
    from .factors.embedder import SBertEmbedder, hash_embed
    if use_hash:
        return lambda t: hash_embed(t)
    return SBertEmbedder()


def _cmd_export(args) -> int:
    recs = load_markdown(args.dump, split=args.md_split)
    gold = set()
    if args.gold:
        gold = {ln.strip() for ln in Path(args.gold).read_text().splitlines() if ln.strip()}
    ann = annotate_memories(recs, embedder=_embedder(args.hash_embed), gold_ids=gold)
    mat = build_matrix([ann], keep_frac=args.keep_frac)
    out = Path(args.out)
    out.write_text(json.dumps(mat, indent=2))
    print(f"wrote {out}  ({len(ann)} memories, {sum(a['gold'] for a in ann)} gold)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="lmfm")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export-factors",
                       help="locally extract memories → numeric factor matrix")
    e.add_argument("dump", help="path to a markdown memory dump")
    e.add_argument("-o", "--out", default="factors.json")
    e.add_argument("--gold", help="file of gold memory ids (one per line)")
    e.add_argument("--keep-frac", type=float, default=0.3)
    e.add_argument("--md-split", default="auto",
                   choices=["heading", "bullet", "dated", "auto"])
    e.add_argument("--hash-embed", action="store_true",
                   help="use deterministic hash embedding (no model download)")
    e.set_defaults(func=_cmd_export)

    args = p.parse_args(argv)
    return args.func(args)
```

**Step 4: Run** → Expected: PASS.

**Step 5: Grep** `! grep -rin "borge" lmfm/cli.py` → clean.

**Step 6: Commit**

```bash
git add lmfm/cli.py tests/test_cli_export.py
git commit -m "feat: lmfm export-factors CLI (local, privacy-preserving)"
```

---

## Phase 5 — Cloud `/learn` (paid layer)

### Task 5.1: The learning objective over an uploaded matrix

**Files:**
- Test: `tests/test_objective.py`
- Create: `lmfm/learn/__init__.py` (empty)
- Create: `lmfm/learn/objective.py`

**Step 1: Write the failing test**

```python
# tests/test_objective.py
from lmfm.learn.objective import gold_retention, matrix_objective


def _case(*mems):
    return {"memories": [{"factors": f, "gold": g} for f, g in mems]}


def test_gold_retention_keeps_high_value_gold():
    # reliability-weighted: the gold item has high reliability
    weights = {"reliability": 1.0}
    case = _case(
        ({"reliability": 0.9}, True),    # gold, high value → kept
        ({"reliability": 0.1}, False),   # non-gold, low → dropped
    )
    r = gold_retention(case, weights, keep_frac=0.5)
    assert r == 1.0


def test_matrix_objective_averages_cases():
    weights = {"reliability": 1.0}
    mat = {"keep_frac": 0.5, "cases": [
        _case(({"reliability": 0.9}, True), ({"reliability": 0.1}, False)),
        _case(({"reliability": 0.8}, True), ({"reliability": 0.2}, False)),
    ]}
    obj = matrix_objective(mat)
    assert obj(weights) == 1.0
```

**Step 2: Run** → Expected: FAIL.

**Step 3: Write `lmfm/learn/objective.py`** (port `gold_retention` from `experiments/lme_real_retention.py`, adapted to the matrix schema)

```python
"""Gold-retention objective over an uploaded numeric factor matrix."""

from __future__ import annotations

from ..value import MemoryValue


def gold_retention(case: dict, weights: dict, *, keep_frac: float):
    mv = MemoryValue(weights=weights)
    scored = [(mv.value(m["factors"]), m) for m in case["memories"]]
    scored.sort(key=lambda x: -x[0])
    k = max(1, int(len(scored) * keep_frac))
    kept = [m for _, m in scored[:k]]
    total = sum(1 for m in case["memories"] if m["gold"])
    if total == 0:
        return None
    return sum(1 for m in kept if m["gold"]) / total


def matrix_objective(matrix: dict):
    """Return a task_return(weights)->float averaging gold retention."""
    kf = matrix["keep_frac"]
    cases = matrix["cases"]

    def obj(weights: dict) -> float:
        vals = [gold_retention(c, weights, keep_frac=kf) for c in cases]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    return obj
```

**Step 4: Run** → Expected: PASS.

**Step 5: Commit**

```bash
git add lmfm/learn/objective.py lmfm/learn/__init__.py tests/test_objective.py
git commit -m "feat: gold-retention objective over factor matrix"
```

### Task 5.2: `learn_from_matrix` — fit weights server-side

**Files:**
- Test: `tests/test_learn_from_matrix.py`
- Create: `lmfm/learn/service.py`

**Step 1: Write the failing test**

```python
# tests/test_learn_from_matrix.py
from lmfm.learn.service import learn_from_matrix


def test_learn_from_matrix_returns_weights_and_score():
    mat = {"keep_frac": 0.5, "factor_order":
           ["emotion","goal_relevance","value_alignment","self_relevance",
            "task_utility","reliability","usage"],
           "cases": [
        {"memories": [
            {"factors": {"reliability": 0.9, "emotion": 0.1}, "gold": True},
            {"factors": {"reliability": 0.1, "emotion": 0.9}, "gold": False},
        ]},
    ]}
    out = learn_from_matrix(mat, iters=60, seed=1)
    assert set(out["weights"]) >= {"reliability", "emotion"}
    assert 0.0 <= out["train_retention"] <= 1.0
    # learner should prefer reliability for this gold structure
    assert out["weights"]["reliability"] >= out["weights"]["emotion"]
```

**Step 2: Run** → Expected: FAIL.

**Step 3: Write `lmfm/learn/service.py`**

```python
"""Server-side: fit weights from an uploaded factor matrix."""

from __future__ import annotations

from ..value import learn_weights, MemoryValue
from .objective import matrix_objective

# Only learn over factors that actually vary in the upload; learning a
# weight for an all-zero factor would multiply out to nothing and just
# add noise.
def _live_factors(matrix: dict) -> tuple[str, ...]:
    order = matrix.get("factor_order") or list(MemoryValue.FACTORS)
    seen = {f: False for f in order}
    for c in matrix["cases"]:
        for m in c["memories"]:
            for f, v in m["factors"].items():
                if v not in (0, 0.0):
                    seen[f] = True
    live = tuple(f for f in order if seen.get(f))
    return live or tuple(order)


def learn_from_matrix(matrix: dict, *, iters: int = 120, seed: int = 1) -> dict:
    obj = matrix_objective(matrix)
    live = _live_factors(matrix)
    weights, hist = learn_weights(obj, live, seed=seed, iters=iters)
    return {
        "weights": {f: round(weights[f], 4) for f in live},
        "train_retention": round(hist[-1]["best_return"], 4),
        "model_version": "v1",
        "factors_learned": list(live),
    }
```

**Step 4: Run** → Expected: PASS.

**Step 5: Commit**

```bash
git add lmfm/learn/service.py tests/test_learn_from_matrix.py
git commit -m "feat: learn_from_matrix server-side fitter"
```

### Task 5.3: FastAPI `/learn` endpoint with API-key auth + metering

**Files:**
- Test: `tests/test_api.py`
- Create: `lmfm/server/__init__.py` (empty)
- Create: `lmfm/server/app.py`
- Create: `lmfm/server/keys.py`

**Step 1: Write the failing test** (FastAPI TestClient via httpx)

```python
# tests/test_api.py
from fastapi.testclient import TestClient
from lmfm.server.app import create_app


def _matrix():
    return {"keep_frac": 0.5,
            "factor_order": ["reliability", "emotion"],
            "cases": [{"memories": [
                {"factors": {"reliability": 0.9, "emotion": 0.1}, "gold": True},
                {"factors": {"reliability": 0.1, "emotion": 0.9}, "gold": False},
            ]}]}


def test_learn_requires_api_key():
    app = create_app(valid_keys={"k-good": {"quota": 5}})
    c = TestClient(app)
    r = c.post("/learn", json=_matrix())
    assert r.status_code == 401


def test_learn_with_valid_key_returns_weights():
    app = create_app(valid_keys={"k-good": {"quota": 5}})
    c = TestClient(app)
    r = c.post("/learn", json=_matrix(), headers={"Authorization": "Bearer k-good"})
    assert r.status_code == 200
    body = r.json()
    assert "weights" in body and "train_retention" in body


def test_quota_decrements_and_blocks():
    app = create_app(valid_keys={"k-lim": {"quota": 1}})
    c = TestClient(app)
    h = {"Authorization": "Bearer k-lim"}
    assert c.post("/learn", json=_matrix(), headers=h).status_code == 200
    assert c.post("/learn", json=_matrix(), headers=h).status_code == 429
```

**Step 2: Run** → Expected: FAIL.

**Step 3: Write `lmfm/server/keys.py`**

```python
"""In-memory API-key store + metering. Swap for a DB in production."""

from __future__ import annotations


class KeyStore:
    def __init__(self, valid_keys: dict):
        # {key: {"quota": int}}  (quota = remaining /learn calls)
        self._keys = {k: dict(v) for k, v in valid_keys.items()}

    def authorize(self, key: str | None) -> str:
        """Return 'ok' | 'unauthorized' | 'exhausted'."""
        if key is None or key not in self._keys:
            return "unauthorized"
        if self._keys[key].get("quota", 0) <= 0:
            return "exhausted"
        return "ok"

    def consume(self, key: str) -> None:
        self._keys[key]["quota"] -= 1
```

**Step 4: Write `lmfm/server/app.py`**

```python
"""FastAPI app exposing the paid /learn endpoint."""

from __future__ import annotations

from fastapi import FastAPI, Request, HTTPException

from ..learn.service import learn_from_matrix
from .keys import KeyStore


def _bearer(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    return auth[7:] if auth.startswith("Bearer ") else None


def create_app(valid_keys: dict) -> FastAPI:
    app = FastAPI(title="lmfm learn service")
    store = KeyStore(valid_keys)

    @app.post("/learn")
    async def learn(request: Request):
        key = _bearer(request)
        status = store.authorize(key)
        if status == "unauthorized":
            raise HTTPException(status_code=401, detail="invalid or missing API key")
        if status == "exhausted":
            raise HTTPException(status_code=429, detail="quota exhausted")
        matrix = await request.json()
        result = learn_from_matrix(matrix)
        store.consume(key)
        return result

    return app
```

**Step 5: Run** `pytest tests/test_api.py -v` → Expected: PASS (3 passed).

**Step 6: Grep** `! grep -rin "borge" lmfm/server/` → clean.

**Step 7: Commit**

```bash
git add lmfm/server/ tests/test_api.py
git commit -m "feat: /learn FastAPI endpoint with API-key auth + quota"
```

---

## Phase 6 — Client round-trip: upload + load weights

### Task 6.1: `lmfm learn` client command (upload matrix → save weights)

**Files:**
- Test: `tests/test_cli_learn.py`
- Modify: `lmfm/cli.py` (add `learn` subcommand)
- Create: `lmfm/client.py`

**Step 1: Write the failing test** (monkeypatch the HTTP POST so no network)

```python
# tests/test_cli_learn.py
import json
from pathlib import Path
import lmfm.client as client
from lmfm.cli import main


def test_learn_uploads_matrix_and_saves_weights(tmp_path, monkeypatch):
    mat = tmp_path / "factors.json"
    mat.write_text(json.dumps({"keep_frac": 0.5, "factor_order": ["reliability"],
        "cases": [{"memories": [{"factors": {"reliability": 0.9}, "gold": True}]}]}))
    out = tmp_path / "weights.json"

    def fake_post(url, payload, api_key):
        assert "text" not in json.dumps(payload)   # privacy guard
        return {"weights": {"reliability": 0.7}, "train_retention": 1.0}

    monkeypatch.setattr(client, "post_learn", fake_post)
    rc = main(["learn", str(mat), "--endpoint", "https://x/learn",
               "--key", "k-good", "-o", str(out)])
    assert rc == 0
    assert json.loads(out.read_text())["weights"]["reliability"] == 0.7
```

**Step 2: Run** → Expected: FAIL.

**Step 3: Write `lmfm/client.py`** (stdlib urllib POST)

```python
"""Client helpers for the paid /learn service."""

from __future__ import annotations

import json
import urllib.request


def post_learn(url: str, payload: dict, api_key: str) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))
```

**Step 4: Add the `learn` subcommand to `lmfm/cli.py`**

```python
# in main(), after the export-factors parser:
    l = sub.add_parser("learn", help="upload a factor matrix → learned weights")
    l.add_argument("matrix", help="factors.json from export-factors")
    l.add_argument("--endpoint", required=True)
    l.add_argument("--key", required=True)
    l.add_argument("-o", "--out", default="weights.json")
    l.set_defaults(func=_cmd_learn)
```

```python
# new handler:
def _cmd_learn(args) -> int:
    from . import client
    payload = json.loads(Path(args.matrix).read_text())
    result = client.post_learn(args.endpoint, payload, args.key)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"learned weights → {args.out}  (train_retention={result.get('train_retention')})")
    return 0
```

**Step 5: Run** `pytest tests/test_cli_learn.py -v` → Expected: PASS.

**Step 6: Commit**

```bash
git add lmfm/client.py lmfm/cli.py tests/test_cli_learn.py
git commit -m "feat: lmfm learn client (upload matrix → save weights)"
```

### Task 6.2: Load learned weights back into the value model

**Files:**
- Test: `tests/test_weights_roundtrip.py`
- Create: `lmfm/weights.py`

**Step 1: Write the failing test**

```python
# tests/test_weights_roundtrip.py
import json
from lmfm.weights import load_value_from_weights_file


def test_load_weights_file_into_value(tmp_path):
    wf = tmp_path / "weights.json"
    wf.write_text(json.dumps({"weights": {"reliability": 0.9, "emotion": 0.2}}))
    mv = load_value_from_weights_file(wf)
    # learned weights override defaults
    assert mv.weights["reliability"] == 0.9
    assert mv.weights["emotion"] == 0.2
    # untouched factors fall back to default
    assert mv.weights["self_relevance"] == 0.23
    # scoring works
    v = mv.value({"reliability": 1.0, "emotion": 0.0})
    assert round(v, 2) == 0.9
```

**Step 2: Run** → Expected: FAIL.

**Step 3: Write `lmfm/weights.py`**

```python
"""Load a learned weights.json back into a MemoryValue."""

from __future__ import annotations

import json
from pathlib import Path

from .value import default_memory_value


def load_value_from_weights_file(path):
    data = json.loads(Path(path).read_text())
    weights = data.get("weights", data)   # accept bare {factor: w} too
    return default_memory_value(override=weights)
```

**Step 4: Run** → Expected: PASS.

**Step 5: Commit**

```bash
git add lmfm/weights.py tests/test_weights_roundtrip.py
git commit -m "feat: load learned weights into value model (round-trip closed)"
```

---

## Phase 7 — Full-suite guard + naming audit

### Task 7.1: End-to-end local test (export → learn-local → load)

**Files:**
- Test: `tests/test_e2e_local.py`

**Step 1: Write the test** (runs `learn_from_matrix` directly — no server — to validate the whole local chain)

```python
# tests/test_e2e_local.py
import json
from pathlib import Path
from lmfm.cli import main
from lmfm.learn.service import learn_from_matrix
from lmfm.weights import load_value_from_weights_file


def test_export_then_learn_then_load(tmp_path: Path):
    md = tmp_path / "mem.md"
    md.write_text("## A\nUser is allergic to penicillin.\n\n## B\nok thanks\n")
    gold = tmp_path / "gold.txt"; gold.write_text("mem#a\n")
    fac = tmp_path / "factors.json"
    assert main(["export-factors", str(md), "--gold", str(gold),
                 "-o", str(fac), "--hash-embed"]) == 0

    mat = json.loads(fac.read_text())
    result = learn_from_matrix(mat, iters=40)
    wf = tmp_path / "weights.json"; wf.write_text(json.dumps(result))

    mv = load_value_from_weights_file(wf)
    assert 0.0 <= result["train_retention"] <= 1.0
    assert mv.value({"reliability": 1.0}) >= 0.0
```

**Step 2: Run** `pytest tests/test_e2e_local.py -v` → Expected: PASS.

**Step 3: Commit**

```bash
git add tests/test_e2e_local.py
git commit -m "test: end-to-end local export→learn→load chain"
```

### Task 7.2: Naming audit — fail loudly on any `borge`

**Files:**
- Test: `tests/test_no_borge.py`

**Step 1: Write the test**

```python
# tests/test_no_borge.py
import pathlib
import re


def test_no_forbidden_naming():
    root = pathlib.Path(__file__).resolve().parents[1] / "lmfm"
    pat = re.compile(r"borge", re.IGNORECASE)
    offenders = []
    for p in root.rglob("*.py"):
        if pat.search(p.read_text(encoding="utf-8")):
            offenders.append(str(p))
        if pat.search(p.name):
            offenders.append(p.name)
    assert not offenders, f"'borge' found in: {offenders}"
```

**Step 2: Run** `pytest tests/test_no_borge.py -v` → Expected: PASS. If FAIL, scrub the named files and re-run.

**Step 3: Run the full suite**

Run: `pytest -q`
Expected: all green.

**Step 4: Commit**

```bash
git add tests/test_no_borge.py
git commit -m "test: naming audit forbids any borge string"
```

---

## Done criteria

- `pytest -q` all green.
- `lmfm export-factors mem.md --gold gold.txt -o factors.json` produces a text-free numeric matrix.
- `factors.json` contains zero raw memory text (privacy guard test passes).
- `lmfm learn factors.json --endpoint ... --key ...` returns `weights.json`.
- `load_value_from_weights_file` round-trips learned weights into `MemoryValue`.
- `tests/test_no_borge.py` passes — no `borge` anywhere under `lmfm/`.

## Explicitly out of scope (separate plans)

- `/evolve` continuous-evolution feedback loop + scheduled re-learning.
- Custom user-defined factors (factor plugin registry).
- Billing/Stripe integration (metering here is an in-memory quota only).
- Persisting the KeyStore to a database.
- Publishing to PyPI.
