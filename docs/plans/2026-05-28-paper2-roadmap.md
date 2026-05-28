# paper2 Roadmap (API-free track) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Strengthen paper2 ("Learning What to Remember") from a 74-case pilot to a full-500 LongMemEval-S study with a keep-fraction sweep, per-case bootstrap CIs, and a neural interaction ablation — all API-free — keeping the interpretable linear value as the headline.

**Architecture:** Split the pipeline into (a) one expensive SBert **annotation pass** that caches per-turn factor vectors to disk, and (b) cheap **evaluation** scripts that read the cache. This fixes the killed full-500 run (it re-encoded ~275k turns every time) and makes every downstream experiment run in seconds. The neural ablation swaps only the scalar scoring function `g_θ` behind the existing `.value(factors)->float` interface; nothing else in encode/forget/retrieve changes.

**Tech Stack:** Python, sentence-transformers (all-MiniLM-L6-v2, already used), numpy, torch 2.10 (MLP + pairwise ranking loss), matplotlib. Branch: `multi-factor-eval`.

**Scope:** API-free items only. The QA-accuracy loop (answerer + judge LLM, costs $) is explicitly DEFERRED to a later gated step. The linear model stays the headline; the MLP is an interaction ablation, never the default.

**Invariants (do not violate):**
- The 74-case numbers in the current paper must remain reproducible from the cache (regression guard).
- `MemoryValue` linear path and its weights stay the interpretable headline.
- No fabricated numbers — every paper number traces to a results JSON.
- Each task commits on `multi-factor-eval`.

---

### Task 1: Factor cache builder (the enabler)

**Files:**
- Create: `experiments/build_factor_cache.py`
- Create (output): `results/lme_factor_cache.jsonl` (gitignore-large? keep — few MB)
- Reuse: `borge/eval/longmemeval.py` (`load_longmemeval`, `flatten_to_messages`), `borge/affective/signal_extractor.py`, `borge/values/self_model.py` (`SBertEmbedder`, `cosine`)

**Design:** Lift the per-turn factor computation out of `experiments/lme_blind_forgetting.py::annotate_dual` into a standalone cache builder. For each case, encode question + all turn texts ONCE, compute the regime-independent factors plus BOTH goal variants, write one JSONL record per case:
```json
{"qid": "...", "turns": [{"emotion":0.0,"self":0.61,"reliability":0.7,
  "goal_oracle":0.83,"goal_blind":0.55,"has_answer":false,"sidx":3}, ...]}
```
`goal_oracle = sim01(turn, question)`, `goal_blind = sim01(turn, session-user-centroid)`, `self = sim01(turn, μ_user)`, `emotion = |ΔV|·(0.5+ΔA)` clamped, `reliability = 0.7 user / 0.4 assistant`. (Identical formulas to current `annotate_dual` — copy them verbatim so numbers match.)

**Steps:**
1. Write `build_factor_cache.py` with `--n-cases` (default 500) and `--out results/lme_factor_cache.jsonl`. Only keep cases with ≥1 `has_answer` turn (same filter as now). Print progress every 25 cases.
2. Dry-run on `--n-cases 5` → confirm JSONL parses, factor keys present, values in [0,1].
3. Add `tests/test_factor_cache.py::test_cache_record_schema` — build cache for 2 tiny synthetic cases (monkeypatch a fake embedder returning fixed vectors), assert record keys + value ranges + determinism.
4. Run test → PASS.
5. Launch full build in background: `python experiments/build_factor_cache.py --n-cases 500` (slow, ~once; ~MB output). **This is the only SBert-heavy step.**
6. Commit `build_factor_cache.py` + test (NOT the large cache yet — commit cache after Task 2 verifies it).

**Verify:** `wc -l results/lme_factor_cache.jsonl` ≈ number of usable cases (expect ~450–470 of 500); each line parses.

---

### Task 2: Cache-backed blind-forgetting + full-500 headline

**Files:**
- Modify: `experiments/lme_blind_forgetting.py` (add `--cache PATH` mode that reads the JSONL instead of running SBert; keep the existing SBert path as fallback)
- Output: `results/lme_blind_forgetting_full.json`

**Steps:**
1. Add a `load_from_cache(path)` that yields the same per-case annotated structure `annotate_dual` returns (`factors_oracle`, `factors_blind`, `has_answer`, `timestamp_idx`), so the rest of `main()` is unchanged.
2. **Regression guard:** run `--cache ... --n-cases 74` style (first 74 usable cases) and confirm learned/uniform/reliability/recency match the committed `lme_blind_forgetting.json` (0.811 / 0.634 / 0.541 / 0.380) within float tol. If mismatch, STOP — the cache formulas drifted.
3. Run full: read entire cache → `results/lme_blind_forgetting_full.json`. Record new mean±std + paired diffs + learned weights on ~450+ cases.
4. Commit the cache file + `lme_blind_forgetting.py` change + `lme_blind_forgetting_full.json`.

**Verify:** full-run learned_V still clearly beats uniform/reliability/recency with non-overlapping CIs and ~100% paired win-rate; if the gap shrinks materially on 500 cases, report honestly (do not hide).

---

### Task 3: Keep-fraction sweep + figure

**Files:**
- Create: `experiments/lme_keepfrac_sweep.py` (reads cache)
- Output: `results/lme_keepfrac_sweep.json`, `paper2/figures/fig_keepfrac.pdf`

**Steps:**
1. For `keep_frac ∈ {0.1, 0.2, 0.3, 0.4, 0.5}`: learn blind weights on train split, eval test retention for learned / uniform / reliability_only / recency. 20 resampled splits, mean±std (reuse helpers).
2. Write JSON: per-keepfrac policy means.
3. Render `fig_keepfrac.pdf` (matplotlib): x=keep_frac, y=retention, one line per policy, error bands. Add to `paper2/figures/render_paper2.py`.
4. Commit script + JSON + figure + render update.

**Verify:** learned_V dominates across all keep fractions (effect not budget-specific) — the point of the sweep.

---

### Task 4: Per-case bootstrap CI

**Files:**
- Modify: `experiments/lme_blind_forgetting.py` (add `--bootstrap N` over CASES, default 0=off)
- Output: extend `lme_blind_forgetting_full.json` with a `bootstrap` block

**Steps:**
1. After learning weights on the full set (or a fixed train split), bootstrap-resample CASES with replacement B=1000; for each resample compute test retention of each policy + the learned−uniform / learned−recency / learned−reliability gaps. Report 2.5/97.5 percentile CIs + fraction-of-resamples-positive (a sign-test-style p-proxy).
2. This replaces the "split variance only" caveat (L5/§4.2) with a proper case-level CI.
3. Commit.

**Verify:** gap CIs exclude 0 → the advantage is real at the benchmark-sampling level, not just split-resampling.

---

### Task 5: Neural interaction ablation (MLP `g_θ`)

**Files:**
- Create: `borge/memory/value_net.py` (`MemoryValueNet`)
- Create: `tests/test_value_net.py`
- Modify: `experiments/lme_blind_forgetting.py` (`--model linear|mlp`)
- Output: `results/lme_value_net_ablation.json`

**Design — `MemoryValueNet`:** torch MLP over the factor vector, exposing the SAME `.value(factors: dict)->float` interface so all consumers are untouched. Train with a **pairwise ranking loss** (gold turns should score above non-gold within a case), since top-κ keep is non-differentiable:

```python
import torch, torch.nn as nn

FACTORS = ("emotion","goal_relevance","self_relevance","reliability")  # live factors

class MemoryValueNet(nn.Module):
    def __init__(self, factors=FACTORS, hidden=(16, 16)):
        super().__init__()
        self.factors = tuple(factors)
        dims = [len(self.factors), *hidden, 1]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers = layers[:-1]  # drop final ReLU
        self.net = nn.Sequential(*layers)

    def _vec(self, factors: dict):
        return torch.tensor([[factors.get(f, 0.0) for f in self.factors]],
                            dtype=torch.float32)

    @torch.no_grad()
    def value(self, factors: dict) -> float:        # mirrors MemoryValue.value
        self.eval()
        return float(self.net(self._vec(factors)).item())

def train_value_net(train_cases, *, factors=FACTORS, epochs=200, lr=1e-2,
                    weight_decay=1e-3, seed=1):
    """Pairwise logistic ranking loss: within each case, gold(has_answer)
    turns ranked above non-gold. weight_decay guards the tiny-data regime."""
    torch.manual_seed(seed)
    net = MemoryValueNet(factors)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    # Precompute (pos, neg) factor tensors per case once.
    pairs = []  # list of (pos_tensor, neg_tensor) stacked per case
    for case in train_cases:
        pos = [t for t in case if t["has_answer"]]
        neg = [t for t in case if not t["has_answer"]]
        if not pos or not neg:
            continue
        P = torch.tensor([[t[f] for f in factors] for t in pos], dtype=torch.float32)
        N = torch.tensor([[t[f] for f in factors] for t in neg], dtype=torch.float32)
        pairs.append((P, N))
    for _ in range(epochs):
        opt.zero_grad(); loss = 0.0
        for P, N in pairs:
            sp, sn = net.net(P), net.net(N)            # [|pos|,1], [|neg|,1]
            diff = sp.unsqueeze(1) - sn.unsqueeze(0)    # all pos-neg pairs
            loss = loss + torch.nn.functional.softplus(-diff).mean()
        loss = loss / max(1, len(pairs))
        loss.backward(); opt.step()
    return net
```

**Steps:**
1. Write `value_net.py` as above (note: experiment maps cache keys `goal_blind`→`goal_relevance`, `self`→`self_relevance` when building the per-turn dict for training/scoring).
2. `tests/test_value_net.py`: (a) `.value()` returns float, shape ok; (b) on a toy dataset where gold has high reliability, after `train_value_net` the mean gold score > mean non-gold score (ranking loss works).
3. Run tests → PASS.
4. Add `--model mlp` branch: train `MemoryValueNet` on the train split (blind factors), evaluate test retention exactly like the linear policy (rank by `.value()`, keep top-κ). Compare linear vs mlp on full-500 → `results/lme_value_net_ablation.json`.
5. Commit.

**Verify + honest reporting:** report linear vs MLP retention. Two acceptable outcomes — (a) MLP ≈ linear → "factors combine near-additively; the interpretable linear model suffices" (strengthens the headline); (b) MLP > linear → name the captured interaction (inspect via permutation importance / a 2-factor probe). EITHER is reported truthfully; the MLP never replaces the linear headline.

---

### Task 6: `value_forget_score` policy sanity (small)

**Files:**
- Modify: `experiments/lme_blind_forgetting.py` (add a `forget_formula` policy option)

**Steps:**
1. Add a policy that ranks turns by the deployed `value_forget_score` (Eq.2: recency·usage·1/(1+βV)) rather than raw V, to confirm the actual forget formula (not just the abstract ranking) retains gold. In the static eval recency/usage are uniform, so this should ≈ the learned_V ranking — a consistency check, reported in one line.
2. Commit.

---

### Task 7: Fold results into paper2 + recompile

**Files:**
- Modify: `paper2/sections/04_experiments.tex` (full-500 numbers; keep-frac figure; bootstrap CI; new "Interaction ablation" paragraph), `paper2/sections/06_limitations.tex` (L5 — replace "future work" NN/bootstrap caveats with the now-done results), `paper2/main.tex` (abstract numbers if they shift)
- Run: `latexmk`; regenerate `arxiv_submission.tar.gz`

**Steps:**
1. Update every number to the full-500 values (or keep 74-pilot as a labeled subset + add full-500 as the main result — decide based on whether numbers hold).
2. Add the interaction-ablation result + keep-frac figure + bootstrap CIs.
3. Recompile clean (no undefined refs / overfull); regenerate tarball; commit.
4. Optional: re-run `auto-review-loop` round to re-score the strengthened draft.

**Verify:** `tools`-free — paper compiles to PDF, every new number matches its JSON, `git grep` finds no stale 74-only claims left unlabeled.

---

## Dependency order
Task 1 → 2 (cache must exist) → {3, 4, 5, 6 in any order, all read cache} → 7 (paper, after experiments land).

## Out of scope (deferred / gated)
- QA-accuracy loop (answerer + judge LLM, API $).
- Generalization to tool-use/coding agents.
- Replacing the linear headline with the MLP.
