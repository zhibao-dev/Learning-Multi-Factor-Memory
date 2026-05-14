# Self-FEP Memory — Narrative Report

**Pipeline**: idea-discovery → implement → run-experiment → Codex review → narrative
**Date**: 2026-05-15
**Substrate**: BorgeAgent v0.2-cleaner + Self-FEP extension
**Venue target**: NeurIPS (AUTO_WRITE pending; this report is the handoff)

---

## 1. Problem Statement and Core Claim

**Problem.** Existing computational memory models (CMR3, ACT-R, LLM-agent retrieval-augmented systems) modulate memory by emotional valence/arousal but lack a principled "self" model that interacts mechanistically with emotion. The cognitive psychology literature has documented the Self-Reference Effect (SRE; Rogers, Kuiper & Kirker 1977; Symons & Johnson 1997 meta-analysis) for 40+ years without unifying it with the Free Energy Principle (FEP; Friston 2010), even though FEP work on self (Apps & Tsakiris 2014; Limanowski & Blankenburg 2013) explicitly invokes self-as-generative-model.

**Core claim.** Treat the agent's "self" as a generative model `M_self = (μ_self, π_self)` under FEP. Let self-precision `π_self` and self-relevance `sr = cos(memory_embedding, μ_self)` multiplicatively gate memory dynamics:

```
encoding_depth ←  emotion-driven tier, bumped one level if sr > 0.65
forget_score   ←  recency × usage × imp_res × emotion_res × self_res
                  where self_res = 1 / (1 + λ · sr)
retrieval rank ←  mood_sim + recency + relevance + ΔF_bonus
                  + w_s · cos(memory.embedding, current μ_self)
```

`μ_self` updates via EMA only on user content containing first-person tokens (theory-driven gating: self is identity-constitutive, not topic-chasing). `π_self` updates via `π ∝ 1/(1 + γ·Var[PE])` over a 50-turn window.

## 2. Method Summary

**Implementation locus** ([commit `8a17480..ee62067`](https://github.com/zhibao-dev/BorgeAgent)):

| File | Role |
|------|------|
| `borge/values/self_model.py` | `SelfModel(μ_self, π_self)` + bag-of-tokens embedding |
| `borge/memory/store.py` | `borge_memories` schema adds `self_relevance_score`, `embedding` |
| `borge/memory/consolidation.py:262-288` | Step 3 computes embedding + sr per memory; updates μ_self on self-referencing user content |
| `borge/memory/forgetting.py:148-156` | `self_resistance = 1/(1 + λ·sr)` factor in `_compute_score` |
| `borge/memory/retrieval.py:153-170` | `_self_similarity()` adds `w_s · cos(emb, μ_self)` to rank |
| `borge/agent.py:81-117` | `BorgeAgent.self_model` bootstrapped from SOUL.md values |

**Engineering posture.** All changes preserve back-compat: Hermes plugin mode (no SelfModel wired) gets `self_resistance = 1.0` (no-op). Existing 30 tests + 7 new self-FEP tests all pass.

## 3. Experiments and Results

Three CPU-only experiments. Each is deterministic given seed; total runtime ~5 s.

### E1 — SRE replication (mechanical formulation)

**Design.** 4 encoding conditions (structural / phonemic / semantic / self-ref) × 50 items × 10 seeds. Each condition pre-assigned a depth-of-processing self-relevance (0.05 / 0.20 / 0.50 / 0.85). Run forgetting pass, measure mean forget_score per condition.

**Result.** Forget-score ordering matches the human SRE direction in **10/10 seeds**:

| Condition | self_relevance | mean forget |
|-----------|---------------:|------------:|
| structural | 0.05 | 2.366 |
| phonemic   | 0.20 | 1.859 |
| semantic   | 0.50 | 1.302 |
| **self_ref** | 0.85 | **0.964** |

SRE recall-proxy (semantic forget − self_ref forget) = **+0.337**.

**Honest framing.** This is a *mechanical sanity check*, not an empirical SRE replication. We hand-set sr per condition and verify the formula responds in the human-observed direction. The harder test — whether the forward pass (raw stimulus → encoding task → sr) recovers the condition-appropriate sr without manual setting — requires sentence-transformer embeddings rather than our bag-of-tokens hash; the included E1b stub documents this gap.

### E2 — Mood × Self factorial (additive vs multiplicative)

**Design.** 2×2 factorial: emotion ∈ {low (|V|·A ≈ 0), high (≈ 0.7)} × self ∈ {low sr=0.1, high sr=0.85}, 100 items per cell.

**Result.**

| Cell | mean forget |
|------|------------:|
| emotion_low, self_low   | 2.169 |
| emotion_low, self_high  | 0.964 |
| emotion_high, self_low  | 0.889 |
| **emotion_high, self_high** | **0.395** |

**Interaction diagnostic**: `delta_low = -1.205`, `delta_high = -0.494`, `|delta_low − delta_high| = 0.711`. Pure additivity predicts 0; the 0.71 gap is the multiplicative signature.

**OLS model comparison (raw scale, primary test)**:

| Model | AIC | Interaction coef |
|-------|----:|------:|
| additive (E + S) | −1376 | — |
| with interaction (E + S + E·S) | **−11044** | **+0.711** |
| **ΔAIC (interaction − additive)** | **−9669** | |

The with-interaction model wins decisively. Log-scale companion check finds the log-interaction coef rounded to 0.0, confirming the data has pure-product structure (no super-multiplicative interaction).

### E3 — π_self ablation (primary) + λ sensitivity (secondary)

**E3a — π_self ablation** (theoretical test from IDEA_REPORT prediction P3). For each π_self ∈ {0.0, 0.25, 0.5, 0.75, 1.0}, build SelfModel pinned at that precision with seed = "I am honest careful patient kind myself"; pass condition-tagged texts through SelfModel.self_relevance() and run forgetting.

| π_self | sr (self_ref) | sr (semantic) | SRE size |
|-------:|---------:|---------:|---------:|
| 0.00 | 0.500 | 0.500 | **0.000** |
| 0.25 | 0.625 | 0.592 | 0.036 |
| 0.50 | 0.750 | 0.683 | 0.059 |
| 0.75 | 0.875 | 0.775 | 0.075 |
| 1.00 | 1.000 | 0.866 | **0.085** |

**Monotonic increase: TRUE**. At π_self=0, all conditions collapse to sr=0.5 and the SRE vanishes — matching the clinical prediction (dissociation/severe depression attenuate SRE).

**E3b — λ sensitivity** (secondary, formula sweep). SRE size peaks at λ=2.0 (the production default) and decreases at both extremes (λ=0 → no self gating; λ→∞ → all conditions crushed to zero). This is not a theoretical claim; it bounds the regime where the formula produces measurable separation.

## 4. Evidence for Each Claim

| Claim from IDEA_REPORT P1-P5 | Evidence | Verdict |
|---|---|---|
| P1: SRE direction replicated | E1: 10/10 seeds, ordering self_ref < semantic < phonemic < structural in forget-space | **PARTIAL** — mechanical, not from forward-pass on raw stimuli |
| P2: Mood × Self multiplicative not additive | E2: raw-scale ΔAIC −9669; interaction coef +0.71 | **SUPPORTED** |
| P3: SRE attenuates as π_self → 0 | E3a: monotone increase from 0.000 to 0.085 across π_self ∈ [0, 1] | **SUPPORTED** |
| P4: F_total drops on self-relevant turns (self-evidencing) | Not run in this session — needs longer simulated dialogues | **PENDING** |
| P5: Self-FEP beats CMR3 on self-referenced subsets | Not run — CMR3 not reimplemented this session | **PENDING** |

## 5. Figure/Table Inventory

| Item | Status |
|------|--------|
| Table 1: E1 4-condition forget means | Generated from `results/e1_sre_replication.json`, table in section 3 |
| Table 2: E2 2×2 cell means | Generated from `results/e2_mood_self_factorial.json`, table in section 3 |
| Table 3: E2 AIC comparison | Generated from same JSON, table in section 3 |
| Table 4: E3a π_self ablation | Generated from `results/e3_pi_self_ablation.json`, table in section 3 |
| Figure 1: Architecture diagram (BorgeAgent + SelfModel) | **NEEDS MANUAL CREATION** — Mermaid or TikZ |
| Figure 2: SRE bar chart | **NEEDS MANUAL CREATION** — matplotlib from `e1_*.json` |
| Figure 3: Mood × Self interaction plot | **NEEDS MANUAL CREATION** — matplotlib from `e2_*.json` |
| Figure 4: π_self ablation curve | **NEEDS MANUAL CREATION** — matplotlib from `e3_*.json` |

## 6. Limitations (brutally honest)

1. **No human-data fit.** All experiments are agent-internal simulations. We have not fit our model to Symons & Johnson (1997) effect sizes, Rogers et al. (1977) raw data, or any clinical SRE attenuation dataset. The "matches human direction" claim is qualitative, not quantitative.

2. **Embedding is bag-of-tokens.** `self_model.embed()` uses SHA1-hashed bag-of-tokens. This cannot separate the encoding task framing ("Does X describe ME?") from the content word (X) — confirmed by a forward-pass diagnostic where self-ref and phonemic prompts both containing the trait word "honest" got near-identical self_relevance. Sentence-transformers (or BERT-class encoder) is the necessary swap before any forward-pass SRE replication is meaningful.

3. **No CMR3 comparison.** The IDEA_REPORT promised to fit both Self-FEP and CMR3 to held-out emotional-memory data. CMR3 is not packaged as a library; reimplementation is a multi-day task. We have not done it.

4. **π_self update rule is engineered, not derived.** `π ∝ 1/(1 + γ·Var[PE])` is a plausible heuristic but not derived from variational free energy first principles. A reviewer's first question will be "where does this come from?"

5. **N=1 model, no random initialisation noise.** All experiments are deterministic given seed. Cell-level variance reported is exactly zero in some cases (E1) because the formula response to fixed inputs is fixed. Adding realistic encoding noise is a paper-readiness fix.

6. **Forward pass μ_self updates need self-tokens.** The chosen gating (`has_self_reference()`) is a crude English+light-zh heuristic. In multilingual or oblique-self contexts ("the writer was tired"), it will miss. Treat as a v0.1 simplification.

## 7. Remaining Follow-up Items

Critical (before submission):
- [ ] Implement sentence-transformers-based `embed()` (lazy import); rerun E1 as a *true* forward-pass SRE test → currently the killer gap.
- [ ] Add noise to E1/E2/E3 (per-item Gaussian on forget_score) for honest variance estimates.
- [ ] Derive π_self update rule from a variational objective (or weaken the FEP grounding claim in the narrative).
- [ ] Fit cell means to Symons & Johnson (1997) Cohen's d ≈ 0.50; report log-likelihood + parameter constraints.

Important (for completeness):
- [ ] CMR3 comparison on at least one published emotional-memory dataset (E5).
- [ ] F_total trajectory experiment (E4) — track ΔF when self-relevant content is encoded.
- [ ] Multilingual / oblique-self extension of `has_self_reference()`.

Nice-to-have:
- [ ] Markov-blanket extension (deferred to v2 from IDEA_REPORT).
- [ ] Source-monitoring + reconsolidation hook (also deferred to v2).

## 8. Codex Review Verdict (Stage 4 record)

**Initial review**: 3/10 — "Reject in current form."

**Killer concerns identified**:
1. E1 circular (now reframed as mechanical sanity check) ✓ addressed
2. E2 log-scale interaction was mathematically erased (now raw + log dual comparison) ✓ addressed
3. E3 contradicted stated monotonicity (now actual π_self ablation, monotone) ✓ addressed
4. FEP grounding too thin → **NOT FIXED** in this session
5. Retrieval doesn't match proposal (uses memory.embedding not memory.μ_self_at_encoding) → **NOT FIXED**

**Estimated post-fix score**: ~5/10. Major revisions still required: human-data fit (limitation #1), real embeddings (limitation #2), derived precision update (limitation #4). Without these, this is "submit after major revisions" — not "ready for submission".

## 9. Stage 6 Handoff

This narrative is the input for Workflow 3 (`/paper-writing`). Given `VENUE=NeurIPS` and `AUTO_WRITE=true` was set:

**Gate 2 — Writing Checkpoint:**

```
📝 Research pipeline complete. Ready for Workflow 3.

  Venue: NeurIPS
  Input: NARRATIVE_REPORT.md
  Manual figures required:
    - Figure 1 (architecture diagram)
    - Figure 2 (SRE bar chart from e1_*.json)
    - Figure 3 (Mood × Self interaction plot from e2_*.json)
    - Figure 4 (π_self ablation curve from e3_*.json)
  Next step: /paper-writing "NARRATIVE_REPORT.md — venue: NeurIPS"
```

**My recommendation**: Pause before invoking `/paper-writing`. Current state would yield a paper that Codex (and likely real reviewers) would still reject. The four critical follow-up items above are 2-3 days of additional work. Better to fix at least #1 (sentence-transformers + real SRE forward pass) before writing, otherwise the narrative will need substantive rewriting after results arrive.

If the user wants to proceed regardless — `/paper-writing` is one command away.
