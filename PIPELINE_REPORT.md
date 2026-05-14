# Research Pipeline Report

**Direction**: 研究"情绪"和"自我"对记忆的提取和遗忘的量化建模；"自我"用自由能理论（FEP）建模
**Chosen Idea**: Self-FEP Memory — `M_self = (μ_self, π_self)` driving forget + retrieval
**Date**: 2026-05-14 → 2026-05-15
**Pipeline**: idea-discovery → implement → run-experiment → auto-review-loop → narrative

## Journey Summary

| Stage | Outcome | Compute |
|-------|---------|---------|
| 1. Idea discovery | 10 ideas generated → 8 surviving → top 3 detailed (`idea-stage/IDEA_REPORT.md`) | 0 GPU; web-search + in-session brainstorming |
| 2. Implementation | 7 sub-tasks completed: `SelfModel`, schema, embedding util, consolidation Step 3, forget_score self_resistance, retrieval self_similarity, BorgeAgent wiring | Local; 37/37 tests pass |
| 3. Experiments | E1 (mechanical SRE), E2 (Mood × Self factorial), E3a/b (π_self ablation + λ sensitivity) | CPU; ~5 s total |
| 4. Auto-review (1 round, Codex MCP) | 3/10 reject → fixed E2 raw-scale + E3 real π_self → ~5/10 with named limitations | Codex MCP read-only |
| 5. Narrative | `NARRATIVE_REPORT.md` with brutally honest limitations | Local |
| 6. Paper writing | **HOLD** — recommended pause; see Gate 2 below | n/a |

## Implementation

- 1 new module: `borge/values/self_model.py` (SelfModel + embed util)
- 4 modified modules: `consolidation.py`, `forgetting.py`, `retrieval.py`, `store.py`, `agent.py`
- 1 new test file: `tests/test_self_fep_memory.py` (7 cases)
- 3 experiment scripts: `experiments/e1_sre_replication.py`, `e2_mood_self_factorial.py`, `e3_pi_self_ablation.py`
- Results: 3 JSON files in `results/`

## Experiments

| ID | Compute | Result | Verdict on claim |
|----|---------|--------|------------------|
| E1 | CPU < 1s | Forget-score ordering matches SRE in 10/10 seeds | PARTIAL — mechanical, not raw-stimulus |
| E2 | CPU < 1s | ΔAIC −9669 favours interaction in raw space | SUPPORTED — multiplicative confirmed |
| E3a | CPU < 1s | SRE size monotonically increases 0 → 0.085 over π_self ∈ [0, 1] | SUPPORTED — matches FEP-clinical prediction |
| E3b | CPU < 1s | SRE size peaks at λ=2.0 (production default) | Sensitivity bound — not a theoretical claim |

Total runtime: ~5 seconds.

## Codex Review (Stage 4)

**Round 1**: 3/10 — reject in current form. Three killer concerns:
1. E1 circular validation
2. E2 log-scale math broken
3. E3 contradicts stated π_self monotonicity

**Fixes applied** (~30 min):
- E1 reframed as mechanical sanity check; E1b sentence-transformers test noted as future work
- E2 added raw-scale primary comparison (now properly tests multiplicative claim)
- E3 split into E3a (true π_self ablation, monotone ✓) + E3b (λ sensitivity, secondary)

**Post-fix self-assessed score**: ~5/10. Three of five concerns addressed. Two remain (#4 FEP derivation, #5 retrieval uses embedding not μ_self_at_encoding) and would push toward 7-8/10 if fixed before submission.

## Writing Handoff

- `NARRATIVE_REPORT.md`: ✅ generated (this run)
- Venue: NeurIPS (per pipeline args)
- Manual figures needed:
  - Figure 1: architecture diagram
  - Figure 2: SRE bar chart (matplotlib from `e1_*.json`)
  - Figure 3: Mood × Self interaction plot (matplotlib from `e2_*.json`)
  - Figure 4: π_self ablation curve (matplotlib from `e3_*.json`)

## Remaining TODOs (from Codex + self-audit)

Critical (block submission):
- [ ] Sentence-transformers embedding → real forward-pass SRE replication (E1b)
- [ ] Fit cell means to Symons & Johnson (1997) Cohen's d
- [ ] Derive π_self update rule from variational free energy (or soften FEP claim)
- [ ] Add noise to E1/E2/E3 for honest variance estimates

Important (for completeness):
- [ ] CMR3 comparison (E5)
- [ ] F_total trajectory experiment (E4)
- [ ] `retrieval._self_similarity()` should use `memory.μ_self_at_encoding`, not memory embedding — requires storing μ_self snapshot per memory

Nice-to-have:
- [ ] Markov blanket extension (v2)
- [ ] Source monitoring / reconsolidation hook (v2)

## Recommendation on Stage 6

**HOLD on /paper-writing.** Current state would produce a paper Codex (and likely human reviewers) would reject. The critical TODOs are 2-3 days of additional work that **should land before writing**, not after — they will change the narrative's main story, not just polish it.

If you nevertheless want the writing pipeline triggered now, the input is ready (`NARRATIVE_REPORT.md`) and you only need to invoke:

```
/paper-writing "NARRATIVE_REPORT.md" — venue: NeurIPS
```

## Budget & Audit

- GPU hours used: 0
- Codex MCP calls: 1
- Tests: 37 → 37 (no regression)
- Commits in this pipeline: 4 (`feat(self)`, `feat(experiments)`, `fix(experiments)`, narrative)
- Files added: 6 (1 module, 1 test, 3 experiments, 1 narrative)
- Lines added: ~1500
- Lines removed: 0
