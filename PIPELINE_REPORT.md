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
| 6. Paper writing | **DONE** — 12 pp NeurIPS-format PDF at `paper/main.pdf` (~5.7/10) | 4 Codex MCP calls |

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

## Stage 6 — DONE

Triggered after the user explicitly chose "write a v0.1 draft now" at
Gate 2. Full sub-pipeline executed:

| Phase | Outcome | Compute |
|-------|---------|---------|
| 1. PAPER_PLAN.md | ✅ 9-page section plan + claims-evidence matrix + 25-entry citation scaffold | Local |
| 2. Figures | ✅ 3 matplotlib PDFs from results JSON (`fig2_sre`, `fig3_factorial`, `fig4_pi_ablation`) | Local |
| 3. LaTeX writing | ✅ `main.tex` + 7 section files + `references.bib` + `math_commands.tex` | Local |
| 4. Compilation | ✅ 12 pp PDF, 0 unresolved refs/cites | TeX Live 2026 |
| 4.7 Claim audit | ✅ WARN — **23/23 numeric matches**, 0 mismatch | 1 Codex call |
| 5. Improvement loop | ✅ 2 rounds: 4/10 → 5/10 → 5.7/10 | 2 Codex calls |
| 5.8 Citation audit | ✅ 17 KEEP / 2 FIX-META / 3 REPLACE / 2 REMOVE — all applied | 1 Codex call |
| 6. Final report | ✅ `paper/PAPER_IMPROVEMENT_LOG.md` | Local |

**Final paper artifacts**:
- `paper/main.pdf` — 12 pp final PDF
- `paper/main_round0_original.pdf` — initial compile baseline
- `paper/main_round1.pdf` — after Codex Round 1 (5 fixes)
- `paper/main_round2.pdf` — after Codex Round 2 (3 fixes) + citation audit (7 fixes)
- `paper/PAPER_IMPROVEMENT_LOG.md` — full round-by-round log

**Final score**: ~5.7/10 NeurIPS, ~6/10 workshop/arXiv preprint
quality.

## Budget & Audit

- GPU hours used: 0
- Codex MCP calls: **5** (1 research review, 1 claim audit, 2 paper review rounds, 1 citation audit)
- Tests: 37 → 37 (no regression)
- Commits in this pipeline: 5 (research feat, fix, narrative, paper, this report)
- Files added: **17** (1 module, 1 test, 3 experiments, 2 reports, 1 plan, 3 figures, 1 figure script, 1 paper improvement log, 1 main.tex, 7 sections, 1 bib, 1 math_commands, 3 round PDFs)
- Lines added: ~3500
- Lines removed: 0 in this pipeline (1 in claim-audit-driven citation pruning)
