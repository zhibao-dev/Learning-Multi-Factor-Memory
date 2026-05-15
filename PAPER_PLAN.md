# PAPER_PLAN.md — Self-FEP Memory (v0.1)

**Venue**: NeurIPS 2026 (anonymous submission format)
**Page budget**: 9 main + unlimited refs + unlimited appendix
**Track**: Main conference (technical track)
**Title (proposed)**: *Self-FEP Memory: A Precision-Weighted Generative Self Drives Memory Encoding, Forgetting, and Retrieval*

## Claims-Evidence Matrix

| # | Claim | Evidence | Status |
|---|-------|---------|--------|
| C1 | Self can be formalised under FEP as `M_self = (μ_self, π_self)` with embedding-centroid mean + PE-variance-based precision | `borge/values/self_model.py`; Section 3 (Method) | implemented |
| C2 | Self-relevance enters memory forget_score as a multiplicative `self_resistance = 1/(1+λ·sr)` factor | `borge/memory/forgetting.py`; E1 result | implemented + tested |
| C3 | Mechanically wiring self_resistance into forget_score reproduces the Self-Reference Effect ordering (self_ref < semantic < phonemic < structural in forget-space) | E1 results: 10/10 seeds, SRE-proxy +0.337 | supported (mechanical) |
| C4 | The forget surface is multiplicative in (emotion, self), not additive | E2 raw-scale ΔAIC −9669, interaction coef +0.71 | supported |
| C5 | SRE magnitude monotonically increases with self-precision π_self, vanishing at π_self=0 | E3a results: monotone 0.000 → 0.085 across π ∈ [0,1] | supported |
| C6 | Same code substrate supports both standalone agent and Hermes plugin deployments; back-compat preserved | 37/37 tests pass before+after, BorgeAgent v0.2-cleaner tag | supported |
| L1 | (Limitation) Bag-of-tokens embedding cannot replicate SRE from raw stimuli without sentence-transformers | Forward-pass diagnostic; L1 in NARRATIVE | acknowledged in §6 |
| L2 | (Limitation) `π_self ∝ 1/(1+γ·Var[PE])` is engineering heuristic, not derived from variational free energy | self_model.py docstring; L2 in NARRATIVE | acknowledged in §6 |
| L3 | (Limitation) No human-data fit (Symons & Johnson 1997 d, CMR3 comparison); all results are agent-internal | NARRATIVE §6 | acknowledged in §6 |

## Section Plan (NeurIPS 9-page budget)

| # | Section | Length | Content |
|---|---------|-------:|---------|
| 1 | Introduction | ~0.75 pg | Motivation: 4 decades of SRE without unified computational/FEP account; CMR3 is emotion-only; FEP-self work has no memory tie. Contribution bullets: (i) the formalisation `M_self`, (ii) wiring into forget+retrieve, (iii) three CPU experiments showing the SRE direction, the multiplicative emotion×self interaction, and the monotone π_self attenuation. |
| 2 | Related Work | ~0.75 pg | Three threads: (a) memory models (CMR3, ACT-R, retrieval-augmented LLMs), (b) FEP-self (Friston, Apps & Tsakiris, Limanowski), (c) SRE empirical (Rogers, Symons & Johnson, recent). |
| 3 | Method | ~2.5 pg | 3.1 Generative self model. 3.2 Self-relevance as precision-weighted cosine. 3.3 Forget_score = recency × usage × imp_res × emotion_res × self_res. 3.4 Retrieval ranking with self_similarity term. 3.5 Self-token gating: μ_self updates only on first-person utterances. |
| 4 | Experiments | ~2.5 pg | 4.1 Setup (substrate, embedding, hyperparameters table). 4.2 E1 mechanical SRE replication (Table 1, Figure 2). 4.3 E2 Mood × Self factorial (Table 2, Figure 3). 4.4 E3 π_self ablation (Figure 4). |
| 5 | Discussion | ~1.0 pg | What the multiplicative emotion×self interaction predicts about clinical populations. Connection to encoding-specificity (Tulving) and reconsolidation (Nader). |
| 6 | Limitations | ~0.5 pg | Carry forward NARRATIVE §6 verbatim (5 named limitations). |
| 7 | Conclusion | ~0.25 pg | One paragraph. |
| — | References | unlimited | ~25 entries. |
| — | Appendix A | — | Full hyperparameter table, exact E1/E2/E3 reproduction commands. |
| — | Appendix B | — | Codex Round-1 review log (transparency). |

## Figure / Table Plan

| ID | Type | Source | Caption sketch |
|----|------|--------|----------------|
| Fig 1 | Architecture | TikZ (Phase 2b) | The Self-FEP Memory loop. User content → SelfModel update (gated by self-tokens) → consolidation writes sr → forget_score / retrieval use sr. |
| Fig 2 | Bar chart | `results/e1_sre_replication.json` | Mean forget_score per encoding condition, 10 seeds. SRE direction matches Rogers et al. (1977). |
| Fig 3 | 2×2 interaction plot | `results/e2_mood_self_factorial.json` | Emotion × Self cell means. Non-parallel lines show multiplicative interaction; ΔAIC = −9669 favours interaction. |
| Fig 4 | Line chart | `results/e3_pi_self_ablation.json` | E3a: SRE size vs π_self; monotone increase from 0 at π=0 to 0.085 at π=1. |
| Tab 1 | Numeric | `results/e1_*.json` | Per-condition mean forget_score (in §4.2). |
| Tab 2 | Numeric | `results/e2_*.json` | 2×2 cell means + OLS AIC comparison (in §4.3). |
| Tab 3 | Hyperparameters | self_model.py defaults | In Appendix A. |

## Citation Scaffold (≈25 entries)

**Memory models**:
- Cohen & Kahana 2022 — CMR3
- Howard & Kahana 2002 — TCM
- Polyn et al 2009 — CMR
- Anderson 2007 — ACT-R memory
- Sumers et al 2024 — Cognitive Architectures for Language Agents

**FEP-self / predictive self**:
- Friston 2010 — Free Energy Principle review
- Friston et al 2017 — Active inference: process theory
- Apps & Tsakiris 2014 — free-energy self
- Limanowski & Blankenburg 2013 — minimal self models
- Hohwy 2016 — self-evidencing
- Seth & Tsakiris 2018 — being a beast machine

**SRE empirical**:
- Rogers, Kuiper, Kirker 1977 — original SRE
- Symons & Johnson 1997 — SRE meta-analysis
- Conway 2005 — self-memory system
- Klein 2012 — self-reference and elaboration

**Encoding / forgetting theory**:
- Craik & Lockhart 1972 — levels of processing
- Ebbinghaus 1885 — forgetting curve
- Tulving 1972 — episodic/semantic
- Tulving & Thomson 1973 — encoding specificity
- Bower 1981 — mood and memory
- Nader & Hardt 2011 — reconsolidation review

**Adjacent agent memory work (2024-2025)**:
- LUFY (emotional arousal + surprise) 2025
- Nemori 2025
- ComoRAG 2025
- MemOS 2025

## Pipeline TODO (within paper-writing skill scope)

- [x] Phase 0: assurance=draft (default; AUTO_WRITE=true, no `--assurance` flag)
- [ ] Phase 1: PAPER_PLAN.md ← this file
- [ ] Phase 2: figures (matplotlib) for Fig 2, 3, 4
- [ ] Phase 2b: architecture diagram (TikZ inline)
- [ ] Phase 3: write main.tex + 7 section files + references.bib
- [ ] Phase 4: latexmk compile (now that MacTeX is installed)
- [ ] Phase 4.5: skip — no formal proofs/theorems beyond E2 OLS
- [ ] Phase 4.7: paper-claim-audit (numeric claims exist + raw results exist)
- [ ] Phase 5: auto-paper-improvement-loop (2 rounds Codex)
- [ ] Phase 5.5: rerun paper-claim-audit (mandatory submission gate at draft level — informational only)
- [ ] Phase 5.8: citation-audit (real cites exist → run; we'll record verdicts)
- [ ] Phase 6: PAPER_IMPROVEMENT_LOG.md + Final Report
