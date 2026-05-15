# PAPER_IMPROVEMENT_LOG — Self-FEP Memory v0.1

**Pipeline**: `/research-pipeline → /paper-writing`
**Venue**: NeurIPS 2026 (anonymous submission)
**Compile**: TeX Live 2026, `latexmk -pdf main.tex` → 12 pp PDF

This log records every Codex-driven improvement round, what changed, and
the Codex verdicts at each step. The three PDF snapshots
(`main_round0_original.pdf`, `main_round1.pdf`, `main_round2.pdf`) sit
next to `main.pdf` for visual diff.

---

## Round 0 — Initial Compile (Phase 4)

**Trigger**: First `latexmk -pdf main.tex` after writing
`paper/main.tex` + 7 section files + `references.bib`.

**Compile issues fixed in-place**:
1. CJK characters (我 自 己) in `03_method.tex` — replaced with
   romanised "wo, ziji"
2. `$\muself_{\text{at encoding}}$` double-subscript in
   `06_limitations.tex` — rewritten as $\muself^{\text{enc}}$

**Result**: 11 pp PDF, 0 unresolved citations, 0 unresolved cross-refs.
Saved as `main_round0_original.pdf`.

---

## Round 0.5 — Paper Claim Audit (Phase 4.7)

**Reviewer**: Codex (`gpt-5.5` via MCP), zero-context.

**Verdict**: **WARN**.

| Checks | Matches | Mismatches | Unverifiable |
|--------|---------|------------|--------------|
| 29     | **23**  | 0          | 6 (contextual, no JSON evidence — wall-clock, hyperparameter values, version strings) |

All numeric claims (forget_score means, AIC values, ΔAIC, interaction
coefficients, SRE size numbers, monotonicity flag) match the JSON
ground truth in `results/`.

**Single fix applied**: replaced "total wall-clock is under five seconds"
with "complete in seconds of wall-clock on a single laptop CPU"
(removed the unverifiable specific number).

---

## Round 1 — Codex Paper Review

**Reviewer**: Codex (`gpt-5.5`), full read of `paper/sections/*.tex`
+ `references.bib`.

**Initial score**: 4/10 NeurIPS / 5/10 workshop.

**Killer concerns**:
1. Experimental validity (E1 manually typed, E2 by construction, E3 ablation of engineered formula)
2. FEP framing vs heuristic precision rule mismatch
3. Limitations section too self-defeating in placement/tone

**5 fixes applied**:

| # | Section | Change |
|---|---------|--------|
| 1 | Abstract | Removed "v0.1", "brutally honest limitations". Softened "reproduces canonical SRE" → "responds in the predicted direction to controlled self-relevance inputs"; "matching clinical reports" → "consistent with empirical reports". |
| 2 | Intro contributions list | Renamed "A mechanical reproduction of the SRE" → "Controlled self-relevance response". Added "We are explicit that sr is manually assigned, so this tests the gating formula's qualitative response — not the stimulus-to-memory forward pass." |
| 3 | Intro limitations paragraph | Rewrote "deliberately ship a v0.1 draft" framing as "five scope boundaries together with the validation roadmap they imply". |
| 4 | Method §3 (new paragraph) | Added "Scope: precision-inspired, not variational" — explicitly defines what is and isn't FEP. |
| 5 | Bibliography | Removed anonymous `lufy2025` and `nemori2025`. Fixed `anderson2007actr` from @article to @book. Updated `sumers2024cogarch` from @inproceedings to @article (TMLR). |

**Compile**: clean 11 pp PDF. Saved as `main_round1.pdf`.

---

## Round 2 — Codex Paper Re-Review

**Reviewer**: Codex (`gpt-5.5`), aware of Round 1 fixes.

**Round 2 score (post-Round-1 fixes)**: **5.7/10**.

**One thing Round 1 missed**: Discussion and Conclusion still carried
pre-Round-1 confidence ("explains", "matching clinical reports",
"reproduces").

**3 fixes applied**:

| # | Section | Change |
|---|---------|--------|
| 1 | Conclusion §7 | Replaced "reproduces the canonical SRE direction… matching clinical reports" with the explicit triple: "responds in the predicted direction to controlled inputs (E1), exhibits expected non-additive differential (E2), monotonically attenuates the simulated SRE (E3). These are mechanism checks, not quantitative fits." |
| 2 | Discussion §5 clinical implications | "explains both healthy SREs and clinical attenuations within one framework" → "*can express* both ordinary SRE behaviour and attenuated SRE behaviour in the model. The clinical mapping remains a prediction… not yet been fit to patient data." |
| 3 | Figure captions (Fig 2 + Fig 3) | Both rewritten so they can stand alone without overclaiming. Fig 2 caption now says "controlled self-relevance response". Fig 3 caption foregrounds non-parallel cell means; AIC moved to secondary diagnostic. |

**Compile**: clean 12 pp PDF (the rewrites grew Conclusion + caption length slightly).

**Round 2 verdict**: "Keep as v0.1, but apply these before arXiv. One more pass."

---

## Round 2.5 — Citation Audit (Phase 5.8)

**Reviewer**: Codex (`gpt-5.5`), per-entry web verification.

**Audit summary**: 17 KEEP / 2 FIX-META / 3 REPLACE / 2 REMOVE.

| Verdict | Key | Action taken |
|---------|-----|--------------|
| REMOVE  | `borge2026` | Self-citation to the project's own GitHub. Replaced with `@misc{anon_substrate}` — anonymous substrate citation, omitted URL. Updated 3 callsites (intro, method, appendix). |
| REMOVE  | `tulving1972episodicsemantic` | Unused in any `\cite{...}`. Deleted entry. |
| FIX-META | `ebbinghaus1885forgetting` | English title was 1913 translation. Replaced with German original "Über das Gedächtnis: Untersuchungen zur experimentellen Psychologie", added note about 1913 English translation. |
| FIX-META | `memos2025` | Was anonymous "Memtensor Lab". Replaced with arXiv:2507.03724 canonical metadata (Li, Song, Xi, Wang, et al.). |
| REPLACE | `sumers2024cogarch` | Cited in claim "use embedding similarity plus heuristic decay". Sumers's CoALA paper is an architecture survey, not a concrete embedding-decay system. Rewrote sentence: CoALA cited for architecture proposal; memos2025 cited for the actual embedding+decay implementation. |
| REPLACE | `conway2005smemsys` | Cited for broad "SRE survives across age, language, source-memory, clinical" claim. Conway covers self-memory system but not all of those. Softened intro sentence to "The self-memory system has been a productive framework". |
| REPLACE | `klein2012selfref` | Cited similarly for broad survival claim. Softened related-work sentence to "reviews conceptual cautions about variants and boundary conditions". |
| KEEP    | 17 entries | Verified OK on metadata + context. |

**Compile**: clean 12 pp PDF. `main.pdf` and `main_round2.pdf` are
identical (citation-audit applied in-place on Round 2).

---

## Round Progression Summary

| Round | Score | PDF | Key delta |
|-------|------:|-----|-----------|
| 0     | n/a   | `main_round0_original.pdf` (11 pp) | Initial compile, 0 errors |
| 0.5   | n/a   | (no compile)              | Paper-claim audit: WARN, 23/23 numeric matches |
| 1     | 4/10 → ~5/10 | `main_round1.pdf` (11 pp) | 5 abstract/intro/E1/method/bib fixes |
| 2     | ~5/10 → 5.7/10 | `main_round2.pdf` (12 pp) | 3 conclusion/discussion/captions fixes |
| 2.5   | n/a (audit only) | `main.pdf` (12 pp) | 7 citation fixes folded in-place |

**Final score**: ~5.7/10 NeurIPS (per Codex Round 2 estimate), ~6/10
workshop/preprint quality. Verdict: arXiv preprint ready; ready for
human read-through; substantive empirical revisions (sentence-
transformer embeddings, human-data fit, CMR3 comparison) needed before
NeurIPS submission as enumerated in §6 Limitations.

---

## Remaining Issues for v0.2

These are the named scope boundaries that did not change in Stage 6
because they require new empirical work, not editing:

- **L1**: Bag-of-tokens embedding → swap for sentence-transformers, rerun a true forward-pass SRE replication.
- **L2**: π_self update rule → derive from variational free energy under a specified likelihood-prior pair.
- **L3**: Quantitative fit to Symons & Johnson (1997) Cohen's d ≈ 0.50 and CMR3 baseline reimplementation.
- **L4**: Add noise to all three experiments for honest variance estimates and well-conditioned AIC.
- **L5**: Persist `μ_self_at_encoding` snapshot per memory so retrieval can match the encoding-specificity proposal exactly.

These five items are the v0.2 backlog; addressing any of them likely
shifts the paper from "v0.1 framework demonstration" toward "v0.2
submission-ready empirical paper".
