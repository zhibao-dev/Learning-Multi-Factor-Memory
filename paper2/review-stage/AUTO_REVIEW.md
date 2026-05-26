# Auto Review Loop — paper2: Multi-Factor Memory Value for Agentic Agents

Target: `paper2/main.tex` — "Learning What to Remember: A Multi-Factor Value
Function for Agentic Memory" (v0.1, 9 pages, compiles clean).
Reviewer: Codex MCP (`gpt-5.5`-class, xhigh). Difficulty: medium. MAX_ROUNDS=4.
Positive threshold: score ≥ 6/10 AND verdict ready/almost.

Evidence files (numbers must match):
- `results/lme_blind_forgetting.json` — blind vs oracle, 74 cases, 20 splits
- `results/lme_retention.json` — synthetic confound, 60/60 split

Started: 2026-05-26.

---

## Round 1 (2026-05-26)

### Assessment (Summary)
- Score: 6.5/10
- Verdict: Almost (ready for arXiv after fixing one false claim + softening)
- Key criticisms:
  1. [MUST] Missing `reliability_only` baseline → "≤0.39 any single factor" claim FALSE (reliability is largest learned weight; alone ≈0.541).
  2. [MUST] "learned from downstream task return" overstated — return used is retention proxy, not QA accuracy.
  3. [MUST] Synthetic interpretation inconsistent — only value_alignment suppressed (self=1.01, usage=0.63); "single factor cannot" not shown (no goal/task/rel single-factor baseline in synth).
  4. [MUST framing] Experiments test value ranking under a keep budget, not the full encode-forget-retrieve system.
  5. [MUST framing] "first-80 subset" reads broader than it is → call it a 74-case LongMemEval-S pilot.
  6. [strengthen] Stats descriptive; add paired split differences / win-rate.
  7. [MUST wording] "lower bound on full value function" too strong → conservative four-factor result.
- Evidence check: all reported numbers matched the JSONs; the only gap was the omitted reliability_only baseline (Codex estimated 0.541 — exact match on rerun).

### Reviewer Raw Response
<details><summary>Codex gpt-5.5 xhigh, round 1</summary>

Score: 6.5/10. Oracle/blind framing is a real contribution — cleanly exposes an eval artifact (policy seeing held-out question measures retrieval, not consolidation-time forgetting). Ready for arXiv: Almost; would not post as-is because one headline claim is unsupported/likely false.

Evidence check: blind learned 0.8106→0.811; uniform 0.6335→0.633 (suggest 0.634); recency 0.3798→0.380; oracle goal-only 0.9662→0.966; synthetic 1.00/0.6208/0.00 all match. CRITICAL: paper claims "≤0.39 every single factor" but JSON has no reliability_only despite reliability being a live factor and the largest learned weight; API-free recompute ≈0.541±0.051. Remove or rerun with all live single-factor baselines.

Ranked weaknesses + min fixes: (1) missing reliability_only — must-fix; (2) "learned from downstream task return" overstated (return = retention not QA) — must-fix wording; (3) synthetic suppress-confounds claim inconsistent (self 1.0081, usage 0.6291; only value_alignment suppressed) + "single factor cannot" not shown — must-fix; (4) experiments isolate budgeted retention, not full system — must-fix framing, value_forget_score policy would strengthen; (5) first-80 subset → call it 74-case pilot, run all 500 (API-free) to strengthen; (6) stats descriptive — add paired diffs / bootstrap; (7) "lower bound" too strong → conservative four-factor result. Verdict: Almost.
</details>

### Actions Taken (all 7 addressed)
- #1: Added `reliability_only` (+ paired diffs) to experiments/lme_blind_forgetting.py; reran 74-case pool (deterministic → all prior numbers identical). reliability_only = 0.541 ± 0.051. Corrected abstract, intro contribution 4, conclusion, Table 5 (new row), §4.2 result paragraph. Removed every "≤0.39 / any single factor" claim.
- #2: Softened "downstream task return" → "downstream objective (gold-evidence retention here; task-QA return in general)" in abstract, intro ×2, conclusion.
- #3: Rewrote synthetic interpretation — only value_alignment suppressed; self not suppressed; framed as separating-mixture sanity check, not "single factor cannot."
- #4: §4.2 now states it isolates the budgeted keep/drop decision (no answerer / full loop; retention not QA).
- #5: Reframed as a 74-case LongMemEval-S pilot; full-500 noted as API-free extension.
- #6: Added paired per-split diffs: learned−uniform +0.18±0.05, learned−reliability +0.27±0.06, learned−recency +0.43±0.07, ALL 100% win-rate over 20 splits. In §4.2 + JSON.
- #7: "lower bound" → "conservative, four-factor, API-free estimate" in L2.

### Results
- New JSON: results/lme_blind_forgetting.json now includes reliability_only (0.541±0.051) + paired_blind_diffs (all 100% win). Headline unchanged (learned 0.811±0.047).
- Recompiles clean: 9 pages, no undefined refs, no overfull >15pt.

### Status
- Continuing to Round 2 (re-review the corrected draft). Difficulty: medium.

## Round 2 (2026-05-26)

### Assessment (Summary)
- Score: 7.3/10 (arXiv/workshop calibration)
- Verdict: READY for arXiv — Yes
- All 7 Round-1 fixes confirmed landed correctly (reviewer verified each).
- Evidence re-check: blind learned 0.811±0.047, reliability_only 0.541±0.051, paired gaps uniform +0.177/reliability +0.270/recency +0.431 all win_frac=1.0, synthetic 1.00/0.6208 — all match JSON. Only nit: uniform 0.6335 should round to 0.634 (fixed).
- Remaining = polish nits (fixed) + would-strengthen (future work): full-500 run, QA-accuracy loop, keep-fraction sweep, per-case bootstrap, §3 anchor wording (clarified).

### Reviewer Raw Response
<details><summary>Codex gpt-5.5 xhigh, round 2 (fresh thread — round-1 thread expired on MCP reconnect)</summary>

Verdict: READY for arXiv, with tiny polish nits. Score 7.3/10 for arXiv/agents-memory workshop; for main COLM/ICML/NeurIPS still workshop-level because eval is retention-only on a 74-case pilot.

Evidence check: all headline numbers match JSON (blind learned 0.8106→0.811; reliability-only 0.5409→0.541; paired uniform 0.1771, reliability 0.2697, recency 0.4308, all win_frac=1.0; synthetic learned 1.0, uniform 0.6208, weights value_alignment=0.0889, self=1.0081). Only nit: uniform_V blind 0.6335 → conventional rounding 0.634, draft said 0.633.

Seven fixes: (1) reliability_only baseline LANDED; (2) downstream-return wording MOSTLY landed (soften two residual "task-success/task return" phrases); (3) synthetic interpretation LANDED; (4) ranking-vs-full-system LANDED; (5) 74-case pilot LANDED; (6) paired diffs LANDED; (7) lower-bound overclaim LANDED.

Remaining must-fix: none substantive (fix 0.633 rounding + soften two phrases). Would strengthen: full 500 cases; QA-accuracy loop; multiple keep fractions; per-case bootstrap/sign-test; clarify §3 "retrieval-time goal anchor". Bottom line: Ready for arXiv: Yes.
</details>

### Actions Taken (Round-2 nits, all applied)
- 0.633 → 0.634 (uniform_V) in abstract, intro, conclusion, §4.2 text + Table 5.
- Softened residual phrases: intro "task-success signal" → "the chosen objective"; discussion "fit to task return" → "fit to the chosen objective"; related-work "learned from task return" → "from a downstream objective".
- §3.2 retrieval paragraph: clarified V(m) ranking uses the regime-specific goal anchor (question under oracle, session topic under blind).
- Recompiles clean: 9 pages, no undefined refs, no overfull, no stale numbers.

### Status
- STOP — positive assessment (7.3/10, ready). Loop complete at Round 2/4.

---

## Final Summary

Score progression: Round 1 **6.5/10 (Almost)** → Round 2 **7.3/10 (READY: Yes)**.
The loop fixed one genuine falsifiable error (missing reliability_only baseline → false "≤0.39 any single factor" claim) and tightened honesty throughout (downstream-objective wording, synthetic interpretation, pilot framing, conservative-estimate limitation). Added paired per-split statistics (learned_V wins 100% of 20 splits vs every baseline). Draft is arXiv-ready for a COLM / agents-memory-workshop target; main-track would need the full-500 run + a QA-accuracy loop (both future work, both flagged in §6).

## Method Description
The method is a multi-factor memory value V(m) = Σ_i w_i f_i(m): a non-negative-weighted linear score over seven interpretable factors (emotional intensity, goal relevance, value alignment, self/user relevance, task utility, reliability, usage history). One scalar V(m) uniformly drives three memory operations: encoding depth (normalised V mapped to four levels-of-processing tiers), forgetting (forget = recency^0.7 · 1/(1+r) · 1/(1+βV), so high value resists forgetting), and retrieval rank. Weights w are learned, not hand-set: a gradient-free random-restart hill-climb maximises a downstream objective (gold-evidence retention here; task-QA return in general) over the live factors, since the encode→forget→retrieve→answer pipeline is non-differentiable. Data flow: turns → API-free factor annotation (SBert embeddings for goal/self relevance, lexical extractor for emotion, role heuristic for reliability) → V(m) per turn → top-κ kept under budget → gold-retention measured. The headline experiment contrasts an oracle goal anchor (cos to the held-out question; a retrieval ceiling) with a blind anchor (cos to the session topic; realistic consolidation-time forgetting), isolating the budgeted keep/drop decision.

## Would-Strengthen Backlog (future work, non-blocking)
1. Run all 500 LongMemEval-S cases (API-free; current is a 74-case pilot) — biggest credibility lift.
2. Close the loop to QA accuracy (needs answerer + judge LLM; unlocks the task_utility factor).
3. Keep-fraction sweep (show effect is not specific to κ=0.30).
4. Per-case bootstrap / sign-test CI (20 resampled splits are not independent benchmark samples).
