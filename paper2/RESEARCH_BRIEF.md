# RESEARCH_BRIEF — Sister Paper: Multi-Factor Memory Value for Agentic Agents

**Relationship to self-FEP paper**: sister / superset. self-FEP stays as-is
(CogSci-flavoured, SRE + FEP). This paper drops FEP, reframes the same
substrate as an engineering + decision-theory contribution for agentic agents.

## Core claim

A **multi-factor memory value function** estimates each memory's expected
utility to future agent behaviour, and that single scalar uniformly controls
**encoding depth, forgetting, and retrieval**. Multi-factor value beats
similarity-only / recency-only / emotion-only / LLM-judge baselines on
long-horizon task performance and preference consistency.

## Value function

```
V(m) = Σ_i  w_i · factor_i(m)
```

Seven candidate factors:
1. emotional intensity        |V|·A          (have: emotion_resistance)
2. goal relevance             cos(m, active_goal)        NEW
3. value-alignment            SOUL.md value match         NEW (ValueSystem exists)
4. self / user relevance      cos(m, μ_self / μ_user)     (have: self-relevance)
5. task utility               marginal task-success contribution   NEW
6. reliability / source       provenance + confidence     NEW
7. usage history              retrieval_count, recency    (have: usage, recency)

## Theory base (NO FEP)

- Value-directed remembering (Castel) — humans preferentially retain high-value items
- Levels of processing (Craik & Lockhart) — self/goal/value relevance = deeper processing
- Emotional arousal & consolidation (McGaugh)
- Adaptive memory / rational analysis (Anderson & Schooler 1991) — memory tracks need-probability
- RL / expected utility — value = marginal contribution to future task success, user satisfaction, error avoidance

## Keep from self-FEP

- multi-factor forget formula
- unified encode/forget/retrieve under one scalar
- emotion × self/value interaction
- ablation methodology (drop-one-factor, bootstrap CI)
- baselines (similarity / recency / emotion only) + ADD LLM-judge baseline
- agent-memory substrate (BorgeAgent)

## Replace from self-FEP

- M_self = (μ_self, π_self): no longer FEP self model → just a self-relevance feature
- π_self: rename `confidence` / `identity_stability` or delete
- free_energy / prediction-error-variance: DELETE unless rigorously derived
- focus shifts: "explain human SRE" → "build agentic memory valuation system"

## New core contributions

1. Multi-factor memory value function for expected utility to future behaviour
2. Unify value → encoding depth + forget risk + retrieval rank
3. Beat similarity/recency/emotion-only on long-horizon agent tasks
4. Interpretable ablation: each factor's marginal contribution
5. A deployable memory-eval framework for agentic agents

## LOCKED DECISIONS

- **Fork A = A2**: weights w_i LEARNED from downstream task return (credit
  assignment). Value function fit to maximise task success, not hand-set.
- **Fork B = B2**: LongMemEval (Wu et al., ICLR 2025) public benchmark.
  500 questions, multi-session chat histories, 5 abilities
  (info-extraction, multi-session reasoning, temporal, knowledge-update,
  abstention). Task return = QA correctness → trains w_i.
- **Target = arXiv** preprint (then COLM main as paper venue later).

## Build plan (staged)

### Stage A — machinery (no API, this session)
1. Add 4 factor columns: goal_relevance, value_alignment, task_utility, reliability
2. `MemoryValue` class: V(m) = Σ w_i·factor_i, weights as a vector
3. Unify V → encoding depth + forget risk + retrieval rank (replace the
   5 fixed resistances with one learned-weighted value)
4. Weight-learning harness (A2): optimise w to maximise a task-return
   signal. Gradient-free (CMA-ES / coordinate ascent / REINFORCE-style)
   since the encode→forget→retrieve→answer pipeline is non-differentiable.
5. Validate harness on a SYNTHETIC proxy task (deterministic, free) —
   prove learned w recovers the planted optimal weighting.

### Stage B — real eval (NEEDS API budget + data download)
6. LongMemEval loader/adapter
7. Run agent over LongMemEval under: learned-V, similarity-only,
   recency-only, emotion-only, LLM-judge baselines
8. Report QA accuracy + preference-consistency per config + ablation

## Status: forks locked. Stage A buildable now; Stage B gated on API + data.
