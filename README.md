<div align="center">

<img src="assets/title.svg" alt="Learning Multi-Factor Memory" width="860"/>

<br/>

[![arXiv](https://img.shields.io/badge/arXiv-2606.12945-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/pdf/2606.12945)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![CPU only](https://img.shields.io/badge/compute-CPU%20only-orange?style=flat-square)]()
[![No API calls](https://img.shields.io/badge/API%20calls-none-blueviolet?style=flat-square)]()

<br/>

*What should an AI agent remember — and what should it forget?*

[English](README.md) · [中文](README.zh.md) · [日本語](README.ja.md) · [Português](README.pt.md)

</div>

---

## Vision

Every AI agent deployed over days or weeks accumulates an interaction history far larger than any context window. Today's systems answer with one of two heuristics: **keep the most recent** or **keep the most similar to the current query**. Both are wrong for the forgetting decision, which happens at consolidation time — before any future query exists.

Human memory is not single-factor. Decades of cognitive psychology show that what survives forgetting is governed by the **value** of an item: its emotional weight, its goal relevance, its reliability, its connection to the self. No single cue dominates — it's an ensemble.

**This project brings that insight to agentic memory.** We define a learned, multi-factor value function that governs all three memory operations — encoding depth, forget risk, and retrieval rank — through a single interpretable scalar. The weights are not hand-set; they are learned from a downstream objective. And the whole system runs on a laptop CPU with no API calls.

---

## Core Idea

### The Value Function

Every stored memory $m$ receives a score:

$$V(m) = \sum_{i=1}^{7} w_i \, f_i(m)$$

Seven interpretable factors, each a determinant of human retention:

| # | Factor | Cognitive grounding |
|---|--------|-------------------|
| 1 | **Emotional intensity** | Arousal modulates consolidation (McGaugh 2000) |
| 2 | **Goal relevance** | Value-directed remembering (Castel 2008) |
| 3 | **Value alignment** | Levels of processing (Craik & Lockhart 1972) |
| 4 | **Self / user relevance** | Self-reference effect (Rogers 1977) |
| 5 | **Task utility** | Adaptive memory (Anderson 1991) |
| 6 | **Reliability** | Provenance heuristic (user-stated > model-stated) |
| 7 | **Usage history** | Need-probability retrieval (Anderson & Milson 1989) |

One scalar $V(m)$ controls three operations:

```
encode  →  depth tier ∝ V(m)
forget  →  drop lowest V(m) under keep-budget κ
retrieve →  rank by V(m) + query match
```

### Learning the Weights

The encode → forget → retrieve → answer pipeline is non-differentiable, so weights $\mathbf{w}$ are learned by a **gradient-free optimiser** (random-restart hill-climb) that maximises gold-evidence retention under a fixed memory budget. No LLM calls required during training.

### The Oracle / Blind Distinction

A key methodological contribution: standard retention benchmarks score goal relevance against the **held-out evaluation question** — an oracle that peeks at the future query. This saturates at ~0.98 and measures retrieval, not forgetting. We evaluate in the **blind regime**: the consolidation policy never sees the evaluation question, matching how real systems operate.

```
oracle regime (unfair):  goal-only → 0.979   ← measures retrieval
blind  regime (honest):  goal-only → 0.286   ← actual forgetting quality
```

---

## Effectiveness — LongMemEval Benchmark

479 real multi-session chat cases, keep fraction κ = 0.30, blind regime:

<div align="center">

| Policy | Gold retention | vs. learned |
|--------|:--------------:|:-----------:|
| **Learned multi-factor (ours)** | **0.770 ± 0.011** | — |
| Uniform weights | 0.657 | −0.113 |
| Best single factor (self-relevance) | 0.518 | −0.252 |
| Recency baseline | 0.368 | −0.402 |

*Every gap's 95 % bootstrap CI is strictly above zero (20 resampled 50/50 splits).*

</div>

**Learned weights are interpretable** — reliability (0.64), emotional intensity (0.55), and self/user relevance (0.23) dominate; query-time goal similarity is correctly down-weighted (0.00) because it is unavailable at consolidation time.

**A neural MLP over the same factors ties the linear model (+0.003 ± 0.013)** — confirming factors combine near-additively and the interpretable linear value is not a compromise.

---

## Repository Layout

```
borge/memory/value.py        ← value function V(m) + default learned weights
borge/memory/value_net.py    ← MLP ablation model
borge/eval/                  ← LongMemEval annotator + retention metric
borge/audit/                 ← memory hygiene audit CLI
borge/affective/             ← emotional intensity factor extractor
borge/values/                ← SBert embedder, value system, soul parser
experiments/                 ← all paper experiments (CPU-only, no API)
results/                     ← cached experiment outputs
figures/                     ← paper figures (PDF)
paper/                       ← LaTeX source + compiled PDF
packaging/borge-audit/       ← standalone audit wheel
assets/                      ← readme assets
```

---

## Installation

```bash
git clone https://github.com/zhibao-dev/Learning-Multi-Factor-Memory.git
cd Learning-Multi-Factor-Memory

# Full dev install (experiments + audit + tests)
pip install -e ".[dev]"

# Audit CLI only
pip install -e ".[audit]"
```

> **No GPU required.** Embeddings use a local sentence-transformer (downloads once to your HuggingFace cache). All experiments run on CPU in minutes.

---

## Reproduce Paper Results

```bash
# Table 2 headline — blind forgetting on LongMemEval
python experiments/lme_blind_forgetting.py

# Per-case bootstrap CI (Figure 3)
python experiments/lme_bootstrap_ci.py

# Keep-fraction sweep κ ∈ {0.1 … 0.9} (Figure 2)
python experiments/lme_keepfrac_sweep.py

# Synthetic confound study — Table 1
python experiments/e2_mood_self_factorial.py

# MLP ablation
python experiments/lme_blind_forgetting.py --model mlp
```

All results are cached in `results/` — scripts are re-runnable without raw LongMemEval data.

---

## Memory Audit CLI

`borge-audit` finds bloat, contradictions, duplicates, and stale entries in any agent memory store. It runs entirely locally — no memory data ever leaves your machine. **Read-only and dry-run: nothing is ever deleted.**

```bash
# JSON dump (any agent: Mem0, Zep, pgvector, custom)
borge-audit memories.json

# Markdown dump (heading-split)
borge-audit memories.md

# Keep 30 % (more aggressive)
borge-audit memories.json --budget 0.3

# With optional LLM judge for contradiction precision
borge-audit memories.json --llm-endpoint http://localhost:11434/v1 --llm-model qwen3:8b
```

Output: `memories.audit.md` (human-readable report) + `memories.audit.forget.json` (reversible forget list).

---

## Use the Value Function Directly

```python
from borge.memory.value import default_memory_value, memory_factors

mv = default_memory_value()          # learned weights from the LongMemEval blind fit

# A memory row carries the factor inputs (valence/arousal → emotion,
# reliability, self-relevance, retrieval_count → usage, ...).
reliable_fact = {
    "emotional_valence": 0.3, "emotional_arousal": 0.6,
    "reliability": 1.0, "self_relevance_score": 0.8, "retrieval_count": 4,
}
chatter = {
    "emotional_valence": 0.0, "emotional_arousal": 0.1,
    "reliability": 0.4, "self_relevance_score": 0.1, "retrieval_count": 0,
}

v_fact    = mv.value(memory_factors(reliable_fact))   # high  → keep
v_chatter = mv.value(memory_factors(chatter))         # low   → forget candidate
print(round(v_fact, 3), round(v_chatter, 3))
```

---

## Citation

```bibtex
@article{chen2026multifactor,
  title   = {Learning What to Remember: A Cognitively Grounded
             Multi-Factor Value Model for Agentic Memory},
  author  = {Chen, Zhibao and Cheng, Qian},
  journal = {arXiv preprint arXiv:2606.12945},
  year    = {2026},
  url     = {https://arxiv.org/pdf/2606.12945}
}
```

---

## License

MIT © 2026 Zhibao Chen, Qian Cheng
