# Multi-Factor-Value-Memory

**A cognitively grounded multi-factor value model for agentic memory.**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2606.12945-b31b1b)](https://arxiv.org/pdf/2606.12945)

---

## Paper

**Learning What to Remember: A Cognitively Grounded Multi-Factor Value Model for Agentic Memory**
Zhibao Chen, Qian Cheng — arXiv 2026
[https://arxiv.org/pdf/2606.12945](https://arxiv.org/pdf/2606.12945)

---

## What is this?

Long-running LLM agents accumulate histories far larger than any context window. Production systems forget by recency or semantic similarity to the current query — both are mis-specified: the forgetting decision is made at consolidation time, before the future query is known.

We propose a **multi-factor memory value function**:

```
V(m) = Σ wᵢ fᵢ(m)
```

over seven interpretable factors drawn from cognitive psychology:

| Factor | Source |
|--------|--------|
| Emotional intensity | McGaugh (2000) — emotional consolidation |
| Goal relevance | Value-directed remembering |
| Value alignment | Levels of processing |
| Self / user relevance | Self-reference effect |
| Task utility | Adaptive memory |
| Reliability | Provenance heuristic |
| Usage history | Need-probability (Anderson 1991) |

A **single scalar** V(m) uniformly controls encoding depth, forget risk, and retrieval rank. Weights are learned from a downstream objective by a gradient-free optimiser — not hand-set.

**Key result (LongMemEval, blind regime):**

| Policy | Gold retention |
|--------|---------------|
| **Learned multi-factor** | **0.770 ± 0.011** |
| Uniform weights | 0.657 |
| Best single factor | 0.518 |
| Recency | 0.368 |

All experiments run on a single CPU with no API calls.

---

## Repository layout

```
borge/memory/value.py        ← core value function + default weights
borge/memory/value_net.py    ← MLP ablation (ties linear model)
borge/eval/                  ← LongMemEval annotator + retention metric
borge/audit/                 ← memory hygiene audit CLI (bloat/contradiction/duplicate/stale)
borge/affective/             ← emotional intensity factor (signal extractor)
borge/values/                ← SBert embedder, soul parser, value system
experiments/                 ← all paper experiments (CPU-only, no API)
results/                     ← cached experiment outputs (JSON)
figures/                     ← paper figures
paper/                       ← LaTeX source + compiled PDF
packaging/borge-audit/       ← standalone pip-installable audit wheel
```

---

## Install

```bash
git clone https://github.com/zhibao-dev/Multi-Factor-Value-Memory.git
cd Multi-Factor-Value-Memory
pip install -e ".[dev]"
```

---

## Reproduce paper results

```bash
# Blind forgetting experiment (Table 2 headline)
python experiments/lme_blind_forgetting.py

# Bootstrap CI (all gaps above zero)
python experiments/lme_bootstrap_ci.py

# Keep-fraction sweep (Figure 2)
python experiments/lme_keepfrac_sweep.py

# Synthetic confound study (Table 1)
python experiments/e2_mood_self_factorial.py
```

All results cached in `results/` — scripts re-run from scratch without LongMemEval access.

---

## Memory audit CLI

Find bloat, contradictions, duplicates, and stale entries in any agent memory store. Read-only and dry-run — nothing is ever deleted.

```bash
pip install -e ".[audit]"
borge-audit memories.json          # → report.md + forget-list.json
borge-audit memories.md            # markdown memory dump supported
```

See `packaging/borge-audit/README.md` for full docs.

---

## Citation

```bibtex
@article{chen2026multifactor,
  title   = {Learning What to Remember: A Cognitively Grounded Multi-Factor Value Model for Agentic Memory},
  author  = {Chen, Zhibao and Cheng, Qian},
  journal = {arXiv preprint arXiv:2606.12945},
  year    = {2026}
}
```

---

## License

MIT
