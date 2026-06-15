<div align="center">

# borge-audit

**Self-hosted memory hygiene for LLM agents**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red?style=flat-square)]()
[![CPU only](https://img.shields.io/badge/compute-CPU%20only-orange?style=flat-square)]()
[![No cloud calls](https://img.shields.io/badge/data-stays%20local-22c55e?style=flat-square)]()

<br/>

*Find the bloat, contradictions, and stale facts hiding in your agent's memory — before they cost you.*

</div>

---

## What It Does

Every production LLM agent accumulates a memory store that grows unbounded. Over weeks, it fills with:

- **Bloat** — low-value chatter and redundant context that wastes tokens on every call
- **Contradictions** — conflicting facts the agent silently carries ("prefers vegetarian" alongside "loved the steak")
- **Duplicates** — the same fact stored 4 different ways
- **Stale entries** — instructions, credentials, and sprint plans that are months out of date

`borge-audit` scores every memory with the same **multi-factor value model** proven on LongMemEval (0.770 gold retention at κ = 0.30, blind regime) and surfaces a prioritised forget list — in under a minute, on a laptop, with no data leaving your infrastructure.

**Read-only and dry-run by default. Nothing is ever deleted automatically.**

---

## Installation

```bash
pip install borge_audit-0.1.0-py3-none-any.whl
```

> **No GPU required.** Sentence-transformer embeddings download once to your local HuggingFace cache; every subsequent run is fully offline.

---

## Usage

```bash
# Audit a JSON memory dump (Mem0 / Zep / pgvector / custom)
borge-audit memories.json

# Audit a Markdown memory file (Claude Code CLAUDE.md, Hermes, Cursor, etc.)
borge-audit memories.md

# More aggressive keep budget (keep only 30 %)
borge-audit memories.json --budget 0.3

# Add an LLM judge for higher-precision contradiction detection
borge-audit memories.json \
    --llm-endpoint http://localhost:11434/v1 \
    --llm-model qwen3:8b
```

**Outputs** (written next to the input file):

| File | Contents |
|------|----------|
| `memories.audit.md` | Human-readable report — executive summary, tiered forget list, contradiction pairs, near-duplicate clusters, stale entries, token-savings estimate |
| `memories.audit.forget.json` | Machine-readable forget list for your deletion workflow — reversible, review before applying |

---

## How It Scores

Every memory receives a single value score:

$$V(m) = \sum_{i=1}^{7} w_i \, f_i(m)$$

Seven interpretable factors — the same ones that govern human long-term retention:

| # | Factor | Signal |
|---|--------|--------|
| 1 | **Emotional intensity** | Arousal × valence from linguistic rules |
| 2 | **Goal relevance** | Semantic distance to the agent's task centroid |
| 3 | **Value alignment** | Proximity to the agent's stated principles (SOUL.md) |
| 4 | **Self / user relevance** | Self-reference effect — how much is this *about* the user |
| 5 | **Task utility** | LLM-gated usefulness score (optional; 0 in static pass) |
| 6 | **Reliability** | Role heuristic — user-stated facts outrank assistant assertions |
| 7 | **Usage history** | Retrieval count, saturating in [0, 1] |

Weights were learned on LongMemEval (blind regime) by gradient-free optimisation. The top three: **reliability (0.64)**, **emotional intensity (0.55)**, **self-relevance (0.23)**. Goal similarity is correctly zeroed out — it is unavailable at consolidation time.

---

## Privacy

- All computation runs **in your infrastructure** — no memory record, embedding, or report fragment is transmitted to any external service.
- The optional LLM judge (`--llm-endpoint`) calls a **locally-hosted** model you specify; by default no LLM is invoked at all.
- `borge-audit` is read-only: it never writes to, modifies, or deletes your memory store.

---

## Sample Report

```
## Executive Summary

- 23 memories analysed.
- Recommended forget set (moderate tier, keep 50 %): 8 memories flagged.
- 2 candidate contradiction(s) for human review.
- 0 near-duplicate cluster(s) detected.
- 1 memory older than staleness threshold.
- Estimated savings: ~4 920 tokens/month ($0.015/month).

Nothing in this report has been changed. Every item below is a
recommendation — no memory is touched.
```

Full sample: [`demo/sample_audit_report.md`](demo/sample_audit_report.md)

---

## Input Formats

**JSON** — one object per memory row, fields mapped automatically:

```json
[
  {
    "id": "mem-001",
    "text": "User is allergic to penicillin.",
    "role": "user",
    "timestamp": "2026-01-10T14:30:00Z",
    "retrieval_count": 7
  }
]
```

**Markdown** — heading-split, bullet-split, dated-log, or auto-detected:

```markdown
---
modified: 2026-05-20T09:00:00Z
tags: [profile]
---

## Allergy
I am allergic to penicillin.

## Deadline
Project deadline is March 15.
```

---

## Citation

If you use the underlying value model in research, please cite the paper:

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

Proprietary — for authorised use only. Contact [maxzhibao@gmail.com](mailto:maxzhibao@gmail.com) for licensing.
