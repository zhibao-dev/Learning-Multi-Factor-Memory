# Citation Audit Report — paper2

**Date**: 2026-05-26
**Bib file**: `paper2/references.bib`
**Reviewer**: Codex `gpt-5.5` xhigh, fresh thread, live web/DBLP/arXiv/CrossRef lookups
**Cited entries audited**: 16 (3 uncited entries handled separately)

## Summary
| Verdict | Count |
|---------|------|
| KEEP    | 13   |
| FIX     | 2    |
| REWORD (context) | 1 |
| REPLACE | 0    |
| REMOVE  | 0    |

Overall: **WARN** (metadata drift only; no hallucinated refs, no wrong-context refs left). All findings applied; clean rebuild, no undefined citations.

## Fixes applied

### FIX `memos2025` (metadata)
- Title `A memory operating system for AI systems` → **`MemOS: A Memory OS for AI System`** (verified against arXiv:2507.03724).
- Author list corrected/reordered (Li, Xi, Li, Chen, Chen, Song, … + `and others`).
- `primaryClass` cs.AI → **cs.CL**; added DOI `10.48550/arXiv.2507.03724`. Removed the "re-verify" note.

### FIX `castel2008value` (metadata)
- Year **2008 → 2007** (DOI `10.1016/S0079-7421(07)48006-9`, ScienceDirect online-first 2007).
- Added editors (Benjamin & Ross) + DOI; booktitle `The Psychology…` → `Psychology of Learning and Motivation`. Cite key label kept (`castel2008value`) since numerical natbib hides it.

### REWORD `bjork1994memory` (context)
- Was cited for "directed forgetting is a feature, not a bug" — the chapter is real (MIT Press, Metcalfe & Shimamura eds.) but its actual thrust is metamemory / desirable difficulties, not adaptive directed forgetting per se.
- Sentence reworded in §2: "the metamemory and desirable-difficulties tradition argues that forgetting and selective retention serve memory rather than merely degrade it." Reference kept.

## Uncited entries (resolved)
- `tulving1973encodingspec` — **pruned** (unused in paper2).
- `sumers2024cogarch` — **now cited** in §2 (cognitive-architecture view of language agents).
- `chen2026borge` — **now cited** in §7 (open-source substrate).

## All-clean (KEEP) entries
reimers2019sbert, wu2025longmemeval, park2023generative, lewis2020rag,
packer2023memgpt, zhong2024memorybank, craik1972lop, rogers1977sre,
mcgaugh2000consolidation, anderson1991adaptive, ebbinghaus1885forgetting,
sutton2018rl, hansen2016cmaes.

## Notes / optional (not blocking)
- `reimers2019sbert`: cite supports SBert generally, not the exact `all-MiniLM-L6-v2` checkpoint (WEAK, acceptable).
- Optional metadata adds skipped (park DOI/pages, etc.) — not required for arXiv.
- `wu2025longmemeval` confirmed ICLR 2025 (arXiv:2410.10813) — the most submission-sensitive entry, verified.
