# borge-audit: markdown ingest + per-customer weights — Design

> Branch: `multi-factor-eval-business`. Validated via brainstorm 2026-06-01.
> Scope: **A (markdown adapter)** + **B (manual/configurable weights)**. Both make the
> audit *usable per customer*, not just *connectable*. Orthogonal concerns, one TDD round.

## Why

1. **A — format reality.** Target customers (Claude Code, Hermes, Codex) store agent
   memory in **markdown**, not JSON. The audit currently only ingests canonical JSON.
   Format is orthogonal to the value logic (it's pure ingest), so the fix is a markdown
   front-end that produces the same `MemoryRecord` list — the pipeline is unchanged.
2. **B — 千人千面.** A single shipped weight vector (the LongMemEval blind fit) is wrong
   for most workloads (companion vs coding vs support weight factors differently). Weights
   must be settable per customer's business. This is also the monetization ladder.

## A — markdown ingest adapter

**New module** `borge/audit/ingest_md.py` → `load_markdown_dump(path, *, split="auto") -> list[MemoryRecord]`.
Produces the SAME `MemoryRecord(id, text, timestamp, role, metadata)` as `ingest.load_dump`,
so `build_audit` and the whole value pipeline are unchanged downstream.

**Split modes** (`--md-split heading|bullet|dated|auto`, default `auto`):
- `heading` — each `##`/`###` section = 1 record (CLAUDE.md / AGENTS.md style).
- `bullet` — each top-level `-`/`*` list item = 1 record (fact-list style).
- `dated` — each dated block (`## 2026-05-30`, `- [2026-05-30] ...`) = 1 record (append-log style).
- `auto` — detect: dated headings present → `dated`; else mostly bullets → `bullet`; else `heading`.

**YAML frontmatter** (`---\n...\n---` at file top): parse into per-file metadata
(`created`/`modified`/`timestamp`/`tags`/`role`/`retrieval_count` if present). Standard
markdown metadata mechanism (Obsidian / agent memory).

**Timestamp recovery chain** (first hit wins):
1. record-level inline/heading date (dated mode)
2. frontmatter `modified`/`created`/`timestamp`
3. file mtime (whole-file fallback)
4. (future) `git blame` per-block date — deferred (subprocess, fragile).
Missing → empty string → stale/recency degrade gracefully (the value path already
handles missing fields via `.get(f, 0)`).

**Per-record fields:** `id` = `<filestem>#<heading-slug or index>` (stable across runs);
`text` = block content; `timestamp` = recovered (or ""); `role` = frontmatter role else
`"user"` (markdown memory is usually agent-curated); `metadata` = `{source_file, heading_path, tags?}`.

**CLI routing:** `.md` → `load_markdown_dump`, `.json` → `load_dump` (or explicit `--format md|json`).

### Honest scope on markdown (from brainstorm)

- Text + recovered timestamp ⇒ **contradiction / dedup / stale / recency** all work fully.
- `usage` (retrieval_count) is a runtime recall stat, usually **absent** from a static `.md`
  (unless frontmatter has it). `usage` was the value model's strongest bloat separator, so
  **bloat-by-value is best-effort on markdown** — recency does NOT replace it (recency separates
  old/new, not trivial/substantive). The report says so honestly.
- **Deferred (NOT this plan):** a markdown-native "triviality" bloat factor (short/low-info =
  bloat). It would touch `MemoryValue.FACTORS` (the vendored core) — out of A+B scope; revisit
  after real markdown audits show it's needed.

## B — per-customer / configurable weights

Three-tier ladder (maps to the monetization ladder):

| tier | mechanism | who sets | monetization | this plan? |
|------|-----------|----------|--------------|-----------|
| 0 scenario presets | named weight profiles | customer picks | free | **deferred** — crystallize from real audits, don't guess |
| 1 manual/fixed | `--weights file.json` | you (in the paid audit) | audit service | **YES** |
| 2 learned/dynamic | `learn_weights` on their data | automatic | pro module (license) | **deferred** — pro module, later |

**Build now = tier 1 only.** `build_audit(..., weights: dict | None = None)` →
`mv = default_memory_value(weights)` (the factory already merges an override over
`SHIPPED_DEFAULT`). CLI `--weights path.json` loads the dict. The report's Methodology/header
shows the exact weights used + a "tunable per your business — defaults may not fit your
scenario" note (also the upsell hook). Presets + learned are later tiers.

## Anchor points (verified 2026-06-01)

- `borge/audit/ingest.py`: `MemoryRecord(id,text,timestamp,role,metadata)`, `load_dump(path)`.
- `borge/audit/report.py::build_audit(dump_path, *, now_iso, soul_centroid=None, budget=0.5,
  retrieval_freq=30.0, price_per_1k=0.003)` — calls `load_dump` + `default_memory_value()`
  (no override) + `_render_markdown(...)`.
- `borge/memory/value.py`: `SHIPPED_DEFAULT` dict, `default_memory_value(override=None)`.
- `borge/audit/__main__.py`: argparse `dump, --soul, --budget, --retrieval-freq, -o`.

## Out of scope (YAGNI)

scenario presets · learned-weight (pro) · triviality factor · git-blame timestamps ·
live/online memory · non-markdown/non-JSON formats (CSV/SQLite/vector) — all deferred
until a paying customer pulls them.
