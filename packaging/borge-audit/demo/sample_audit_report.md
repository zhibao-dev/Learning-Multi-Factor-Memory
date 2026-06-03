# Borge Memory Audit

Source dump: `packaging/borge-audit/demo/sample_agent_memory.md`  ·  as of `2026-06-03T13:57:56+00:00`

## Executive Summary

- **23** memories analysed.
- Recommended forget set (`moderate` tier, keep 50%): **8** memories flagged as low-value candidates to forget.
- **2** candidate contradiction(s) flagged for human review.
- **0** near-duplicate cluster(s) detected.
- **1** memory(ies) older than the staleness threshold.
- Estimated savings: **~4920 tokens/month** (**$0.0148/month**), under the assumptions in the Methodology appendix.

_Nothing in this report has been changed. Every item below is a recommendation or candidate for your review — no memory is touched._

## Bloat / Forget Candidates

Memories ranked by the paper2 multi-factor value model (LongMemEval-validated). Lower value = earlier forget candidate. This ranking is **informed by** that model but is **not guaranteed** on your data — treat it as a prioritised review queue, not a verdict.

Tiered forget counts (each tier = a keep-fraction):

| Tier | Keep fraction | Forget candidates |
| --- | --- | --- |
| safe | 70% | 7 |
| moderate (chosen) | 50% | 11 |
| aggressive | 30% | 16 |

3 memory(ies) that appear as contradiction candidates were excluded from this forget list and deferred to the Pollution review below.

Top forget candidates in the chosen `moderate` tier (lowest value first):

| Memory id | Role | Value | Text |
| --- | --- | --- | --- |
| `sample_agent_memory#database` | user | 0.619 | ## Database Primary datastore is PostgreSQL 15. Connection pooling via PgBounce… |
| `sample_agent_memory#_preamble` | user | 0.631 | # Project memory — ACME API |
| `sample_agent_memory#ci` | user | 0.631 | ## CI CI runs on GitHub Actions. Required checks: lint, typecheck, unit tests. |
| `sample_agent_memory#old-sprint-plan-2025-02-10` | user | 0.634 | ## Old sprint plan (2025-02-10) Current sprint focuses on the billing rewrite; … |
| `sample_agent_memory#testing` | user | 0.635 | ## Testing Integration tests hit a throwaway Postgres in CI; never the prod DB. |
| `sample_agent_memory#auth` | user | 0.635 | ## Auth Auth uses Auth0 with JWT access tokens, 15-minute expiry, refresh rotat… |
| `sample_agent_memory#secrets` | user | 0.636 | ## Secrets Secrets live in GCP Secret Manager. Never hardcode keys; the user is… |
| `sample_agent_memory#code-style` | user | 0.638 | ## Code style 2-space indent, no semicolons omitted, prefer named exports, no d… |

## Pollution (Contradictions, Duplicates, Stale)

### LLM-verified contradictions

**These pairs survived a local NLI prefilter AND an LLM-judge confirmation step.** Nothing here is auto-resolved, merged, or forgotten; `likely stale` is only a hint (the older of the two timestamps, unless the judge says otherwise). LLM-verified; review still recommended — a human decides.

| Memory A | Memory B | Score | Likely stale | Texts |
| --- | --- | --- | --- | --- |
| `sample_agent_memory#tech-stack` | `sample_agent_memory#stack-migration-2026-04-12` | 1.00 | `sample_agent_memory#tech-stack` | ## Tech stack The user prefers TypeScri… / ## Stack migration (2026-04-12) Team de… |
| `sample_agent_memory#deployment-target` | `sample_agent_memory#infra-change-2026-05-02` | 1.00 | `sample_agent_memory#deployment-target` | ## Deployment target Deploy the API to … / ## Infra change (2026-05-02) Moved off … |

Note: the NLI detector can over-fire on entity/name slots (e.g. two different names in the same role) — verify each candidate; a high score is not proof of a real conflict.

### Near-duplicate clusters

_No near-duplicate clusters detected._

### Stale memories

- `sample_agent_memory#old-sprint-plan-2025-02-10` (2025-02-10) — ## Old sprint plan (2025-02-10) Current sprint focuses on t…

## Safety

This audit is **read-only and dry-run**. The input dump was not modified and no memory was touched. The companion `*.forget.json` is a reversible list of forget *candidate* ids only — applying it is entirely at your discretion and can be undone.

## Methodology

**Forgetting.** Memories are scored by the paper2 multi-factor value model (seven interpretable factors; weights from the LongMemEval blind fit). The model is validated on LongMemEval but its ranking is **not guaranteed** on your specific data — use the tiers as a prioritised review queue.

Note: markdown sources usually lack per-item usage counts, so the bloat ranking is best-effort; contradiction, duplicate, and stale detection are unaffected.

Weights used: emotion=0.55, goal_relevance=0, value_alignment=0, self_relevance=0.23, task_utility=0, reliability=0.64, usage=0.1.

Weights are tunable per your business — the defaults are fit to a general benchmark and may not match your scenario.

Assistant-authored memories receive a lower reliability prior (0.4 vs 0.7 for user-authored), and reliability is the most heavily weighted value factor — so on dumps mixing user and assistant turns, authorship is a strong driver of the forget order. Review assistant-authored rows on their own merits rather than by rank alone.

**Contradictions.** A local NLI cross-encoder flags conflicting assertions. This is a **product feature for human review**, not a paper2-proven result; candidates are never auto-resolved or auto-applied.

**Savings assumptions:**
- Assumed retrieval frequency: **30.0** re-injections/month.
- Price per 1k tokens: **$0.003**.
- Tokenizer: **char4**.
- Assumes the forgotten memories would otherwise be re-injected 30.0 times/month; savings scale linearly with this assumed retrieval frequency.

Bloat, duplicate, and stale detection run 100% locally. The LLM-judge step sent the text of the contradiction *candidate* pairs above (NLI survivors only, not your whole dump) to the LLM endpoint you configured — your own/local model, never a Borge-hosted API.
