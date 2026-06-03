"""Markdown audit report + dry-run forget script for ``borge audit``.

Orchestrates the whole audit pipeline (ingest → factors → bloat → contradiction
→ hygiene → savings) into a single human-readable markdown report plus a
reversible, dry-run ``forget_script`` dict. **Nothing is ever deleted.**

Honesty rules (the product depends on not overclaiming):
  * The report only ever *recommends forgetting* / flags *candidates* — it never
    says a memory was "deleted" or "removed".
  * Contradiction detection is a PRODUCT feature for human review, not a
    paper2-proven result and never auto-resolved.
  * Forgetting is INFORMED BY the paper2 value model (LongMemEval-validated),
    but not guaranteed on the customer's own data.
  * Savings rest on the explicit assumptions returned by ``estimate_savings``.

One ``SBertEmbedder`` is constructed here and shared across annotate / contradiction
/ duplicates so the model loads once, not three times.
"""

from __future__ import annotations

from ..memory.value import default_memory_value
from ..values.self_model import SBertEmbedder
from .bloat import forget_ranking
from .contradiction import MIN_TOKENS as _CONTRA_MIN_TOKENS
from .contradiction import _body as _contra_body
from .contradiction import find_contradictions
from .factors import annotate_dump
from .hygiene import find_duplicates, find_stale
from .ingest import load_dump
from .ingest_md import load_markdown_dump
from .savings import estimate_savings


def _pick_tier(tiers: dict, budget: float) -> str:
    """Name of the tier whose keep-fraction is closest to ``budget``."""
    return min(tiers, key=lambda name: abs(tiers[name]["keep_frac"] - budget))


def build_audit(
    dump_path,
    *,
    now_iso,
    soul_centroid=None,
    budget: float = 0.5,
    retrieval_freq: float = 30.0,
    price_per_1k: float = 0.003,
    md_split: str = "auto",
    weights: dict | None = None,
    judge=None,
    llm_endpoint: str | None = None,
    llm_model: str | None = None,
    llm_key: str | None = None,
) -> dict:
    """Run the full audit pipeline and render a markdown report + forget script.

    Returns ``{"markdown": str, "forget_script": dict}``. The ``forget_script``
    is a dry-run dict (chosen-tier forget ids + provenance note); it never
    deletes or modifies anything. ``budget`` selects the headline forget set by
    picking the tier whose keep-fraction is closest to it.

    ``judge`` is an OPTIONAL contradiction precision filter (see
    :func:`borge.audit.contradiction.find_contradictions`). When omitted but
    ``llm_endpoint`` is given, an OpenAI-compatible judge is built from
    ``llm_endpoint`` / ``llm_model`` / ``llm_key`` (the customer's own LLM or a
    local Ollama). With a judge active the contradiction section is relabelled
    "LLM-verified"; without one the NLI-only "candidates for human review"
    wording is preserved verbatim.
    """
    if judge is None and llm_endpoint:
        from .judge import make_openai_judge

        judge = make_openai_judge(llm_endpoint, llm_model, llm_key)
    is_markdown = str(dump_path).lower().endswith(".md")
    if is_markdown:
        records = load_markdown_dump(dump_path, split=md_split)
    else:
        records = load_dump(dump_path)
    by_id = {r.id: r for r in records}

    # One embedder, shared across all three SBert-backed passes.
    embedder = SBertEmbedder()

    factors = annotate_dump(records, embedder=embedder, soul_centroid=soul_centroid)
    mv = default_memory_value(weights)
    ranking = forget_ranking(records, factors, mv)

    # Keep bloat out of contradiction NLI (cheaper, fewer false positives for
    # the judge). Skip = SAFE-tier bloat (highest-confidence, bottom ~30% by
    # value) that is EITHER ultra-short chatter OR assistant-authored
    # boilerplate. Crucially we never skip a substantive USER assertion just
    # because it scored low-value: contradictions here are conflicting user
    # facts, and on usage-count-free dumps a real assertion can rank low —
    # skipping it would suppress a genuine contradiction pair (breaking the
    # planted-pair / forget-disjointness contracts).
    safe_forget = set(ranking["tiers"]["safe"]["forget_ids"])
    skip_ids = {
        i for i in safe_forget
        if by_id[i].role != "user"
        or len(_contra_body(by_id[i].text).split()) < _CONTRA_MIN_TOKENS
    }
    contradictions = find_contradictions(
        records, embedder=embedder, judge=judge, skip_ids=skip_ids
    )
    duplicates = find_duplicates(records, embedder=embedder)
    stale_ids = find_stale(records, now_iso)

    chosen_name = _pick_tier(ranking["tiers"], budget)
    chosen_tier = ranking["tiers"][chosen_name]

    # Defer any id that is also a contradiction candidate to human review
    # (the Pollution section already flags it). Otherwise the report could
    # recommend forgetting one member of a pair while flagging the other as
    # likely-stale — contradictory advice on the same pair.
    contradiction_ids = {
        cid for p in contradictions for cid in (p.a_id, p.b_id)
    }
    forget_ids = [
        i for i in chosen_tier["forget_ids"] if i not in contradiction_ids
    ]
    n_deferred = len(chosen_tier["forget_ids"]) - len(forget_ids)
    forget_texts = [by_id[i].text for i in forget_ids]

    savings = estimate_savings(
        forget_texts,
        retrieval_freq=retrieval_freq,
        price_per_1k=price_per_1k,
    )

    markdown = _render_markdown(
        dump_path=dump_path,
        now_iso=now_iso,
        records=records,
        by_id=by_id,
        ranking=ranking,
        chosen_name=chosen_name,
        chosen_tier=chosen_tier,
        forget_ids=forget_ids,
        n_deferred=n_deferred,
        contradictions=contradictions,
        duplicates=duplicates,
        stale_ids=stale_ids,
        savings=savings,
        weights_used=mv.weights,
        is_markdown=is_markdown,
        judge_used=judge is not None,
    )

    forget_script = {
        "forget_ids": forget_ids,
        "generated_from": dump_path,
        "note": "dry-run; apply at your own discretion, reversible",
    }

    return {"markdown": markdown, "forget_script": forget_script}


def _trunc(text: str, n: int = 80) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def _render_markdown(
    *,
    dump_path,
    now_iso,
    records,
    by_id,
    ranking,
    chosen_name,
    chosen_tier,
    forget_ids,
    n_deferred,
    contradictions,
    duplicates,
    stale_ids,
    savings,
    weights_used,
    is_markdown=False,
    judge_used: bool = False,
) -> str:
    n = len(records)
    n_forget = len(forget_ids)
    keep_frac = chosen_tier["keep_frac"]
    lines: list[str] = []

    # ── 1. Executive Summary ──
    lines += [
        "# Borge Memory Audit",
        "",
        f"Source dump: `{dump_path}`  ·  as of `{now_iso}`",
        "",
        "## Executive Summary",
        "",
        f"- **{n}** memories analysed.",
        f"- Recommended forget set (`{chosen_name}` tier, keep {keep_frac:.0%}): "
        f"**{n_forget}** memories flagged as low-value candidates to forget.",
        f"- **{len(contradictions)}** candidate contradiction(s) flagged for human review.",
        f"- **{len(duplicates)}** near-duplicate cluster(s) detected.",
        f"- **{len(stale_ids)}** memory(ies) older than the staleness threshold.",
        f"- Estimated savings: **~{int(savings['tokens_saved_per_month'])} tokens/month** "
        f"(**${savings['usd_per_month']:.4f}/month**), under the assumptions in the Methodology appendix.",
        "",
        "_Nothing in this report has been changed. Every item below is a "
        "recommendation or candidate for your review — no memory is touched._",
        "",
    ]

    # ── 2. Bloat / Forget ──
    lines += [
        "## Bloat / Forget Candidates",
        "",
        "Memories ranked by the paper2 multi-factor value model (LongMemEval-validated). "
        "Lower value = earlier forget candidate. This ranking is **informed by** that "
        "model but is **not guaranteed** on your data — treat it as a prioritised "
        "review queue, not a verdict.",
        "",
        "Tiered forget counts (each tier = a keep-fraction):",
        "",
        "| Tier | Keep fraction | Forget candidates |",
        "| --- | --- | --- |",
    ]
    for name, tier in ranking["tiers"].items():
        marker = " (chosen)" if name == chosen_name else ""
        lines.append(
            f"| {name}{marker} | {tier['keep_frac']:.0%} | {len(tier['forget_ids'])} |"
        )
    lines.append("")
    if n_deferred:
        lines.append(
            f"{n_deferred} memory(ies) that appear as contradiction candidates "
            "were excluded from this forget list and deferred to the Pollution "
            "review below."
        )
        lines.append("")
    lines += [
        f"Top forget candidates in the chosen `{chosen_name}` tier (lowest value first):",
        "",
        "| Memory id | Role | Value | Text |",
        "| --- | --- | --- | --- |",
    ]
    value_by_id = ranking["value_by_id"]
    for mid in forget_ids[:10]:
        rec = by_id[mid]
        lines.append(
            f"| `{mid}` | {rec.role} | {value_by_id[mid]:.3f} | {_trunc(rec.text)} |"
        )
    if not forget_ids:
        lines.append("| _(none)_ | | | |")
    lines.append("")

    # ── 3. Pollution ──
    lines += [
        "## Pollution (Contradictions, Duplicates, Stale)",
        "",
    ]
    if judge_used:
        lines += [
            "### LLM-verified contradictions",
            "",
            "**These pairs survived a local NLI prefilter AND an LLM-judge "
            "confirmation step.** Nothing here is auto-resolved, merged, or "
            "forgotten; `likely stale` is only a hint (the older of the two "
            "timestamps, unless the judge says otherwise). LLM-verified; review "
            "still recommended — a human decides.",
            "",
        ]
    else:
        lines += [
            "### Candidate contradictions",
            "",
            "**These are candidates for human review — a product feature, NOT a "
            "paper2-proven result.** Nothing here is auto-resolved, merged, or forgotten; "
            "`likely stale` is only a hint (the older of the two timestamps). A human "
            "decides.",
            "",
        ]
    if contradictions:
        lines += [
            "| Memory A | Memory B | Score | Likely stale | Texts |",
            "| --- | --- | --- | --- | --- |",
        ]
        for p in contradictions:
            ta = _trunc(by_id[p.a_id].text, 40)
            tb = _trunc(by_id[p.b_id].text, 40)
            lines.append(
                f"| `{p.a_id}` | `{p.b_id}` | {p.contradiction_score:.2f} | "
                f"`{p.likely_stale_id}` | {ta} / {tb} |"
            )
    else:
        lines.append("_No candidate contradictions surfaced._")
    lines += [
        "",
        "Note: the NLI detector can over-fire on entity/name slots (e.g. two "
        "different names in the same role) — verify each candidate; a high score "
        "is not proof of a real conflict.",
        "",
        "### Near-duplicate clusters",
        "",
    ]
    if duplicates:
        for cluster in duplicates:
            ids = ", ".join(f"`{i}`" for i in cluster)
            sample = _trunc(by_id[cluster[0]].text, 60)
            lines.append(f"- {ids} — e.g. {sample}")
    else:
        lines.append("_No near-duplicate clusters detected._")
    lines += [
        "",
        "### Stale memories",
        "",
    ]
    if stale_ids:
        for mid in stale_ids:
            lines.append(f"- `{mid}` ({by_id[mid].timestamp}) — {_trunc(by_id[mid].text, 60)}")
    else:
        lines.append("_No stale memories beyond the age threshold._")
    lines.append("")

    # ── 4. Safety / dry-run ──
    lines += [
        "## Safety",
        "",
        "This audit is **read-only and dry-run**. The input dump was not modified "
        "and no memory was touched. The companion `*.forget.json` is a "
        "reversible list of forget *candidate* ids only — applying it is entirely "
        "at your discretion and can be undone.",
        "",
    ]

    # ── 5. Methodology appendix ──
    assumptions = savings["assumptions"]
    lines += [
        "## Methodology",
        "",
        "**Forgetting.** Memories are scored by the paper2 multi-factor value model "
        "(seven interpretable factors; weights from the LongMemEval blind fit). The "
        "model is validated on LongMemEval but its ranking is **not guaranteed** on "
        "your specific data — use the tiers as a prioritised review queue.",
        "",
    ]
    if is_markdown:
        lines += [
            "Note: markdown sources usually lack per-item usage counts, so the bloat "
            "ranking is best-effort; contradiction, duplicate, and stale detection are "
            "unaffected.",
            "",
        ]
    lines += [
        "Weights used: "
        + ", ".join(f"{f}={w:g}" for f, w in weights_used.items())
        + ".",
        "",
        "Weights are tunable per your business — the defaults are fit to a general "
        "benchmark and may not match your scenario.",
        "",
        "Assistant-authored memories receive a lower reliability prior (0.4 vs 0.7 "
        "for user-authored), and reliability is the most heavily weighted value "
        "factor — so on dumps mixing user and assistant turns, authorship is a "
        "strong driver of the forget order. Review assistant-authored rows on their "
        "own merits rather than by rank alone.",
        "",
        "**Contradictions.** A local NLI cross-encoder flags conflicting assertions. "
        "This is a **product feature for human review**, not a paper2-proven result; "
        "candidates are never auto-resolved or auto-applied.",
        "",
        "**Savings assumptions:**",
        f"- Assumed retrieval frequency: **{assumptions['retrieval_freq']}** re-injections/month.",
        f"- Price per 1k tokens: **${assumptions['price_per_1k']}**.",
        f"- Tokenizer: **{assumptions['tokenizer']}**.",
        f"- {assumptions['note']}",
        "",
        (
            "Bloat, duplicate, and stale detection run 100% locally. The "
            "LLM-judge step sent the text of the contradiction *candidate* pairs "
            "above (NLI survivors only, not your whole dump) to the LLM endpoint "
            "you configured — your own/local model, never a Borge-hosted API."
            if judge_used
            else "All memory processing is local; your memory data is not sent "
            "to any external API by this audit."
        ),
        "",
    ]

    return "\n".join(lines)
