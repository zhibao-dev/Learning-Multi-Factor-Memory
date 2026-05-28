"""
LongMemEval QA-accuracy harness for paper2 (multi-factor memory value).

The honest end-to-end test for paper2: do the turns a memory policy KEEPS
actually let an LLM answer the question? Gold-retention (lme_blind_forgetting.py)
measures whether the needle survives the forgetting pass; this harness asks the
downstream question — given ONLY the kept turns, can the question be answered
correctly?

API-FREE BY DESIGN. The LLM answer+judge step is NOT done here — it is driven
externally (Codex MCP) by the controller, because MCP calls can't be issued
from a python loop. So this file is the two API-free halves:

  --mode prep   read LongMemEval cases, rank each case's turns by a memory
                policy (learned_V / recency / uniform), keep the top-K, and
                write a tasks JSONL (question + kept-turn TEXT + gold per
                (case,policy)). No LLM.

  --mode score  read the controller's verdicts JSONL ({qid,policy,correct})
                → per-policy QA accuracy + paired learned-vs-recency contrast
                → results JSON. No LLM.

The middle step (answer from ONLY the kept context, judge vs gold, emit
verdicts) is the controller's job.

Ranking matches experiments/lme_blind_forgetting.py EXACTLY:
  - cache↔dataset joined by question_id (== qid), same order (case i ↔ line i)
  - blind-factor mapping: emotion→emotion, goal_blind→goal_relevance,
    self→self_relevance, reliability→reliability; the three API-free-inert
    factors (value_alignment, task_utility, usage) = 0.0
  - learned_V uses learned_weights_blind_mean from
    results/lme_blind_forgetting_full.json (the blind regime)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from borge.eval.longmemeval import load_longmemeval, flatten_to_messages  # noqa: E402
from borge.memory.value import MemoryValue  # noqa: E402

# The factors the API-free annotator can populate (mirrors
# lme_blind_forgetting.LIVE_FACTORS). value_alignment / task_utility / usage
# need an LLM / SOUL / access history, so they stay 0.0 — the inert three.
LIVE_FACTORS = ("emotion", "goal_relevance", "self_relevance", "reliability")

DEFAULT_LEARNED_JSON = "results/lme_blind_forgetting_full.json"


# ── cache → blind-factor mapping (matches lme_blind_forgetting._record_to_annotated) ──

def _cache_turn_to_blind_factors(t: dict) -> dict:
    """Map ONE cached turn onto the full MemoryValue factor dict, blind regime.

    EXACTLY mirrors lme_blind_forgetting._record_to_annotated's `factors_blind`:
        emotion        ← emotion
        goal_relevance ← goal_blind   (blind = session-topic centroid, no peek)
        self_relevance ← self
        reliability    ← reliability
    The three factors the API-free annotator can't populate stay 0.0.
    """
    return {
        "emotion":         t["emotion"],
        "goal_relevance":  t["goal_blind"],
        "value_alignment": 0.0,
        "self_relevance":  t["self"],
        "task_utility":    0.0,
        "reliability":     t["reliability"],
        "usage":           0.0,
    }


def load_cache_index(path: str | Path) -> list[dict]:
    """Read the factor cache JSONL into an ordered list of records.

    Order is preserved (line i ↔ dataset case i, same as load_from_cache in
    lme_blind_forgetting.py). Each record: {"qid": str, "turns": [..]}.
    """
    out: list[dict] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_learned_weights(path: str | Path) -> dict[str, float]:
    """learned_weights_blind_mean from the full blind-forgetting JSON."""
    d = json.loads(Path(path).read_text())
    w = d.get("learned_weights_blind_mean")
    if not w:
        raise SystemExit(f"no learned_weights_blind_mean in {path}")
    return {f: float(w.get(f, 0.0)) for f in LIVE_FACTORS}


# ── ranking policies ─────────────────────────────────────────────────────

def _rank_indices(
    turns: list[dict],          # per-turn blind-factor dicts (chronological)
    sidx: list[int],            # session_idx per turn (== cache sidx)
    policy: str,
    learned: dict[str, float],
    k: int,
) -> list[int]:
    """Return the indices of the top-K turns under `policy`.

    Indices are into the chronological turn list. Caller re-sorts the kept
    indices chronologically before emission.

    learned_V / uniform: rank by MemoryValue(weights).value(blind_factors) desc.
        Tie-break: later turn first (so a stable, recency-leaning tie-break,
        matching that higher original index = later in the haystack).
    recency: highest sidx first, tie-break later original turn — mirrors
        lme_blind_forgetting.recency_retention's `sorted(key=-timestamp_idx)`
        with a deterministic within-session tie-break by original order.
    """
    n = len(turns)
    if policy in ("learned_V", "uniform"):
        weights = (learned if policy == "learned_V"
                   else {f: 1.0 for f in LIVE_FACTORS})
        mv = MemoryValue(weights=weights)
        scored = [(mv.value(turns[i]), i) for i in range(n)]
        # sort by value desc, then later turn first (larger i) as tie-break
        scored.sort(key=lambda x: (-x[0], -x[1]))
        return [i for _, i in scored[:k]]
    if policy == "recency":
        # highest sidx first; within a session, later original turn first
        order = sorted(range(n), key=lambda i: (-sidx[i], -i))
        return order[:k]
    raise SystemExit(f"unknown policy: {policy}")


# ── prep mode ────────────────────────────────────────────────────────────

def run_prep(args) -> None:
    learned = load_learned_weights(args.learned_json)
    cache = load_cache_index(args.cache)
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    for p in policies:
        if p not in ("learned_V", "recency", "uniform"):
            raise SystemExit(f"unknown policy in --policies: {p}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # join cache ↔ dataset by ORDER (case i ↔ cache line i), keep first
    # n_cases usable cases (usable = ≥1 has_answer turn), matching the cache's
    # own usable-case ordering in lme_blind_forgetting.load_from_cache.
    n_target = args.n_cases
    rows_out: list[dict] = []
    # per-policy accumulators for the summary
    kept_counts: dict[str, list[int]] = {p: [] for p in policies}
    gold_present: dict[str, int] = {p: 0 for p in policies}
    n_usable = 0
    n_skipped_mismatch = 0

    ds_iter = load_longmemeval(args.data)
    for i, case in enumerate(ds_iter):
        if n_usable >= n_target:
            break
        if i >= len(cache):
            print(f"  [warn] dataset case {i} has no cache line; stopping join")
            break
        cache_rec = cache[i]
        # join sanity: qids must match position-for-position
        if str(cache_rec["qid"]) != str(case.question_id):
            raise SystemExit(
                f"cache↔dataset qid mismatch at index {i}: "
                f"cache={cache_rec['qid']!r} dataset={case.question_id!r} "
                f"(order assumption broken — abort rather than misalign)")

        msgs = flatten_to_messages(case)
        cache_turns = cache_rec["turns"]

        # per-case alignment assertion: skip + log on mismatch (never misalign)
        if len(msgs) != len(cache_turns):
            n_skipped_mismatch += 1
            print(f"  [skip] qid={case.question_id} len mismatch: "
                  f"text_turns={len(msgs)} cache_turns={len(cache_turns)}")
            continue

        # usable = has ≥1 has_answer turn (use cache flag; identical to dataset)
        if not any(t["has_answer"] for t in cache_turns):
            continue
        n_usable += 1

        T = len(msgs)
        K = min(args.keep, math.ceil(args.keep_frac * T))
        K = max(1, min(K, T))

        blind_factors = [_cache_turn_to_blind_factors(t) for t in cache_turns]
        sidx = [int(t["sidx"]) for t in cache_turns]

        for policy in policies:
            keep_idx = _rank_indices(blind_factors, sidx, policy, learned, K)
            # chronological order for the answerer: sort kept by (sidx, orig idx)
            keep_idx_chrono = sorted(keep_idx, key=lambda j: (sidx[j], j))
            kept = [{
                "sidx":    sidx[j],
                "role":    msgs[j]["role"],
                "content": msgs[j]["content"],   # verbatim, no truncation
            } for j in keep_idx_chrono]

            kept_counts[policy].append(len(kept))
            if any(cache_turns[j]["has_answer"] for j in keep_idx):
                gold_present[policy] += 1

            rows_out.append({
                "qid":          case.question_id,
                "policy":       policy,
                "question":     case.question,
                "gold":         case.answer,
                "n_turns_total": T,
                "n_kept":       len(kept),
                "kept":         kept,
            })

    with out_path.open("w") as f:
        for row in rows_out:
            f.write(json.dumps(row) + "\n")

    # ── summary ──────────────────────────────────────────────────────────
    print(f"\n=== prep summary → {out_path} ===")
    print(f"  data:           {args.data}")
    print(f"  cache:          {args.cache}")
    print(f"  learned weights (blind): " +
          ", ".join(f"{f}={learned[f]:.3f}" for f in LIVE_FACTORS))
    print(f"  usable cases:   {n_usable} (target {n_target})")
    if n_skipped_mismatch:
        print(f"  skipped (len mismatch): {n_skipped_mismatch}")
    print(f"  keep:           K = min({args.keep}, ceil({args.keep_frac}*T))")
    print(f"  rows written:   {len(rows_out)}  "
          f"({n_usable} cases × {len(policies)} policies)")
    print(f"\n  {'policy':12s} {'mean n_kept':>11s}  "
          f"{'gold-in-kept frac':>17s}")
    for p in policies:
        ks = kept_counts[p]
        mk = sum(ks) / len(ks) if ks else 0.0
        gf = gold_present[p] / n_usable if n_usable else 0.0
        print(f"  {p:12s} {mk:11.1f}  {gf:17.3f}")
    print("\n  (gold-in-kept frac: fraction of cases whose kept set holds ≥1 "
          "has_answer turn;\n   learned_V should exceed recency — this previews "
          "whether QA accuracy tracks retention)")


# ── score mode ───────────────────────────────────────────────────────────

def load_verdicts(path: str | Path) -> list[dict]:
    """Read the controller's verdicts JSONL: {qid, policy, correct}."""
    out: list[dict] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.append({
                "qid":     str(r["qid"]),
                "policy":  str(r["policy"]),
                "correct": bool(r["correct"]),
            })
    return out


def run_score(args) -> None:
    verdicts = load_verdicts(args.verdicts)
    if not verdicts:
        raise SystemExit(f"no verdicts in {args.verdicts}")

    # index by (qid, policy) → correct; collect policies + qids
    by_qp: dict[tuple[str, str], bool] = {}
    policies: list[str] = []
    qids: set[str] = set()
    for v in verdicts:
        key = (v["qid"], v["policy"])
        if key in by_qp and by_qp[key] != v["correct"]:
            print(f"  [warn] conflicting verdicts for {key}; "
                  f"keeping last ({v['correct']})")
        by_qp[key] = v["correct"]
        if v["policy"] not in policies:
            policies.append(v["policy"])
        qids.add(v["qid"])

    # per-policy accuracy
    per_policy: dict[str, dict] = {}
    for p in policies:
        vals = [c for (q, pp), c in by_qp.items() if pp == p]
        n = len(vals)
        acc = sum(1 for c in vals if c) / n if n else 0.0
        per_policy[p] = {"accuracy": round(acc, 4), "n": n}

    # paired contrasts: learned_V vs each of {recency, uniform} present.
    # A case is paired only if BOTH policies have a verdict for that qid.
    def paired(a: str, b: str) -> dict | None:
        if a not in policies or b not in policies:
            return None
        wins = ties = losses = 0
        n_pair = 0
        a_correct = b_correct = 0
        for q in sorted(qids):
            ka, kb = (q, a), (q, b)
            if ka not in by_qp or kb not in by_qp:
                continue
            n_pair += 1
            ca, cb = by_qp[ka], by_qp[kb]
            a_correct += int(ca)
            b_correct += int(cb)
            if ca and not cb:
                wins += 1
            elif cb and not ca:
                losses += 1
            else:
                ties += 1
        if n_pair == 0:
            return None
        return {
            "vs":            b,
            "n_paired":      n_pair,
            "wins":          wins,   # learned_V correct, b wrong
            "ties":          ties,
            "losses":        losses,  # b correct, learned_V wrong
            "acc_a":         round(a_correct / n_pair, 4),
            "acc_b":         round(b_correct / n_pair, 4),
            "acc_gap":       round((a_correct - b_correct) / n_pair, 4),
        }

    contrasts: dict[str, dict] = {}
    for b in ("recency", "uniform"):
        c = paired("learned_V", b)
        if c is not None:
            contrasts[f"learned_V_vs_{b}"] = c

    # ── markdown summary ─────────────────────────────────────────────────
    md_lines = ["**LongMemEval QA accuracy (paper2)**", ""]
    md_lines.append(f"- cases: {len(qids)}; policies: {', '.join(policies)}")
    for p in policies:
        pp = per_policy[p]
        md_lines.append(f"- `{p}`: {pp['accuracy']:.1%} accuracy (n={pp['n']})")
    for name, c in contrasts.items():
        md_lines.append(
            f"- `{name}`: gap {c['acc_gap']:+.1%} "
            f"(learned_V {c['acc_a']:.1%} vs {c['acc_b']:.1%}); "
            f"W/T/L = {c['wins']}/{c['ties']}/{c['losses']} of {c['n_paired']} paired")
    md = "\n".join(md_lines)

    payload = {
        "experiment": "LongMemEval QA accuracy — multi-factor memory value (paper2)",
        "scored_at":  datetime.now().isoformat(),
        "verdicts":   str(args.verdicts),
        "n_cases":    len(qids),
        "policies":   policies,
        "per_policy": per_policy,
        "paired_contrasts": contrasts,
        "summary_md": md,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    print(f"\n=== QA accuracy → {out} ===")
    for p in policies:
        pp = per_policy[p]
        print(f"  {p:12s}: {pp['accuracy']:.3f}  (n={pp['n']})")
    for name, c in contrasts.items():
        print(f"  {name}: gap {c['acc_gap']:+.3f}  "
              f"W/T/L={c['wins']}/{c['ties']}/{c['losses']} "
              f"(n_paired={c['n_paired']})")
    print(f"  → wrote {out}")


# ── cli ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("prep", "score"), required=True)
    # prep args
    ap.add_argument("--data", default="data/longmemeval_s_cleaned.json")
    ap.add_argument("--cache", default="results/lme_factor_cache.jsonl")
    ap.add_argument("--learned-json", default=DEFAULT_LEARNED_JSON,
                    help="JSON with learned_weights_blind_mean for learned_V")
    ap.add_argument("--n-cases", type=int, default=40)
    ap.add_argument("--keep", type=int, default=40, help="max kept turns")
    ap.add_argument("--keep-frac", type=float, default=0.3)
    ap.add_argument("--policies", default="learned_V,recency,uniform")
    # score args
    ap.add_argument("--verdicts", default="results/lme_qa_verdicts.jsonl")
    # shared
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.mode == "prep":
        if args.out is None:
            args.out = "results/lme_qa_tasks.jsonl"
        run_prep(args)
    else:
        if args.out is None:
            args.out = "results/lme_qa_pilot.json"
        run_score(args)


if __name__ == "__main__":
    main()
