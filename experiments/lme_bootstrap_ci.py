"""
Task 4 — per-CASE bootstrap confidence intervals (API-free, paper2).

Codex critique of the headline: the 20 resampled 50/50 splits in
`lme_blind_forgetting.py` are NOT i.i.d. draws from the benchmark — they
reshuffle the SAME 479 cases — so the reported ± std understates the real
sampling uncertainty over the benchmark. This script answers the honest
uncertainty question with a per-CASE bootstrap.

Design (documented so the split is auditable):
  1. Fit ONE set of blind learned weights on a FIXED split: the FIRST HALF
     of the 479 cases (the first floor(479/2) = 239 cases in cache order),
     via learn_weights(blind obj, LIVE_FACTORS, seed=1, iters=120). This
     mirrors the headline's per-rep recipe (50/50, learn on the first
     half of a shuffle) but with a single fixed, documented split so the
     bootstrapped weights are held constant across resamples. Bootstrap
     uncertainty is then PURELY over which cases land in the benchmark,
     not over the optimiser — exactly the quantity the critique asks for.
  2. Score EVERY one of the 479 cases once: per-case blind retention for
     learned_V, uniform_V, recency_only, reliability_only. (gold_retention
     scores each case independently, so a per-case retention is well
     defined and resampling cases just resamples these scalars.)
  3. Bootstrap B = 1000, random.Random(0): resample the 479 cases WITH
     replacement; per resample, average each policy's per-case retention
     and each paired gap (learned-uniform, learned-recency,
     learned-reliability) over the resampled cases.
  4. Report per policy and per gap: bootstrap mean, 2.5 / 97.5 percentile
     (95% CI), and the fraction of resamples with gap > 0. The
     learned-uniform and learned-recency CIs should exclude 0.

keep_frac = 0.3 (the headline operating point).

The retention metric, cache loader, weight learner and one-hot helper are
imported from `lme_blind_forgetting` — not re-implemented — so this shares
the headline's code path.

Output: results/lme_bootstrap_ci.json
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

from lme_blind_forgetting import (  # noqa: E402
    LIVE_FACTORS,
    _single,
    gold_retention,
    learn_weights,
    load_from_cache,
    mean,
    recency_retention,
)

POLICIES = ("learned_V", "uniform_V", "recency_only", "reliability_only")
# gap name -> baseline policy subtracted from learned_V
GAPS = {
    "learned_minus_uniform":     "uniform_V",
    "learned_minus_recency":     "recency_only",
    "learned_minus_reliability": "reliability_only",
}


def percentile(sorted_xs, q):
    """Linear-interpolation percentile (q in [0,1]) over a sorted list."""
    if not sorted_xs:
        return float("nan")
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    pos = q * (len(sorted_xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_xs) - 1)
    frac = pos - lo
    return sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/lme_factor_cache.jsonl")
    ap.add_argument("--n-cases", type=int, default=None)
    ap.add_argument("--keep-frac", type=float, default=0.3)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default="results/lme_bootstrap_ci.json")
    args = ap.parse_args()
    kf = args.keep_frac

    print(f"Loading cases from factor cache {args.cache} …")
    cases = load_from_cache(args.cache, n_cases=args.n_cases)
    n = len(cases)
    print(f"  usable cases (with has_answer gold): {n}")

    # ── 1. fit ONE set of blind learned weights on a fixed split ─────────
    # Fixed = first half of the cases in cache order (no shuffle), so the
    # weights are reproducible and held constant across all resamples.
    split = n // 2
    train = cases[:split]
    print(f"Fitting blind learned weights on first {len(train)} cases "
          f"(fixed split), iters={args.iters} …")

    def obj(w):
        return mean(gold_retention(a, w, regime="blind", keep_frac=kf)
                    for a in train)

    learned, _ = learn_weights(obj, LIVE_FACTORS, seed=1, iters=args.iters)
    weights = {
        "learned_V":        learned,
        "uniform_V":        {f: 1.0 for f in LIVE_FACTORS},
        "reliability_only": _single("reliability"),
    }

    # ── 2. per-case blind retention over ALL cases (computed once) ───────
    # gold_retention / recency_retention score each case independently, so
    # these per-case scalars are exactly what the bootstrap resamples.
    per_case = {p: [] for p in POLICIES}
    for a in cases:
        per_case["learned_V"].append(
            gold_retention(a, weights["learned_V"], regime="blind", keep_frac=kf))
        per_case["uniform_V"].append(
            gold_retention(a, weights["uniform_V"], regime="blind", keep_frac=kf))
        per_case["reliability_only"].append(
            gold_retention(a, weights["reliability_only"], regime="blind", keep_frac=kf))
        per_case["recency_only"].append(
            recency_retention(a, keep_frac=kf))

    # load_from_cache filters to cases with gold, so none should be None;
    # assert it so a silent data change can't quietly bias the bootstrap.
    for p, xs in per_case.items():
        assert all(x is not None for x in xs), f"unexpected None in {p}"

    # point estimates over the full benchmark (no resampling)
    point = {p: round(mean(per_case[p]), 4) for p in POLICIES}
    point_gaps = {
        g: round(point["learned_V"] - point[base], 4)
        for g, base in GAPS.items()
    }

    # ── 3. per-case bootstrap (B resamples WITH replacement) ─────────────
    rng = random.Random(0)
    boot_pol = {p: [] for p in POLICIES}
    boot_gap = {g: [] for g in GAPS}
    for _ in range(args.boot):
        idx = [rng.randrange(n) for _ in range(n)]
        means = {p: sum(per_case[p][i] for i in idx) / n for p in POLICIES}
        for p in POLICIES:
            boot_pol[p].append(means[p])
        for g, base in GAPS.items():
            boot_gap[g].append(means["learned_V"] - means[base])

    # ── 4. summarise: mean, 95% CI, P(gap > 0) ───────────────────────────
    def summarise(samples):
        s = sorted(samples)
        return {
            "mean":     round(sum(s) / len(s), 4),
            "ci_2.5":   round(percentile(s, 0.025), 4),
            "ci_97.5":  round(percentile(s, 0.975), 4),
        }

    policy_ci = {p: summarise(boot_pol[p]) for p in POLICIES}
    gap_ci = {}
    for g in GAPS:
        s = summarise(boot_gap[g])
        s["frac_gap_gt_0"] = round(
            sum(1 for d in boot_gap[g] if d > 0) / args.boot, 4)
        s["excludes_zero"] = bool(s["ci_2.5"] > 0)
        gap_ci[g] = s

    payload = {
        "experiment": "per-case bootstrap CI — BLIND multi-factor forgetting (API-free)",
        "ran_at": datetime.now().isoformat(),
        "cache": args.cache,
        "n_cases_used": n,
        "keep_frac": kf,
        "iters": args.iters,
        "n_bootstrap": args.boot,
        "bootstrap_seed": 0,
        "regime": "blind",
        "live_factors": list(LIVE_FACTORS),
        "weight_fit_split": (
            f"fixed: first {len(train)} of {n} cases in cache order "
            f"(no shuffle); learn_weights(blind obj, LIVE_FACTORS, seed=1, "
            f"iters={args.iters})"),
        "learned_weights_blind": {f: round(learned.get(f, 0.0), 4)
                                  for f in LIVE_FACTORS},
        "point_estimate_full_benchmark": {
            "retention": point,
            "gaps": point_gaps,
        },
        "bootstrap_policy_ci": policy_ci,
        "bootstrap_gap_ci": gap_ci,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    print(f"\n=== per-case bootstrap (B={args.boot}, {n} cases, "
          f"keep {kf:.0%}, blind) ===")
    print(f"  fixed weight-fit split: first {len(train)} cases")
    print("  learned blind weights: " +
          ", ".join(f"{f}={learned.get(f,0.0):.2f}" for f in LIVE_FACTORS))
    print("  policy            point     boot-mean   95% CI")
    for p in POLICIES:
        c = policy_ci[p]
        print(f"  {p:16s} {point[p]:.3f}    {c['mean']:.3f}      "
              f"[{c['ci_2.5']:.3f}, {c['ci_97.5']:.3f}]")
    print("  paired gaps (learned - baseline):")
    for g in GAPS:
        c = gap_ci[g]
        mark = "EXCLUDES 0" if c["excludes_zero"] else "includes 0"
        print(f"    {g:28s}: {c['mean']:+.3f}  "
              f"[{c['ci_2.5']:+.3f}, {c['ci_97.5']:+.3f}]  "
              f"P(>0)={c['frac_gap_gt_0']:.3f}  {mark}")
    print(f"  → wrote {out}")


if __name__ == "__main__":
    main()
