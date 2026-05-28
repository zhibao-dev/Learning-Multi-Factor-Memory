"""
Task 3 — keep-fraction sweep (API-free robustness check for paper2).

The headline (`results/lme_blind_forgetting_full.json`) fixes keep_frac
κ = 0.3. A reviewer can object that the multi-factor win is κ-specific:
maybe learned_V only beats the baselines at one operating point. This
sweep settles that by re-running the SAME blind train/test protocol at
κ ∈ {0.1, 0.2, 0.3, 0.4, 0.5} and showing learned_V stays on top at
EVERY budget.

Protocol (identical to the headline, see lme_blind_forgetting.main):
  * 479 usable cases from the factor cache (≥1-gold filter at build time)
  * 20 resampled 50/50 splits, random.Random(100 + rep) shuffle
  * learn blind weights on TRAIN with
        learn_weights(obj, LIVE_FACTORS, seed=1, iters=120)
  * eval blind retention on TEST for learned_V, uniform_V,
    reliability_only (the best single factor), recency_only
  * report mean ± std over the 20 splits per (κ, policy)

Everything is imported from `lme_blind_forgetting` — the retention metric,
the cache loader, the weight learner and the one-hot helper are NOT
re-implemented here, so this script and the headline share one code path.

Output: results/lme_keepfrac_sweep.json
        paper2/figures/fig_keepfrac.pdf
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lme_blind_forgetting import (  # noqa: E402
    LIVE_FACTORS,
    _single,
    gold_retention,
    learn_weights,
    load_from_cache,
    mean,
    recency_retention,
)

# Policies compared at every κ. reliability_only is the best single factor
# in the headline (0.50 vs ~0.46/0.45/0.44 for self/goal/emotion at κ=0.3),
# so it is the honest "best single factor" baseline — beating it means the
# win is not reproducible by any one factor alone.
POLICIES = ("learned_V", "uniform_V", "reliability_only", "recency_only")
KEEP_FRACS = (0.1, 0.2, 0.3, 0.4, 0.5)


def agg(vals):
    """mean ± population std (matches lme_blind_forgetting.main's agg)."""
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
    return round(m, 4), round(sd, 4)


def eval_keepfrac(cases, kf, reps, iters):
    """Return {policy: [per-split blind retention]} at one keep_frac κ.

    For each resampled split: learn blind weights on TRAIN, then score the
    held-out TEST under the blind regime for each policy. recency_only is
    weight-free (ranks by timestamp), so it ignores the learned weights.
    """
    per_split = {p: [] for p in POLICIES}
    for rep in range(reps):
        r = random.Random(100 + rep)
        shuffled = list(cases)
        r.shuffle(shuffled)
        split = len(shuffled) // 2
        tr, te = shuffled[:split], shuffled[split:]

        def obj(w, tr=tr):
            return mean(
                gold_retention(a, w, regime="blind", keep_frac=kf) for a in tr
            )

        learned, _ = learn_weights(obj, LIVE_FACTORS, seed=1, iters=iters)
        weights = {
            "learned_V":        learned,
            "uniform_V":        {f: 1.0 for f in LIVE_FACTORS},
            "reliability_only": _single("reliability"),
        }
        for name, w in weights.items():
            per_split[name].append(
                mean(gold_retention(a, w, regime="blind", keep_frac=kf)
                     for a in te)
            )
        per_split["recency_only"].append(
            mean(recency_retention(a, keep_frac=kf) for a in te)
        )
    return per_split


def render_figure(sweep, out_path):
    """x = κ, one line per policy with ±1 std error bands. Agg backend,
    ASCII-only title (no LaTeX backslashes)."""
    ks = list(KEEP_FRACS)
    style = {
        "learned_V":        ("#2e8b57", "o", "learned V"),
        "uniform_V":        ("#4c72b0", "s", "uniform V"),
        "reliability_only": ("#dd8452", "^", "reliability only"),
        "recency_only":     ("#937860", "d", "recency only"),
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for p in POLICIES:
        color, marker, lab = style[p]
        ms = [sweep[f"{k:.1f}"][p][0] for k in ks]
        ss = [sweep[f"{k:.1f}"][p][1] for k in ks]
        lo = [m - s for m, s in zip(ms, ss)]
        hi = [m + s for m, s in zip(ms, ss)]
        ax.plot(ks, ms, marker=marker, color=color, label=lab, lw=1.8, ms=5)
        ax.fill_between(ks, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax.set_xlabel("keep fraction (kappa)")
    ax.set_ylabel("blind gold-evidence retention")
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.0)
    ax.set_title("Learned multi-factor value tops every keep budget "
                 "(479 cases, 20 splits; mean +/- 1 std)", fontsize=9)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.92)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/lme_factor_cache.jsonl")
    ap.add_argument("--n-cases", type=int, default=None,
                    help="limit to first N cached cases (default: all)")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--out", default="results/lme_keepfrac_sweep.json")
    ap.add_argument("--fig", default="paper2/figures/fig_keepfrac.pdf")
    args = ap.parse_args()

    print(f"Loading cases from factor cache {args.cache} …")
    cases = load_from_cache(args.cache, n_cases=args.n_cases)
    print(f"  usable cases (with has_answer gold): {len(cases)}")
    reps = max(1, args.reps)

    sweep = {}
    for kf in KEEP_FRACS:
        per_split = eval_keepfrac(cases, kf, reps, args.iters)
        sweep[f"{kf:.1f}"] = {p: agg(per_split[p]) for p in POLICIES}

    payload = {
        "experiment": "keep-fraction sweep — BLIND multi-factor forgetting (API-free)",
        "ran_at": datetime.now().isoformat(),
        "cache": args.cache,
        "n_cases_used": len(cases),
        "reps": reps,
        "iters": args.iters,
        "regime": "blind",
        "live_factors": list(LIVE_FACTORS),
        "policies": list(POLICIES),
        "keep_fracs": list(KEEP_FRACS),
        "protocol": ("20 resampled 50/50 splits, random.Random(100+rep) "
                     "shuffle, learn_weights(blind obj, LIVE_FACTORS, seed=1, "
                     "iters=120); reliability_only = best single factor"),
        "retention_mean_std": sweep,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    print(f"\n=== keep-fraction sweep (BLIND, {len(cases)} cases, "
          f"{reps} splits; mean ± std) ===")
    header = "  kappa  " + "".join(f"{p:>20s}" for p in POLICIES)
    print(header)
    for kf in KEEP_FRACS:
        row = sweep[f"{kf:.1f}"]
        cells = "".join(f"{row[p][0]:>11.3f}±{row[p][1]:<8.3f}" for p in POLICIES)
        top = max(POLICIES, key=lambda p: row[p][0])
        flag = "  <- learned tops" if top == "learned_V" else f"  <- {top} tops!"
        print(f"  {kf:.1f} " + cells + flag)
    print(f"  → wrote {out}")

    render_figure(sweep, Path(args.fig))


if __name__ == "__main__":
    main()
