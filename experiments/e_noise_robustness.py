"""
L4 — Noise robustness of the E1b Cohen's d match.

E1b reported a single noise level (σ=0.05 added to forget_score) and a
single d = 0.67 ± 0.16 across 5 seeds. A reviewer's next question is:
"how brittle is this to the noise hyperparameter?"

This experiment sweeps noise standard deviation σ ∈
{0.0, 0.025, 0.05, 0.10, 0.20, 0.40} with 10 seeds per cell and reports

    Cohen's d  per σ:  mean ± 95% bootstrap CI

The prediction: d is large at σ=0 (deterministic regime), stays inside
the Symons & Johnson [0.30, 0.70] band across the "reasonable" noise
range, and degrades only at extreme noise (σ ≥ 0.40) where the
forget-score signal itself drowns.

Output: results/e_noise_robustness.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.e1b_sre_forward_pass import run_experiment  # noqa: E402
from borge.values.self_model import SBertEmbedder  # noqa: E402


def _bootstrap_ci(xs: list[float], n_boot: int = 5000,
                  alpha: float = 0.05, seed: int = 1) -> tuple[float, float]:
    """95% percentile bootstrap CI on the mean."""
    rng = random.Random(seed)
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    means = []
    for _ in range(n_boot):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", nargs="+", type=float,
                    default=[0.0, 0.025, 0.05, 0.10, 0.20, 0.40])
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--n-per-condition", type=int, default=30)
    ap.add_argument("--out", default="results/e_noise_robustness.json")
    args = ap.parse_args()

    print("Loading SBert (cached if already downloaded)…")
    embedder = SBertEmbedder()
    _ = embedder("warm-up")

    sweep = []
    print(f"\n=== Noise robustness sweep: {len(args.sigmas)} σ × {args.n_seeds} seeds ===")
    print(f"  {'σ':>6s}  {'mean d':>9s}  {'95% CI':>20s}  {'within SJ97?':>13s}  ordering")
    SJ_lo, SJ_hi = 0.30, 0.70
    for sigma in args.sigmas:
        ds: list[float] = []
        ordering_hits = 0
        for s in range(args.n_seeds):
            r = run_experiment(
                n_items_per_condition=args.n_per_condition,
                seed=42 + s,
                age_days=7,
                noise_std=sigma,
                embedder=embedder,
            )
            ds.append(r["cohens_d_self_vs_semantic"])
            if r["ordering_ok"]:
                ordering_hits += 1
        mean = sum(ds) / len(ds)
        lo, hi = _bootstrap_ci(ds)
        within = SJ_lo <= mean <= SJ_hi
        sweep.append({
            "sigma": sigma,
            "n_seeds": args.n_seeds,
            "d_mean": round(mean, 4),
            "d_ci_lo": round(lo, 4),
            "d_ci_hi": round(hi, 4),
            "d_per_seed": [round(x, 4) for x in ds],
            "ordering_hits": f"{ordering_hits}/{args.n_seeds}",
            "within_sj97_band": within,
        })
        print(f"  {sigma:>6.3f}  {mean:>+9.4f}  [{lo:>+7.4f}, {hi:>+7.4f}]  "
              f"{str(within):>13s}  {ordering_hits}/{args.n_seeds}")

    payload = {
        "experiment":           "L4 — Noise robustness of E1b Cohen's d",
        "ran_at":               datetime.now().isoformat(),
        "embedder":             "sentence-transformers/all-MiniLM-L6-v2",
        "sj97_reference_band":  [SJ_lo, SJ_hi],
        "sj97_meta_d":          0.50,
        "n_seeds_per_sigma":    args.n_seeds,
        "n_per_condition":      args.n_per_condition,
        "sweep":                sweep,
        "summary": {
            "sigmas_within_sj_band": [r["sigma"] for r in sweep if r["within_sj97_band"]],
            "sigma_max_within":      max((r["sigma"] for r in sweep if r["within_sj97_band"]), default=None),
            "robustness_verdict":    "robust" if sum(1 for r in sweep if r["within_sj97_band"]) >= len(sweep)//2 else "brittle",
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print()
    print(f"  Within S&J [0.30, 0.70] band: σ ∈ {payload['summary']['sigmas_within_sj_band']}")
    print(f"  Robustness verdict: {payload['summary']['robustness_verdict']}")
    print(f"  → wrote {out}")


if __name__ == "__main__":
    main()
