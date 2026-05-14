"""
E3 — λ_self ablation: how strongly does self-resistance scale the SRE?

The forget_score formula carries

  self_resistance = 1 / (1 + λ · self_relevance)

A clinical prediction follows: in populations with impaired self-precision
(dissociation, severe depression), the empirical SRE should attenuate.
Mechanistically, lower π_self at encoding pushes the stored
self_relevance toward 0.5 (neutral) AND a lower λ in the formula
flattens the slope between sr and forget_score.

This experiment varies the formula coefficient λ across a grid and
measures the SRE magnitude (self_ref − semantic gap in forget space).

Expected pattern: monotonic increase in SRE magnitude with λ; at λ=0
the SRE collapses (forget_score insensitive to self_relevance).

Output: results/e3_pi_self_ablation.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from borge.memory import forgetting as fgmod  # noqa: E402
from borge.memory.forgetting import ForgettingEngine  # noqa: E402
from borge.memory.store import MemoryStore  # noqa: E402

CONDITION_SR = {
    "structural": 0.05,
    "phonemic":   0.20,
    "semantic":   0.50,
    "self_ref":   0.85,
}


def run_lambda(lam: float, n_per_condition: int, age_days: int) -> dict:
    # Patch the module-level constant. This is a deliberate ablation knob —
    # in production the constant is fixed at 2.0; here we sweep.
    original = fgmod.SELF_RESISTANCE_LAMBDA
    fgmod.SELF_RESISTANCE_LAMBDA = lam
    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            store = MemoryStore(db_path)
            ts = (datetime.now() - timedelta(days=age_days)).isoformat()
            for cond, sr in CONDITION_SR.items():
                for i in range(n_per_condition):
                    store.insert({
                        "id":                   f"{cond}-{i}",
                        "session_id":           "ablation",
                        "role":                 "user",
                        "content":              "x",
                        "timestamp":            ts,
                        "emotional_valence":    0.0,
                        "emotional_arousal":    0.5,
                        "self_relevance_score": sr,
                        "encoding_depth":       3,
                        "importance_score":     0.5,
                        "retrieval_count":      0,
                    })
            ForgettingEngine().run_forgetting_pass(db_path)

            cell_means: dict[str, float] = {}
            for cond in CONDITION_SR:
                scores = [
                    r["forget_score"]
                    for r in store.all()
                    if r["id"].startswith(f"{cond}-")
                ]
                cell_means[cond] = sum(scores) / len(scores) if scores else 0.0
            sre = cell_means["semantic"] - cell_means["self_ref"]
            return {
                "lambda":     lam,
                "cell_means": {k: round(v, 4) for k, v in cell_means.items()},
                "sre_size":   round(sre, 4),
            }
        finally:
            try:
                os.unlink(db_path)
            except FileNotFoundError:
                pass
    finally:
        fgmod.SELF_RESISTANCE_LAMBDA = original


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-condition", type=int, default=50)
    ap.add_argument("--age-days", type=int, default=7)
    ap.add_argument("--lambdas", nargs="+", type=float,
                    default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--out", default="results/e3_pi_self_ablation.json")
    args = ap.parse_args()

    runs = []
    print(f"=== E3 λ ablation — N={args.n_per_condition} per condition ===")
    print(f"  {'λ':>6s}  {'structural':>10s}  {'phonemic':>10s}  {'semantic':>10s}  {'self_ref':>10s}  {'SRE_size':>10s}")
    for lam in args.lambdas:
        r = run_lambda(lam, args.n_per_condition, args.age_days)
        runs.append(r)
        cm = r["cell_means"]
        print(f"  {lam:6.2f}  {cm['structural']:10.4f}  {cm['phonemic']:10.4f}  "
              f"{cm['semantic']:10.4f}  {cm['self_ref']:10.4f}  {r['sre_size']:>10.4f}")

    payload = {
        "experiment":      "E3 — λ_self ablation",
        "ran_at":          datetime.now().isoformat(),
        "n_per_condition": args.n_per_condition,
        "lambdas":         args.lambdas,
        "runs":            runs,
        "monotonic":       all(runs[i]["sre_size"] <= runs[i+1]["sre_size"]
                               for i in range(len(runs) - 1)),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"  Monotonic increase: {payload['monotonic']}")
    print(f"  → wrote {out}")


if __name__ == "__main__":
    main()
