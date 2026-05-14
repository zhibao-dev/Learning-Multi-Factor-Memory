"""
E3 — π_self ablation + λ sensitivity.

The forget_score formula multiplies in

  self_resistance = 1 / (1 + λ · self_relevance)

where self_relevance is set at encoding time via SelfModel:

  sr_encoded = π_self · raw_cosine_sim + (1 - π_self) · 0.5

Two independent knobs gate the effective SRE size:

  (a) π_self  — the self-precision the agent has at encoding time.
                π_self → 0  ⇒  every memory's stored sr → 0.5 neutral,
                collapsing the encoded distinction between self-ref
                and other conditions. This is the FEP-theoretical
                prediction the proposal commits to (impaired self-
                precision attenuates the SRE).

  (b) λ       — the multiplicative coefficient in forget_score.
                A pure formula coefficient; varying it tests how
                strongly stored sr is allowed to influence forgetting.

Experiment (a) — π_self ablation:
   For each π_self ∈ {0.0, 0.25, 0.5, 0.75, 1.0}, freshly build a
   SelfModel with the seed text "self_ref" (high cosine for self_ref
   items) and run the consolidation+forgetting pipeline through
   condition-tagged stimuli. Measure SRE size in forget space.
   Predicts: monotonic increase with π_self.

Experiment (b) — λ sensitivity:
   With π_self pinned at the production default 1.0, sweep λ across
   {0, 0.25, 0.5, 1, 2, 4, 8}. Measure SRE size. This is *not* a
   theoretical claim — it's a sensitivity test that bounds the
   regime where the SRE is non-trivial in the model.

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
from borge.values.self_model import SelfModel, embed  # noqa: E402

# For (b) λ sensitivity we use fixed stored sr per condition (same as E1).
CONDITION_SR = {
    "structural": 0.05,
    "phonemic":   0.20,
    "semantic":   0.50,
    "self_ref":   0.85,
}

# For (a) π_self ablation we use condition-tagged text whose embedding
# distance to μ_self varies. Seed μ_self with a self-ref-style sentence;
# self_ref items hug μ_self, structural items live in unrelated embedding
# space. SelfModel.self_relevance() applies the π_self blend.
CONDITION_TEXT = {
    "structural": "noise random unrelated wallpaper tabletop curtain",
    "phonemic":   "rhyme echo sound noise random wallpaper",
    "semantic":   "trait meaning honest careful patient kind",
    "self_ref":   "I am honest careful patient kind myself",
}
SELF_SEED = "I am honest careful patient kind myself"


def run_lambda(lam: float, n_per_condition: int, age_days: int) -> dict:
    """(b) Pure λ formula sensitivity — sr fixed per condition (E1-style)."""
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
                        "session_id":           "lambda_sens",
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
            cell_means = _cell_forget_means(store)
            sre = cell_means["semantic"] - cell_means["self_ref"]
            return {
                "lambda":     lam,
                "cell_means": {k: round(v, 4) for k, v in cell_means.items()},
                "sre_size":   round(sre, 4),
            }
        finally:
            try: os.unlink(db_path)
            except FileNotFoundError: pass
    finally:
        fgmod.SELF_RESISTANCE_LAMBDA = original


def run_pi_self(pi: float, n_per_condition: int, age_days: int) -> dict:
    """
    (a) π_self ablation — the central theoretical test.

    Build a SelfModel pinned at the target π_self and a μ_self seeded with
    a self-ref sentence. Compute self_relevance for each condition's text
    by passing it through SelfModel.self_relevance() — which applies the
    π · raw_sim + (1−π) · 0.5 blend.

    Then store memories with these π-blended sr values and run the
    forgetting pass. Lower π pulls all conditions toward neutral 0.5,
    shrinking the SRE.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        self_model = SelfModel.from_seed(SELF_SEED)
        self_model.pi_self = pi   # pin precision for the ablation

        # Pre-compute condition-specific sr via the SelfModel forward pass
        cond_sr = {
            cond: self_model.self_relevance(embed(text, dim=self_model.dim))
            for cond, text in CONDITION_TEXT.items()
        }

        store = MemoryStore(db_path)
        ts = (datetime.now() - timedelta(days=age_days)).isoformat()
        for cond, sr in cond_sr.items():
            for i in range(n_per_condition):
                store.insert({
                    "id":                   f"{cond}-{i}",
                    "session_id":           "pi_ablation",
                    "role":                 "user",
                    "content":              CONDITION_TEXT[cond],
                    "timestamp":            ts,
                    "emotional_valence":    0.0,
                    "emotional_arousal":    0.5,
                    "self_relevance_score": sr,
                    "encoding_depth":       3,
                    "importance_score":     0.5,
                    "retrieval_count":      0,
                })

        ForgettingEngine().run_forgetting_pass(db_path)
        cell_means = _cell_forget_means(store)
        sre = cell_means["semantic"] - cell_means["self_ref"]
        return {
            "pi_self":            pi,
            "condition_sr":       {k: round(v, 4) for k, v in cond_sr.items()},
            "cell_forget_means":  {k: round(v, 4) for k, v in cell_means.items()},
            "sre_size":           round(sre, 4),
        }
    finally:
        try: os.unlink(db_path)
        except FileNotFoundError: pass


def _cell_forget_means(store: MemoryStore) -> dict[str, float]:
    means = {}
    for cond in CONDITION_SR:
        scores = [
            r["forget_score"] for r in store.all()
            if r["id"].startswith(f"{cond}-")
        ]
        means[cond] = sum(scores) / len(scores) if scores else 0.0
    return means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-condition", type=int, default=50)
    ap.add_argument("--age-days", type=int, default=7)
    ap.add_argument("--lambdas", nargs="+", type=float,
                    default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--pis", nargs="+", type=float,
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--out", default="results/e3_pi_self_ablation.json")
    args = ap.parse_args()

    # ── (a) π_self ablation — primary theoretical test ───────────────
    pi_runs = []
    print(f"\n=== E3a π_self ablation — N={args.n_per_condition} per condition ===")
    print(f"  {'π_self':>7s}  {'sr_struct':>10s}  {'sr_phon':>10s}  {'sr_sem':>10s}  {'sr_self':>10s}  {'SRE_size':>10s}")
    for pi in args.pis:
        r = run_pi_self(pi, args.n_per_condition, args.age_days)
        pi_runs.append(r)
        csr = r["condition_sr"]
        print(f"  {pi:7.2f}  {csr['structural']:10.4f}  {csr['phonemic']:10.4f}  "
              f"{csr['semantic']:10.4f}  {csr['self_ref']:10.4f}  {r['sre_size']:>10.4f}")
    pi_monotone = all(pi_runs[i]["sre_size"] <= pi_runs[i+1]["sre_size"]
                      for i in range(len(pi_runs) - 1))
    print(f"  Monotonic increase with π_self: {pi_monotone}   "
          f"(predicted: True — low π collapses sr → 0.5 → no SRE)")

    # ── (b) λ sensitivity — secondary formula sweep ──────────────────
    lam_runs = []
    print(f"\n=== E3b λ sensitivity — N={args.n_per_condition} per condition ===")
    print(f"  {'λ':>6s}  {'struct':>9s}  {'phon':>9s}  {'sem':>9s}  {'self':>9s}  {'SRE_size':>10s}")
    for lam in args.lambdas:
        r = run_lambda(lam, args.n_per_condition, args.age_days)
        lam_runs.append(r)
        cm = r["cell_means"]
        print(f"  {lam:6.2f}  {cm['structural']:9.4f}  {cm['phonemic']:9.4f}  "
              f"{cm['semantic']:9.4f}  {cm['self_ref']:9.4f}  {r['sre_size']:>10.4f}")
    lam_peak_idx = max(range(len(lam_runs)), key=lambda i: lam_runs[i]["sre_size"])
    print(f"  SRE size peaks at λ = {lam_runs[lam_peak_idx]['lambda']}   "
          f"(non-monotonic by design — large λ crushes all conditions toward 0)")

    payload = {
        "experiment":      "E3 — π_self ablation (primary) + λ sensitivity (secondary)",
        "ran_at":          datetime.now().isoformat(),
        "n_per_condition": args.n_per_condition,
        "pi_self_runs":    pi_runs,
        "pi_self_monotone_increase": pi_monotone,
        "lambda_runs":     lam_runs,
        "lambda_peak":     lam_runs[lam_peak_idx]["lambda"],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n  → wrote {out}")


if __name__ == "__main__":
    main()
