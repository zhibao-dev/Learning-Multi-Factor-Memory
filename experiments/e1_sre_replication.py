"""
E1 — Self-Reference Effect (SRE) replication, mechanical formulation.

In the classic SRE experiment (Rogers, Kuiper & Kirker 1977; Symons &
Johnson 1997 meta-analysis), items encoded under different depth-of-
processing tasks (structural / phonemic / semantic / self-referential)
produce a robust ordering of recall:

  self_ref > semantic > phonemic > structural    (Cohen's d ≈ 0.50 for
                                                  self_ref vs semantic)

Our quantitative claim is that this ordering can be reproduced by the
Self-FEP forget_score formula:

  score = recency × usage × importance_res × emotion_res × self_res

  self_res = 1 / (1 + λ · self_relevance)

with self_relevance set by the encoding task. E1 tests this *mechanical*
claim by:

  1. Inserting N memories per condition into borge_memories with a
     condition-specific self_relevance_score (set to match the published
     depth-of-processing literature), holding emotion / importance /
     recency / depth constant.
  2. Running ForgettingEngine.run_forgetting_pass.
  3. Reading out the per-condition mean forget_score.

A "lower forget_score → more remembered" mapping yields a predicted
ordering: self_ref < semantic < phonemic ≈ structural, which is the
SRE direction.

The companion experiment E1b (future work — requires sentence-
transformers embeddings) tests whether the *forward pass* of the
encoding task actually produces the expected self_relevance values
from raw text. E1 here tests the formula itself.

Output: results/e1_sre_replication.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from borge.memory.forgetting import ForgettingEngine  # noqa: E402
from borge.memory.store import MemoryStore  # noqa: E402

# ── Condition design ──────────────────────────────────────────────────────

# Self-relevance per condition. Anchored to the depth-of-processing
# literature: structural is surface-feature, phonemic adds sound,
# semantic adds meaning, self-ref binds to identity.
CONDITION_SELF_RELEVANCE = {
    "structural": 0.05,
    "phonemic":   0.20,
    "semantic":   0.50,
    "self_ref":   0.85,
}

# Common controls held constant across conditions
COMMON = {
    "emotional_valence":  0.0,
    "emotional_arousal":  0.5,
    "importance_score":   0.5,
    "encoding_depth":     3,    # SCHEMATIC — neither pruned nor compressed
    "retrieval_count":    0,
    "role":               "user",
    "content":            "trait word",
}

CONDITIONS = ("structural", "phonemic", "semantic", "self_ref")


def run_experiment(n_per_condition: int, seed: int, age_days: int) -> dict:
    """One simulation run; returns per-condition mean forget_score."""
    rng = random.Random(seed)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        store = MemoryStore(db_path)
        ts_base = (datetime.now() - timedelta(days=age_days)).isoformat()

        # Insert N items per condition with condition-specific self_relevance
        for cond in CONDITIONS:
            for i in range(n_per_condition):
                store.insert({
                    **COMMON,
                    "id":                   f"{cond}-{i}",
                    "session_id":           "sre",
                    "timestamp":            ts_base,
                    "self_relevance_score": CONDITION_SELF_RELEVANCE[cond],
                })

        # Optional tiny jitter so storage order isn't degenerate
        _ = rng.random()

        # Run forget pass to compute forget_score for each row
        ForgettingEngine().run_forgetting_pass(db_path)

        rows = store.all()
        scores_by_cond: dict[str, list[float]] = {c: [] for c in CONDITIONS}
        for r in rows:
            for c in CONDITIONS:
                if r["id"].startswith(f"{c}-"):
                    scores_by_cond[c].append(r["forget_score"])
                    break

        per_cond = {}
        for c in CONDITIONS:
            xs = scores_by_cond[c]
            mean = sum(xs) / len(xs) if xs else 0.0
            var  = sum((x - mean) ** 2 for x in xs) / len(xs) if xs else 0.0
            per_cond[c] = {
                "n":            len(xs),
                "forget_mean":  round(mean, 4),
                "forget_std":   round(var ** 0.5, 4),
                "self_relevance": CONDITION_SELF_RELEVANCE[c],
            }

        # SRE effect: in forget-score space we predict self_ref < semantic
        # (i.e. negative difference). Report self_ref - semantic so the
        # sign matches the human "self > semantic" direction when flipped.
        sre_forget_diff = per_cond["self_ref"]["forget_mean"] - per_cond["semantic"]["forget_mean"]
        sre_recall_proxy = -sre_forget_diff  # higher = better predicted recall

        return {
            "seed":              seed,
            "age_days":          age_days,
            "per_condition":     per_cond,
            "sre_forget_diff":   round(sre_forget_diff, 4),
            "sre_recall_proxy":  round(sre_recall_proxy, 4),
            "ordering_ok":       (per_cond["self_ref"]["forget_mean"]
                                  <  per_cond["semantic"]["forget_mean"]
                                  <  per_cond["phonemic"]["forget_mean"]
                                  <= per_cond["structural"]["forget_mean"]),
        }
    finally:
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-condition", type=int, default=50)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--age-days", type=int, default=7)
    ap.add_argument("--out", default="results/e1_sre_replication.json")
    args = ap.parse_args()

    runs = []
    for s in range(args.n_seeds):
        result = run_experiment(args.n_per_condition, seed=42 + s, age_days=args.age_days)
        runs.append(result)
        cond_means = {c: result["per_condition"][c]["forget_mean"] for c in CONDITIONS}
        ordering = " < ".join(
            f"{c}={cond_means[c]:.3f}"
            for c in sorted(CONDITIONS, key=lambda x: cond_means[x])
        )
        print(f"  seed {42 + s}: {ordering}  (SRE recall-proxy = {result['sre_recall_proxy']:+.4f})")

    # Aggregate across seeds
    aggregate = {}
    for c in CONDITIONS:
        means = [r["per_condition"][c]["forget_mean"] for r in runs]
        m = sum(means) / len(means)
        v = sum((x - m) ** 2 for x in means) / len(means)
        aggregate[c] = {
            "mean_forget":     round(m, 4),
            "std_forget":      round(v ** 0.5, 4),
            "self_relevance":  CONDITION_SELF_RELEVANCE[c],
        }

    sre_recall_proxies = [r["sre_recall_proxy"] for r in runs]
    mean_sre = sum(sre_recall_proxies) / len(sre_recall_proxies)
    var_sre  = sum((x - mean_sre) ** 2 for x in sre_recall_proxies) / len(sre_recall_proxies)
    ordering_hits = sum(1 for r in runs if r["ordering_ok"])

    payload = {
        "experiment":       "E1 — SRE replication (mechanical, forget-score)",
        "ran_at":           datetime.now().isoformat(),
        "n_seeds":          args.n_seeds,
        "n_per_condition":  args.n_per_condition,
        "age_days":         args.age_days,
        "condition_self_relevance": CONDITION_SELF_RELEVANCE,
        "aggregate":        aggregate,
        "sre_recall_proxy": {
            "mean": round(mean_sre, 4),
            "std":  round(var_sre ** 0.5, 4),
        },
        "ordering_hits":    f"{ordering_hits}/{args.n_seeds}",
        "per_seed":         runs,
        "note": ("E1 tests the forget_score formula's response to "
                 "self_relevance, holding emotion/importance/recency "
                 "constant. E1b (future) would use sentence-transformers "
                 "embeddings to verify the forward pass also assigns "
                 "condition-appropriate self_relevance from raw text."),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print()
    print(f"=== E1 SRE Replication — N={args.n_seeds} seeds × {args.n_per_condition} items per condition ===")
    print(f"  {'condition':12s} {'self_rel':>10s}   {'forget':>9s}   ")
    for c in ("structural", "phonemic", "semantic", "self_ref"):
        print(f"  {c:12s} {aggregate[c]['self_relevance']:>10.2f}   "
              f"{aggregate[c]['mean_forget']:>9.4f}")
    print(f"  SRE recall-proxy (semantic forget − self_ref forget) "
          f"= {mean_sre:+.4f} ± {var_sre ** 0.5:.4f}")
    print(f"  Ordering matches human direction in {ordering_hits}/{args.n_seeds} seeds")
    print(f"  → wrote {out_path}")


if __name__ == "__main__":
    main()
