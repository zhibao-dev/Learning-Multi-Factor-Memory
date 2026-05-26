"""
REAL LongMemEval gold-retention (API-free): learned multi-factor value
vs baselines on the actual benchmark haystacks.

For each LongMemEval case: annotate every turn with free factors (SBert
cos-to-question, cos-to-μ_user, emotion rules, role-reliability), rank
turns by memory value V, keep top keep_frac, measure fraction of the
dataset's `has_answer` gold turns retained.

A2: learn weights on a TRAIN split to maximise gold retention; report on
held-out TEST split for learned-V vs uniform-V vs goal-only vs
emotion-only vs recency-only.

Needs the dataset (data/longmemeval_s_cleaned.json) + SBert (local, no
API). Subset with --n-cases for speed.

Output: results/lme_real_retention.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from borge.eval.longmemeval import load_longmemeval  # noqa: E402
from borge.eval.annotate import annotate_case  # noqa: E402
from borge.memory.value import MemoryValue, learn_weights  # noqa: E402
from borge.values.self_model import SBertEmbedder  # noqa: E402

FACTORS = MemoryValue.FACTORS


def gold_retention(annotated: list[dict], weights: dict, *, keep_frac: float):
    """Rank turns by V (desc), keep top frac, return fraction of
    has_answer gold turns kept."""
    mv = MemoryValue(weights=weights)
    scored = [(mv.value(a["factors"]), a) for a in annotated]
    scored.sort(key=lambda x: -x[0])
    k = max(1, int(len(scored) * keep_frac))
    kept = [a for _, a in scored[:k]]
    total = sum(1 for a in annotated if a["has_answer"])
    if total == 0:
        return None  # case has no flagged gold; skip
    return sum(1 for a in kept if a["has_answer"]) / total


def recency_retention(annotated: list[dict], *, keep_frac: float):
    """Keep most-recent turns (highest session/time index)."""
    ordered = sorted(annotated, key=lambda a: -a["timestamp_idx"])
    k = max(1, int(len(ordered) * keep_frac))
    kept = ordered[:k]
    total = sum(1 for a in annotated if a["has_answer"])
    if total == 0:
        return None
    return sum(1 for a in kept if a["has_answer"]) / total


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/longmemeval_s_cleaned.json")
    ap.add_argument("--n-cases", type=int, default=80)
    ap.add_argument("--keep-frac", type=float, default=0.3)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--out", default="results/lme_real_retention.json")
    args = ap.parse_args()

    print("Loading SBert…")
    embedder = SBertEmbedder()
    _ = embedder("warm-up")

    print(f"Loading + annotating {args.n_cases} cases (batched SBert)…")
    cases = []
    for i, case in enumerate(load_longmemeval(args.data)):
        if i >= args.n_cases:
            break
        ann = annotate_case(case, embedder)
        if any(a["has_answer"] for a in ann):   # only keep cases with flagged gold
            cases.append(ann)
        if (i + 1) % 20 == 0:
            print(f"  annotated {i+1}…")
    print(f"  usable cases (with has_answer gold): {len(cases)}")

    rng = random.Random(0)
    rng.shuffle(cases)
    split = len(cases) // 2
    train, test = cases[:split], cases[split:]

    kf = args.keep_frac
    def train_obj(w):
        return mean(gold_retention(a, w, keep_frac=kf) for a in train)
    learned, hist = learn_weights(train_obj, FACTORS, seed=1, iters=args.iters)

    policies = {
        "learned_V":     learned,
        "uniform_V":     {f: 1.0 for f in FACTORS},
        "goal_only":     {f: (1.0 if f == "goal_relevance" else 0.0) for f in FACTORS},
        "emotion_only":  {f: (1.0 if f == "emotion" else 0.0) for f in FACTORS},
        "self_only":     {f: (1.0 if f == "self_relevance" else 0.0) for f in FACTORS},
    }
    results = {name: round(mean(gold_retention(a, w, keep_frac=kf) for a in test), 4)
               for name, w in policies.items()}
    results["recency_only"] = round(mean(recency_retention(a, keep_frac=kf) for a in test), 4)
    results["random_keep"]  = round(kf, 4)   # expected gold retention if keep is random

    payload = {
        "experiment": "REAL LongMemEval gold-retention (API-free)",
        "ran_at": datetime.now().isoformat(),
        "data": args.data,
        "n_cases_used": len(cases),
        "n_train": len(train), "n_test": len(test),
        "keep_frac": kf,
        "learned_weights": {f: round(learned[f], 4) for f in FACTORS},
        "test_gold_retention": results,
        "train_final": round(hist[-1]["best_return"], 4),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    print(f"\n=== REAL LongMemEval gold retention (keep {kf:.0%}, {len(test)} test cases) ===")
    for name in ("learned_V", "uniform_V", "goal_only", "self_only",
                 "emotion_only", "recency_only", "random_keep"):
        print(f"  {name:14s}: {results[name]:.3f}")
    print(f"\n  learned weights: " + ", ".join(f"{f}={learned[f]:.2f}" for f in FACTORS))
    print(f"  → wrote {out}")


if __name__ == "__main__":
    main()
