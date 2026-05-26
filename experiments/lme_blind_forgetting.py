"""
REAL LongMemEval — BLIND vs ORACLE forgetting (API-free).

The honest negative from `lme_real_retention.py`: LongMemEval gold is
QUESTION-DEFINED, so a memory that scores goal_relevance = cos(turn,
question) retrieves the gold trivially (retention ≈ 1.0). That measures
RETRIEVAL, not FORGETTING — at consolidation time a real agent has NOT
yet seen the future eval question.

This experiment separates the two regimes:

  ORACLE goal_relevance = cos(turn, eval question)
        → the retrieval ceiling. A policy that peeks at the question.

  BLIND  goal_relevance = cos(turn, session-topic centroid)
        → realistic forgetting. "What is this session about?" is known
          at consolidation; the future eval question is NOT.

Everything else (emotion, self_relevance, reliability) is identical and
already query-agnostic. We embed each turn ONCE, derive both goal
variants, then compare policies under each regime:

  learned_V    : weights fit on TRAIN (in the SAME regime) to maximise
                 gold retention
  goal_only    : goal_relevance alone
  self_only / emotion_only / recency_only / random

Headline question: in the BLIND regime — the one that matches how
forgetting actually happens — does learned multi-factor V retain gold
better than recency / single-factor baselines, and how far below the
ORACLE ceiling does it sit?

Output: results/lme_blind_forgetting.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from borge.affective.signal_extractor import EmotionalSignalExtractor  # noqa: E402
from borge.eval.longmemeval import load_longmemeval, flatten_to_messages  # noqa: E402
from borge.memory.value import MemoryValue, learn_weights  # noqa: E402
from borge.values.self_model import SBertEmbedder, cosine  # noqa: E402

FACTORS = MemoryValue.FACTORS
# Factors actually populated by the API-free annotator. task_utility,
# usage, value_alignment are 0 here (need an LLM / access history / SOUL),
# so we LEARN only over the live factors — otherwise the optimiser leaves
# vacuous non-zero weights on dead (×0) factors.
LIVE_FACTORS = ("emotion", "goal_relevance", "self_relevance", "reliability")
_EXTRACTOR = EmotionalSignalExtractor()


def _sim01(a, b) -> float:
    return 0.5 + 0.5 * cosine(a, b)


def _centroid(vs):
    if not vs:
        return []
    dim = len(vs[0])
    return [sum(v[i] for v in vs) / len(vs) for i in range(dim)]


def annotate_dual(case, embedder) -> list[dict]:
    """
    One annotation per turn carrying BOTH goal variants.

    Query-agnostic factors (emotion, self_relevance, reliability) are
    computed once. goal_oracle = cos(turn, question); goal_blind =
    cos(turn, that turn's session-topic centroid over its user turns).
    """
    msgs = flatten_to_messages(case)
    contents = [m["content"] for m in msgs]
    if not contents:
        return []

    q_emb = embedder(case.question)
    turn_embs = [embedder(c) for c in contents]

    # global μ_user (self-relevance anchor)
    user_embs = [e for e, m in zip(turn_embs, msgs) if m["role"] == "user"]
    mu_user = _centroid(user_embs)

    # per-session user-turn centroid = "what is this session about" (blind)
    by_sess: dict[int, list] = {}
    for e, m in zip(turn_embs, msgs):
        if m["role"] == "user":
            by_sess.setdefault(m["session_idx"], []).append(e)
    sess_anchor = {s: _centroid(es) for s, es in by_sess.items()}

    out = []
    for m, emb in zip(msgs, turn_embs):
        dv, da = _EXTRACTOR.extract(m["content"], [])
        emotion = max(0.0, min(1.0, abs(dv) * (0.5 + da)))
        self_rel = _sim01(emb, mu_user) if mu_user else 0.5
        reliability = 0.7 if m["role"] == "user" else 0.4
        goal_oracle = _sim01(emb, q_emb)
        anchor = sess_anchor.get(m["session_idx"]) or mu_user
        goal_blind = _sim01(emb, anchor) if anchor else 0.5

        common = {
            "emotion":         emotion,
            "value_alignment": 0.0,
            "self_relevance":  self_rel,
            "task_utility":    0.0,
            "reliability":     reliability,
            "usage":           0.0,
        }
        out.append({
            "factors_oracle": {**common, "goal_relevance": goal_oracle},
            "factors_blind":  {**common, "goal_relevance": goal_blind},
            "has_answer":     m["has_answer"],
            "timestamp_idx":  m["session_idx"],
        })
    return out


def gold_retention(annotated, weights, *, regime: str, keep_frac: float):
    """Rank by V (desc) under the chosen regime's factors, keep top
    frac, return fraction of has_answer gold kept (None if no gold)."""
    key = f"factors_{regime}"
    mv = MemoryValue(weights=weights)
    scored = [(mv.value(a[key]), a) for a in annotated]
    scored.sort(key=lambda x: -x[0])
    k = max(1, int(len(scored) * keep_frac))
    kept = [a for _, a in scored[:k]]
    total = sum(1 for a in annotated if a["has_answer"])
    if total == 0:
        return None
    return sum(1 for a in kept if a["has_answer"]) / total


def recency_retention(annotated, *, keep_frac: float):
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


def _single(factor):
    return {f: (1.0 if f == factor else 0.0) for f in FACTORS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/longmemeval_s_cleaned.json")
    ap.add_argument("--n-cases", type=int, default=50)
    ap.add_argument("--keep-frac", type=float, default=0.3)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--reps", type=int, default=20, help="resampled train/test splits for CI")
    ap.add_argument("--out", default="results/lme_blind_forgetting.json")
    args = ap.parse_args()

    print("Loading SBert…")
    embedder = SBertEmbedder()
    _ = embedder("warm-up")

    print(f"Loading + annotating ≤{args.n_cases} cases (dual goal, batched)…")
    cases = []
    for i, case in enumerate(load_longmemeval(args.data)):
        if i >= args.n_cases:
            break
        ann = annotate_dual(case, embedder)
        if any(a["has_answer"] for a in ann):
            cases.append(ann)
        if (i + 1) % 10 == 0:
            print(f"  annotated {i+1}…")
    print(f"  usable cases (with has_answer gold): {len(cases)}")

    kf = args.keep_frac
    reps = max(1, args.reps)
    # Report ALL four live factors as single-factor baselines (reliability
    # included — it carries the largest learned weight, so omitting it would
    # let an "any single factor" claim cheat).
    policies = ("learned_V", "uniform_V", "goal_only", "self_only",
                "emotion_only", "reliability_only")

    def eval_on(test_set, regime, learned):
        pol = {
            "learned_V":         learned,
            "uniform_V":         {f: 1.0 for f in LIVE_FACTORS},
            "goal_only":         _single("goal_relevance"),
            "self_only":         _single("self_relevance"),
            "emotion_only":      _single("emotion"),
            "reliability_only":  _single("reliability"),
        }
        return {n: mean(gold_retention(a, w, regime=regime, keep_frac=kf)
                        for a in test_set)
                for n, w in pol.items()}

    # ── resampled train/test splits for a confidence interval ──────────
    # Annotation is the expensive (SBert) step and is done ONCE above;
    # each rep only reshuffles the split + re-fits weights (cheap dot
    # products), so a real CI is essentially free.
    rows = []
    for rep in range(reps):
        r = random.Random(100 + rep)
        shuffled = list(cases)
        r.shuffle(shuffled)
        split = len(shuffled) // 2
        tr, te = shuffled[:split], shuffled[split:]

        def obj(regime, tr=tr):
            return lambda w: mean(
                gold_retention(a, w, regime=regime, keep_frac=kf) for a in tr
            )
        # Learn over LIVE factors only (dead factors would get vacuous weights).
        lo, _ = learn_weights(obj("oracle"), LIVE_FACTORS, seed=1, iters=args.iters)
        lb, _ = learn_weights(obj("blind"), LIVE_FACTORS, seed=1, iters=args.iters)
        rows.append({
            "oracle":  eval_on(te, "oracle", lo),
            "blind":   eval_on(te, "blind", lb),
            "recency": mean(recency_retention(a, keep_frac=kf) for a in te),
            "w_blind": lb,
        })

    def agg(vals):
        m = sum(vals) / len(vals)
        sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
        return round(m, 4), round(sd, 4)

    ci = {"oracle": {}, "blind": {}}
    for regime in ("oracle", "blind"):
        for p in policies:
            ci[regime][p] = agg([row[regime][p] for row in rows])
    ci_recency = agg([row["recency"] for row in rows])
    avg_w_blind = {f: round(sum(row["w_blind"].get(f, 0.0) for row in rows) / reps, 4)
                   for f in LIVE_FACTORS}

    # Paired per-split differences (blind regime): same test split scored by
    # both policies, so the difference is paired and removes split variance.
    def paired(a_key, b_is_recency=False):
        diffs = [row["blind"]["learned_V"] -
                 (row["recency"] if b_is_recency else row["blind"][a_key])
                 for row in rows]
        m, s = agg(diffs)
        win = round(sum(1 for d in diffs if d > 0) / reps, 3)
        return {"mean": m, "std": s, "win_frac": win}
    paired_blind = {
        "learned_minus_uniform":  paired("uniform_V"),
        "learned_minus_recency":  paired(None, b_is_recency=True),
        "learned_minus_reliability_only": paired("reliability_only"),
    }

    payload = {
        "experiment": "REAL LongMemEval — BLIND vs ORACLE forgetting (API-free)",
        "ran_at": datetime.now().isoformat(),
        "data": args.data,
        "n_cases_used": len(cases),
        "reps": reps,
        "n_test_per_rep": len(cases) - len(cases) // 2,
        "keep_frac": kf,
        "live_factors": list(LIVE_FACTORS),
        "regimes": {
            "oracle": {
                "goal_relevance": "cos(turn, eval question) — peeks (retrieval ceiling)",
                "retention_mean_std": {p: ci["oracle"][p] for p in policies},
            },
            "blind": {
                "goal_relevance": "cos(turn, session-topic centroid) — realistic forgetting",
                "retention_mean_std": {p: ci["blind"][p] for p in policies},
            },
        },
        "recency_only_mean_std": ci_recency,
        "random_keep": round(kf, 4),
        "learned_weights_blind_mean": avg_w_blind,
        "paired_blind_diffs": paired_blind,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    ntest = len(cases) - len(cases) // 2
    print(f"\n=== BLIND vs ORACLE gold retention (keep {kf:.0%}, "
          f"{ntest} test/rep, {reps} reps; mean ± std) ===")
    for n in policies:
        m, s = ci["oracle"][n]
        print(f"  ORACLE  {n:14s}: {m:.3f} ± {s:.3f}")
    print("  " + "-" * 40)
    for n in policies:
        m, s = ci["blind"][n]
        print(f"  BLIND   {n:14s}: {m:.3f} ± {s:.3f}")
    m, s = ci_recency
    print(f"  BLIND   {'recency_only':14s}: {m:.3f} ± {s:.3f}")
    print(f"  {'':8s}{'random_keep':14s}: {kf:.3f}")
    print(f"\n  mean learned blind weights: " +
          ", ".join(f"{f}={avg_w_blind[f]:.2f}" for f in LIVE_FACTORS))
    print("  paired blind diffs (learned − X; mean ± std, win-frac):")
    for k, d in paired_blind.items():
        print(f"    {k:32s}: {d['mean']:+.3f} ± {d['std']:.3f}  ({d['win_frac']:.0%} of splits)")
    print(f"  → wrote {out}")


if __name__ == "__main__":
    main()
