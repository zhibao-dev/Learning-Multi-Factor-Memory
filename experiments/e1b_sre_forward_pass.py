"""
E1b — True forward-pass SRE replication with sentence-transformers.

E1 (mechanical) hand-set self_relevance per condition. E1b puts the
encoding-task framing prompts through the actual SelfModel forward pass
and asks whether the agent's stored self_relevance score per condition
matches the depth-of-processing prediction WITHOUT any manual typing.

Design (mirrors the classic SRE paradigm):

  Subject's identity is seeded once: "I am an honest careful learner..."

  For each of 50 trait words (honest, brave, kind, …) under each of the
  4 encoding-task framings:
      structural : "Does the word 'X' contain any letters with curves?"
      phonemic   : "Does the word 'X' rhyme with 'wind'?"
      semantic   : "Is the word 'X' a synonym for 'patient'?"
      self_ref   : "Does the word 'X' describe me?"

  → The full prompt is embedded by SBertEmbedder; the SelfModel
    computes self_relevance against the seeded μ_self.

  → No condition-typed sr is set externally. The entire experiment is
    a forward pass from raw text to forget_score.

  → Insert into borge_memories; run forgetting pass; read out mean
    forget_score per condition. Compare to E1.

  → ALSO compute Cohen's d between self_ref and semantic conditions
    (the standard SRE effect-size metric). Compare to Symons &
    Johnson (1997) meta-analytic d ≈ 0.50.

Output: results/e1b_sre_forward_pass.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from borge.memory.forgetting import ForgettingEngine  # noqa: E402
from borge.memory.store import MemoryStore  # noqa: E402
from borge.values.self_model import SBertEmbedder, SelfModel  # noqa: E402


PERSONAL_TRAIT_WORDS = [
    "honest", "curious", "patient", "stubborn", "kind", "anxious", "brave",
    "lazy", "thoughtful", "impulsive", "cautious", "generous", "selfish",
    "creative", "shy", "outgoing", "loyal", "moody", "calm", "fierce",
    "humble", "proud", "playful", "serious", "warm", "cold", "diligent",
    "messy", "punctual", "forgetful", "decisive", "indecisive", "open",
    "guarded", "optimistic", "pessimistic", "reliable", "spontaneous",
    "analytical", "intuitive", "empathetic", "detached", "ambitious",
    "content", "restless", "focused", "scattered", "compassionate",
    "harsh", "gentle",
]

CONDITIONS = ("structural", "phonemic", "semantic", "self_ref")
SELF_SEED = ("I am an honest careful learner who values truth, "
             "curiosity, and depth over speed. I take responsibility "
             "for my mistakes and try to grow from them.")


def encoding_prompt(word: str, condition: str) -> str:
    if condition == "structural":
        return f"Does the word '{word}' contain any letters with curves like o or b?"
    if condition == "phonemic":
        return f"Does the word '{word}' rhyme with the word 'wind'?"
    if condition == "semantic":
        return f"Is the word '{word}' a synonym for 'patient'?"
    if condition == "self_ref":
        return f"Does the word '{word}' describe me?"
    raise ValueError(condition)


def run_experiment(
    n_items_per_condition: int,
    seed: int,
    age_days: int,
    noise_std: float,
    embedder: SBertEmbedder,
) -> dict:
    """One run: embed every trait+condition prompt, forward sr, forget, score."""
    rng = random.Random(seed)

    # Random word subset, deterministic per seed
    words = list(PERSONAL_TRAIT_WORDS)
    rng.shuffle(words)
    words = words[: n_items_per_condition]

    # Fresh SelfModel per run with the SBert embedder
    self_model = SelfModel.from_seed(SELF_SEED, embedder=embedder)
    # Pin precision at 1 so the forward sr is the raw cosine-derived score
    self_model.pi_self = 1.0

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        store = MemoryStore(db_path)
        ts = (datetime.now() - timedelta(days=age_days)).isoformat()
        sr_by_cond: dict[str, list[float]] = {c: [] for c in CONDITIONS}

        # Encoding phase — embed each prompt, compute sr, store
        for cond in CONDITIONS:
            for i, w in enumerate(words):
                prompt = encoding_prompt(w, cond)
                emb = embedder(prompt)
                sr  = self_model.self_relevance(emb)
                sr_by_cond[cond].append(sr)
                store.insert({
                    "id":                   f"{cond}-{i}",
                    "session_id":           "e1b",
                    "role":                 "user",
                    "content":              prompt,
                    "timestamp":            ts,
                    "emotional_valence":    0.0,
                    "emotional_arousal":    0.5,
                    "self_relevance_score": sr,
                    "encoding_depth":       3,
                    "importance_score":     0.5,
                    "retrieval_count":      0,
                })

        # Forgetting pass with optional Gaussian noise on forget_score
        ForgettingEngine().run_forgetting_pass(db_path)
        rng_noise = random.Random(seed * 31 + 7)
        forget_by_cond: dict[str, list[float]] = {c: [] for c in CONDITIONS}
        rows = store.all()
        for r in rows:
            for cond in CONDITIONS:
                if r["id"].startswith(f"{cond}-"):
                    base = float(r["forget_score"] or 0.0)
                    noisy = base + rng_noise.gauss(0.0, noise_std)
                    forget_by_cond[cond].append(noisy)
                    break

        # Cell summaries
        per_cond = {}
        for cond in CONDITIONS:
            srs = sr_by_cond[cond]
            fs = forget_by_cond[cond]
            per_cond[cond] = {
                "sr_mean":      round(sum(srs) / len(srs), 4) if srs else 0.0,
                "sr_std":       round(_std(srs), 4),
                "forget_mean":  round(sum(fs) / len(fs), 4) if fs else 0.0,
                "forget_std":   round(_std(fs), 4),
                "n":            len(srs),
            }

        # Cohen's d between self_ref and semantic, in forget space.
        # Lower forget = better remembered; flip sign so positive d means
        # self_ref is better remembered (matches the human convention).
        d = _cohens_d_signed_for_recall(
            forget_by_cond["self_ref"],
            forget_by_cond["semantic"],
        )

        return {
            "seed":          seed,
            "n_items":       n_items_per_condition,
            "per_condition": per_cond,
            "ordering_ok": (per_cond["self_ref"]["forget_mean"]
                            < per_cond["semantic"]["forget_mean"]),
            "cohens_d_self_vs_semantic": round(d, 4),
        }
    finally:
        try: os.unlink(db_path)
        except FileNotFoundError: pass


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _cohens_d_signed_for_recall(self_ref: list[float], semantic: list[float]) -> float:
    """
    Cohen's d for forget_score difference, flipped so positive d = better
    self_ref recall (matching the empirical convention).
    """
    if len(self_ref) < 2 or len(semantic) < 2:
        return 0.0
    m_self, m_sem = sum(self_ref)/len(self_ref), sum(semantic)/len(semantic)
    s_self, s_sem = _std(self_ref), _std(semantic)
    pooled = math.sqrt(((len(self_ref) - 1) * s_self ** 2
                      + (len(semantic) - 1) * s_sem ** 2)
                      / max(1, len(self_ref) + len(semantic) - 2))
    if pooled < 1e-9:
        return 0.0
    raw = (m_self - m_sem) / pooled       # negative in forget space when self_ref wins
    return -raw                            # flip → positive = self_ref better recalled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-condition", type=int, default=30)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--age-days", type=int, default=7)
    ap.add_argument("--noise-std", type=float, default=0.05,
                    help="Gaussian std added to each forget_score (default 0.05)")
    ap.add_argument("--out", default="results/e1b_sre_forward_pass.json")
    args = ap.parse_args()

    print("Loading SBert (first call downloads ~80 MB)…")
    embedder = SBertEmbedder()
    _ = embedder("warm-up")

    runs = []
    print(f"\n=== E1b SBert forward-pass SRE — N={args.n_per_condition} per condition × {args.n_seeds} seeds ===")
    print(f"  {'seed':>6s}  {'self_ref_sr':>12s}  {'sem_sr':>9s}  {'self_forget':>12s}  {'sem_forget':>11s}  {'d':>7s}")
    for s in range(args.n_seeds):
        r = run_experiment(args.n_per_condition,
                           seed=42 + s,
                           age_days=args.age_days,
                           noise_std=args.noise_std,
                           embedder=embedder)
        runs.append(r)
        pc = r["per_condition"]
        print(f"  {42+s:>6d}  {pc['self_ref']['sr_mean']:>12.4f}  {pc['semantic']['sr_mean']:>9.4f}  "
              f"{pc['self_ref']['forget_mean']:>12.4f}  {pc['semantic']['forget_mean']:>11.4f}  "
              f"{r['cohens_d_self_vs_semantic']:>+7.3f}")

    # Aggregate
    agg = {}
    for cond in CONDITIONS:
        srs    = [r["per_condition"][cond]["sr_mean"]     for r in runs]
        forgets= [r["per_condition"][cond]["forget_mean"] for r in runs]
        agg[cond] = {
            "sr_mean_across_seeds":     round(sum(srs)/len(srs), 4),
            "forget_mean_across_seeds": round(sum(forgets)/len(forgets), 4),
        }
    ds = [r["cohens_d_self_vs_semantic"] for r in runs]
    d_mean = sum(ds) / len(ds)
    d_std  = _std(ds)
    ordering_hits = sum(1 for r in runs if r["ordering_ok"])

    SJ_d = 0.50  # Symons & Johnson 1997 meta-analytic Cohen's d for SRE
    SJ_band = (0.30, 0.70)  # rough 95% CI bracket from the meta-analysis

    payload = {
        "experiment":       "E1b — True forward-pass SRE with SBert",
        "embedder":         "sentence-transformers/all-MiniLM-L6-v2",
        "ran_at":           datetime.now().isoformat(),
        "n_seeds":          args.n_seeds,
        "n_per_condition":  args.n_per_condition,
        "noise_std":        args.noise_std,
        "self_seed":        SELF_SEED,
        "aggregate":        agg,
        "cohens_d": {
            "mean":     round(d_mean, 4),
            "std":      round(d_std, 4),
            "per_seed": ds,
        },
        "symons_johnson_1997": {
            "d":             SJ_d,
            "ci_95":         SJ_band,
            "within_band":   SJ_band[0] <= d_mean <= SJ_band[1],
        },
        "ordering_hits":    f"{ordering_hits}/{args.n_seeds}",
        "per_seed":         runs,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"\n=== Aggregate across {args.n_seeds} seeds ===")
    for cond in CONDITIONS:
        print(f"  {cond:11s}: sr={agg[cond]['sr_mean_across_seeds']:.4f}   "
              f"forget={agg[cond]['forget_mean_across_seeds']:.4f}")
    print(f"\n  Cohen's d (self_ref vs semantic): {d_mean:+.4f} ± {d_std:.4f}")
    print(f"  Symons & Johnson 1997 meta-analysis d ≈ {SJ_d}  (CI ~{SJ_band})")
    print(f"  Within human band: {payload['symons_johnson_1997']['within_band']}")
    print(f"  Ordering matches human direction: {ordering_hits}/{args.n_seeds}")
    print(f"  → wrote {out}")


if __name__ == "__main__":
    main()
