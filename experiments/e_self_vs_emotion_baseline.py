"""
Head-to-head: Self-FEP Memory vs CMR-style emotion-only baseline.

CMR3 (Cohen & Kahana 2022) models emotion-modulated retrieval with no
self representation. A faithful CMR3 reimplementation is days of work
that v0.5 doesn't budget. The lightweight defensible alternative:
ablate the self component of our own model and see what's left. This
isolates "what does adding the self component buy?" — the most direct
test of the paper's contribution.

Two model conditions, run on the same E1b paradigm:

  (A) Self-FEP Memory (full):
      forget_score includes self_resistance = 1/(1 + λ · sr),
      retrieval ranking includes w_s · self_similarity.

  (B) Emotion-only ablation (CMR-style):
      self_relevance forced to 0.5 (neutral) for every memory,
      retrieval w_s = 0 (no self ranking).
      Only emotion_resistance × recency × usage × importance survive.

Prediction: ablation B should produce Cohen's d ≈ 0 for self-ref vs
semantic (no self mechanism → no SRE). The full A model recovers the
S&J-matching d. The d-gap A−B is the model's self-contribution.

Output: results/e_self_vs_emotion_baseline.json
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
             "curiosity, and depth over speed.")


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


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _cohens_d(self_ref: list[float], semantic: list[float]) -> float:
    if len(self_ref) < 2 or len(semantic) < 2:
        return 0.0
    m_self, m_sem = sum(self_ref)/len(self_ref), sum(semantic)/len(semantic)
    s_self, s_sem = _std(self_ref), _std(semantic)
    pooled = math.sqrt(
        ((len(self_ref) - 1) * s_self ** 2 + (len(semantic) - 1) * s_sem ** 2)
        / max(1, len(self_ref) + len(semantic) - 2)
    )
    return 0.0 if pooled < 1e-9 else -(m_self - m_sem) / pooled


def run_one_seed(seed: int, n_per_condition: int, noise_std: float,
                  embedder: SBertEmbedder, *, ablate_self: bool) -> float:
    """Run one E1b-style trial under either the full or ablated model."""
    rng = random.Random(seed)
    words = list(PERSONAL_TRAIT_WORDS)
    rng.shuffle(words)
    words = words[:n_per_condition]

    self_model = SelfModel.from_seed(SELF_SEED, embedder=embedder)
    self_model.pi_self = 1.0

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        store = MemoryStore(db_path)
        ts = (datetime.now() - timedelta(days=7)).isoformat()
        for cond in CONDITIONS:
            for i, w in enumerate(words):
                prompt = encoding_prompt(w, cond)
                emb = embedder(prompt)
                # ↓↓↓  THE ABLATION  ↓↓↓
                if ablate_self:
                    # CMR-style: force self_relevance to neutral 0.5
                    # → self_resistance always 1/(1 + λ·0.5) = constant
                    # → forget_score depends only on emotion + usage + recency + importance
                    sr = 0.5
                else:
                    sr = self_model.self_relevance(emb)
                store.insert({
                    "id":                   f"{cond}-{i}",
                    "session_id":           "head_to_head",
                    "role":                 "user",
                    "content":              prompt,
                    "timestamp":            ts,
                    "emotional_valence":    0.0,
                    "emotional_arousal":    0.5,
                    "self_relevance_score": sr,
                    "encoding_depth":       3,
                    "importance_score":     0.5,
                    "retrieval_count":      0,
                    "embedding":            emb,
                })

        ForgettingEngine().run_forgetting_pass(db_path)

        rng_noise = random.Random(seed * 31 + 7)
        forget_by_cond: dict[str, list[float]] = {c: [] for c in CONDITIONS}
        for r in store.all():
            for c in CONDITIONS:
                if r["id"].startswith(f"{c}-"):
                    noisy = float(r["forget_score"] or 0.0) + rng_noise.gauss(0.0, noise_std)
                    forget_by_cond[c].append(noisy)
                    break

        return _cohens_d(forget_by_cond["self_ref"], forget_by_cond["semantic"])
    finally:
        try: os.unlink(db_path)
        except FileNotFoundError: pass


def _bootstrap_ci(xs: list[float], n_boot: int = 5000,
                   alpha: float = 0.05, seed: int = 1) -> tuple[float, float]:
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
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--n-per-condition", type=int, default=20)
    ap.add_argument("--noise-std", type=float, default=0.05)
    ap.add_argument("--out", default="results/e_self_vs_emotion_baseline.json")
    args = ap.parse_args()

    print("Loading SBert…")
    embedder = SBertEmbedder()
    _ = embedder("warm-up")

    full_ds: list[float] = []
    ablated_ds: list[float] = []
    print(f"\n=== Self-FEP (full) vs CMR-style (emotion-only) — N={args.n_seeds} seeds ===")
    print(f"  {'seed':>6s}  {'full d':>9s}  {'ablated d':>11s}  {'gap':>8s}")
    for s in range(args.n_seeds):
        d_full    = run_one_seed(42+s, args.n_per_condition,
                                  args.noise_std, embedder, ablate_self=False)
        d_ablated = run_one_seed(42+s, args.n_per_condition,
                                  args.noise_std, embedder, ablate_self=True)
        full_ds.append(d_full)
        ablated_ds.append(d_ablated)
        print(f"  {42+s:>6d}  {d_full:>+9.4f}  {d_ablated:>+11.4f}  "
              f"{d_full - d_ablated:>+8.4f}")

    full_mean    = sum(full_ds) / len(full_ds)
    ablated_mean = sum(ablated_ds) / len(ablated_ds)
    gap_per_seed = [a - b for a, b in zip(full_ds, ablated_ds)]
    gap_mean     = sum(gap_per_seed) / len(gap_per_seed)
    full_ci      = _bootstrap_ci(full_ds)
    ablated_ci   = _bootstrap_ci(ablated_ds)
    gap_ci       = _bootstrap_ci(gap_per_seed)

    payload = {
        "experiment":            "Self-FEP (full) vs CMR-style emotion-only baseline",
        "ran_at":                datetime.now().isoformat(),
        "n_seeds":               args.n_seeds,
        "n_per_condition":       args.n_per_condition,
        "noise_std":             args.noise_std,
        "embedder":              "sentence-transformers/all-MiniLM-L6-v2",
        "full_self_fep": {
            "d_mean":   round(full_mean, 4),
            "d_ci":     [round(full_ci[0], 4), round(full_ci[1], 4)],
            "d_per_seed": [round(x, 4) for x in full_ds],
        },
        "ablated_emotion_only": {
            "d_mean":   round(ablated_mean, 4),
            "d_ci":     [round(ablated_ci[0], 4), round(ablated_ci[1], 4)],
            "d_per_seed": [round(x, 4) for x in ablated_ds],
        },
        "self_contribution_gap": {
            "mean":     round(gap_mean, 4),
            "ci":       [round(gap_ci[0], 4), round(gap_ci[1], 4)],
            "per_seed": [round(x, 4) for x in gap_per_seed],
        },
        "interpretation": (
            "The emotion-only ablation is the CMR-style baseline (no self "
            "representation). The full model's Cohen's d minus the ablated "
            "model's Cohen's d isolates the self mechanism's contribution "
            "to the SRE effect size. A 95% CI on the gap that excludes 0 "
            "indicates the self component is responsible for the observed "
            "SRE-direction effect."
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    print()
    print(f"=== Aggregate ===")
    print(f"  Full Self-FEP:    d = {full_mean:+.4f}  CI {full_ci}")
    print(f"  CMR-style baseline: d = {ablated_mean:+.4f}  CI {ablated_ci}")
    print(f"  Self contribution (gap): {gap_mean:+.4f}  CI {gap_ci}")
    print(f"  Gap CI excludes 0: {gap_ci[0] > 0 or gap_ci[1] < 0}")
    print(f"  → wrote {out}")


if __name__ == "__main__":
    main()
