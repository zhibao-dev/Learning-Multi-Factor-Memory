"""
E2 — Mood × Self factorial: additive vs multiplicative.

Borge's forget_score combines emotion and self-relevance multiplicatively:

  forget = recency × usage × imp_res × emotion_res × self_res

The competing additive hypothesis (Bower 1981 associative network style)
would predict the two factors enter as separable terms.

This experiment runs a 2 × 2 factorial:

  emotion ∈ {low (|V|·A ≈ 0.0), high (|V|·A ≈ 0.7)}
  self    ∈ {low (sr = 0.1),       high (sr = 0.85)}

For each of the 4 cells, N memories with controlled features are
inserted; forgetting pass populates forget_score; we then fit two
linear models to the log-forget surface

  additive:        log(forget) = α + β_e·E + β_s·S          (+ Gaussian noise)
  multiplicative:  log(forget) = α + β_e·E + β_s·S + β_es·E·S

and report ΔAIC. A negative ΔAIC favours the multiplicative form,
which is the model's prediction.

Output: results/e2_mood_self_factorial.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from borge.memory.forgetting import ForgettingEngine  # noqa: E402
from borge.memory.store import MemoryStore  # noqa: E402

# 2×2 cell definitions
CELLS = {
    ("emotion_low", "self_low"):   {"V": 0.0,  "A": 0.5, "sr": 0.10},
    ("emotion_low", "self_high"):  {"V": 0.0,  "A": 0.5, "sr": 0.85},
    ("emotion_high","self_low"):   {"V": 0.8,  "A": 0.9, "sr": 0.10},
    ("emotion_high","self_high"):  {"V": 0.8,  "A": 0.9, "sr": 0.85},
}


@dataclass
class CellResult:
    emotion: str          # "low" / "high"
    self_lvl: str         # "low" / "high"
    n: int
    forget_mean: float
    forget_std: float


def run_cell(db_path: str, cell_id: tuple, n: int, age_days: int) -> CellResult:
    cfg = CELLS[cell_id]
    store = MemoryStore(db_path)
    ts = (datetime.now() - timedelta(days=age_days)).isoformat()
    e_lvl, s_lvl = cell_id
    for i in range(n):
        store.insert({
            "id":                   f"{e_lvl}_{s_lvl}_{i}",
            "session_id":           "factorial",
            "role":                 "user",
            "content":              "x",
            "timestamp":            ts,
            "emotional_valence":    cfg["V"],
            "emotional_arousal":    cfg["A"],
            "self_relevance_score": cfg["sr"],
            "encoding_depth":       3,   # SCHEMATIC (no compression/deletion)
            "importance_score":     0.5,
            "retrieval_count":      0,
        })
    return CellResult(emotion=e_lvl.split("_")[1],
                       self_lvl=s_lvl.split("_")[1],
                       n=n, forget_mean=0, forget_std=0)


def run_experiment(n_per_cell: int, age_days: int) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        store = MemoryStore(db_path)
        for cell_id in CELLS:
            run_cell(db_path, cell_id, n_per_cell, age_days)

        ForgettingEngine().run_forgetting_pass(db_path)

        # Collect forget_score by cell + build regression data
        cell_data: dict[tuple, list[float]] = {c: [] for c in CELLS}
        all_rows: list[dict] = []
        for r in store.all():
            for cell_id in CELLS:
                e_lvl, s_lvl = cell_id
                if r["id"].startswith(f"{e_lvl}_{s_lvl}_"):
                    cell_data[cell_id].append(r["forget_score"])
                    all_rows.append({
                        "E": 1.0 if e_lvl == "emotion_high" else 0.0,
                        "S": 1.0 if s_lvl == "self_high"    else 0.0,
                        "log_forget": math.log(max(r["forget_score"], 1e-6)),
                    })
                    break

        # Two model comparisons:
        #
        # (i)  Raw-scale: tests whether forget_score is recovered by an
        #      additive vs. additive+interaction linear model. Because the
        #      true formula is multiplicative in linear space, we expect
        #      the interaction term to carry non-trivial weight here.
        #
        # (ii) Log-scale: tests whether log(forget_score) needs an
        #      interaction. Because the true formula is a product of
        #      terms, log(forget_score) is additive in log of each term,
        #      so the interaction here is NOT predicted to win. This
        #      comparison serves as a sanity check that the experimental
        #      data has the multiplicative structure we believe (positive
        #      coefficients in log-space, near-zero interaction).
        all_raw = [
            {"E": r["E"], "S": r["S"], "forget": math.exp(r["log_forget"])}
            for r in all_rows
        ]
        for r in all_raw:
            r["target"] = r["forget"]
        for r in all_rows:
            r["target"] = r["log_forget"]

        raw_add_aic, raw_add_coefs, _ = _fit_linear(all_raw,  ["E", "S"],          target="target")
        raw_mul_aic, raw_mul_coefs, _ = _fit_linear(all_raw,  ["E", "S", "ES"],    target="target")
        log_add_aic, log_add_coefs, _ = _fit_linear(all_rows, ["E", "S"],          target="target")
        log_mul_aic, log_mul_coefs, _ = _fit_linear(all_rows, ["E", "S", "ES"],    target="target")

        delta_aic_raw = raw_mul_aic - raw_add_aic   # primary test
        delta_aic_log = log_mul_aic - log_add_aic   # sanity check

        # Cell summaries
        per_cell = {}
        for cell_id, scores in cell_data.items():
            m = sum(scores) / len(scores) if scores else 0
            v = sum((x - m) ** 2 for x in scores) / len(scores) if scores else 0
            per_cell[f"{cell_id[0]}_{cell_id[1]}"] = {
                "n":           len(scores),
                "forget_mean": round(m, 4),
                "forget_std":  round(v ** 0.5, 4),
            }

        # Interaction-magnitude diagnostic: |delta_high - delta_low|
        # delta_low  = forget(self_high | emotion_low)  - forget(self_low | emotion_low)
        # delta_high = forget(self_high | emotion_high) - forget(self_low | emotion_high)
        # Pure additivity => delta_low == delta_high. Multiplicative gives
        # |delta_low| > |delta_high| (or vice versa) by a non-trivial margin.
        m = per_cell
        delta_low  = m["emotion_low_self_high"]["forget_mean"]  - m["emotion_low_self_low"]["forget_mean"]
        delta_high = m["emotion_high_self_high"]["forget_mean"] - m["emotion_high_self_low"]["forget_mean"]
        interaction_magnitude = abs(delta_low - delta_high)

        return {
            "n_per_cell":     n_per_cell,
            "per_cell":       per_cell,
            "interaction_diagnostic": {
                "delta_low":             round(delta_low, 4),
                "delta_high":            round(delta_high, 4),
                "interaction_magnitude": round(interaction_magnitude, 4),
                "interpretation":        ("Pure additivity predicts "
                                          "delta_low ≈ delta_high; any "
                                          "non-trivial difference signals "
                                          "interaction."),
            },
            "model_fit_raw": {
                "additive":       {"AIC": raw_add_aic, "coefs": raw_add_coefs},
                "with_interaction": {"AIC": raw_mul_aic, "coefs": raw_mul_coefs},
                "delta_AIC":      delta_aic_raw,
                "winner":         "with_interaction" if delta_aic_raw < -2 else
                                  "additive"        if delta_aic_raw >  2 else
                                  "tie",
                "note":           "Primary test: raw-scale linear regression. "
                                  "Multiplicative ground-truth predicts the "
                                  "interaction term wins on raw scale.",
            },
            "model_fit_log": {
                "additive":         {"AIC": log_add_aic, "coefs": log_add_coefs},
                "with_interaction": {"AIC": log_mul_aic, "coefs": log_mul_coefs},
                "delta_AIC":        delta_aic_log,
                "winner":           "with_interaction" if delta_aic_log < -2 else
                                    "additive"        if delta_aic_log >  2 else
                                    "tie",
                "note":             "Sanity check: log-scale. Because the true "
                                    "formula is a product of terms, "
                                    "log-additivity is predicted; the "
                                    "interaction term should ROT win here.",
            },
        }
    finally:
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass


# ── Tiny OLS implementation ──────────────────────────────────────────────

def _fit_linear(rows: list[dict], features: list[str], target: str = "log_forget") -> tuple[float, dict, list[float]]:
    """OLS fit; returns (AIC, coefs_dict, residuals)."""
    n = len(rows)
    if n == 0:
        return 0.0, {}, []
    # Build X (n × p+1) with intercept, y (n)
    p = len(features) + 1
    X = [[1.0] + [_feature(r, f) for f in features] for r in rows]
    y = [r[target] for r in rows]
    # Normal equations: (XᵀX) β = Xᵀy
    XtX = [[sum(X[i][k] * X[i][j] for i in range(n)) for j in range(p)] for k in range(p)]
    Xty: list[float] = [float(sum(X[i][k] * y[i] for i in range(n))) for k in range(p)]
    beta = _solve_linear(XtX, Xty)
    # Residuals
    y_hat = [sum(beta[k] * X[i][k] for k in range(p)) for i in range(n)]
    resid = [y[i] - y_hat[i] for i in range(n)]
    rss   = sum(r * r for r in resid)
    # AIC = n·ln(RSS/n) + 2·p   (Gaussian likelihood, ignoring constants)
    aic = n * math.log(max(rss / n, 1e-12)) + 2 * p
    coefs = {"intercept": round(beta[0], 4)}
    for j, f in enumerate(features):
        coefs[f] = round(beta[j + 1], 4)
    return round(aic, 4), coefs, resid


def _feature(row: dict, name: str) -> float:
    if name in row:
        return row[name]
    if name == "ES":
        return row["E"] * row["S"]
    raise ValueError(f"unknown feature: {name}")


def _solve_linear(A: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination for small symmetric positive-semi-def systems."""
    n = len(A)
    # Augment
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        # Pivot
        pivot = M[i][i]
        for j in range(i + 1, n):
            if abs(M[j][i]) > abs(pivot):
                M[i], M[j] = M[j], M[i]
                pivot = M[i][i]
        if abs(pivot) < 1e-12:
            return [0.0] * n
        for j in range(i + 1, n):
            factor = M[j][i] / pivot
            for k in range(i, n + 1):
                M[j][k] -= factor * M[i][k]
    # Back-substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n]
        for j in range(i + 1, n):
            s -= M[i][j] * x[j]
        x[i] = s / M[i][i]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-cell", type=int, default=100)
    ap.add_argument("--age-days",   type=int, default=7)
    ap.add_argument("--out", default="results/e2_mood_self_factorial.json")
    args = ap.parse_args()

    result = run_experiment(args.n_per_cell, args.age_days)
    payload = {
        "experiment":  "E2 — Mood × Self factorial (additive vs multiplicative)",
        "ran_at":      datetime.now().isoformat(),
        **result,
        "note": ("Predicts ΔAIC < -2 (multiplicative wins) because the "
                 "forget_score formula multiplies emotion_resistance and "
                 "self_resistance."),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"\n=== E2 Mood × Self Factorial — {args.n_per_cell} items per cell ===")
    for cell, stats in result["per_cell"].items():
        print(f"  {cell:30s}  forget = {stats['forget_mean']:.4f}")

    diag = result["interaction_diagnostic"]
    print(f"\n  Interaction diagnostic (cell means in raw forget-score space):")
    print(f"    delta_low  (S_high - S_low @ E_low)  = {diag['delta_low']:+.4f}")
    print(f"    delta_high (S_high - S_low @ E_high) = {diag['delta_high']:+.4f}")
    print(f"    |delta_low - delta_high|             = {diag['interaction_magnitude']:.4f}   "
          f"(pure additive predicts 0)")

    raw = result["model_fit_raw"]
    log = result["model_fit_log"]
    print(f"\n  Raw-scale OLS (primary test):")
    print(f"    additive          AIC = {raw['additive']['AIC']:.2f}   {raw['additive']['coefs']}")
    print(f"    with_interaction  AIC = {raw['with_interaction']['AIC']:.2f}   {raw['with_interaction']['coefs']}")
    print(f"    ΔAIC = {raw['delta_AIC']:+.2f}   winner = {raw['winner']}")
    print(f"\n  Log-scale OLS (sanity check):")
    print(f"    additive          AIC = {log['additive']['AIC']:.2f}   {log['additive']['coefs']}")
    print(f"    with_interaction  AIC = {log['with_interaction']['AIC']:.2f}   {log['with_interaction']['coefs']}")
    print(f"    ΔAIC = {log['delta_AIC']:+.2f}   winner = {log['winner']}")
    print(f"  → wrote {out}")


if __name__ == "__main__":
    main()
