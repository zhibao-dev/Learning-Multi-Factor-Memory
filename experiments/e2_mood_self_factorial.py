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

        # Fit additive: log_forget = α + β_e·E + β_s·S
        # Fit multiplicative: log_forget = α + β_e·E + β_s·S + β_es·E·S
        add_aic, add_coefs, add_residuals = _fit_linear(all_rows, ["E", "S"])
        mul_aic, mul_coefs, mul_residuals = _fit_linear(all_rows, ["E", "S", "ES"])

        delta_aic = mul_aic - add_aic   # negative = multiplicative wins

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

        return {
            "n_per_cell":     n_per_cell,
            "per_cell":       per_cell,
            "model_fit": {
                "additive":       {"AIC": add_aic, "coefs": add_coefs},
                "multiplicative": {"AIC": mul_aic, "coefs": mul_coefs},
                "delta_AIC":      delta_aic,     # negative = mul wins
                "winner":         "multiplicative" if delta_aic < -2 else
                                  "additive"       if delta_aic >  2 else
                                  "tie",
            },
        }
    finally:
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass


# ── Tiny OLS implementation ──────────────────────────────────────────────

def _fit_linear(rows: list[dict], features: list[str]) -> tuple[float, dict, list[float]]:
    """OLS fit; returns (AIC, coefs_dict, residuals)."""
    n = len(rows)
    if n == 0:
        return 0.0, {}, []
    # Build X (n × p+1) with intercept, y (n)
    p = len(features) + 1
    X = [[1.0] + [_feature(r, f) for f in features] for r in rows]
    y = [r["log_forget"] for r in rows]
    # Normal equations: (XᵀX) β = Xᵀy
    XtX = [[sum(X[i][k] * X[i][j] for i in range(n)) for j in range(p)] for k in range(p)]
    Xty = [sum(X[i][k] * y[i] for i in range(n)) for k in range(p)]
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
    print()
    fit = result["model_fit"]
    print(f"  Additive       AIC = {fit['additive']['AIC']:.2f}   coefs={fit['additive']['coefs']}")
    print(f"  Multiplicative AIC = {fit['multiplicative']['AIC']:.2f}   coefs={fit['multiplicative']['coefs']}")
    print(f"  ΔAIC (mul − add)  = {fit['delta_AIC']:+.2f}   winner = {fit['winner']}")
    print(f"  → wrote {out}")


if __name__ == "__main__":
    main()
