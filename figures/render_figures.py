"""Render Figures 2-4 from the three E1/E2/E3 results JSONs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family":   "serif",
    "font.size":      10,
    "axes.linewidth": 0.8,
    "lines.linewidth":1.6,
    "figure.dpi":     150,
})

ROOT     = Path(__file__).resolve().parent
RESULTS  = ROOT.parent / "results"
OUT_DIR  = ROOT


def fig2_sre_bars():
    data = json.loads((RESULTS / "e1_sre_replication.json").read_text())
    conds = ["structural", "phonemic", "semantic", "self_ref"]
    means = [data["aggregate"][c]["mean_forget"] for c in conds]
    stds  = [data["aggregate"][c]["std_forget"]  for c in conds]
    srs   = [data["aggregate"][c]["self_relevance"] for c in conds]
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    bars = ax.bar(conds, means, yerr=stds, capsize=3,
                  color=["#cbd5e1", "#94a3b8", "#64748b", "#1e293b"],
                  edgecolor="black", linewidth=0.6)
    for b, sr in zip(bars, srs):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
                f"sr={sr:.2f}", ha="center", fontsize=8)
    ax.set_ylabel("Mean forget_score (lower = better remembered)")
    ax.set_title("E1 — SRE replication (10 seeds × 50 items)")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    out = OUT_DIR / "fig2_sre.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def fig3_factorial_interaction():
    data = json.loads((RESULTS / "e2_mood_self_factorial.json").read_text())
    pc   = data["per_cell"]
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    x = [0, 1]
    low_emo  = [pc["emotion_low_self_low"]["forget_mean"],
                pc["emotion_low_self_high"]["forget_mean"]]
    high_emo = [pc["emotion_high_self_low"]["forget_mean"],
                pc["emotion_high_self_high"]["forget_mean"]]
    ax.plot(x, low_emo,  "o-", label="emotion = low",  color="#3b82f6", markersize=7)
    ax.plot(x, high_emo, "s-", label="emotion = high", color="#ef4444", markersize=7)
    for xi, yi in zip(x, low_emo):
        ax.text(xi, yi + 0.05, f"{yi:.2f}", ha="center", fontsize=8, color="#3b82f6")
    for xi, yi in zip(x, high_emo):
        ax.text(xi, yi - 0.1, f"{yi:.2f}", ha="center", fontsize=8, color="#ef4444")
    ax.set_xticks(x)
    ax.set_xticklabels(["self_low (sr=0.1)", "self_high (sr=0.85)"])
    ax.set_ylabel("Mean forget_score")
    ax.set_title("E2 — Mood × Self factorial")
    raw = data["model_fit_raw"]
    ax.text(0.02, 0.02,
            f"raw-scale ΔAIC = {raw['delta_AIC']:+.0f}   "
            f"interaction coef = {raw['with_interaction']['coefs']['ES']:.3f}",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(facecolor="#fafafa", edgecolor="black", linewidth=0.4, pad=3))
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    out = OUT_DIR / "fig3_factorial.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def fig4_pi_ablation():
    data = json.loads((RESULTS / "e3_pi_self_ablation.json").read_text())
    pi_runs = data["pi_self_runs"]
    lam_runs = data["lambda_runs"]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.7))

    # E3a — π_self ablation (primary)
    pis  = [r["pi_self"]  for r in pi_runs]
    sre  = [r["sre_size"] for r in pi_runs]
    ax_a.plot(pis, sre, "o-", color="#7c3aed", markersize=7)
    ax_a.fill_between(pis, 0, sre, alpha=0.15, color="#7c3aed")
    ax_a.set_xlabel(r"$\pi_{\mathrm{self}}$  (self-precision)")
    ax_a.set_ylabel("SRE size  (semantic − self_ref forget gap)")
    ax_a.set_title("E3a — $\\pi_\\mathrm{self}$ ablation (primary)")
    ax_a.grid(alpha=0.3, linewidth=0.5)
    ax_a.set_axisbelow(True)
    ax_a.set_xlim(-0.05, 1.05)

    # E3b — λ sensitivity (secondary)
    lams = [r["lambda"]   for r in lam_runs]
    sreL = [r["sre_size"] for r in lam_runs]
    ax_b.plot(lams, sreL, "s-", color="#0891b2", markersize=7)
    peak = max(range(len(sreL)), key=lambda i: sreL[i])
    ax_b.axvline(lams[peak], color="#999", linestyle="--", linewidth=0.6)
    ax_b.annotate(f"peak λ={lams[peak]:.1f}",
                  xy=(lams[peak], sreL[peak]), xytext=(4, 5),
                  textcoords="offset points", fontsize=8)
    ax_b.set_xlabel(r"$\lambda$  (forget formula coef)")
    ax_b.set_ylabel("SRE size")
    ax_b.set_title("E3b — $\\lambda$ sensitivity (secondary)")
    ax_b.grid(alpha=0.3, linewidth=0.5)
    ax_b.set_axisbelow(True)

    plt.tight_layout()
    out = OUT_DIR / "fig4_pi_ablation.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def fig5_cohens_d_vs_sj():
    data = json.loads((RESULTS / "e1b_sre_forward_pass.json").read_text())
    d_per_seed = data["cohens_d"]["per_seed"]
    d_mean     = data["cohens_d"]["mean"]
    d_std      = data["cohens_d"]["std"]
    sj_d       = data["symons_johnson_1997"]["d"]
    sj_lo, sj_hi = data["symons_johnson_1997"]["ci_95"]

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    # Symons & Johnson 1997 reference band
    ax.axhspan(sj_lo, sj_hi, color="#fde68a", alpha=0.6,
               label=f"S&J 1997 d≈{sj_d} (CI {sj_lo}–{sj_hi})")
    ax.axhline(sj_d, color="#b45309", linewidth=1.2, linestyle="--")
    # Our per-seed dots
    xs = list(range(len(d_per_seed)))
    ax.scatter(xs, d_per_seed, s=42, color="#7c3aed",
               edgecolors="black", linewidths=0.6, zorder=3,
               label="Self-FEP Memory (per seed)")
    # Mean line
    ax.axhline(d_mean, color="#7c3aed", linewidth=1.6,
               label=f"Self-FEP mean d = {d_mean:.2f} ± {d_std:.2f}")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"seed {42 + i}" for i in xs], fontsize=8)
    ax.set_ylabel("Cohen's d  (self_ref vs semantic)")
    ax.set_title("E1b — Forward-pass SRE size vs S&J 1997")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    plt.tight_layout()
    out = OUT_DIR / "fig5_cohens_d.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def fig6_noise_robustness():
    data = json.loads((RESULTS / "e_noise_robustness.json").read_text())
    sweep = data["sweep"]
    sigmas = [r["sigma"] for r in sweep]
    means  = [r["d_mean"] for r in sweep]
    lows   = [r["d_ci_lo"] for r in sweep]
    highs  = [r["d_ci_hi"] for r in sweep]
    sj_lo, sj_hi = data["sj97_reference_band"]

    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    ax.axhspan(sj_lo, sj_hi, color="#fde68a", alpha=0.6,
               label=f"S&J 1997 band [{sj_lo}, {sj_hi}]")
    ax.errorbar(sigmas, means,
                yerr=[[m - lo for m, lo in zip(means, lows)],
                      [hi - m for hi, m in zip(highs, means)]],
                fmt="o-", color="#7c3aed", linewidth=1.6, markersize=6,
                capsize=4, label="Self-FEP (per σ, mean ± 95% CI)")
    ax.axhline(0.5, color="#b45309", linewidth=0.9, linestyle="--",
               label=f"S&J meta d = 0.50")
    ax.set_xlabel(r"Forget-score noise $\sigma$")
    ax.set_ylabel("Cohen's $d$  (self_ref vs semantic)")
    ax.set_title("L4 — Noise robustness of E1b Cohen's d")
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    plt.tight_layout()
    out = OUT_DIR / "fig6_noise_robustness.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def fig7_self_vs_baseline():
    data = json.loads((RESULTS / "e_self_vs_emotion_baseline.json").read_text())
    full = data["full_self_fep"]
    abl  = data["ablated_emotion_only"]
    gap  = data["self_contribution_gap"]

    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    labels = ["Full Self-FEP", "CMR-style\n(emotion only)", "Self contribution\n(full − ablated)"]
    means  = [full["d_mean"], abl["d_mean"], gap["mean"]]
    lows   = [full["d_ci"][0], abl["d_ci"][0], gap["ci"][0]]
    highs  = [full["d_ci"][1], abl["d_ci"][1], gap["ci"][1]]
    colors = ["#7c3aed", "#94a3b8", "#16a34a"]

    xs = list(range(3))
    for x, m, lo, hi, c in zip(xs, means, lows, highs, colors):
        ax.bar(x, m, color=c, edgecolor="black", linewidth=0.6,
               yerr=[[m - lo], [hi - m]], capsize=5)
        ax.text(x, hi + 0.04, f"{m:+.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.axhspan(0.30, 0.70, color="#fde68a", alpha=0.45,
               label="S&J 1997 band")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Cohen's $d$  (self_ref vs semantic)")
    ax.set_title("Self-FEP vs CMR-style baseline")
    ax.set_ylim(-0.4, 1.2)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    out = OUT_DIR / "fig7_self_vs_baseline.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


if __name__ == "__main__":
    print("Rendering figures …")
    fig2_sre_bars()
    fig3_factorial_interaction()
    fig4_pi_ablation()
    fig5_cohens_d_vs_sj()
    fig6_noise_robustness()
    fig7_self_vs_baseline()
    print("Done.")
