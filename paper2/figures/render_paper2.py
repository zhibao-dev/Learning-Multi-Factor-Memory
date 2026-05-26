"""
Render paper2 data figure(s) from results JSON. CPU, deterministic.

Fig: blind-vs-oracle gold retention grouped bars (the headline).
Reads results/lme_blind_forgetting.json, writes paper2/figures/fig_blind_oracle.pdf.

Run:  python paper2/figures/render_paper2.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # repo root


def load():
    with open(os.path.join(ROOT, "results", "lme_blind_forgetting.json")) as f:
        return json.load(f)


def fig_blind_oracle(d):
    oracle = d["regimes"]["oracle"]["retention_mean_std"]
    blind = d["regimes"]["blind"]["retention_mean_std"]
    rec = d["recency_only_mean_std"]
    chance = d["random_keep"]

    order = ["learned_V", "uniform_V", "reliability_only", "self_only",
             "recency_only", "goal_only", "emotion_only"]
    label = {"learned_V": "learned\nV", "uniform_V": "uniform\nV",
             "reliability_only": "reliability\nonly", "self_only": "self\nonly",
             "recency_only": "recency\nonly", "goal_only": "goal\nonly",
             "emotion_only": "emotion\nonly"}

    def get(reg, p):
        if p == "recency_only":
            return (rec[0], rec[1]) if reg == "blind" else (None, None)
        src = oracle if reg == "oracle" else blind
        return tuple(src[p]) if p in src else (None, None)

    def mz(v):
        return [0.0 if x is None else x for x in v]

    x = np.arange(len(order))
    w = 0.38
    o_m = [get("oracle", p)[0] for p in order]
    o_s = [get("oracle", p)[1] for p in order]
    b_m = [get("blind", p)[0] for p in order]
    b_s = [get("blind", p)[1] for p in order]

    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.bar(x - w / 2, mz(o_m), w, yerr=mz(o_s), capsize=2,
           color="#c8d6e8", edgecolor="#5a6b85", label="oracle (peeks at $Q$)")
    ax.bar(x + w / 2, mz(b_m), w, yerr=mz(b_s), capsize=2,
           color="#2e8b57", edgecolor="#16432a", label="blind (realistic)")
    ax.axhline(chance, ls="--", c="gray", lw=1.0,
               label=f"chance ({chance:.2f})")

    # annotate the learned_V blind bar (the headline)
    lv = get("blind", "learned_V")[0]
    ax.annotate(f"{lv:.2f}", (0 + w / 2, lv + 0.02), ha="center",
                fontsize=8, fontweight="bold", color="#16432a")

    ax.set_xticks(x)
    ax.set_xticklabels([label[p] for p in order], fontsize=8)
    ax.set_ylabel("gold-evidence retention")
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.92)
    ax.set_title("Blind vs. oracle forgetting "
                 "(keep 30%, 74-case pilot, 20 splits; mean " r"$\pm$ 1 std)",
                 fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = os.path.join(HERE, "fig_blind_oracle.pdf")
    plt.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    d = load()
    fig_blind_oracle(d)
