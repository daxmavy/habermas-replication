"""Figures for the Fig. 4C replication report. Reads report/values.json only -- no literal numbers.

    uv run python report/figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "figures"

# Okabe-Ito: the standard CVD-safe qualitative palette. Fixed order, never cycled.
C_PAPER, C_BASE, C_LARGE = "#0072B2", "#E69F00", "#009E73"
C_RULE, C_INK, C_MUTED = "#333333", "#222222", "#666666"

PHASES = ["initial_candidates", "initial_winner", "revised_candidates", "revised_winner"]
LABELS = {"opinions": "Opinions\n(sanity check)", "initial_candidates": "Initial\nstatements",
          "initial_winner": "Initial\nwinner", "revised_candidates": "Revised\nstatements",
          "revised_winner": "Revised\nwinner"}

plt.rcParams.update({
    "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#999999", "axes.labelcolor": C_INK, "text.color": C_INK,
    "xtick.color": C_MUTED, "ytick.color": C_MUTED, "figure.dpi": 200,
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.6, "axes.axisbelow": True,
})


def _bar_labels(ax, bars, vals, fmt="{:.2f}", dy=0.006, ses=None):
    """Label above the bar, clearing the error-bar cap where there is one."""
    ses = ses or [None] * len(vals)
    for b, v, se in zip(bars, vals, ses):
        top = v + (se or 0)
        ax.text(b.get_x() + b.get_width() / 2, top + dy, fmt.format(v), ha="center", va="bottom", fontsize=6.5, color=C_INK)


def fig_paper(V, path):
    """The paper's own Fig. 4C values (SM Fig. S60, position-embedding panel), redrawn."""
    p = V["paper"]
    keys = ["opinions_sanity"] + PHASES
    vals = [p[k] for k in keys]
    names = [LABELS["opinions"]] + [LABELS[k] for k in PHASES]
    fig, ax = plt.subplots(figsize=(5.0, 2.5))
    bars = ax.bar(names, vals, color=C_PAPER, width=0.62)
    ax.errorbar(len(vals) - 1, p["revised_winner"], yerr=p["revised_winner_se"], fmt="none", ecolor=C_INK, capsize=2.5, lw=1)
    ax.axhline(p["minority_share"], ls=":", lw=1, color=C_RULE)
    ax.text(len(vals) - 0.45, p["minority_share"], " true share", va="center", fontsize=6.5, color=C_RULE)
    _bar_labels(ax, bars, vals, ses=[None] * 4 + [p["revised_winner_se"]])
    ax.set_ylabel("Weight of minority opinions")
    ax.set_ylim(0, 0.45)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_contrast(V, path):
    """Paper vs. our reproduction under the primary embedding model, across the four phases.
    Error bars: +/- 1 SE of the estimated regression coefficients (SM Fig. S60 convention)."""
    p, m = V["paper"], V["primary_model"]
    o = V["ours"][m]
    keys = ["opinions_sanity"] + PHASES
    series = [("Paper (Tessler et al.)", [p[k] for k in keys], [None] * 4 + [p["revised_winner_se"]], C_PAPER),
              (f"Ours ({m})", [o["opinions_sanity"]] + [o[k] for k in PHASES], [None] + [o["phase_se"][k] for k in PHASES], C_LARGE)]

    x = np.arange(len(keys))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    for i, (name, vals, ses, c) in enumerate(series):
        off = (i - 0.5) * w
        bars = ax.bar(x + off, vals, width=w * 0.92, color=c, label=name)
        for xi, v, se in zip(x + off, vals, ses):
            if se:
                ax.errorbar(xi, v, yerr=se, fmt="none", ecolor=C_INK, capsize=2, lw=0.9)
        _bar_labels(ax, bars, vals, dy=0.004, ses=ses)

    ax.axhline(p["minority_share"], ls=":", lw=1, color=C_PAPER)
    ax.axhline(o["minority_share"], ls=(0, (1, 2)), lw=1, color=C_RULE)
    # The two reference lines sit ~0.02 apart: label one above its line and one below, so they never collide.
    ax.text(x[-1] + 0.45, p["minority_share"], "true share (paper)", va="bottom", fontsize=6, color=C_PAPER)
    ax.text(x[-1] + 0.45, o["minority_share"], "true share (ours)", va="top", fontsize=6, color=C_RULE)
    ax.set_xticks(x, [LABELS["opinions"]] + [LABELS[k] for k in PHASES])
    ax.set_ylabel("Weight of minority opinions")
    ax.set_ylim(0, 0.46)
    ax.set_xlim(-0.55, len(keys) + 0.55)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", ncol=1)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_sensitivity(V, path):
    """Mean weight at each phase across the specifications run under each model; whiskers span the full range."""
    models = V["models"]
    x = np.arange(len(PHASES))
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    for i, (m, c) in enumerate(zip(models, (C_BASE, C_LARGE))):
        s = V["sensitivity"]["per_model"][m]
        mean = np.array([s[k]["mean"] for k in PHASES])
        lo = np.array([s[k]["min"] for k in PHASES])
        hi = np.array([s[k]["max"] for k in PHASES])
        off = (i - 0.5) * 0.16
        ax.errorbar(x + off, mean, yerr=[mean - lo, hi - mean], fmt="o", ms=5, color=c, lw=1.4,
                    capsize=3, label=f"{m}  (n={s['n_runs']} specifications)")
    ref = V["sensitivity"]["pooled"]["true_share_mean"]
    ax.axhline(ref, ls=":", lw=1, color=C_RULE)
    ax.text(x[-1] + 0.28, ref + 0.004, " mean true\n share", va="bottom", fontsize=6, color=C_RULE)
    ax.set_xticks(x, [LABELS[k] for k in PHASES])
    ax.set_ylabel("Weight of minority opinions")
    ax.set_ylim(0, 0.35)
    ax.set_xlim(-0.45, len(PHASES) - 0.35)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    V = json.loads((ROOT / "report" / "values.json").read_text())
    for name, fn in (("fig_paper", fig_paper), ("fig_contrast", fig_contrast), ("fig_sensitivity", fig_sensitivity)):
        path = OUT / f"{name}.pdf"
        fn(V, path)
        fn(V, OUT / f"{name}.png")
        print("wrote", path)


if __name__ == "__main__":
    main()
