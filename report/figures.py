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

# Two entities get categorical hues (paper = orange, ours = blue); model size is ordinal, so the four
# sizes are evenly spaced steps (250/400/550/700) of the blue ramp, light -> dark with size.
C_PAPER = "#eb6834"
MODEL_COLORS = {"st5-base": "#86b6ef", "st5-large": "#3987e5", "st5-xl": "#1c5cab", "st5-xxl": "#0d366b"}
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
    """Paper vs. our reproduction under every embedding model, across the four phases.
    Error bars: paper, +/- 1 SE as printed (SM Fig. S60); ours, +/- 1 cluster-bootstrap SE over rounds.
    No per-bar value labels: the numbers are in the phases table of the report."""
    p, models = V["paper"], V["models"]
    keys = ["opinions_sanity"] + PHASES
    series = [("Paper (Tessler et al.)", [p[k] for k in keys], [None] * 4 + [p["revised_winner_se"]], C_PAPER)]
    for m in models:
        o = V["ours"][m]
        series.append((f"Ours, {m}", [o["opinions_sanity"]] + [o[k] for k in PHASES],
                       [None] + [o["phase_boot_se"][k] for k in PHASES], MODEL_COLORS[m]))

    x = np.arange(len(keys))
    n = len(series)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    for i, (name, vals, ses, c) in enumerate(series):
        off = (i - (n - 1) / 2) * w
        ax.bar(x + off, vals, width=w * 0.9, color=c, label=name)
        for xi, v, se in zip(x + off, vals, ses):
            if se:
                ax.errorbar(xi, v, yerr=se, fmt="none", ecolor=C_INK, capsize=1.5, lw=0.8)

    share_ours = V["ours"][V["primary_model"]]["minority_share"]
    ax.axhline(p["minority_share"], ls=":", lw=1, color=C_PAPER)
    ax.axhline(share_ours, ls=(0, (1, 2)), lw=1, color=C_RULE)
    # The two reference lines sit ~0.02 apart: label one above its line and one below, so they never collide.
    ax.text(x[-1] + 0.45, p["minority_share"], "true share (paper)", va="bottom", fontsize=6, color=C_PAPER)
    ax.text(x[-1] + 0.45, share_ours, "true share (ours)", va="top", fontsize=6, color=C_RULE)
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
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    for i, m in enumerate(models):
        s = V["sensitivity"]["per_model"][m]
        mean = np.array([s[k]["mean"] for k in PHASES])
        lo = np.array([s[k]["min"] for k in PHASES])
        hi = np.array([s[k]["max"] for k in PHASES])
        off = (i - (len(models) - 1) / 2) * 0.14
        ax.errorbar(x + off, mean, yerr=[mean - lo, hi - mean], fmt="o", ms=4.5, color=MODEL_COLORS[m], lw=1.3,
                    capsize=2.5, label=f"{m}  (n={s['n_runs']} specifications)")
    ref = V["sensitivity"]["pooled"]["true_share_mean"]
    ax.axhline(ref, ls=":", lw=1, color=C_RULE)
    ax.text(x[-1] + 0.3, ref + 0.004, " mean true\n share", va="bottom", fontsize=6, color=C_RULE)
    ax.set_xticks(x, [LABELS[k] for k in PHASES])
    ax.set_ylabel("Weight of minority opinions")
    ax.set_ylim(0, 0.35)
    ax.set_xlim(-0.5, len(PHASES) - 0.3)
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
