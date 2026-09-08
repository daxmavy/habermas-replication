"""Figures for the Fig. 4C replication report. Reads report/values.json only -- no literal numbers.

All three figures are drawn by the same routine, `grouped_bars`: grouped bars with error bars, one figure size, one
palette, one y-axis, the legend in the same corner and the true-share reference lines drawn the same way.

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
PAPER_NAME = "Paper (Tessler et al.)"

FIGSIZE = (6.6, 3.0)   # every figure is included at \linewidth, so one size keeps the type the same size throughout
YLIM = (0, 0.46)       # clears the paper's revised winner plus its error bar, and leaves room for the legend
LINE_PAPER, LINE_OURS = ":", (0, (1, 2))   # the paper's true share is drawn ":" in its colour, ours is a finer dotted rule

plt.rcParams.update({
    "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#999999", "axes.labelcolor": C_INK, "text.color": C_INK,
    "xtick.color": C_MUTED, "ytick.color": C_MUTED, "figure.dpi": 200,
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.6, "axes.axisbelow": True,
})


def grouped_bars(ax, groups: list[str], series: list[dict], group_width: float = 0.8) -> np.ndarray:
    """Draw one bar per (group, series) with an optional error bar, and return the group centres.

    Each series is {"name", "vals", "err", "color"}; an err entry is None (no bar), a half-width (symmetric) or a
    (below, above) pair (asymmetric, e.g. a min-max range). Series are drawn left to right in the order given."""
    x = np.arange(len(groups))
    n = len(series)
    w = group_width / n
    for i, s in enumerate(series):
        xs = x + (i - (n - 1) / 2) * w
        ax.bar(xs, s["vals"], width=w * 0.9, color=s["color"], label=s["name"])
        for xi, v, e in zip(xs, s["vals"], s["err"]):
            if e is None:
                continue
            yerr = np.array([[e[0]], [e[1]]]) if isinstance(e, (tuple, list)) else e
            ax.errorbar(xi, v, yerr=yerr, fmt="none", ecolor=C_INK, capsize=1.5, lw=0.8)
    ax.set_xticks(x, groups)
    ax.set_xlim(-0.55, len(groups) + 0.55)   # the space past the last group holds the reference-line labels
    ax.set_ylim(*YLIM)
    ax.set_ylabel("Weight of minority opinions")
    if n > 1:
        ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    return x


def reference_line(ax, x_text: float, y: float, label: str, color: str, ls, va: str, pad: float = 0.004) -> None:
    """A horizontal true-share rule, labelled just past the last group; va="bottom" puts the label above the rule,
    va="top" below it, each offset by `pad` so the dotted rule never runs through the text."""
    ax.axhline(y, ls=ls, lw=1, color=color)
    ax.text(x_text, y + (pad if va == "bottom" else -pad), label, va=va, fontsize=6, color=color)


def paper_series(V, keys: list[str]) -> dict:
    """The paper's values at the given phases (SM Fig. S60), with its printed SE on the revised winner only."""
    p = V["paper"]
    return {"name": PAPER_NAME, "vals": [p[k] for k in keys], "color": C_PAPER,
            "err": [p["revised_winner_se"] if k == "revised_winner" else None for k in keys]}


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_paper(V, path):
    """The paper's own Fig. 4C values (SM Fig. S60, position-embedding panel), redrawn."""
    keys = ["opinions_sanity"] + PHASES
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = grouped_bars(ax, [LABELS["opinions"]] + [LABELS[k] for k in PHASES], [paper_series(V, keys)], group_width=0.6)
    reference_line(ax, x[-1] + 0.45, V["paper"]["minority_share"], "true share (paper)", C_PAPER, LINE_PAPER, "bottom")
    save(fig, path)


def fig_contrast(V, path):
    """Paper vs. our reproduction under every embedding model, across the four phases.
    Error bars: paper, +/- 1 SE as printed (SM Fig. S60); ours, +/- 1 cluster-bootstrap SE over rounds.
    No per-bar value labels: the numbers are in the phases table of the report."""
    keys = ["opinions_sanity"] + PHASES
    series = [paper_series(V, keys)]
    for m in V["models"]:
        o = V["ours"][m]
        series.append({"name": f"Ours, {m}", "vals": [o["opinions_sanity"]] + [o[k] for k in PHASES],
                       "err": [None] + [o["phase_boot_se"][k] for k in PHASES], "color": MODEL_COLORS[m]})
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = grouped_bars(ax, [LABELS["opinions"]] + [LABELS[k] for k in PHASES], series)
    # The two reference lines sit ~0.02 apart: label one above its line and one below, so they never collide.
    reference_line(ax, x[-1] + 0.45, V["paper"]["minority_share"], "true share (paper)", C_PAPER, LINE_PAPER, "bottom")
    reference_line(ax, x[-1] + 0.45, V["ours"][V["primary_model"]]["minority_share"], "true share (ours)", C_RULE, LINE_OURS, "top")
    save(fig, path)


def fig_sensitivity(V, path):
    """The paper against the sensitivity sweep: for each model the mean weight at each phase over the specifications
    run under it, with whiskers spanning their full range (min to max). The paper's bar carries its printed SE."""
    series = [paper_series(V, PHASES)]
    for m in V["models"]:
        s = V["sensitivity"]["per_model"][m]
        series.append({"name": f"Ours, {m}", "vals": [s[k]["mean"] for k in PHASES], "color": MODEL_COLORS[m],
                       "err": [(s[k]["mean"] - s[k]["min"], s[k]["max"] - s[k]["mean"]) for k in PHASES]})
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = grouped_bars(ax, [LABELS[k] for k in PHASES], series)
    reference_line(ax, x[-1] + 0.45, V["paper"]["minority_share"], "true share (paper)", C_PAPER, LINE_PAPER, "bottom")
    reference_line(ax, x[-1] + 0.45, V["sensitivity"]["pooled"]["true_share_mean"], "mean true share (sweep)", C_RULE, LINE_OURS, "top")
    save(fig, path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    V = json.loads((ROOT / "report" / "values.json").read_text())
    for name, fn in (("fig_paper", fig_paper), ("fig_contrast", fig_contrast), ("fig_sensitivity", fig_sensitivity)):
        path = OUT / f"{name}.pdf"
        fn(V, path)
        print("wrote", path)


if __name__ == "__main__":
    main()
