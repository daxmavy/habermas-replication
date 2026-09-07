"""Fig. 4D of Tessler et al. (2024): the Habermas Machine's per-round "majority bias" against the group's
movement toward the majority position between the pre- and post-deliberation position ratings.

Definitions (main text RQ3; Methods "Embedding geometry"; SM 4.1.2.1):
  * majority direction of a round = the side of Neutral (rating 4) holding more pre-deliberation ratings;
    a tie is set to AGREE, as in SM 4.1.2.1;
  * majority-aligned rating x' = 8 - x when the majority direction is DISAGREE, else x (SM 4.1.2.1);
  * majority bias = fraction of the candidate group statements shown in the round (4 initial + 4 revised)
    whose position score falls on the majority side of the median opinion score of the group;
  * group movement to majority = mean over the group's participants of sign(post' - pre'), i.e. the share of
    participants who moved toward the majority minus the share who moved away (multiples of 1/n in [-1, 1],
    which is the banding visible in the paper's panel D). Two alternatives are kept as extra columns:
    the mean change in majority-aligned rating and the change in the Group Agreement Index (SM 4.1.2.2).
  Paper: no association, b = 0.058, SE = 0.07, z = 0.9, P = 0.37; one point per group x question.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .pipeline import KEY, select_cohort

PAPER = {"b": 0.058, "se": 0.07, "z": 0.9, "p": 0.37}
MOVEMENT_COLS = {"movement_sign": "mean sign(post' - pre')  [primary]",
                 "movement_mean": "mean (post' - pre')",
                 "movement_gai": "Group Agreement Index post - pre"}


def _majority_direction(pre: np.ndarray, scores: np.ndarray, by: str) -> int:
    """+1 = AGREE side is the majority, -1 = DISAGREE. by='rating' uses the pre-deliberation ratings
    (neutral raters ignored, tie -> AGREE); by='score' uses the sign of the opinion position scores."""
    if by == "rating":
        n_ag, n_dis = int((pre > 4).sum()), int((pre < 4).sum())
    elif by == "score":
        n_ag, n_dis = int((scores > 0).sum()), int((scores < 0).sum())
    else:
        raise ValueError(by)
    return 1 if n_ag >= n_dis else -1


def _gai(x_aligned: np.ndarray) -> float:
    """Group Agreement Index (SM 4.1.2.2) on majority-aligned ratings: share on the majority side minus share on the minority side."""
    return float(((x_aligned > 4).sum() - (x_aligned < 4).sum()) / len(x_aligned))


def round_table(opinions: pd.DataFrame, candidates: pd.DataFrame, cohort: str = "cohorts_1_3", prereg_only: bool = True,
                majority_by: str = "rating") -> pd.DataFrame:
    """One row per round (group x question) with the HM majority bias and the group's movement toward the majority.
    Rounds are dropped when any opinion or candidate score is missing, or when no participant has a pre-deliberation rating."""
    op = select_cohort(opinions, cohort, prereg_only)
    ca = select_cohort(candidates, cohort, prereg_only)
    cand = {k: g for k, g in ca.groupby(KEY)}
    rows = []
    for k, g in op.groupby(KEY):
        if k not in cand or g["score"].isna().any() or cand[k]["score"].isna().any():
            continue
        rated = g.dropna(subset=["pre_rating"])
        if not len(rated):
            continue
        pre, scores = rated["pre_rating"].to_numpy(), g["score"].to_numpy()
        d = _majority_direction(pre, scores, majority_by)
        median = float(np.median(scores))
        c = cand[k]
        on_majority = (c["score"].to_numpy() - median) * d > 0
        both = rated.dropna(subset=["post_rating"])
        pre_a = np.where(d > 0, both["pre_rating"], 8 - both["pre_rating"]).astype(float)
        post_a = np.where(d > 0, both["post_rating"], 8 - both["post_rating"]).astype(float)
        delta = post_a - pre_a
        n_ag, n_dis = int((pre > 4).sum()), int((pre < 4).sum())
        pre_all = np.where(d > 0, pre, 8 - pre).astype(float)  # majority-aligned pre ratings of every rated participant
        rows.append({**dict(zip(KEY, k)), "question_id": g["question_id"].iloc[0], "n": len(g), "n_rated": len(rated), "n_both": len(both),
                     "majority_dir": d, "tie": n_ag == n_dis, "has_minority": min(n_ag, n_dis) > 0, "k_min": min(n_ag, n_dis),
                     "gai_pre": _gai(pre_all), "pre_mean": float(pre_all.mean()),
                     "median_score": median, "majority_bias": float(on_majority.mean()), "n_candidates": len(c),
                     "bias_initial": float(on_majority[(c["phase"] == "initial").to_numpy()].mean()),
                     "bias_revised": float(on_majority[(c["phase"] == "revised").to_numpy()].mean()),
                     "movement_sign": float(np.sign(delta).mean()) if len(delta) else np.nan,
                     "movement_mean": float(delta.mean()) if len(delta) else np.nan,
                     "movement_gai": _gai(post_a) - _gai(pre_a) if len(delta) else np.nan})
    return pd.DataFrame(rows)


def fit_association(df: pd.DataFrame, x: str = "majority_bias", y: str = "movement_sign", covariates: tuple[str, ...] = (),
                    crossed: bool = True) -> pd.DataFrame:
    """Slope of y on x (optionally adjusted for `covariates`) under up to four estimators; z = b / SE as in the paper.
    Rows run from the simplest (OLS on rounds) to a mixed model with crossed random intercepts for group and question
    (the slowest fit; skipped when crossed=False)."""
    import statsmodels.formula.api as smf

    d = df.dropna(subset=[x, y, *covariates]).copy()
    rhs = " + ".join([x, *covariates])
    out = []
    out.append(("OLS (rounds independent)", smf.ols(f"{y} ~ {rhs}", d).fit()))
    out.append(("OLS, cluster-robust SE by group", smf.ols(f"{y} ~ {rhs}", d).fit(cov_type="cluster", cov_kwds={"groups": d["launch_id"]})))
    try:
        out.append(("mixed: random intercept by group", smf.mixedlm(f"{y} ~ {rhs}", d, groups=d["launch_id"]).fit(reml=True)))
    except Exception as e:  # singular fits etc.
        out.append(("mixed: random intercept by group", e))
    if crossed:
        try:
            m = smf.mixedlm(f"{y} ~ {rhs}", d, groups=np.ones(len(d)), re_formula="0",
                            vc_formula={"group": "0 + C(launch_id)", "question": "0 + C(question_id)"}).fit(reml=True)
            out.append(("mixed: crossed random intercepts by group and question", m))
        except Exception as e:
            out.append(("mixed: crossed random intercepts by group and question", e))
    rows = []
    for name, m in out:
        if isinstance(m, Exception):
            rows.append({"model": name, "error": str(m)[:80]})
            continue
        b, se = float(m.params[x]), float(m.bse[x])
        rows.append({"model": name, "b": b, "se": se, "z": b / se, "p": float(m.pvalues[x]), "n_rounds": int(len(d)),
                     "n_groups": int(d["launch_id"].nunique()), "n_questions": int(d["question_id"].nunique()),
                     "intercept": float(m.params["Intercept"])})
    return pd.DataFrame(rows)


def plot_fig4d(df: pd.DataFrame, fit: pd.DataFrame, x: str = "majority_bias", y: str = "movement_sign", ax=None, title: str | None = None, seed: int = 0):
    """Scatter of rounds (jittered) with the OLS line and its 95% band, in the layout of the paper's panel D."""
    import matplotlib.pyplot as plt
    import statsmodels.formula.api as smf

    d = df.dropna(subset=[x, y])
    rng = np.random.default_rng(seed)
    if ax is None:
        _, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.scatter(d[x] + rng.uniform(-.02, .02, len(d)), d[y] + rng.uniform(-.03, .03, len(d)), s=7, color="k", alpha=.35, lw=0)
    grid = pd.DataFrame({x: np.linspace(0, 1, 50)})
    pred = smf.ols(f"{y} ~ {x}", d).fit().get_prediction(grid).summary_frame(alpha=.05)
    ax.fill_between(grid[x], pred["mean_ci_lower"], pred["mean_ci_upper"], color="grey", alpha=.35, lw=0)
    ax.plot(grid[x], pred["mean"], color="k", lw=1.6)
    ax.axhline(0, ls="--", color="k", lw=0.8)
    r = fit.iloc[0]
    ax.set_xlabel("Habermas Machine 'majority bias'"); ax.set_ylabel("Group movement to majority")
    ax.set_xlim(-.08, 1.08); ax.set_ylim(-1.15, 1.15); ax.set_xticks([0, .5, 1]); ax.set_yticks([-1, -.5, 0, .5, 1])
    ax.set_title(title or f"n = {int(r['n_rounds'])} rounds; b = {r['b']:.3f}, SE = {r['se']:.3f}, z = {r['z']:.2f}\n(paper: b = 0.058, SE = 0.07, z = 0.9)", fontsize=8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    return ax
