"""Position-axis projection and convex regression for the Fig. 4C analysis.

Method (Tessler et al. 2024, Methods "Embedding geometry" + SM 5.4/5.5):
  * embed opinions, group statements and each question's affirming/negating position statements;
  * the segment negating -> affirming defines a per-question "position axis"; the projection of a text's
    embedding onto that axis is its "position component score";
  * minority participants are those whose pre-deliberation position rating falls on the smaller side of
    neutral within their group (neutral participants are excluded);
  * for each level of division (n participants, k in minority) the group-statement scores are regressed on
    convex combinations (weights >= 0, sum to 1) of the constituent opinion scores; the minority weight is the
    sum of the minority coefficients; levels are averaged (weighted by number of rounds).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, minimize


# ----------------------------------------------------------------------------- position scores
def position_axis_scores(emb: np.ndarray, e_neg: np.ndarray, e_aff: np.ndarray, method: str = "unit") -> np.ndarray:
    """Project rows of `emb` onto the axis from e_neg to e_aff.

    unit     : scalar projection onto the unit axis vector (SM eq. 3 & 5; with normalised embeddings the affirming and
               negating endpoints score +/- the same value, so 0 is the neutral midpoint).
    affine   : 0 at the negating statement, 1 at the affirming statement (projection of e - e_neg onto the axis,
               divided by the axis length).
    """
    u = e_aff - e_neg
    if u @ u == 0:  # degenerate axis (one question in the data has identical affirming/negating text)
        return np.full(len(emb), np.nan)
    if method == "affine":
        return (emb - e_neg) @ u / (u @ u)
    if method == "unit":
        return emb @ (u / np.linalg.norm(u))
    raise ValueError(method)


def score_texts(df: pd.DataFrame, text_col: str, questions: pd.DataFrame, lookup: dict, mat: np.ndarray,
                text_id_fn, method: str = "unit", endpoint_style: str = "prefixed") -> pd.Series:
    """Position score for each row of df (needs question_id + text column)."""
    from .data import endpoint_texts
    q = questions.set_index("question_id")

    def endpoint(text: str):
        """Embedding of an endpoint text, or None if it is not in the cache."""
        tid = text_id_fn(text)
        return mat[lookup[tid]] if tid in lookup else None

    endpoints = {}
    out = np.full(len(df), np.nan)
    for i, (qid, text) in enumerate(zip(df["question_id"], df[text_col])):
        if not isinstance(text, str):
            continue
        tid = text_id_fn(text)
        if qid not in endpoints:
            a_txt, n_txt = endpoint_texts(q.at[qid, "affirming"], q.at[qid, "negating"], endpoint_style)
            endpoints[qid] = (endpoint(a_txt), endpoint(n_txt))
        aff, neg = endpoints[qid]
        if tid in lookup and aff is not None and neg is not None:
            out[i] = position_axis_scores(mat[lookup[tid]][None, :], neg, aff, method)[0]
    return pd.Series(out, index=df.index)


# ----------------------------------------------------------------------------- group structure
# The minority side in a round where the two sides are equal.  SM 4.1.2.1 resolves such a round rather than
# dropping it: "In the case of a tie (e.g., 2 agree, 2 disagree, 1 neutral), we arbitrarily set the majority
# direction to AGREE", which leaves the disagree side as the minority.
TIE_RULES = {"majority_agree": -1, "majority_disagree": 1, "exclude": 0}


def assign_minority(opinions: pd.DataFrame, neutral: str = "as_majority", ties: str = "exclude", by: str = "rating") -> pd.DataFrame:
    """Add is_minority / n_div / k_min columns. Rows are participant-rounds with a pre_rating.

    neutral = 'drop_participant': neutral raters are removed, the group is kept.
              'drop_group'      : any group containing a neutral rater is removed.
              'as_majority'     : neutral raters are kept and counted as non-minority (SM 5.4.1).
    ties    = 'majority_agree'  : SM 4.1.2.1's rule -- when the two sides are equal the majority direction is
                                  set to AGREE, so the disagree side is the minority and the round is kept.
              'majority_disagree': the mirror of that arbitrary choice, for the sensitivity analysis.
              'exclude'         : rounds with equal agree/disagree counts are dropped instead.
    by      = 'rating'          : sides from the pre-deliberation Likert rating (SM Fig. S60).
              'score'           : sides from the sign of the opinion's position score (SM Fig. S62; no neutrals).
    Rounds with no dissent are always dropped.
    """
    key = ["metadata.version", "launch_id", "round_id"]
    if by == "score":
        df = opinions.dropna(subset=["score"]).copy()
        df["side"] = np.sign(df["score"]).astype(int)
    else:
        df = opinions.dropna(subset=["pre_rating"]).copy()
        df["side"] = np.sign(df["pre_rating"] - 4).astype(int)  # -1 disagree, 0 neutral, +1 agree
    if neutral == "drop_group":
        bad = df.loc[df["side"] == 0, key].drop_duplicates()
        df = df.merge(bad.assign(_bad=1), on=key, how="left")
        df = df[df["_bad"].isna()].drop(columns="_bad")
    if neutral in ("drop_participant", "drop_group"):
        df = df[df["side"] != 0]
    g = df.groupby(key)["side"]
    n_ag = g.transform(lambda s: (s > 0).sum())
    n_dis = g.transform(lambda s: (s < 0).sum())
    if ties not in TIE_RULES:
        raise ValueError(f"unknown tie rule {ties!r}; expected one of {sorted(TIE_RULES)}")
    tie_side = TIE_RULES[ties]
    minority_side = np.where(n_ag < n_dis, 1, np.where(n_dis < n_ag, -1, tie_side))
    df["is_minority"] = (df["side"] == minority_side) & (minority_side != 0)
    df["k_min"] = np.minimum(n_ag, n_dis)
    df["n_div"] = g.transform("size")
    df = df[(minority_side != 0) & (df["k_min"] > 0)]
    return df


# ----------------------------------------------------------------------------- convex regression
def convex_lstsq(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """argmin ||Xw - y||^2  s.t. w >= 0, sum(w) = 1."""
    n = X.shape[1]
    lam = 1e3 * max(1.0, np.abs(X).max())
    Xa = np.vstack([X, lam * np.ones((1, n))]); ya = np.concatenate([y, [lam]])
    w0 = lsq_linear(Xa, ya, bounds=(0, 1)).x
    w0 = np.clip(w0, 0, None); w0 /= w0.sum()
    res = minimize(lambda w: np.sum((X @ w - y) ** 2), w0, jac=lambda w: 2 * X.T @ (X @ w - y), method="SLSQP",
                   bounds=[(0, 1)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}],
                   options={"ftol": 1e-12, "maxiter": 500})
    w = res.x if res.success else w0
    w = np.clip(w, 0, None); return w / w.sum()


def convex_fit_with_se(X: np.ndarray, y: np.ndarray, contrast: np.ndarray):
    """Fit the convex regression and return (w, contrast'w, SE of contrast'w, sigma2, r2).

    SE: treat the fit as an equality-constrained least squares on the active set (coefficients > 0),
    reparametrise by eliminating one active coefficient via the sum-to-one constraint and use the OLS covariance.
    """
    w = convex_lstsq(X, y)
    active = np.where(w > 1e-9)[0]
    if len(active) < 2:  # degenerate: everything on one coefficient
        resid = y - X @ w
        return w, contrast @ w, np.nan, resid @ resid / max(len(y) - 1, 1), np.nan
    ref = active[-1]; others = active[:-1]
    Z = X[:, others] - X[:, [ref]]; yy = y - X[:, ref]
    ZtZ_inv = np.linalg.pinv(Z.T @ Z)
    resid = yy - Z @ w[others]
    dof = max(len(y) - len(others), 1)
    sigma2 = resid @ resid / dof
    c = contrast[others] - contrast[ref]
    se = float(np.sqrt(sigma2 * c @ ZtZ_inv @ c))
    full_resid = y - X @ w
    r2 = 1 - full_resid @ full_resid / np.sum((y - y.mean()) ** 2) if len(y) > 1 else np.nan
    return w, float(contrast @ w), se, sigma2, r2


def design_row(minority_scores: np.ndarray, other_scores: np.ndarray, columns: str = "per_opinion") -> np.ndarray:
    """One row of the design matrix from a round's minority and non-minority opinion scores.

    per_opinion : the scores themselves, minority first, so each column is one opinion slot.
    blocks      : the two block means, so every opinion of a block carries the same weight and the row -- and
                  therefore the fit -- does not depend on the order of the opinions within either block.
    """
    if columns == "per_opinion":
        return np.concatenate([minority_scores, other_scores])
    if columns == "blocks":
        return np.array([minority_scores.mean(), other_scores.mean()])
    raise ValueError(columns)


def minority_contrast(n: int, k: int, columns: str = "per_opinion") -> np.ndarray:
    """The vector c with c @ w = the summed minority weight, for a level with n opinions of which k are minority."""
    if columns == "per_opinion":
        return np.array([1.0] * k + [0.0] * (n - k))
    if columns == "blocks":
        return np.array([1.0, 0.0])
    raise ValueError(columns)


def build_design(opinions_div: pd.DataFrame, targets: pd.DataFrame, score_col: str = "score", target_col: str = "score",
                 order: str = "data", seed: int = 0, columns: str = "per_opinion"):
    """Per (n_div, k_min) level: X (targets x columns) of opinion scores, y = target scores, keys = round key per row
    (several targets per round are allowed, e.g. all candidate statements), contrast = the vector that sums the
    minority weight.  `columns` picks the design: see `design_row`."""
    key = ["metadata.version", "launch_id", "round_id"]
    tg = targets.dropna(subset=[target_col]).groupby(key)[target_col].apply(list)
    rng = np.random.default_rng(seed)
    levels = {}
    for k, g in opinions_div.groupby(key):
        if k not in tg.index or g[score_col].isna().any():
            continue
        mn, mj = g[g["is_minority"]], g[~g["is_minority"]]
        if order == "random":
            mn, mj = mn.sample(frac=1, random_state=rng.integers(1 << 31)), mj.sample(frac=1, random_state=rng.integers(1 << 31))
        row = design_row(mn[score_col].values, mj[score_col].values, columns)
        lvl = (int(g["n_div"].iloc[0]), int(g["k_min"].iloc[0]))
        d = levels.setdefault(lvl, {"X": [], "y": [], "keys": []})
        for yv in tg[k]:
            d["X"].append(row); d["y"].append(yv); d["keys"].append(k)
    return {l: {"X": np.array(v["X"]), "y": np.array(v["y"]), "keys": v["keys"], "contrast": minority_contrast(*l, columns)}
            for l, v in levels.items()}


@dataclass
class MinorityWeightResult:
    per_level: pd.DataFrame
    weight: float
    se: float
    n_rounds: int
    true_share: float

    @property
    def t_vs_true(self):
        return (self.weight - self.true_share) / self.se


def minority_weight(design: dict, min_rounds: int = 10) -> MinorityWeightResult:
    rows = []
    for (n, k), d in sorted(design.items()):
        if len(d["y"]) < min_rounds:
            continue
        w, mw, se, s2, r2 = convex_fit_with_se(d["X"], d["y"], d["contrast"])
        rows.append(dict(n=n, k=k, n_rounds=len(d["y"]), true_share=k / n, minority_weight=mw, se=se, r2=r2, coefs=np.round(w, 3).tolist()))
    per = pd.DataFrame(rows)
    wts = per["n_rounds"] / per["n_rounds"].sum()
    agg_w = float((wts * per["minority_weight"]).sum())
    agg_se = float(np.sqrt(((wts * per["se"]) ** 2).sum()))
    true = float((wts * per["true_share"]).sum())
    return MinorityWeightResult(per, agg_w, agg_se, int(per["n_rounds"].sum()), true)


def cluster_bootstrap(designs: dict, n_boot: int = 500, seed: int = 0, min_rounds: int = 10) -> np.ndarray:
    """Bootstrap over rounds, resampling the same rounds for every phase in `designs` (phase -> design).
    Returns an array (n_boot, n_phases) of aggregate minority weights (levels weighted by the number of resampled
    rounds present in that phase, as in the point estimate), phases in dict order."""
    rng = np.random.default_rng(seed)
    phases = list(designs)
    levels = sorted(set.intersection(*[set(d) for d in designs.values()]))
    prep = []
    for l in levels:
        rounds = sorted(set.union(*[set(designs[p][l]["keys"]) for p in phases]))
        if len(rounds) < min_rounds:
            continue
        idx = {}
        for p in phases:
            m = {}
            for i, k in enumerate(designs[p][l]["keys"]):
                m.setdefault(k, []).append(i)
            idx[p] = m
        prep.append((l, rounds, idx, designs[phases[0]][l]["contrast"]))
    out = np.empty((n_boot, len(phases)))
    for b in range(n_boot):
        num = np.zeros(len(phases)); den = np.zeros(len(phases))
        for (n, k), rounds, idx, contrast in prep:
            samp = rng.choice(len(rounds), len(rounds), replace=True)
            for pi, p in enumerate(phases):
                present = [j for j in samp if rounds[j] in idx[p]]
                if not present:
                    continue
                rows = np.concatenate([idx[p][rounds[j]] for j in present]).astype(int)
                w = convex_lstsq(designs[p][(n, k)]["X"][rows], designs[p][(n, k)]["y"][rows])
                num[pi] += len(present) * (contrast @ w); den[pi] += len(present)
        out[b] = num / np.where(den > 0, den, np.nan)
    return out


def opinion_self_regression(opinions_div: pd.DataFrame, score_col: str = "score", min_rounds: int = 10,
                            columns: str = "per_opinion") -> pd.DataFrame:
    """SM sanity check: regress each individual opinion on convex combinations of the same group's opinions
    (one target per opinion); the summed minority weight should equal the proportion of opinions in the minority.
    `columns` picks the design, as in `build_design`; the targets are the individual opinions either way."""
    key = ["metadata.version", "launch_id", "round_id"]
    levels = {}
    for k, g in opinions_div.groupby(key):
        if g[score_col].isna().any():
            continue
        mn, mj = g[g["is_minority"]], g[~g["is_minority"]]
        targets = np.concatenate([mn[score_col].values, mj[score_col].values])
        row = design_row(mn[score_col].values, mj[score_col].values, columns)
        lvl = (int(g["n_div"].iloc[0]), int(g["k_min"].iloc[0]))
        for target in targets:
            levels.setdefault(lvl, {"X": [], "y": []}); levels[lvl]["X"].append(row); levels[lvl]["y"].append(target)
    rows = []
    for (n, k), d in sorted(levels.items()):
        X, y = np.array(d["X"]), np.array(d["y"])
        if len(y) < min_rounds:
            continue
        w = convex_lstsq(X, y)
        rows.append(dict(n=n, k=k, n_targets=len(y), true_share=k / n, minority_weight=minority_contrast(n, k, columns) @ w))
    per = pd.DataFrame(rows)
    wts = per["n_targets"] / per["n_targets"].sum()
    per.loc[len(per)] = dict(n="all", k="-", n_targets=per["n_targets"].sum(), true_share=(wts * per["true_share"]).sum(), minority_weight=(wts * per["minority_weight"]).sum())
    return per
