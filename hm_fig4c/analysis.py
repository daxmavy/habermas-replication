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
def position_axis_scores(emb: np.ndarray, e_neg: np.ndarray, e_aff: np.ndarray, method: str = "affine") -> np.ndarray:
    """Project rows of `emb` onto the axis from e_neg to e_aff.

    affine   : 0 at the negating statement, 1 at the affirming statement (projection of e - e_neg onto the axis,
               divided by the axis length).
    unit     : plain dot product with the unit axis vector (no re-centring), in embedding units.
    """
    u = e_aff - e_neg
    if method == "affine":
        return (emb - e_neg) @ u / (u @ u)
    if method == "unit":
        return emb @ (u / np.linalg.norm(u))
    raise ValueError(method)


def score_texts(df: pd.DataFrame, text_col: str, questions: pd.DataFrame, lookup: dict, mat: np.ndarray,
                text_id_fn, method: str = "affine") -> pd.Series:
    """Position score for each row of df (needs question_id + text column)."""
    q = questions.set_index("question_id")
    out = np.full(len(df), np.nan)
    for i, (qid, text) in enumerate(zip(df["question_id"], df[text_col])):
        if not isinstance(text, str):
            continue
        tid = text_id_fn(text)
        aff, neg = text_id_fn(q.at[qid, "affirming"]), text_id_fn(q.at[qid, "negating"])
        if tid in lookup and aff in lookup and neg in lookup:
            out[i] = position_axis_scores(mat[lookup[tid]][None, :], mat[lookup[neg]], mat[lookup[aff]], method)[0]
    return pd.Series(out, index=df.index)


# ----------------------------------------------------------------------------- group structure
def assign_minority(opinions: pd.DataFrame, neutral: str = "drop_participant") -> pd.DataFrame:
    """Add is_minority / n_div / k_min columns. Rows are participant-rounds with a pre_rating.

    neutral = 'drop_participant': neutral raters are removed, the group is kept.
              'drop_group'      : any group containing a neutral rater is removed.
              'as_majority'     : neutral raters are kept and counted as non-minority.
    Rounds with a tie (equal agree/disagree) or no dissent are dropped (no minority).
    """
    key = ["metadata.version", "launch_id", "round_id"]
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
    minority_side = np.where(n_ag < n_dis, 1, np.where(n_dis < n_ag, -1, 0))
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
    r2 = 1 - resid @ resid / np.sum((y - y.mean()) ** 2) if len(y) > 1 else np.nan
    return w, float(contrast @ w), se, sigma2, r2


def build_design(opinions_div: pd.DataFrame, statements: pd.DataFrame, score_col: str, stmt_col: str,
                 order: str = "data", seed: int = 0):
    """For each (n_div, k_min) level: X (rounds x n_div) of opinion scores with minority columns first, y statement scores."""
    key = ["metadata.version", "launch_id", "round_id"]
    st = statements.set_index(key)[stmt_col]
    rng = np.random.default_rng(seed)
    levels = {}
    for k, g in opinions_div.groupby(key):
        if k not in st.index or not np.isfinite(st[k]) or g[score_col].isna().any():
            continue
        mn, mj = g[g["is_minority"]], g[~g["is_minority"]]
        if order == "sorted":
            mn, mj = mn.sort_values(score_col), mj.sort_values(score_col)
        elif order == "random":
            mn, mj = mn.sample(frac=1, random_state=rng.integers(1 << 31)), mj.sample(frac=1, random_state=rng.integers(1 << 31))
        row = np.concatenate([mn[score_col].values, mj[score_col].values])
        lvl = (int(g["n_div"].iloc[0]), int(g["k_min"].iloc[0]))
        levels.setdefault(lvl, {"X": [], "y": [], "keys": []})
        levels[lvl]["X"].append(row); levels[lvl]["y"].append(st[k]); levels[lvl]["keys"].append(k)
    return {l: {"X": np.array(v["X"]), "y": np.array(v["y"]), "keys": v["keys"]} for l, v in levels.items()}


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
        contrast = np.array([1.0] * k + [0.0] * (n - k))
        w, mw, se, s2, r2 = convex_fit_with_se(d["X"], d["y"], contrast)
        rows.append(dict(n=n, k=k, n_rounds=len(d["y"]), true_share=k / n, minority_weight=mw, se=se, r2=r2, coefs=np.round(w, 3).tolist()))
    per = pd.DataFrame(rows)
    wts = per["n_rounds"] / per["n_rounds"].sum()
    agg_w = float((wts * per["minority_weight"]).sum())
    agg_se = float(np.sqrt(((wts * per["se"]) ** 2).sum()))
    true = float((wts * per["true_share"]).sum())
    return MinorityWeightResult(per, agg_w, agg_se, int(per["n_rounds"].sum()), true)


def paired_bootstrap(design_a: dict, design_b: dict, n_boot: int = 500, seed: int = 0, min_rounds: int = 10):
    """Joint bootstrap over rounds (same resampled rounds for both designs) -> arrays (w_a, w_b) of aggregate minority weights.

    Rounds present in only one design are dropped so that the difference b - a is a paired statistic."""
    rng = np.random.default_rng(seed)
    levels = []
    for l in sorted(set(design_a) & set(design_b)):
        ka = {k: i for i, k in enumerate(design_a[l]["keys"])}; kb = {k: i for i, k in enumerate(design_b[l]["keys"])}
        common = [k for k in design_a[l]["keys"] if k in kb]
        if len(common) < min_rounds:
            continue
        ia = np.array([ka[k] for k in common]); ib = np.array([kb[k] for k in common])
        levels.append((l, design_a[l]["X"][ia], design_a[l]["y"][ia], design_b[l]["X"][ib], design_b[l]["y"][ib]))
    N = sum(len(y) for _, _, y, _, _ in levels)
    out = np.empty((n_boot, 2))
    for b in range(n_boot):
        acc = np.zeros(2)
        for (n, k), Xa, ya, Xb, yb in levels:
            idx = rng.integers(0, len(ya), len(ya))
            acc[0] += len(ya) / N * convex_lstsq(Xa[idx], ya[idx])[:k].sum()
            acc[1] += len(ya) / N * convex_lstsq(Xb[idx], yb[idx])[:k].sum()
        out[b] = acc
    return out


def bootstrap_minority_weight(design: dict, n_boot: int = 500, seed: int = 0, min_rounds: int = 10):
    """Resample rounds within each division level; return bootstrap distribution of the aggregate minority weight."""
    rng = np.random.default_rng(seed)
    levels = [(l, d) for l, d in sorted(design.items()) if len(d["y"]) >= min_rounds]
    N = sum(len(d["y"]) for _, d in levels)
    out = np.empty(n_boot)
    for b in range(n_boot):
        acc = 0.0
        for (n, k), d in levels:
            idx = rng.integers(0, len(d["y"]), len(d["y"]))
            w = convex_lstsq(d["X"][idx], d["y"][idx])
            acc += len(d["y"]) / N * w[:k].sum()
        out[b] = acc
    return out
