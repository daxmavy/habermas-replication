"""End-to-end Fig. 4C pipeline: prepared tables + embedding cache -> position scores -> convex regression."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import assign_minority, build_design, cluster_bootstrap, minority_weight, opinion_self_regression, score_texts
from .data import COHORTS, text_id
from .embed import load_embeddings

KEY = ["metadata.version", "launch_id", "round_id"]
PHASES = ["initial_candidates", "initial_winner", "revised_candidates", "revised_winner"]


def load_prepared(prep_dir: Path):
    prep_dir = Path(prep_dir)
    return (pd.read_parquet(prep_dir / "opinions.parquet"), pd.read_parquet(prep_dir / "statements.parquet"),
            pd.read_parquet(prep_dir / "questions.parquet"), pd.read_parquet(prep_dir / "candidates.parquet"))


def score_all(prep_dir: Path, emb_dir: Path, method: str = "unit", endpoint_style: str = "prefixed"):
    """Attach position scores: opinions.score, statements.initial_score/revised_score, candidates.score."""
    opinions, statements, questions, candidates = load_prepared(prep_dir)
    lookup, mat = load_embeddings(emb_dir)
    kw = dict(method=method, endpoint_style=endpoint_style)
    opinions["score"] = score_texts(opinions, "opinion_text", questions, lookup, mat, text_id, **kw)
    statements["initial_score"] = score_texts(statements, "initial_text", questions, lookup, mat, text_id, **kw)
    statements["revised_score"] = score_texts(statements, "revised_text", questions, lookup, mat, text_id, **kw)
    qmap = statements.set_index(KEY)["question_id"]
    candidates["question_id"] = [qmap.get(k) for k in zip(candidates["metadata.version"], candidates["launch_id"], candidates["round_id"])]
    candidates["score"] = score_texts(candidates, "candidate_text", questions, lookup, mat, text_id, **kw)
    return opinions, statements, questions, candidates


def select_cohort(df: pd.DataFrame, cohort: str, prereg_only: bool = True) -> pd.DataFrame:
    """Rows of the cohort; for the evaluation cohorts 1-4 restrict to the pre-registered groups unless prereg_only=False."""
    d = df[df["metadata.version"].isin(COHORTS[cohort])]
    if prereg_only and cohort in ("cohort1", "cohort2", "cohort3", "cohort4", "cohorts_1_3"):
        d = d[d["prereg"]]
    return d


def phase_targets(candidates: pd.DataFrame, phase: str) -> pd.DataFrame:
    stage, kind = phase.split("_")
    d = candidates[candidates["phase"] == stage]
    return d[d["is_winner"]] if kind == "winner" else d


def fig4a_correlation(opinions: pd.DataFrame) -> dict:
    d = opinions.dropna(subset=["score", "pre_rating"])
    r = np.corrcoef(d["score"], d["pre_rating"])[0, 1] if len(d) > 2 else np.nan
    return {"r": float(r), "r2": float(r ** 2), "n": int(len(d))}


def marginal_r2(opinions: pd.DataFrame) -> dict:
    """Paper SM eq. 7: y_ij = a + b*x_position + u_i + e_ij, random intercept per round.

    Marginal R^2 (Nakagawa) = var(fixed prediction) / (var_fixed + var_round + var_resid), the
    quantity the paper reports as 0.41.  Pearson r is Fig. 4A.
    """
    import statsmodels.formula.api as smf

    d = select_cohort(opinions, "cohorts_1_3", prereg_only=True).dropna(subset=["score", "pre_rating"]).copy()
    d["round_key"] = d["metadata.version"].astype(str) + "|" + d["launch_id"].astype(str) + "|" + d["round_id"].astype(str)
    fit = smf.mixedlm("pre_rating ~ score", d, groups=d["round_key"]).fit(reml=True)
    var_f = float(np.var(fit.predict(d), ddof=0))  # MixedLM.predict gives the fixed-effects part only
    var_u = float(fit.cov_re.iloc[0, 0])
    var_e = float(fit.scale)
    r = float(np.corrcoef(d["score"], d["pre_rating"])[0, 1])
    return {"marginal_r2": var_f / (var_f + var_u + var_e), "conditional_r2": (var_f + var_u) / (var_f + var_u + var_e),
            "beta": float(fit.params["score"]), "beta_se": float(fit.bse["score"]),
            "pearson_r": r, "pearson_r2": r ** 2, "n": int(len(d)), "n_rounds": int(d["round_key"].nunique())}


def fig4b_within_range(opinions: pd.DataFrame, statements: pd.DataFrame) -> dict:
    """Fraction of group-statement scores lying within [min, max] of their group's opinion scores."""
    rng = opinions.dropna(subset=["score"]).groupby(KEY)["score"].agg(["min", "max"])
    st = statements.set_index(KEY).join(rng, how="inner")
    out = {}
    for col in ["initial_score", "revised_score"]:
        d = st.dropna(subset=[col])
        out[col] = {"within": float(((d[col] >= d["min"]) & (d[col] <= d["max"])).mean()), "n": int(len(d))}
    both = st.dropna(subset=["initial_score", "revised_score"])
    out["both"] = {"within": float((((both["initial_score"] >= both["min"]) & (both["initial_score"] <= both["max"])).sum() +
                                    ((both["revised_score"] >= both["min"]) & (both["revised_score"] <= both["max"])).sum()) / (2 * len(both))), "n": int(2 * len(both))}
    return out


def run_minority_analysis(opinions: pd.DataFrame, candidates: pd.DataFrame, cohort: str = "cohorts_1_3", neutral: str = "as_majority",
                          ties: str = "exclude", order: str = "data", min_rounds: int = 10, n_boot: int = 0, seed: int = 0,
                          prereg_only: bool = True, phases: list[str] = PHASES, include_opinions: bool = True, minority_by: str = "rating",
                          columns: str = "per_opinion") -> dict:
    op = select_cohort(opinions, cohort, prereg_only); ca = select_cohort(candidates, cohort, prereg_only)
    op_div = assign_minority(op, neutral=neutral, ties=ties, by=minority_by)
    res = {"cohort": cohort, "neutral": neutral, "ties": ties, "order": order, "prereg_only": prereg_only, "minority_by": minority_by, "phases": {}}
    designs = {}
    for phase in phases:
        design = build_design(op_div, phase_targets(ca, phase), "score", "score", order=order, seed=seed, columns=columns)
        designs[phase] = design
        mw = minority_weight(design, min_rounds=min_rounds)
        res["phases"][phase] = {"weight": mw.weight, "se": mw.se, "n_targets": mw.n_rounds,
                                "n_rounds": int(sum(len(set(d["keys"])) for d in design.values())),
                                "true_share": mw.true_share, "t_vs_true": mw.t_vs_true, "per_level": mw.per_level.to_dict(orient="records")}
    if include_opinions:
        sr = opinion_self_regression(op_div, "score", min_rounds=min_rounds, columns=columns)
        last = sr.iloc[-1]
        res["phases"]["opinions"] = {"weight": float(last["minority_weight"]), "se": 0.0, "n_targets": int(last["n_targets"]),
                                     "true_share": float(last["true_share"]), "per_level": sr.iloc[:-1].to_dict(orient="records")}
    if n_boot:
        res["n_boot"] = n_boot
        bs = cluster_bootstrap(designs, n_boot=n_boot, seed=seed, min_rounds=min_rounds)
        for i, phase in enumerate(phases):
            e = res["phases"][phase]
            e["boot_se"] = float(bs[:, i].std(ddof=1)); e["boot_ci95"] = [float(np.quantile(bs[:, i], .025)), float(np.quantile(bs[:, i], .975))]
            e["boot_p_gt_true"] = float((bs[:, i] <= e["true_share"]).mean())
        res["contrasts"] = {}
        for a, b in [("initial_winner", "revised_winner"), ("initial_candidates", "revised_candidates"), ("initial_candidates", "revised_winner")]:
            if a in phases and b in phases:
                d = bs[:, phases.index(b)] - bs[:, phases.index(a)]
                res["contrasts"][f"{b} - {a}"] = {"diff": res["phases"][b]["weight"] - res["phases"][a]["weight"], "boot_se": float(d.std(ddof=1)),
                                                  "boot_ci95": [float(np.quantile(d, .025)), float(np.quantile(d, .975))], "boot_p_le_0": float((d <= 0).mean())}
    return res


# ----------------------------------------------------------------------------- sensitivity grid
NEUTRAL_OPTIONS = {"non-minority": "as_majority", "opinion dropped": "drop_participant"}
ORDER_OPTIONS = {"data": "data", "random": "random"}
SPLIT_OPTIONS = {"Likert rating": "rating", "sign of position score": "score"}


def sensitivity_grid() -> dict[str, dict]:
    """Every combination of the choices the SM leaves open, for one embedding model (model size is the
    remaining axis and is swept by running the notebook once per model).  When the minority side is
    taken from the sign of the position score there are no neutral opinions, so that split is crossed
    with column order only.  All runs use the pre-registered rounds of cohorts 1-3 with tied rounds
    excluded, as the SM specifies."""
    grid = {}
    for split, by in SPLIT_OPTIONS.items():
        neutrals = NEUTRAL_OPTIONS.items() if by == "rating" else [("n/a", "as_majority")]
        for neutral, nv in neutrals:
            for order, ov in ORDER_OPTIONS.items():
                grid[f"split={split} | neutral={neutral} | order={order}"] = dict(minority_by=by, neutral=nv, order=ov)
    return grid


def run_sensitivity(opinions: pd.DataFrame, candidates: pd.DataFrame, grid: dict[str, dict], **kw) -> pd.DataFrame:
    """One row per grid entry: n_rounds, true_share and the minority weight (+ SE) at each phase."""
    rows = []
    for name, spec in grid.items():
        r = run_minority_analysis(opinions, candidates, include_opinions=False, **spec, **kw)
        row = {"variant": name, **spec, "n_rounds": r["phases"]["initial_winner"]["n_rounds"],
               "true_share": r["phases"]["initial_winner"]["true_share"]}
        for ph in PHASES:
            row[ph] = r["phases"][ph]["weight"]; row[ph + "_se"] = r["phases"][ph]["se"]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(res: dict) -> pd.DataFrame:
    rows = []
    for phase in ["opinions"] + PHASES:
        if phase not in res["phases"]:
            continue
        e = res["phases"][phase]
        rows.append({"phase": phase, "minority_weight": e["weight"], "se": e.get("se", np.nan), "boot_se": e.get("boot_se", np.nan),
                     "true_share": e["true_share"], "t_vs_true": e.get("t_vs_true", np.nan), "n_targets": e["n_targets"], "n_rounds": e.get("n_rounds", np.nan)})
    return pd.DataFrame(rows)


def per_level_table(res: dict) -> pd.DataFrame:
    rows = []
    for phase in PHASES:
        for r in res["phases"].get(phase, {}).get("per_level", []):
            rows.append({"phase": phase, **{k: v for k, v in r.items() if k != "coefs"}})
    return pd.DataFrame(rows).pivot(index=["n", "k", "true_share"], columns="phase", values=["minority_weight", "se", "n_rounds"])


def save_results(res: dict, path: Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=float)
