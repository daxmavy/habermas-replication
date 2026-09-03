"""End-to-end Fig. 4C pipeline: prepared tables + embedding cache -> scores -> convex regression -> figure."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import (assign_minority, bootstrap_minority_weight, build_design, minority_weight, paired_bootstrap, score_texts)
from .data import COHORTS, text_id
from .embed import load_embeddings

KEY = ["metadata.version", "launch_id", "round_id"]


def load_prepared(prep_dir: Path):
    prep_dir = Path(prep_dir)
    return (pd.read_parquet(prep_dir / "opinions.parquet"), pd.read_parquet(prep_dir / "statements.parquet"),
            pd.read_parquet(prep_dir / "questions.parquet"))


def score_all(prep_dir: Path, emb_dir: Path, method: str = "affine"):
    """Attach position scores to opinions (score) and statements (initial_score, revised_score)."""
    opinions, statements, questions = load_prepared(prep_dir)
    lookup, mat = load_embeddings(emb_dir)
    opinions["score"] = score_texts(opinions, "opinion_text", questions, lookup, mat, text_id, method)
    statements["initial_score"] = score_texts(statements, "initial_text", questions, lookup, mat, text_id, method)
    statements["revised_score"] = score_texts(statements, "revised_text", questions, lookup, mat, text_id, method)
    return opinions, statements, questions


def select_cohort(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    return df[df["metadata.version"].isin(COHORTS[cohort])]


def fig4a_correlation(opinions: pd.DataFrame) -> dict:
    d = opinions.dropna(subset=["score", "pre_rating"])
    r = np.corrcoef(d["score"], d["pre_rating"])[0, 1] if len(d) > 2 else np.nan
    return {"r": float(r), "r2": float(r ** 2), "n": int(len(d))}


def fig4b_within_range(opinions: pd.DataFrame, statements: pd.DataFrame) -> dict:
    """Fraction of group-statement scores lying within [min, max] of their group's opinion scores."""
    rng = opinions.dropna(subset=["score"]).groupby(KEY)["score"].agg(["min", "max"])
    st = statements.set_index(KEY).join(rng, how="inner")
    out = {}
    for col in ["initial_score", "revised_score"]:
        d = st.dropna(subset=[col])
        out[col] = {"within": float(((d[col] >= d["min"]) & (d[col] <= d["max"])).mean()), "n": int(len(d))}
    return out


def run_minority_analysis(opinions: pd.DataFrame, statements: pd.DataFrame, cohort: str = "cohorts_1_3",
                          neutral: str = "drop_participant", order: str = "data", min_rounds: int = 10,
                          n_boot: int = 0, seed: int = 0, statement_filter: str | None = None) -> dict:
    op = select_cohort(opinions, cohort); st = select_cohort(statements, cohort)
    if statement_filter:  # e.g. "initial_gen_api == 'hydra_70b_generative'"
        st = st.query(statement_filter)
    op_div = assign_minority(op, neutral=neutral)
    res = {"cohort": cohort, "neutral": neutral, "order": order, "statement_filter": statement_filter}
    designs = {}
    for stage, col in [("initial", "initial_score"), ("revised", "revised_score")]:
        design = build_design(op_div, st, "score", col, order=order, seed=seed)
        designs[stage] = design
        mw = minority_weight(design, min_rounds=min_rounds)
        entry = {"weight": mw.weight, "se": mw.se, "n_rounds": mw.n_rounds, "true_share": mw.true_share,
                 "t_vs_true": mw.t_vs_true, "per_level": mw.per_level.to_dict(orient="records")}
        if n_boot:
            bs = bootstrap_minority_weight(design, n_boot=n_boot, seed=seed, min_rounds=min_rounds)
            entry["boot_se"] = float(bs.std(ddof=1)); entry["boot_ci95"] = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
        res[stage] = entry
    # difference revised - initial (levels are the same rounds, so use the per-round paired structure via bootstrap if requested)
    res["diff"] = {"weight": res["revised"]["weight"] - res["initial"]["weight"],
                   "se_indep": float(np.sqrt(res["revised"]["se"] ** 2 + res["initial"]["se"] ** 2))}
    if n_boot:
        pb = paired_bootstrap(designs["initial"], designs["revised"], n_boot=n_boot, seed=seed, min_rounds=min_rounds)
        d = pb[:, 1] - pb[:, 0]
        res["diff"].update({"paired_boot_se": float(d.std(ddof=1)), "paired_boot_ci95": [float(np.quantile(d, .025)), float(np.quantile(d, .975))],
                            "paired_boot_p_revised_gt_initial": float((d <= 0).mean()),
                            "boot_p_revised_gt_true": float((pb[:, 1] <= res["revised"]["true_share"]).mean())})
    return res


def summarize(res: dict) -> pd.DataFrame:
    rows = []
    for stage in ["initial", "revised"]:
        e = res[stage]
        rows.append({"stage": stage, "minority_weight": e["weight"], "se": e["se"], "boot_se": e.get("boot_se", np.nan),
                     "true_share": e["true_share"], "t_vs_true": e["t_vs_true"], "n_rounds": e["n_rounds"]})
    d = res["diff"]
    rows.append({"stage": "revised - initial", "minority_weight": d["weight"], "se": d["se_indep"], "boot_se": d.get("paired_boot_se", np.nan),
                 "true_share": np.nan, "t_vs_true": d["weight"] / d["paired_boot_se"] if "paired_boot_se" in d else np.nan, "n_rounds": np.nan})
    return pd.DataFrame(rows)


def per_level_table(res: dict) -> pd.DataFrame:
    rows = []
    for stage in ["initial", "revised"]:
        for r in res[stage]["per_level"]:
            rows.append({"stage": stage, **{k: v for k, v in r.items() if k != "coefs"}})
    return pd.DataFrame(rows)


def plot_fig4c(res: dict, ax=None, title: str | None = None, show_levels: bool = True):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
    x = np.array([0, 1]); labels = ["Initial\nstatement", "Revised\nstatement"]
    w = [res["initial"]["weight"], res["revised"]["weight"]]; se = [res["initial"]["se"], res["revised"]["se"]]
    if show_levels:
        per = per_level_table(res)
        for (n, k), g in per.groupby(["n", "k"]):
            g = g.set_index("stage").reindex(["initial", "revised"])
            ax.plot(x + 0.12, g["minority_weight"], "o-", ms=3, lw=0.8, alpha=0.35, color="grey", zorder=1)
            ax.annotate(f"{k}/{n}", (1.15, g.loc["revised", "minority_weight"]), fontsize=6, color="grey", va="center")
    ax.errorbar(x, w, yerr=se, fmt="o-", color="#4C72B0", ms=7, capsize=4, lw=2, zorder=3, label="Minority weight ± 1 SE")
    ax.axhline(res["initial"]["true_share"], ls=":", color="k", lw=1.2, label=f"True minority proportion ({res['initial']['true_share']:.2f})")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_xlim(-0.5, 1.7)
    ax.set_ylabel("Weight on minority opinions")
    ax.set_title(title or f"{res['cohort']}  (n = {res['initial']['n_rounds']} rounds)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    return ax


def save_results(res: dict, path: Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=float)
