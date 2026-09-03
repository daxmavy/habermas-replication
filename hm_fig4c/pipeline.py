"""End-to-end Fig. 4C pipeline: prepared tables + embedding cache -> position scores -> convex regression -> figure."""
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
PHASE_LABELS = {"opinions": "Opinions\n(sanity check)", "initial_candidates": "Initial\nstatements", "initial_winner": "Initial\nwinner",
                "revised_candidates": "Revised\nstatements", "revised_winner": "Revised\nwinner"}


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
                          prereg_only: bool = True, phases: list[str] = PHASES, include_opinions: bool = True) -> dict:
    op = select_cohort(opinions, cohort, prereg_only); ca = select_cohort(candidates, cohort, prereg_only)
    op_div = assign_minority(op, neutral=neutral, ties=ties)
    res = {"cohort": cohort, "neutral": neutral, "ties": ties, "order": order, "prereg_only": prereg_only, "phases": {}}
    designs = {}
    for phase in phases:
        design = build_design(op_div, phase_targets(ca, phase), "score", "score", order=order, seed=seed)
        designs[phase] = design
        mw = minority_weight(design, min_rounds=min_rounds)
        res["phases"][phase] = {"weight": mw.weight, "se": mw.se, "n_targets": mw.n_rounds,
                                "n_rounds": int(sum(len(set(d["keys"])) for d in design.values())),
                                "true_share": mw.true_share, "t_vs_true": mw.t_vs_true, "per_level": mw.per_level.to_dict(orient="records")}
    if include_opinions:
        sr = opinion_self_regression(op_div, "score", min_rounds=min_rounds)
        last = sr.iloc[-1]
        res["phases"]["opinions"] = {"weight": float(last["minority_weight"]), "se": 0.0, "n_targets": int(last["n_targets"]),
                                     "true_share": float(last["true_share"]), "per_level": sr.iloc[:-1].to_dict(orient="records")}
    if n_boot:
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


def plot_fig4c(res: dict, ax=None, title: str | None = None, include_opinions: bool = False):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
    phases = (["opinions"] if include_opinions and "opinions" in res["phases"] else []) + [p for p in PHASES if p in res["phases"]]
    colors = {"opinions": "#8da0cb", "initial_candidates": "#c6dbef", "initial_winner": "#9e9ac8", "revised_candidates": "#807dba", "revised_winner": "#6a51a3"}
    x = np.arange(len(phases))
    w = [res["phases"][p]["weight"] for p in phases]; se = [res["phases"][p]["se"] for p in phases]
    ax.bar(x, w, yerr=se, color=[colors[p] for p in phases], edgecolor="k", lw=0.6, capsize=3, error_kw={"lw": 1.2})
    true = res["phases"][phases[-1]]["true_share"]
    ax.axhline(true, ls="--", color="k", lw=1.2)
    ax.annotate("true proportion\nof minority opinions", (x[0] - 0.4, true + 0.01), fontsize=7, style="italic", va="bottom")
    for xi, wi in zip(x, w):
        ax.text(xi, 0.02, f"{wi:.2f}", ha="center", fontsize=7, color="white" if wi > 0.05 else "k")
    ax.set_xticks(x); ax.set_xticklabels([PHASE_LABELS[p] for p in phases], fontsize=8)
    ax.set_ylabel("Weight of minority opinions"); ax.set_xlabel("Group statement type"); ax.set_ylim(0, max(0.45, max(w) + max(se) + 0.05))
    ax.set_title(title or f"{res['cohort']} (n = {res['phases']['initial_winner']['n_rounds']} rounds)", fontsize=9)
    for s_ in ["top", "right"]:
        ax.spines[s_].set_visible(False)
    return ax


def save_results(res: dict, path: Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=float)
