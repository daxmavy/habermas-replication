"""Compute every number the Fig. 4C report quotes, and emit report/values.json.

The report never contains a literal number: values.json -> build_parameters.py -> parameters.tex
-> \\macro in the .tex.  Run from the repo root:

    uv run python report/compute_values.py [--models st5-base st5-large] [--skip-mixed]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hm_fig4c.data import COHORTS  # noqa: E402
from hm_fig4c.pipeline import PHASES, score_all, select_cohort  # noqa: E402
PRIMARY_MODEL = "st5-large"

# Which sensitivity rows vary an *analytic* choice on the primary sample, and which swap the sample.
SAMPLE_VARIANTS = ("all cohorts 1-3 rounds", "cohort 1 only", "cohort 2 only", "cohort 3 only",
                   "cohort 4 (critique exclusion)", "training data", "virtual citizens")


def _is_sample_variant(name: str) -> bool:
    return any(name.startswith(p) for p in SAMPLE_VARIANTS)


def group_sizes(prep_dir: Path) -> dict:
    """Distribution of opinions per round in the primary sample (cohorts 1-3, pre-registered)."""
    op = pd.read_parquet(prep_dir / "opinions.parquet")
    op = select_cohort(op, "cohorts_1_3", prereg_only=True)
    sizes = op.groupby(["metadata.version", "launch_id", "round_id"])["participant_id"].nunique()
    vc = sizes.value_counts().sort_index()
    return {"min": int(sizes.min()), "max": int(sizes.max()), "modal": int(sizes.mode().iloc[0]),
            "n_rounds": int(len(sizes)), "counts": {int(k): int(v) for k, v in vc.items()},
            "modal_share": float((sizes == sizes.mode().iloc[0]).mean())}


def marginal_r2(prep_dir: Path, emb_dir: Path) -> dict:
    """Paper SM eq. 7: y_ij = a + b*x_position + u_i + e_ij, random intercept per round.

    Marginal R^2 (Nakagawa) = var(fixed prediction) / (var_fixed + var_round + var_resid), the
    quantity the paper reports as 0.41.  Pearson r is Fig. 4A.
    """
    import statsmodels.formula.api as smf

    opinions, _, _, _ = score_all(prep_dir, emb_dir)
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


def sensitivity_summary(path: Path) -> dict:
    s_all = pd.read_csv(path)
    s = s_all[s_all["error"].isna()] if "error" in s_all else s_all
    failed = s_all[s_all["error"].notna()]["variant"].tolist() if "error" in s_all else []
    analytic = s[~s["variant"].map(_is_sample_variant)]
    out = {"n_attempted": int(len(s_all)), "n_variants": int(len(s)), "n_failed": int(len(failed)),
           "failed_variants": failed, "n_analytic": int(len(analytic)), "n_sample": int(len(s) - len(analytic))}
    for label, frame in (("all", s), ("analytic", analytic)):
        # "increases from initial to final": stage 1 (initial candidates) -> stage 4 (revised winner),
        # and the winner-to-winner contrast the paper's Fig. 4C emphasises.
        inc_stage = float((frame["revised_winner"] > frame["initial_candidates"]).mean())
        inc_winner = float((frame["revised_winner"] > frame["initial_winner"]).mean())
        above = float((frame["revised_winner"] > frame["true_share"]).mean())
        out[label] = {"prop_increase_initial_to_final": inc_stage, "prop_increase_winner_to_winner": inc_winner,
                      "prop_final_above_true_share": above,
                      "max_revised_winner": float(frame["revised_winner"].max()),
                      "min_true_share": float(frame["true_share"].min()), "max_true_share": float(frame["true_share"].max())}
    for phase in PHASES:
        out[f"{phase}_mean"] = float(s[phase].mean())
        out[f"{phase}_p5"] = float(s[phase].quantile(0.05))
        out[f"{phase}_p95"] = float(s[phase].quantile(0.95))
    out["true_share_mean"] = float(s["true_share"].mean())
    out["variants"] = s["variant"].tolist()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["st5-base", "st5-large"])
    ap.add_argument("--skip-mixed", action="store_true", help="skip the random-effects fit (needs embeddings)")
    ap.add_argument("--out", default=str(ROOT / "report" / "values.json"))
    args = ap.parse_args()

    V: dict = {"models": args.models, "primary_model": PRIMARY_MODEL}

    # Paper-side numbers come from report/paper_citations.json (each with its source), and are
    # cross-checked against the 'paper' block the pipeline carries, so the two cannot drift apart.
    cites = json.loads((ROOT / "report" / "paper_citations.json").read_text())
    first = json.loads((ROOT / "results" / args.models[0] / "summary.json").read_text())
    V["paper"] = {k: v for k, v in cites["fig4c"].items() if not k.startswith("_")}
    for k, v in V["paper"].items():
        if k in first["paper"] and abs(first["paper"][k] - v) > 1e-9:
            raise SystemExit(f"paper value drift for {k}: citations={v} pipeline={first['paper'][k]}")
    V["paper"]["n_rounds"] = first["paper"]["n_rounds"]
    V["paper"]["fig4a_r"] = first["paper"]["fig4a_r"]
    V["paper"]["fig4b_within"] = first["paper"]["fig4b_within"]
    V["paper"]["marginal_r2"] = cites["axis_validation"]["marginal_r2"]
    V["paper"]["conditional_r2"] = cites["axis_validation"]["conditional_r2"]
    V["paper"]["designed_group_size"] = cites["group_design"]["designed_size"]
    V["citations"] = cites

    V["ours"], V["contrasts"], V["sensitivity"], V["axis"] = {}, {}, {}, {}
    for m in args.models:
        rdir = ROOT / "results" / m
        V["ours"][m] = json.loads((rdir / "summary.json").read_text())["ours"]
        prim = json.loads((rdir / "fig4c_primary.json").read_text())
        V["contrasts"][m] = prim.get("contrasts", {})
        V["ours"][m]["phase_se"] = {p: prim["phases"][p]["se"] for p in PHASES}
        V["ours"][m]["phase_boot_ci"] = {p: prim["phases"][p].get("boot_ci95") for p in PHASES}
        V["ours"][m]["opinions_sanity_true_share"] = prim["phases"]["opinions"]["true_share"]
        V["sensitivity"][m] = sensitivity_summary(rdir / "sensitivity.csv")

        vr = pd.read_csv(rdir / "vector_regression.csv")
        vr_all = vr[vr["n"] == "all"].set_index("stage")["minority_weight"]
        V["axis"][m] = {"full768_initial": float(vr_all["initial"]), "full768_revised": float(vr_all["revised"])}
        att = pd.read_csv(rdir / "attenuation.csv")
        V["axis"][m]["attenuation_min_r"] = float(att["achieved_r"].min())
        V["axis"][m]["attenuation_recovered_at_min_r"] = float(att.loc[att["achieved_r"].idxmin(), "recovered_minority_weight"])
        V["axis"][m]["attenuation_true_share"] = float(att["true_share"].iloc[0])
        wp = pd.read_csv(rdir / "winner_position.csv").set_index("stage")
        V["axis"][m]["winner_inside_initial"] = float(wp.loc["initial", "inside"])
        V["axis"][m]["winner_inside_revised"] = float(wp.loc["revised", "inside"])

        if not args.skip_mixed:
            emb = ROOT / "embeddings" / m
            if emb.exists():
                V["axis"][m]["mixed"] = marginal_r2(ROOT / "prepared", emb)
                print(f"[{m}] marginal r2 = {V['axis'][m]['mixed']['marginal_r2']:.3f} "
                      f"(pearson r = {V['axis'][m]['mixed']['pearson_r']:.3f})")

    V["groups"] = group_sizes(ROOT / "prepared")
    V["cohorts_in_primary"] = list(COHORTS["cohorts_1_3"])

    Path(args.out).write_text(json.dumps(V, indent=1, sort_keys=True))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
