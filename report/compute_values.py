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
from hm_fig4c.pipeline import PHASES, run_minority_analysis, score_all, select_cohort  # noqa: E402
PRIMARY_MODEL = "st5-large"

# Minority rules whose true share the report quotes when explaining the gap to the paper's 0.285.
SHARE_RULES = {"neutral_dropped": dict(neutral="drop_participant", ties="exclude"),
               "ties_kept": dict(neutral="as_majority", ties="agree")}


def group_sizes(prep_dir: Path) -> dict:
    """Distribution of opinions per round in the primary sample (cohorts 1-3, pre-registered)."""
    op = pd.read_parquet(prep_dir / "opinions.parquet")
    op = select_cohort(op, "cohorts_1_3", prereg_only=True)
    sizes = op.groupby(["metadata.version", "launch_id", "round_id"])["participant_id"].nunique()
    return {"min": int(sizes.min()), "max": int(sizes.max()), "n_rounds": int(len(sizes))}


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


def true_share_by_rule(opinions: pd.DataFrame, candidates: pd.DataFrame) -> dict:
    """True minority share of the primary sample under each rule in SHARE_RULES (same rounds as the pipeline)."""
    out = {}
    for name, rule in SHARE_RULES.items():
        r = run_minority_analysis(opinions, candidates, phases=["initial_winner"], include_opinions=False, **rule)
        e = r["phases"]["initial_winner"]
        out[name] = {"true_share": e["true_share"], "n_rounds": e["n_rounds"]}
    return out


def _sweep_stats(frame: pd.DataFrame) -> dict:
    out = {"n_runs": int(len(frame)),
           # "increases from initial to final": initial candidates -> revised winner
           "prop_increase_initial_to_final": float((frame["revised_winner"] > frame["initial_candidates"]).mean()),
           "prop_final_above_true_share": float((frame["revised_winner"] > frame["true_share"]).mean()),
           "max_revised_winner": float(frame["revised_winner"].max()),
           "true_share_mean": float(frame["true_share"].mean())}
    for phase in PHASES:
        out[phase] = {"mean": float(frame[phase].mean()), "min": float(frame[phase].min()), "max": float(frame[phase].max())}
    return out


def sensitivity_summary(paths: dict[str, Path]) -> dict:
    """Per-model and pooled statistics of the sensitivity sweep (model size is one of the swept choices)."""
    frames = {m: pd.read_csv(p).assign(model=m) for m, p in paths.items()}
    for m, f in frames.items():
        if "error" in f and f["error"].notna().any():
            raise SystemExit(f"{m}: sensitivity rows failed: {f[f['error'].notna()]['variant'].tolist()}")
    pooled = pd.concat(frames.values(), ignore_index=True)
    out = {"pooled": _sweep_stats(pooled), "per_model": {m: _sweep_stats(f) for m, f in frames.items()},
           "variants": frames[next(iter(frames))]["variant"].tolist()}
    n = {len(f) for f in frames.values()}
    if len(n) != 1:
        raise SystemExit(f"models ran different numbers of specifications: {n}")
    out["n_runs_per_model"] = n.pop()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["st5-base", "st5-large"])
    ap.add_argument("--skip-mixed", action="store_true", help="skip the random-effects fit and rule shares (need embeddings)")
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
    V["citations"] = cites

    V["ours"], V["contrasts"], V["axis"] = {}, {}, {}
    for m in args.models:
        rdir = ROOT / "results" / m
        V["ours"][m] = json.loads((rdir / "summary.json").read_text())["ours"]
        prim = json.loads((rdir / "fig4c_primary.json").read_text())
        V["contrasts"][m] = prim.get("contrasts", {})
        V["ours"][m]["phase_se"] = {p: prim["phases"][p]["se"] for p in PHASES}
        V["ours"][m]["phase_boot_se"] = {p: prim["phases"][p].get("boot_se") for p in PHASES}
        V["ours"][m]["phase_boot_ci"] = {p: prim["phases"][p].get("boot_ci95") for p in PHASES}
        V["ours"][m]["opinions_sanity_true_share"] = prim["phases"]["opinions"]["true_share"]

        if not args.skip_mixed:
            emb = ROOT / "embeddings" / m
            if emb.exists():
                opinions, _, _, candidates = score_all(ROOT / "prepared", emb)
                V["axis"][m] = {"mixed": marginal_r2(opinions)}
                print(f"[{m}] marginal r2 = {V['axis'][m]['mixed']['marginal_r2']:.3f} "
                      f"(pearson r = {V['axis'][m]['mixed']['pearson_r']:.3f})")
                if m == PRIMARY_MODEL:
                    V["share_by_rule"] = true_share_by_rule(opinions, candidates)

    V["sensitivity"] = sensitivity_summary({m: ROOT / "results" / m / "sensitivity.csv" for m in args.models})
    V["groups"] = group_sizes(ROOT / "prepared")
    V["cohorts_in_primary"] = list(COHORTS["cohorts_1_3"])

    Path(args.out).write_text(json.dumps(V, indent=1, sort_keys=True))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
