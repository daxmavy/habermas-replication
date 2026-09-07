"""Generate the analysis notebooks (reusable code lives in hm_fig4c/).

    uv run python notebooks/make_notebook.py --figure 4c   # notebooks/fig4c.ipynb (default)
    uv run python notebooks/make_notebook.py --figure 4d   # notebooks/fig4d.ipynb
"""
import argparse

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))


def build_4c():
    md("""# Replicating Tessler et al. (2024) Fig. 4C — minority weight in Habermas Machine group statements

Pipeline: Sentence-T5 embeddings → per-question position axis (negating → affirming) → position component scores →
convex regression of group-statement scores on constituent opinion scores, per level of division → minority weight
(sum of minority coefficients), averaged over levels.

Paper's numbers to compare against. Fig. 4C / Fig. S60 (position embedding, main-task cohorts 1-3, n = 1047 rounds):
opinions (sanity check) 0.28, initial statements (all candidates) 0.28, initial winner 0.29, revised statements 0.33, revised winner 0.36
(SE 0.03, t = 2.64 vs the true minority proportion, 0.28-0.29). Fig. 4A: r = 0.64 between opinion position score and
pre-deliberation rating. Fig. 4B: 96% of group-statement scores within the range of the group's opinions.

Method (SM 5.1, 5.4.1): Sentence-T5 embeddings; per-question position axis = unit vector from the embedding of
"No, I disagree. <negating statement>" to "Yes, I agree. <affirming statement>"; position score = projection onto that axis;
minority = the side of neutral with fewer pre-deliberation ratings, neutral opinions count as non-minority; convex regression
(weights >= 0, sum = 1) of statement scores on the group's opinion scores, one regression per (group size, minority size) level,
minority weight = sum of minority coefficients, averaged over levels weighted by number of rounds.""")

    code("""import os, sys, json
sys.path.insert(0, os.path.abspath(".."))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from hm_fig4c import pipeline as P
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

EMB_DIR = os.environ.get("HM_EMB_DIR", "../embeddings/st5-base")   # embedding cache to use
AXIS_METHOD = os.environ.get("HM_AXIS_METHOD", "unit")               # 'unit' (SM eq. 5: projection on the unit axis) or 'affine'
N_BOOT = int(os.environ.get("HM_N_BOOT", "500"))
MODEL_TAG = os.path.basename(EMB_DIR.rstrip("/"))
OUT_DIR = f"../results/{MODEL_TAG}"; os.makedirs(OUT_DIR, exist_ok=True)
print(EMB_DIR, AXIS_METHOD, N_BOOT)""")

    md("## 1. Score all texts on the position axis")
    code("""ENDPOINTS = os.environ.get("HM_ENDPOINTS", "prefixed")  # 'prefixed' (SM: generic + question-specific), 'plain', 'generic'
opinions, statements, questions, candidates = P.score_all("../prepared", EMB_DIR, method=AXIS_METHOD, endpoint_style=ENDPOINTS)
cov = pd.DataFrame({"opinions_scored": opinions.groupby("cohort")["score"].apply(lambda s: s.notna().mean()),
                    "initial_scored": statements.groupby("cohort")["initial_score"].apply(lambda s: s.notna().mean()),
                    "revised_scored": statements.groupby("cohort")["revised_score"].apply(lambda s: s.notna().mean()),
                    "candidates_scored": candidates.groupby("cohort")["score"].apply(lambda s: s.notna().mean()),
                    "n_rounds": statements.groupby("cohort").size(), "n_prereg_rounds": statements.groupby("cohort")["prereg"].sum()})
cov.round(3)""")

    md("## 2. Fig. 4A check — opinion position score vs pre-deliberation position rating (paper: r = 0.64)")
    code("""rows = {c: P.fig4a_correlation(P.select_cohort(opinions, c)) for c in ["cohort1", "cohort2", "cohort3", "cohorts_1_3", "cohort4", "training", "vca"]}
fig4a = pd.DataFrame(rows).T; fig4a""")
    code("""d = P.select_cohort(opinions, "cohorts_1_3").dropna(subset=["score", "pre_rating"])
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.scatter(d["pre_rating"] + np.random.uniform(-.15, .15, len(d)), d["score"], s=4, alpha=.25)
means = d.groupby("pre_rating")["score"].mean()
ax.plot(means.index, means.values, "o-", color="k", ms=5)
ax.set_xlabel("Pre-deliberation position rating (1 = strongly disagree, 7 = strongly agree)"); ax.set_ylabel("Position component score")
ax.axhline(0, color="grey", lw=0.8, ls=":")
ax.set_title(f"Cohorts 1-3: r = {fig4a.loc['cohorts_1_3','r']:.2f}  (paper: 0.64)", fontsize=9); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig4a.png", dpi=150)""")

    md("## 3. Fig. 4B check — statement scores relative to the group's opinions (paper: 96% within range)")
    code("""fig4b = {c: P.fig4b_within_range(P.select_cohort(opinions, c), P.select_cohort(statements, c)) for c in ["cohorts_1_3", "training", "vca"]}
pd.DataFrame({(c, s): v for c, dd in fig4b.items() for s, v in dd.items()}).T""")
    code("""d_op = P.select_cohort(opinions, "cohorts_1_3"); d_st = P.select_cohort(statements, "cohorts_1_3")
fig, ax = plt.subplots(figsize=(5, 3.2))
for vals, lab, col in [(d_op["score"], "opinions", "tab:red"), (d_st["initial_score"], "initial statements", "tab:blue"), (d_st["revised_score"], "revised statements", "tab:purple")]:
    ax.hist(vals.dropna(), bins=60, density=True, histtype="step", lw=1.5, label=lab, color=col)
ax.set_xlabel("Position component score (0 = negating, 1 = affirming)"); ax.legend(frameon=False, fontsize=8); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig4b.png", dpi=150)""")

    md("""## 4. Fig. 4C — minority weight via convex regression (primary specification)

Main-task cohorts 1–3, pre-registered groups (n = 1047 rounds); minority = smaller side of neutral on the pre-deliberation
rating, neutral opinions kept as non-minority (SM 5.4.1); rounds with a tie or no dissent excluded; columns ordered as in the data.
Analytic SEs are the OLS standard errors of the constrained fit (as in the paper); bootstrap SEs resample rounds.""")
    code("""res = P.run_minority_analysis(opinions, candidates, cohort="cohorts_1_3", neutral="as_majority", order="data", n_boot=N_BOOT)
P.save_results(res, f"{OUT_DIR}/fig4c_primary.json")
summary = P.summarize(res); summary.round(3)""")
    code("""pd.DataFrame(res["contrasts"]).T.round(3)""")
    code("""P.per_level_table(res).round(3)""")
    code("""fig, ax = plt.subplots(figsize=(5.2, 3.6))
P.plot_fig4c(res, ax=ax, include_opinions=True, title=f"Cohorts 1-3 (pre-registered groups), {MODEL_TAG}, n = {res['phases']['initial_winner']['n_rounds']} rounds")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig4c.png", dpi=200)""")

    md("""## 5. Sensitivity analyses
Every combination of the choices the SM leaves open (`P.sensitivity_grid()`): neutral-opinion treatment × basis for the
minority split × column order of the design matrix; model size is swept by running this notebook once per embedding model.
Sample (pre-registered rounds of cohorts 1–3) and tie handling (excluded) are as the SM specifies and are not varied.""")
    code("""sens = P.run_sensitivity(opinions, candidates, P.sensitivity_grid())
sens.to_csv(f"{OUT_DIR}/sensitivity.csv", index=False); sens.round(3)""")

    md("## 6. Robustness: regress the full 768-d embedding (not just the position score) on convex combinations of opinion embeddings")
    code("""from hm_fig4c.analysis import assign_minority, convex_fit_with_se
from hm_fig4c.data import text_id
from hm_fig4c.embed import load_embeddings
lookup, mat = load_embeddings(EMB_DIR)
op = assign_minority(P.select_cohort(opinions, "cohorts_1_3"), neutral="as_majority")
st = P.select_cohort(statements, "cohorts_1_3").set_index(P.KEY)
def vec_design(stmt_col):
    lv = {}
    for k, g in op.groupby(P.KEY):
        if k not in st.index or not isinstance(st.at[k, stmt_col], str): continue
        ids = [text_id(t) for t in g["opinion_text"]]; sid = text_id(st.at[k, stmt_col])
        if sid not in lookup or any(i not in lookup for i in ids): continue
        mn = g["is_minority"].values
        X = np.stack([mat[lookup[i]] for i in ids], axis=1)  # 768 x n
        X = np.concatenate([X[:, mn], X[:, ~mn]], axis=1)
        y = mat[lookup[sid]]
        key = (int(g["n_div"].iloc[0]), int(g["k_min"].iloc[0]))
        lv.setdefault(key, {"X": [], "y": []}); lv[key]["X"].append(X); lv[key]["y"].append(y)
    return {k: {"X": np.concatenate(v["X"]), "y": np.concatenate(v["y"]), "n_rounds": len(v["y"])} for k, v in lv.items()}
vec_rows = []
for stage, col in [("initial", "initial_text"), ("revised", "revised_text")]:
    D = vec_design(col); tot = sum(d["n_rounds"] for d in D.values()); acc_w = 0; acc_true = 0
    for (n, k), d in sorted(D.items()):
        if d["n_rounds"] < 10: continue
        w, mw, se, _, r2 = convex_fit_with_se(d["X"], d["y"], np.array([1.]*k + [0.]*(n-k)))
        acc_w += d["n_rounds"]/tot*mw; acc_true += d["n_rounds"]/tot*k/n
        vec_rows.append({"stage": stage, "n": n, "k": k, "n_rounds": d["n_rounds"], "minority_weight": mw, "r2": r2})
    vec_rows.append({"stage": stage, "n": "all", "k": "-", "n_rounds": tot, "minority_weight": acc_w, "true_share": acc_true})
vec = pd.DataFrame(vec_rows); vec.to_csv(f"{OUT_DIR}/vector_regression.csv", index=False); vec.round(3)""")

    md("""## 7. Diagnostics
(a) Where do the winning statements fall relative to their group's opinion scores? (b) If statements were *exactly* proportional
convex combinations of latent positions, would measurement noise in the position scores (calibrated to the observed
correlation with ratings) bias the recovered minority weight? A simulation with the real group structure answers this.""")
    code("""from hm_fig4c.analysis import assign_minority, build_design, minority_weight
opd = assign_minority(P.select_cohort(opinions, "cohorts_1_3"), neutral="as_majority")
stp = P.select_cohort(statements, "cohorts_1_3").set_index(P.KEY)
rows = []
for k, g in opd.groupby(P.KEY):
    if k not in stp.index or g["score"].isna().any(): continue
    sign = 1 if g[g["is_minority"]]["pre_rating"].iloc[0] > 4 else -1
    lo, hi = g["score"].min(), g["score"].max(); mn = g[g["is_minority"]]["score"].mean(); mj = g[~g["is_minority"]]["score"].mean()
    for col in ["initial_score", "revised_score"]:
        sc = stp.at[k, col]
        if not np.isfinite(sc): continue
        rows.append(dict(stage=col.split("_")[0], inside=(lo <= sc <= hi), beyond_minority_side=(sc > hi) if sign > 0 else (sc < lo),
                         beyond_majority_side=(sc < lo) if sign > 0 else (sc > hi), closer_to_minority_mean=abs(sc - mn) < abs(sc - mj)))
where = pd.DataFrame(rows).groupby("stage").mean(numeric_only=True)
where.to_csv(f"{OUT_DIR}/winner_position.csv"); where.round(3)""")
    code("""def simulate(target_r, n_rep=3, seed=0):
    rng = np.random.default_rng(seed); ests = []
    for rep in range(n_rep):
        d = opd.copy()
        d["true"] = (d["pre_rating"] - 4) / 3 + rng.normal(0, 0.35, len(d))          # latent position
        base_r = np.corrcoef(d["true"], d["pre_rating"])[0, 1]; var_t = d["true"].var()
        noise_var = var_t * ((base_r / target_r) ** 2 - 1) if target_r < base_r else 0   # noise to hit target r
        d["score"] = d["true"] + rng.normal(0, np.sqrt(max(noise_var, 0)), len(d))
        tg = d.groupby(P.KEY)["true"].mean().rename("score").reset_index()             # exactly proportional statement
        tg["score"] += rng.normal(0, np.sqrt(max(noise_var, 0)), len(tg))
        mw = minority_weight(build_design(d, tg, "score", "score"))
        ests.append((np.corrcoef(d["score"], d["pre_rating"])[0, 1], mw.weight, mw.true_share))
    return np.array(ests).mean(axis=0)
atten = pd.DataFrame([dict(zip(["achieved_r", "recovered_minority_weight", "true_share"], simulate(tr))) for tr in [0.99, 0.8, 0.64, fig4a.loc["cohorts_1_3", "r"], 0.45]])
atten.to_csv(f"{OUT_DIR}/attenuation.csv", index=False); atten.round(3)""")
    md("## 8. Summary vs paper")
    code("""paper = {"n_rounds": 1047, "minority_share": 0.285, "opinions_sanity": 0.28, "initial_candidates": 0.28, "initial_winner": 0.29,
         "revised_candidates": 0.33, "revised_winner": 0.36, "revised_winner_se": 0.03, "revised_winner_t_vs_true": 2.64,
         "fig4a_r": 0.64, "fig4b_within": 0.96}
ph = res["phases"]
ours = {"n_rounds": ph["initial_winner"]["n_rounds"], "n_rounds_total": int(P.select_cohort(statements, "cohorts_1_3")["initial_id"].notna().sum()),
        "minority_share": ph["initial_winner"]["true_share"], "opinions_sanity": ph["opinions"]["weight"],
        **{p: ph[p]["weight"] for p in P.PHASES}, "revised_winner_se": ph["revised_winner"]["se"], "revised_winner_t_vs_true": ph["revised_winner"]["t_vs_true"],
        "fig4a_r": fig4a.loc["cohorts_1_3", "r"], "fig4b_within": fig4b["cohorts_1_3"]["both"]["within"], "model": MODEL_TAG, "endpoints": ENDPOINTS, "axis": AXIS_METHOD}
json.dump({"paper": paper, "ours": ours}, open(f"{OUT_DIR}/summary.json", "w"), indent=1, default=float)
pd.DataFrame({"paper": paper, "ours": ours}).round(3)""")


def build_4d():
    md("""# Replicating Tessler et al. (2024) Fig. 4D — HM "majority bias" vs group movement toward the majority

Main text (RQ3): *"discussants might have gravitated toward the majority view simply because they were asked to judge several
group statements that supported that position. To test this, we measured the relationship between participants' viewpoint change
toward the majority position (from pre- to postdeliberation position ratings) and the fraction of group statements whose position
component scores fell on the majority side of the median opinion ('majority bias'). We found no relationship between fractional
exposure to majority views and subsequent change in viewpoint toward the majority (b = 0.058, SE = 0.07, z score = 0.9, P = 0.37)."*
Caption: *"Individual points represent a single group discussing a single question."* Panel D's x values sit at multiples of 1/8
(4 initial + 4 revised candidates) and its y values at multiples of 1/5 and 1/4 (group sizes), which fixes the y variable as a
group mean of a per-participant {-1, 0, +1} quantity.

Definitions used here (`hm_fig4c/fig4d.py`): majority direction from the pre-deliberation ratings (tie → AGREE, SM 4.1.2.1);
majority-aligned rating x' = 8 − x when the majority is DISAGREE; majority bias = share of the round's 8 candidates whose
position score lies on the majority side of the median opinion score; movement = mean of sign(post' − pre') over the group.
The paper does not state its random-effects structure for this test, so four estimators are reported.""")
    code("""import os, sys, json
sys.path.insert(0, os.path.abspath(".."))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from hm_fig4c import pipeline as P
from hm_fig4c import fig4d as F
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

EMB_DIR = os.environ.get("HM_EMB_DIR", "../embeddings/st5-large")
AXIS_METHOD = os.environ.get("HM_AXIS_METHOD", "unit")
ENDPOINTS = os.environ.get("HM_ENDPOINTS", "prefixed")
MODEL_TAG = os.path.basename(EMB_DIR.rstrip("/"))
OUT_DIR = f"../results/{MODEL_TAG}"; os.makedirs(OUT_DIR, exist_ok=True)
print(EMB_DIR, AXIS_METHOD, ENDPOINTS)""")

    md("## 1. Position scores and the per-round table")
    code("""opinions, statements, questions, candidates = P.score_all("../prepared", EMB_DIR, method=AXIS_METHOD, endpoint_style=ENDPOINTS)
rounds = F.round_table(opinions, candidates, cohort="cohorts_1_3", prereg_only=True, majority_by="rating")
rounds.to_csv(f"{OUT_DIR}/fig4d_rounds.csv", index=False)
print("rounds:", len(rounds), "| groups:", rounds["launch_id"].nunique(), "| questions:", rounds["question_id"].nunique())
print("candidates per round:", rounds["n_candidates"].value_counts().to_dict(), "| group size:", rounds["n"].value_counts().to_dict())
print("participants with both ratings per round:", rounds["n_both"].value_counts().sort_index().to_dict())
print("ties (majority set to AGREE):", int(rounds["tie"].sum()), "| rounds with a minority:", int(rounds["has_minority"].sum()))
rounds.head()""")

    md("## 2. Raw distributions before any model")
    code("""COLS = ["majority_bias", "bias_initial", "bias_revised", "movement_sign", "movement_mean", "movement_gai", "gai_pre"]
fig, axes = plt.subplots(1, 3, figsize=(11, 3))
rounds["majority_bias"].value_counts().sort_index().plot.bar(ax=axes[0], color="grey"); axes[0].set_title("HM majority bias (share of 8 candidates)", fontsize=9)
axes[0].set_xticklabels([f"{v:.3f}" for v in sorted(rounds["majority_bias"].unique())], rotation=90, fontsize=7)
rounds["movement_sign"].round(3).value_counts().sort_index().plot.bar(ax=axes[1], color="grey"); axes[1].set_title("Group movement to majority (mean sign)", fontsize=9)
axes[1].set_xticklabels([f"{v:.2f}" for v in sorted(rounds["movement_sign"].round(3).unique())], rotation=90, fontsize=7)
axes[2].bar(["initial", "revised", "all"], [rounds["bias_initial"].mean(), rounds["bias_revised"].mean(), rounds["majority_bias"].mean()], color=["#c6dbef", "#807dba", "grey"])
axes[2].axhline(.5, ls=":", color="k"); axes[2].set_title("Mean share of candidates on the majority side", fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig4d_distributions.png", dpi=150)
summary_dist = pd.DataFrame({"mean": rounds[COLS].mean(), "sd": rounds[COLS].std(), "share_positive": (rounds[COLS] > 0).mean(), "n": rounds[COLS].notna().sum()})
summary_dist.round(3)""")

    md("## 3. Fig. 4D — association between majority bias and movement toward the majority (paper: b = 0.058, SE = 0.07, z = 0.9, P = 0.37)")
    code("""fit = F.fit_association(rounds, "majority_bias", "movement_sign")
fit.round(4)""")
    code("""fig, ax = plt.subplots(figsize=(3.8, 3.6))
F.plot_fig4d(rounds, fit, ax=ax)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig4d.png", dpi=200)""")
    code("""# mean movement within each majority-bias band (what the regression line averages over)
band = rounds.groupby("majority_bias")["movement_sign"].agg(["mean", "sem", "size"]); band.round(3)""")

    md("""## 4. Sensitivity
Each row changes one choice: the movement measure (three definitions), the side rule (majority by rating vs by the sign of the
opinion scores), the candidate set (initial or revised only), the sample (rounds with a genuine minority; unanimous rounds; ties
excluded; no pre-registration filter; cohort 4), and adjustment for the group's pre-deliberation agreement (a ceiling on movement).
The slope is from the mixed model with a random intercept by group (falling back to OLS if that fit fails), z = b/SE.""")
    code("""def slope(df, x="majority_bias", y="movement_sign", covariates=()):
    f = F.fit_association(df, x, y, covariates=covariates, crossed=False)
    r = f.iloc[2] if "b" in f.columns and pd.notna(f.iloc[2].get("b", np.nan)) else f.iloc[0]
    return {"b": r["b"], "se": r["se"], "z": r["z"], "p": r["p"], "n_rounds": int(r["n_rounds"])}
rounds_score = F.round_table(opinions, candidates, cohort="cohorts_1_3", prereg_only=True, majority_by="score")
sens = {
  "primary: mean sign(post' - pre'), 8 candidates, majority by rating": slope(rounds),
  "movement = mean (post' - pre')": slope(rounds, y="movement_mean"),
  "movement = Group Agreement Index change": slope(rounds, y="movement_gai"),
  "majority side by sign of opinion scores": slope(rounds_score),
  "bias from the 4 initial candidates only": slope(rounds, x="bias_initial"),
  "bias from the 4 revised candidates only": slope(rounds, x="bias_revised"),
  "adjusted for pre-deliberation Group Agreement Index": slope(rounds, covariates=("gai_pre",)),
  "rounds with a minority only": slope(rounds[rounds["has_minority"]]),
  "unanimous rounds only": slope(rounds[~rounds["has_minority"]]),
  "tied rounds excluded": slope(rounds[~rounds["tie"]]),
  "all cohort 1-3 rounds (no pre-registration filter)": slope(F.round_table(opinions, candidates, cohort="cohorts_1_3", prereg_only=False)),
  "cohort 4 (critique exclusion)": slope(F.round_table(opinions, candidates, cohort="cohort4")),
}
sens = pd.DataFrame(sens).T; sens.to_csv(f"{OUT_DIR}/fig4d_sensitivity.csv"); sens.round(3)""")

    md("## 5. Summary vs paper")
    code("""primary = fit.iloc[0]; mixed = fit.iloc[2] if pd.notna(fit.iloc[2].get("b", np.nan)) else fit.iloc[0]
ours = {"n_rounds": int(primary["n_rounds"]), "b_ols": primary["b"], "se_ols": primary["se"], "z_ols": primary["z"], "p_ols": primary["p"],
        "b_mixed": mixed["b"], "se_mixed": mixed["se"], "z_mixed": mixed["z"], "p_mixed": mixed["p"],
        "mean_majority_bias": rounds["majority_bias"].mean(), "mean_movement_sign": rounds["movement_sign"].mean(),
        "model": MODEL_TAG, "endpoints": ENDPOINTS, "axis": AXIS_METHOD}
json.dump({"paper": F.PAPER, "ours": ours, "fits": fit.to_dict(orient="records")}, open(f"{OUT_DIR}/fig4d.json", "w"), indent=1, default=float)
pd.DataFrame({"paper": {"b": F.PAPER["b"], "se": F.PAPER["se"], "z": F.PAPER["z"], "p": F.PAPER["p"]},
              "ours (OLS)": {"b": primary["b"], "se": primary["se"], "z": primary["z"], "p": primary["p"]},
              "ours (mixed, group intercept)": {"b": mixed["b"], "se": mixed["se"], "z": mixed["z"], "p": mixed["p"]}}).round(3)""")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", choices=["4c", "4d"], default="4c")
    a = ap.parse_args()
    {"4c": build_4c, "4d": build_4d}[a.figure]()
    nb = nbf.v4.new_notebook(); nb["cells"] = cells
    out = f"notebooks/fig{a.figure}.ipynb"
    nbf.write(nb, out)
    print("wrote", out, "with", len(cells), "cells")
