"""Generate the Fig. 4C analysis notebook (reusable code lives in hm_fig4c/).

    uv run python notebooks/make_notebook.py     # writes notebooks/fig4c.ipynb

Executing the notebook once per embedding model writes results/<model>/{fig4c_primary.json,
sensitivity.csv, summary.json}, the only results the report reads:

    cd notebooks && HM_EMB_DIR=../embeddings/st5-large \\
        ../.venv/bin/jupyter nbconvert --to notebook --execute fig4c.ipynb --output fig4c_st5-large_out.ipynb
"""
import nbformat as nbf

cells = []


def _add(cell):
    cell["id"] = f"cell{len(cells):02d}"   # deterministic ids: regenerating the notebook is byte-identical
    cells.append(cell)


md = lambda s: _add(nbf.v4.new_markdown_cell(s))
code = lambda s: _add(nbf.v4.new_code_cell(s))


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
ax.set_title(f"Cohorts 1-3: r = {fig4a.loc['cohorts_1_3','r']:.2f}  (paper: 0.64)", fontsize=9); plt.tight_layout()""")

    md("## 3. Fig. 4B check — statement scores relative to the group's opinions (paper: 96% within range)")
    code("""fig4b = {c: P.fig4b_within_range(P.select_cohort(opinions, c), P.select_cohort(statements, c)) for c in ["cohorts_1_3", "training", "vca"]}
pd.DataFrame({(c, s): v for c, dd in fig4b.items() for s, v in dd.items()}).T""")
    code("""d_op = P.select_cohort(opinions, "cohorts_1_3"); d_st = P.select_cohort(statements, "cohorts_1_3")
fig, ax = plt.subplots(figsize=(5, 3.2))
for vals, lab, col in [(d_op["score"], "opinions", "tab:red"), (d_st["initial_score"], "initial statements", "tab:blue"), (d_st["revised_score"], "revised statements", "tab:purple")]:
    ax.hist(vals.dropna(), bins=60, density=True, histtype="step", lw=1.5, label=lab, color=col)
ax.set_xlabel("Position component score (0 = negating, 1 = affirming)"); ax.legend(frameon=False, fontsize=8); plt.tight_layout()""")

    md("""## 4. Fig. 4C — minority weight via convex regression (primary specification)

Main-task cohorts 1–3, pre-registered groups (n = 1047 rounds); minority = smaller side of neutral on the pre-deliberation
rating, neutral opinions kept as non-minority (SM 5.4.1); rounds with a tie or no dissent excluded; columns ordered as in the data.
Analytic SEs are the OLS standard errors of the constrained fit (as in the paper); bootstrap SEs resample rounds.""")
    code("""res = P.run_minority_analysis(opinions, candidates, cohort="cohorts_1_3", neutral="as_majority", order="data", n_boot=N_BOOT)
P.save_results(res, f"{OUT_DIR}/fig4c_primary.json")
summary = P.summarize(res); summary.round(3)""")
    code("""pd.DataFrame(res["contrasts"]).T.round(3)""")
    code("""P.per_level_table(res).round(3)""")

    md("""## 5. Sensitivity analyses
Every combination of the choices the SM leaves open (`P.sensitivity_grid()`): neutral-opinion treatment × basis for the
minority split × column order of the design matrix; model size is swept by running this notebook once per embedding model.
Sample (pre-registered rounds of cohorts 1–3) and tie handling (excluded) are as the SM specifies and are not varied.""")
    code("""sens = P.run_sensitivity(opinions, candidates, P.sensitivity_grid())
sens.to_csv(f"{OUT_DIR}/sensitivity.csv", index=False); sens.round(3)""")

    md("## 6. Summary vs paper")
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


if __name__ == "__main__":
    build_4c()
    nb = nbf.v4.new_notebook(); nb["cells"] = cells
    out = "notebooks/fig4c.ipynb"
    nbf.write(nb, out)
    print("wrote", out, "with", len(cells), "cells")
