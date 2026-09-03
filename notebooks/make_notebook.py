"""Generate notebooks/fig4c.ipynb (analysis notebook; reusable code lives in hm_fig4c/)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

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
minority = the side of neutral with fewer pre-deliberation ratings, neutral raters count as non-minority; convex regression
(weights >= 0, sum = 1) of statement scores on the group's opinion scores, one regression per (group size, minority size) level,
minority weight = sum of minority coefficients, averaged over levels weighted by number of rounds.""")

code("""import os, sys, json
sys.path.insert(0, os.path.abspath(".."))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from hm_fig4c import pipeline as P
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

EMB_DIR = os.environ.get("HM_EMB_DIR", "../embeddings/st5-base")   # embedding cache to use
AXIS_METHOD = os.environ.get("HM_AXIS_METHOD", "affine")             # 'affine' (0 = negating, 1 = affirming) or 'unit'
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
rating, neutral raters kept as non-minority (SM 5.4.1); rounds with a tie or no dissent excluded; columns ordered as in the data.
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
Each row varies one choice relative to the primary specification.""")
code("""variants = {
  "primary (cohorts 1-3 prereg, neutral = non-minority, data order)": dict(cohort="cohorts_1_3"),
  "neutral raters dropped (group kept)": dict(cohort="cohorts_1_3", neutral="drop_participant"),
  "groups with any neutral rater dropped": dict(cohort="cohorts_1_3", neutral="drop_group"),
  "column order: sorted by score": dict(cohort="cohorts_1_3", order="sorted"),
  "column order: random": dict(cohort="cohorts_1_3", order="random"),
  "all cohorts 1-3 rounds (no pre-registration filter)": dict(cohort="cohorts_1_3", prereg_only=False),
  "cohort 1 only": dict(cohort="cohort1"), "cohort 2 only": dict(cohort="cohort2"), "cohort 3 only": dict(cohort="cohort3"),
  "cohort 4 (critique exclusion)": dict(cohort="cohort4"),
  "training data": dict(cohort="training"), "virtual citizens' assembly": dict(cohort="vca"),
}
rows = []
for name, kw in variants.items():
    try:
        r = P.run_minority_analysis(opinions, candidates, include_opinions=False, **kw)
        row = {"variant": name, "n_rounds": r["phases"]["initial_winner"]["n_rounds"], "true_share": r["phases"]["initial_winner"]["true_share"]}
        for ph in P.PHASES:
            row[ph] = r["phases"][ph]["weight"]; row[ph + "_se"] = r["phases"][ph]["se"]
        rows.append(row)
    except Exception as e:
        rows.append({"variant": name, "error": str(e)[:80]})
sens = pd.DataFrame(rows); sens.to_csv(f"{OUT_DIR}/sensitivity.csv", index=False); sens.round(3)""")

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

md("## 7. Summary vs paper")
code("""paper = {"n_rounds": 1047, "minority_share": 0.285, "opinions_sanity": 0.28, "initial_candidates": 0.28, "initial_winner": 0.29,
         "revised_candidates": 0.33, "revised_winner": 0.36, "revised_winner_se": 0.03, "revised_winner_t_vs_true": 2.64,
         "fig4a_r": 0.64, "fig4b_within": 0.96}
ph = res["phases"]
ours = {"n_rounds": ph["initial_winner"]["n_rounds"], "minority_share": ph["initial_winner"]["true_share"], "opinions_sanity": ph["opinions"]["weight"],
        **{p: ph[p]["weight"] for p in P.PHASES}, "revised_winner_se": ph["revised_winner"]["se"], "revised_winner_t_vs_true": ph["revised_winner"]["t_vs_true"],
        "fig4a_r": fig4a.loc["cohorts_1_3", "r"], "fig4b_within": fig4b["cohorts_1_3"]["both"]["within"], "model": MODEL_TAG, "endpoints": ENDPOINTS, "axis": AXIS_METHOD}
json.dump({"paper": paper, "ours": ours}, open(f"{OUT_DIR}/summary.json", "w"), indent=1, default=float)
pd.DataFrame({"paper": paper, "ours": ours}).round(3)""")

nb["cells"] = cells
nbf.write(nb, "notebooks/fig4c.ipynb")
print("wrote notebooks/fig4c.ipynb with", len(cells), "cells")
