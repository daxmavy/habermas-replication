"""values.json -> report/parameters.tex.  Pure function of the input: --check is a byte comparison.

    uv run python report/build_parameters.py           # write
    uv run python report/build_parameters.py --check   # exit 1 if stale or hand-edited
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VALUES = ROOT / "report" / "values.json"
OUT = ROOT / "report" / "parameters.tex"

SUFFIX = {"st5-base": "Base", "st5-large": "Large", "st5-xl": "Xl", "st5-xxl": "Xxl"}
PHASE_MACRO = {"initial_candidates": "InitCand", "initial_winner": "InitWin",
               "revised_candidates": "RevCand", "revised_winner": "RevWin"}


def fmt(v, spec: str) -> str:
    if spec == "int":
        return f"{int(round(v))}"
    if spec == "pct0":
        return f"{100 * v:.0f}\\%"
    if spec.startswith("f"):
        return f"{v:.{int(spec[1:])}f}"
    raise ValueError(spec)


def spec_rows(V) -> list[tuple[str, str]]:
    """(macro name, rendered body). Precision is declared here, never inferred."""
    rows: list[tuple[str, object, str]] = []
    p = V["paper"]
    rows += [
        ("paperShare", p["minority_share"], "f3"), ("paperOpinions", p["opinions_sanity"], "f2"),
        ("paperInitCand", p["initial_candidates"], "f2"), ("paperInitWin", p["initial_winner"], "f2"),
        ("paperRevCand", p["revised_candidates"], "f2"), ("paperRevWin", p["revised_winner"], "f2"),
        ("paperRevWinSE", p["revised_winner_se"], "f2"), ("paperNRounds", p["n_rounds"], "int"),
        ("paperFigAr", p["fig4a_r"], "f2"), ("paperFigBwithin", p["fig4b_within"], "f2"),
        ("paperMarginalRsq", p["marginal_r2"], "f2"), ("paperConditionalRsq", p["conditional_r2"], "f2"),
        ("paperFigArSq", p["fig4a_r"] ** 2, "f2"),
        ("paperNGroups", p["sample"]["n_groups"], "int"), ("paperNParticipants", p["sample"]["n_participants"], "int"),
        ("paperRoundsPerGroup", p["sample"]["rounds_per_group"], "int"),
    ]

    g = V["groups"]
    rows += [("groupMin", g["min"], "int"), ("groupMax", g["max"], "int"), ("groupNRounds", g["n_rounds"], "int"),
             ("groupNGroups", g["n_groups"], "int"), ("groupNParticipants", g["n_participants"], "int")]

    for name, mac in (("neutral_dropped", "NeutralDropped"),):
        r = V["share_by_rule"][name]
        rows += [(f"shareRule{mac}", r["true_share"], "f3"), (f"shareRule{mac}NRounds", r["n_rounds"], "int")]

    for m, sfx in SUFFIX.items():
        if m not in V["ours"]:
            continue
        o, ax, s, c = V["ours"][m], V["axis"][m], V["sensitivity"]["per_model"][m], V["contrasts"][m]
        rows += [(f"oursShare{sfx}", o["minority_share"], "f3"), (f"oursOpinions{sfx}", o["opinions_sanity"], "f3"),
                 (f"oursNRounds{sfx}", o["n_rounds"], "int"), (f"oursNRoundsTotal{sfx}", o["n_rounds_total"], "int"),
                 (f"oursFigAr{sfx}", o["fig4a_r"], "f2"), (f"oursFigBwithin{sfx}", o["fig4b_within"], "f2")]
        for ph, pm in PHASE_MACRO.items():
            rows.append((f"ours{pm}{sfx}", o[ph], "f3"))
            rows.append((f"ours{pm}BootSE{sfx}", o["phase_boot_se"][ph], "f3"))
        mx = ax["mixed"]
        rows += [(f"oursMarginalRsqGap{sfx}", abs(V["paper"]["marginal_r2"] - mx["marginal_r2"]), "f3"),
                 (f"oursMarginalRsq{sfx}", mx["marginal_r2"], "f3"),
                 (f"oursConditionalRsq{sfx}", mx["conditional_r2"], "f3"),
                 (f"oursPearsonR{sfx}", mx["pearson_r"], "f3"), (f"oursPearsonRsq{sfx}", mx["pearson_r2"], "f3"),
                 (f"oursMixedBeta{sfx}", mx["beta"], "f3"), (f"oursMixedBetaSE{sfx}", mx["beta_se"], "f3"),
                 (f"oursMixedN{sfx}", mx["n"], "int"), (f"oursMixedNRounds{sfx}", mx["n_rounds"], "int")]
        for key, mac in (("revised_winner - initial_winner", "WinToWin"),
                         ("revised_winner - initial_candidates", "InitToFinal")):
            rows += [(f"diff{mac}{sfx}", c[key]["diff"], "f3"),
                     (f"diff{mac}CIlo{sfx}", c[key]["boot_ci95"][0], "f3"),
                     (f"diff{mac}CIhi{sfx}", c[key]["boot_ci95"][1], "f3"),
                     (f"diff{mac}P{sfx}", c[key]["boot_p_le_0"], "f3")]
        rows.append((f"sensNRuns{sfx}", s["n_runs"], "int"))
        for ph, pm in PHASE_MACRO.items():
            rows += [(f"sens{pm}Mean{sfx}", s[ph]["mean"], "f3"),
                     (f"sens{pm}Min{sfx}", s[ph]["min"], "f3"), (f"sens{pm}Max{sfx}", s[ph]["max"], "f3")]

    n_boot = {V["ours"][m]["n_boot"] for m in SUFFIX if m in V["ours"]}
    if len(n_boot) != 1:
        raise SystemExit(f"models bootstrapped with different resample counts: {n_boot}")
    rows.append(("nBoot", n_boot.pop(), "int"))
    d = V["sensitivity"]["design"]
    rows += [("sensNModels", d["n_models"], "int"), ("sensNNeutral", d["n_neutral"], "int"), ("sensNOrders", d["n_orders"], "int")]
    sp = V["sensitivity"]["pooled"]
    rows += [("sensNRuns", sp["n_runs"], "int"), ("sensNRunsPerModel", V["sensitivity"]["n_runs_per_model"], "int"),
             ("sensPropInitToFinal", sp["prop_increase_initial_to_final"], "pct0"),
             ("sensPropAboveShare", sp["prop_final_above_true_share"], "pct0"),
             ("sensMaxRevWin", sp["max_revised_winner"], "f3"), ("sensTrueShareMean", sp["true_share_mean"], "f3")]

    seen = set()
    out = []
    for name, val, f in rows:
        if name in seen:
            raise SystemExit(f"duplicate macro {name}")
        seen.add(name)
        if val is None or (isinstance(val, float) and (val != val or abs(val) == float("inf"))):
            raise SystemExit(f"refusing to emit non-finite value for {name}")
        out.append((name, fmt(val, f)))
    return out


def render(V) -> str:
    lines = ["% GENERATED by report/build_parameters.py from report/values.json -- do not edit.",
             "% Every number in the report comes from here.", ""]
    lines += [f"\\newcommand{{\\{n}}}{{{b}}}" for n, b in spec_rows(V)]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = render(json.loads(VALUES.read_text()))
    if args.check:
        cur = OUT.read_text() if OUT.exists() else ""
        if cur != text:
            print("parameters.tex is stale or hand-edited. Run: uv run python report/build_parameters.py", file=sys.stderr)
            sys.exit(1)
        print("parameters.tex up to date")
        return
    OUT.write_text(text)
    print(f"wrote {OUT} ({len(text.splitlines()) - 3} macros)")


if __name__ == "__main__":
    main()
