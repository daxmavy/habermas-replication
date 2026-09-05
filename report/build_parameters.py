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

SUFFIX = {"st5-base": "Base", "st5-large": "Large"}
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
        ("paperGroupSize", p["designed_group_size"], "int"),
    ]

    g = V["groups"]
    rows += [("groupMin", g["min"], "int"), ("groupMax", g["max"], "int"), ("groupModal", g["modal"], "int"),
             ("groupModalShare", g["modal_share"], "pct0"), ("groupNRounds", g["n_rounds"], "int")]
    for size, n in g["counts"].items():
        rows.append((f"groupNof{ {'4':'Four','5':'Five','6':'Six'}[str(size)] }".replace(" ", ""), n, "int"))

    for m, sfx in SUFFIX.items():
        if m not in V["ours"]:
            continue
        o, ax, s, c = V["ours"][m], V["axis"][m], V["sensitivity"][m], V["contrasts"][m]
        rows += [(f"oursShare{sfx}", o["minority_share"], "f3"), (f"oursOpinions{sfx}", o["opinions_sanity"], "f3"),
                 (f"oursNRounds{sfx}", o["n_rounds"], "int"), (f"oursNRoundsTotal{sfx}", o["n_rounds_total"], "int"),
                 (f"oursFigAr{sfx}", o["fig4a_r"], "f2"), (f"oursFigBwithin{sfx}", o["fig4b_within"], "f2")]
        for ph, pm in PHASE_MACRO.items():
            rows.append((f"ours{pm}{sfx}", o[ph], "f3"))
            rows.append((f"ours{pm}SE{sfx}", o["phase_se"][ph], "f3"))
        rows += [(f"oursFullDimInit{sfx}", ax["full768_initial"], "f3"),
                 (f"oursFullDimRev{sfx}", ax["full768_revised"], "f3"),
                 (f"oursAttenMinR{sfx}", ax["attenuation_min_r"], "f2"),
                 (f"oursAttenRecovered{sfx}", ax["attenuation_recovered_at_min_r"], "f3"),
                 (f"oursWinnerInsideInit{sfx}", ax["winner_inside_initial"], "pct0"),
                 (f"oursWinnerInsideRev{sfx}", ax["winner_inside_revised"], "pct0")]
        if "mixed" in ax:
            mx = ax["mixed"]
            rows += [(f"oursMarginalRsqGap{sfx}", abs(V["paper"]["marginal_r2"] - mx["marginal_r2"]), "f3"),
                     (f"oursMarginalRsq{sfx}", mx["marginal_r2"], "f3"),
                     (f"oursConditionalRsq{sfx}", mx["conditional_r2"], "f3"),
                     (f"oursPearsonR{sfx}", mx["pearson_r"], "f3"),
                     (f"oursMixedN{sfx}", mx["n"], "int"), (f"oursMixedNRounds{sfx}", mx["n_rounds"], "int")]
        for key, mac in (("revised_winner - initial_winner", "WinToWin"),
                         ("revised_winner - initial_candidates", "InitToFinal"),
                         ("revised_candidates - initial_candidates", "CandToCand")):
            if key in c:
                rows += [(f"diff{mac}{sfx}", c[key]["diff"], "f3"),
                         (f"diff{mac}CIlo{sfx}", c[key]["boot_ci95"][0], "f3"),
                         (f"diff{mac}CIhi{sfx}", c[key]["boot_ci95"][1], "f3"),
                         (f"diff{mac}P{sfx}", c[key]["boot_p_le_0"], "f3")]
        rows += [(f"sensPropWinToWin{sfx}", s["all"]["prop_increase_winner_to_winner"], "pct0"),
                 (f"sensPropInitToFinal{sfx}", s["all"]["prop_increase_initial_to_final"], "pct0"),
                 (f"sensPropAboveShare{sfx}", s["all"]["prop_final_above_true_share"], "pct0"),
                 (f"sensMaxRevWin{sfx}", s["all"]["max_revised_winner"], "f3"),
                 (f"sensTrueShareMean{sfx}", s["true_share_mean"], "f3")]
        for ph, pm in PHASE_MACRO.items():
            rows += [(f"sens{pm}Mean{sfx}", s[f"{ph}_mean"], "f3"),
                     (f"sens{pm}Pfive{sfx}", s[f"{ph}_p5"], "f3"),
                     (f"sens{pm}Pninefive{sfx}", s[f"{ph}_p95"], "f3")]

    s0 = V["sensitivity"][V["models"][0]]
    rows += [("sensNAttempted", s0["n_attempted"], "int"), ("sensNCompleted", s0["n_variants"], "int"),
             ("sensNFailed", s0["n_failed"], "int"), ("sensNAnalytic", s0["n_analytic"], "int"),
             ("sensNSample", s0["n_sample"], "int")]

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
