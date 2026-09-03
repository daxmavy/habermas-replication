"""Generate the replication report page from results/<model>/ (numbers come from JSON/CSV, never retyped).
Usage: uv run python make_report.py --results results/st5-base [--results results/st5-large] --out report/fig4c_report.html
"""
from __future__ import annotations

import argparse, base64, html, json
from pathlib import Path

import pandas as pd

PHASES = ["initial_candidates", "initial_winner", "revised_candidates", "revised_winner"]
LABEL = {"opinions": "Opinions (sanity)", "initial_candidates": "Initial statements", "initial_winner": "Initial winner",
         "revised_candidates": "Revised statements", "revised_winner": "Revised winner"}
PAPER = {"opinions": 0.28, "initial_candidates": 0.28, "initial_winner": 0.29, "revised_candidates": 0.33, "revised_winner": 0.36}
PAPER_TRUE = 0.285


def load(results_dir: Path):
    d = {"tag": results_dir.name, "primary": json.load(open(results_dir / "fig4c_primary.json")),
         "summary": json.load(open(results_dir / "summary.json"))}
    for f in ["sensitivity.csv", "vector_regression.csv"]:
        if (results_dir / f).exists():
            d[f.split(".")[0]] = pd.read_csv(results_dir / f)
    for f in ["fig4a.png", "fig4b.png", "fig4c.png"]:
        if (results_dir / f).exists():
            d[f.split(".")[0]] = "data:image/png;base64," + base64.b64encode((results_dir / f).read_bytes()).decode()
    return d


def fmt(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "–"


def bar_chart_svg(res: dict, tag: str) -> str:
    """Inline SVG: our estimates (bars ± 1 SE) against the paper's reported values (ticks), sharing one scale."""
    ph = res["phases"]; phases = ["opinions"] + PHASES
    W, H = 640, 300; ml, mr, mt, mb = 46, 12, 26, 58
    ymax = 0.5; yscale = lambda v: mt + (H - mt - mb) * (1 - v / ymax)
    n = len(phases); slot = (W - ml - mr) / n; bw = slot * 0.42
    true = ph["initial_winner"]["true_share"]
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Minority weight by statement phase, {tag} versus paper" font-family="var(--mono)" font-size="11">']
    for v in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        y = yscale(v)
        out.append(f'<line x1="{ml}" x2="{W-mr}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--line)" stroke-width="1"/>')
        out.append(f'<text x="{ml-6}" y="{y+4:.1f}" text-anchor="end" fill="var(--muted)">{v:.1f}</text>')
    yt = yscale(true)
    out.append(f'<line x1="{ml}" x2="{W-mr}" y1="{yt:.1f}" y2="{yt:.1f}" stroke="var(--ink)" stroke-width="1.2" stroke-dasharray="5 4"/>')
    out.append(f'<text x="{W-mr}" y="{yt-5:.1f}" text-anchor="end" fill="var(--ink)" font-style="italic">true minority share {true:.3f}</text>')
    ytp = yscale(PAPER_TRUE)
    out.append(f'<line x1="{ml}" x2="{W-mr}" y1="{ytp:.1f}" y2="{ytp:.1f}" stroke="var(--paper)" stroke-width="1" stroke-dasharray="2 4"/>')
    for i, p in enumerate(phases):
        e = ph.get(p)
        if e is None:
            continue
        x0 = ml + i * slot + slot / 2 - bw / 2 - 8
        y = yscale(e["weight"]); se = e.get("se", 0) or 0
        out.append(f'<rect x="{x0:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{yscale(0)-y:.1f}" fill="var(--accent)" opacity="0.9"/>')
        if se:
            out.append(f'<line x1="{x0+bw/2:.1f}" x2="{x0+bw/2:.1f}" y1="{yscale(e["weight"]+se):.1f}" y2="{yscale(e["weight"]-se):.1f}" stroke="var(--ink)" stroke-width="1.5"/>')
        out.append(f'<text x="{x0+bw/2:.1f}" y="{y-6:.1f}" text-anchor="middle" fill="var(--ink)" font-weight="600">{e["weight"]:.2f}</text>')
        # paper value as a hollow marker to the right of the bar
        xp = x0 + bw + 14; yp = yscale(PAPER[p])
        out.append(f'<line x1="{xp-7:.1f}" x2="{xp+7:.1f}" y1="{yp:.1f}" y2="{yp:.1f}" stroke="var(--paper)" stroke-width="3"/>')
        out.append(f'<text x="{xp:.1f}" y="{yp-6:.1f}" text-anchor="middle" fill="var(--paper)">{PAPER[p]:.2f}</text>')
        lab = LABEL[p].split(" ")
        for j, word in enumerate(lab):
            out.append(f'<text x="{ml + i*slot + slot/2:.1f}" y="{H-mb+16+j*13}" text-anchor="middle" fill="var(--ink)" font-family="var(--body)" font-size="12">{html.escape(word)}</text>')
    out.append(f'<rect x="{ml}" y="{H-14}" width="12" height="10" fill="var(--accent)"/><text x="{ml+16}" y="{H-5}" fill="var(--muted)" font-family="var(--body)" font-size="11">this replication ({html.escape(tag)}), ± 1 SE</text>')
    out.append(f'<line x1="{ml+190}" x2="{ml+204}" y1="{H-9}" y2="{H-9}" stroke="var(--paper)" stroke-width="3"/><text x="{ml+210}" y="{H-5}" fill="var(--muted)" font-family="var(--body)" font-size="11">Tessler et al. (2024), Fig. 4C / S60</text>')
    out.append("</svg>")
    return "\n".join(out)


def table(df: pd.DataFrame, cols: list[str], names: list[str], nd=3) -> str:
    rows = []
    for _, r in df.iterrows():
        cells = "".join(f"<td>{html.escape(str(r[c])) if isinstance(r[c], str) else fmt(r[c], nd)}</td>" for c in cols)
        rows.append(f"<tr>{cells}</tr>")
    head = "".join(f"<th>{html.escape(n)}</th>" for n in names)
    return f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def verdict(res: dict) -> tuple[str, str]:
    ph = res["phases"]; true = ph["initial_winner"]["true_share"]
    rw = ph["revised_winner"]; iw = ph["initial_winner"]
    over = (rw["weight"] - true) / rw["se"] if rw["se"] else float("nan")
    init_ok = abs(iw["weight"] - true) < 2 * iw["se"]
    c = res.get("contrasts", {}).get("revised_winner - initial_winner", {})
    p_diff = c.get("boot_p_le_0")
    if over > 2 and init_ok:
        return "ok", "The pattern replicates: initial statements weight the minority in proportion to its size, and revised winners over-weight it."
    if over > 1.5 and init_ok:
        return "warn", "The pattern is directionally reproduced but the revised-winner over-weighting is weaker than in the paper."
    return "warn", "The paper's pattern is not clearly reproduced under the primary specification; see the sensitivity table."


def build(results: list[dict], out: Path):
    main = results[0]; res = main["primary"]; ph = res["phases"]; s = main["summary"]["ours"]; paper = main["summary"]["paper"]
    level, verdict_text = verdict(res)
    c = res.get("contrasts", {}).get("revised_winner - initial_winner", {})
    comp_rows = [("Rounds (main-task cohorts 1–3, pre-registered)", str(paper["n_rounds"]), str(s["n_rounds"]))]
    comp_rows += [("True minority share (dotted line)", fmt(paper["minority_share"]), fmt(s["minority_share"], 3))]
    for p in ["opinions"] + PHASES:
        key = "opinions_sanity" if p == "opinions" else p
        se = ph[p].get("se")
        comp_rows.append((LABEL[p], fmt(paper[key]), fmt(s[key]) + (f" ± {fmt(se)}" if se else "")))
    comp_rows.append(("Revised winner vs true share (t)", fmt(paper["revised_winner_t_vs_true"]), fmt(s["revised_winner_t_vs_true"])))
    comp_rows.append(("Fig. 4A: r(position score, pre-rating)", fmt(paper["fig4a_r"]), fmt(s["fig4a_r"])))
    comp_rows.append(("Fig. 4B: statements within opinion range", fmt(paper["fig4b_within"]), fmt(s["fig4b_within"])))
    comp_html = "".join(f"<tr><th scope=\"row\">{html.escape(a)}</th><td>{b}</td><td>{c_}</td></tr>" for a, b, c_ in comp_rows)

    other_models = ""
    if len(results) > 1:
        rows = []
        for r in results:
            pr = r["primary"]["phases"]
            rows.append("<tr><th scope=\"row\">" + html.escape(r["tag"]) + "</th>" + "".join(f"<td>{fmt(pr[p]['weight'])} ± {fmt(pr[p]['se'])}</td>" for p in PHASES) + f"<td>{fmt(r['summary']['ours']['fig4a_r'])}</td></tr>")
        other_models = f"""<h2>Embedding model</h2>
<p>The SM names Sentence-T5 but not its size; both public ports were run. Each row is the primary specification.</p>
<div class="tablewrap"><table><thead><tr><th>Model</th>{"".join(f"<th>{LABEL[p]}</th>" for p in PHASES)}<th>Fig. 4A r</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>"""

    sens_html = ""
    if "sensitivity" in main:
        sd = main["sensitivity"].copy()
        cols = ["variant", "n_rounds", "true_share"] + PHASES
        sens_html = table(sd, cols, ["Specification", "Rounds", "True share"] + [LABEL[p] for p in PHASES])
    vec_html = ""
    if "vector_regression" in main:
        vd = main["vector_regression"]; vd = vd[vd["n"].astype(str) == "all"]
        vec_html = table(vd, ["stage", "n_rounds", "minority_weight", "true_share"], ["Winner", "Rounds", "Minority weight (768-d)", "True share"])

    contrast_html = ""
    if c:
        contrast_html = f"""<p class="note">Revised winner − initial winner: {fmt(c['diff'])} (cluster-bootstrap SE {fmt(c['boot_se'])}, 95% CI {fmt(c['boot_ci95'][0])} to {fmt(c['boot_ci95'][1])}; bootstrap p(diff ≤ 0) = {fmt(c['boot_p_le_0'], 3)}). Revised winner above the true share: bootstrap p = {fmt(ph['revised_winner'].get('boot_p_gt_true'), 3)}.</p>"""

    page = f"""<title>Habermas Machine Fig. 4C Replication</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root {{ --bg:#F6F6FA; --surface:#FFFFFF; --ink:#1B1D2A; --muted:#5E6275; --accent:#5B4B9A; --paper:#9C9385; --line:#DCDCE6; --ok:#2F7D5B; --warn:#B7791F;
  --display:'Fraunces', Georgia, 'Times New Roman', serif; --body:'Source Sans 3', 'Segoe UI', Helvetica, Arial, sans-serif; --mono:'JetBrains Mono', Menlo, Consolas, monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#14151C; --surface:#1C1E28; --ink:#E8E8F0; --muted:#A0A3B5; --accent:#A594E6; --paper:#8C857A; --line:#2C2E3B; --ok:#5FBF93; --warn:#E0A94A; }} }}
:root[data-theme="dark"] {{ --bg:#14151C; --surface:#1C1E28; --ink:#E8E8F0; --muted:#A0A3B5; --accent:#A594E6; --paper:#8C857A; --line:#2C2E3B; --ok:#5FBF93; --warn:#E0A94A; }}
body {{ background:var(--bg); color:var(--ink); font-family:var(--body); font-size:16px; line-height:1.55; margin:0; }}
main {{ max-width:70ch; margin:0 auto; padding:32px 20px 64px; }}
h1 {{ font-family:var(--display); font-weight:700; font-size:2rem; line-height:1.15; margin:0 0 6px; text-wrap:balance; }}
h2 {{ font-family:var(--display); font-weight:500; font-size:1.35rem; margin:36px 0 10px; text-wrap:balance; }}
.eyebrow {{ font-family:var(--mono); font-size:0.72rem; letter-spacing:0.08em; text-transform:uppercase; color:var(--muted); margin-bottom:10px; }}
.verdict {{ display:flex; gap:12px; align-items:flex-start; border-left:4px solid var(--{level}); padding:10px 14px; background:var(--surface); margin:18px 0 8px; }}
.verdict .dot {{ width:10px; height:10px; border-radius:50%; background:var(--{level}); margin-top:8px; flex:none; }}
figure {{ margin:18px 0; background:var(--surface); border:1px solid var(--line); padding:12px 8px 6px; }}
figcaption {{ font-size:0.85rem; color:var(--muted); padding:6px 8px 4px; }}
.tablewrap {{ overflow-x:auto; margin:10px 0; }}
table {{ border-collapse:collapse; width:100%; font-size:0.9rem; font-variant-numeric:tabular-nums; }}
th, td {{ text-align:left; padding:6px 10px; border-bottom:1px solid var(--line); vertical-align:top; }}
thead th {{ font-family:var(--mono); font-size:0.7rem; letter-spacing:0.06em; text-transform:uppercase; color:var(--muted); }}
tbody th {{ font-weight:600; }}
td {{ font-family:var(--mono); font-size:0.85rem; }}
.compare td:nth-child(2) {{ color:var(--paper); }}
.compare td:nth-child(3) {{ color:var(--accent); font-weight:600; }}
p {{ margin:8px 0; }}
.note {{ font-size:0.9rem; color:var(--muted); }}
ul {{ padding-left:20px; }} li {{ margin:4px 0; }}
code {{ font-family:var(--mono); font-size:0.85em; }}
img {{ max-width:100%; border:1px solid var(--line); background:#fff; }}
.two {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:12px; }}
</style>
<main>
<div class="eyebrow">Replication · Tessler et al., <em>Science</em> 386 (2024), Fig. 4C · main-task cohorts 1–3</div>
<h1>Does the Habermas Machine over-weight minority opinions after critique?</h1>
<p class="note">Embeddings: Sentence-T5 ({html.escape(main['tag'])}); axis endpoints: “Yes, I agree. …” / “No, I disagree. …”; minority rule from SM 5.4.1; convex regression per (group size, minority size) level, averaged over levels. Pipeline rebuilt from the supplementary methods; the original analysis code no longer exists.</p>
<div class="verdict"><div class="dot"></div><div><strong>{html.escape(verdict_text)}</strong>{contrast_html}</div></div>
<figure>{bar_chart_svg(res, main['tag'])}<figcaption>Weight of minority opinions in each type of group statement (position axis). Bars: this replication ± 1 analytic SE; short horizontal marks: values printed in the paper's Fig. S60. Dashed line: true minority share on these rounds; dotted grey line: the paper's.</figcaption></figure>
<h2>Numbers against the paper</h2>
<div class="tablewrap"><table class="compare"><thead><tr><th>Quantity</th><th>Paper</th><th>This replication</th></tr></thead><tbody>{comp_html}</tbody></table></div>
{other_models}
<h2>Sensitivity</h2>
<p class="note">The SM's literal minority rule (neutral raters are non-minority, tied rounds excluded) gives a true share of 0.26 on these rounds; the paper's dotted line is at 0.28–0.29, which two alternative readings reproduce: dropping neutral raters, or keeping tied rounds with a fixed side as the minority.</p>
{sens_html}
<h2>Robustness: full 768-d embedding instead of the position score</h2>
{vec_html}
<div class="two">
<figure><img src="{main.get('fig4a','')}" alt="Fig 4A check: position score against pre-deliberation rating"><figcaption>Fig. 4A check: opinion position score by pre-deliberation rating.</figcaption></figure>
<figure><img src="{main.get('fig4b','')}" alt="Fig 4B check: distributions of position scores"><figcaption>Fig. 4B check: position-score distributions for opinions, initial and revised winners.</figcaption></figure>
</div>
<h2>What was rebuilt, and the judgement calls</h2>
<ul>
<li><strong>Data.</strong> Public parquet release; the pre-registered preprocessing from the repo's <code>live_loading.py</code> was ported and yields the paper's 1047 rounds exactly.</li>
<li><strong>Statements.</strong> Initial winner = the statement participants critiqued; revised winner = the statement shown in the end-of-round survey; “statements” = all model candidates shown at that iteration.</li>
<li><strong>Embedding.</strong> The SM names Sentence-T5 (768-d) without a size; the public <code>sentence-transformers/sentence-t5-*</code> ports were used with a 512-token limit.</li>
<li><strong>Minority.</strong> Side of neutral with fewer pre-deliberation ratings; neutral raters are non-minority (SM 5.4.1). Tied rounds and unanimous rounds have no minority.</li>
<li><strong>Regression.</strong> Weights ≥ 0 summing to 1, fitted by constrained least squares; SEs are the OLS standard errors of the active-set fit, as the paper reports, plus a cluster bootstrap over rounds.</li>
</ul>
<p class="note">Code and notebook: <code>~/work/habermas-fig4c</code> on the habermas-replication VM (branch <code>habermas-replication/fig4c</code>).</p>
</main>
"""
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(page)
    print("wrote", out, len(page), "bytes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", action="append", required=True)
    ap.add_argument("--out", default="report/fig4c_report.html")
    a = ap.parse_args()
    build([load(Path(r)) for r in a.results], Path(a.out))
