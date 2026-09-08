"""Run the Fig. 4C replication, one step at a time, from the repo root.

    python reproduce.py all                  # data -> prepare -> embed -> results -> report
    python reproduce.py report               # values -> parameters -> figures -> codelinks -> pdf
    python reproduce.py prepare results      # any list of steps, run in the order given
    python reproduce.py check                # parameters.tex still matches values.json?
    python reproduce.py embed --models st5-base st5-large

Steps:
    data        download the public dataset into data/
    prepare     data/ -> prepared/*.parquet
    notebook    regenerate notebooks/fig4c.ipynb from notebooks/make_notebook.py
    embed       prepared/texts.parquet -> embeddings/<model>/   (the slow step; the cache is resumable)
    results     execute the notebook once per model -> results/<model>/
    values      results/ + prepared/ + embeddings/ -> report/values.json
    parameters  report/values.json -> report/parameters.tex
    figures     report/values.json -> report/figures/*.pdf
    codelinks   source line numbers + commit -> report/codelinks.tex, report/codelinks.json
    pdf         report/fig4c_report.tex -> report/fig4c_report.pdf

`results` regenerates the notebook first if notebooks/make_notebook.py is newer than it.
--models applies to embed, results and values.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ["st5-base", "st5-large", "st5-xl", "st5-xxl"]
NOTEBOOK = ROOT / "notebooks" / "fig4c.ipynb"
MAKE_NOTEBOOK = ROOT / "notebooks" / "make_notebook.py"


def py(*args: str, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    """Run a script of this repo with this interpreter; fail fast if it does not exit 0."""
    cmd = [sys.executable, *args]
    print(f"$ {' '.join(cmd)}", flush=True)
    code = subprocess.call(cmd, cwd=cwd, env=env)
    if code:
        raise SystemExit(f"failed ({code}): {' '.join(cmd)}")


def step_data(_: list[str]) -> None:
    py("scripts/download_data.py", "--data-dir", "data")


def step_prepare(_: list[str]) -> None:
    py("-m", "hm_fig4c.data", "--data-dir", "data", "--out-dir", "prepared")


def step_notebook(_: list[str]) -> None:
    py("notebooks/make_notebook.py")


def step_embed(models: list[str]) -> None:
    py("scripts/embed.py", "--models", *models, "--texts", "prepared/texts.parquet", "--out-dir", "embeddings")


def step_results(models: list[str]) -> None:
    if not NOTEBOOK.exists() or MAKE_NOTEBOOK.stat().st_mtime > NOTEBOOK.stat().st_mtime:
        step_notebook(models)
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    work = ROOT / "notebooks"
    for model in models:
        print(f"=== {model} ===", flush=True)
        nb = nbformat.read(NOTEBOOK, as_version=4)
        os.environ["HM_EMB_DIR"] = f"../embeddings/{model}"
        ExecutePreprocessor(timeout=None).preprocess(nb, {"metadata": {"path": str(work)}})
        nbformat.write(nb, work / f"fig4c_{model}_out.ipynb")


def step_values(models: list[str]) -> None:
    py("report/compute_values.py", "--models", *models)


def step_parameters(_: list[str]) -> None:
    py("report/build_parameters.py")


def step_figures(_: list[str]) -> None:
    py("report/figures.py")


def step_codelinks(_: list[str]) -> None:
    py("report/codelinks.py")


def step_pdf(_: list[str]) -> None:
    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "fig4c_report.tex"]
    for _pass in range(2):   # twice, so \ref and the table of contents settle
        print(f"$ {' '.join(cmd)}", flush=True)
        code = subprocess.call(cmd, cwd=ROOT / "report", stdout=subprocess.DEVNULL)
        if code:
            raise SystemExit(f"failed ({code}): {' '.join(cmd)} -- see report/fig4c_report.log")
    print("wrote report/fig4c_report.pdf", flush=True)


def step_check(_: list[str]) -> None:
    py("report/build_parameters.py", "--check")


STEPS = {"data": step_data, "prepare": step_prepare, "notebook": step_notebook, "embed": step_embed,
         "results": step_results, "values": step_values, "parameters": step_parameters,
         "figures": step_figures, "codelinks": step_codelinks, "pdf": step_pdf, "check": step_check}
GROUPS = {"report": ["values", "parameters", "figures", "codelinks", "pdf"],
          "all": ["data", "prepare", "embed", "results", "report"]}


def expand(steps: list[str]) -> list[str]:
    return [s for step in steps for s in (expand(GROUPS[step]) if step in GROUPS else [step])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="+", choices=[*STEPS, *GROUPS], metavar="step",
                    help=f"one or more of: {', '.join([*STEPS, *GROUPS])}")
    ap.add_argument("--models", nargs="+", choices=MODELS, default=MODELS,
                    help="embedding models for embed, results and values (default: all four)")
    a = ap.parse_args()
    for step in expand(a.steps):
        print(f"--- {step} ---", flush=True)
        STEPS[step](a.models)


if __name__ == "__main__":
    main()
