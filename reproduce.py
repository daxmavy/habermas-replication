"""Run the Fig. 4C replication, one step at a time, from the repo root.

    python reproduce.py all                       # data -> prepare -> embed -> results -> report
    python reproduce.py prepare results           # any list of steps, run in the order given
    python reproduce.py embed --models st5-base st5-large

Steps:
    data        download the public dataset into data/
    prepare     execute notebooks/prepare.ipynb: data/ -> prepared/*.parquet
    embed       prepared/texts.parquet -> embeddings/<model>/   (the slow step; the cache is resumable)
    results     execute notebooks/analysis.ipynb once per model -> results/<model>/
    report      execute notebooks/report.ipynb (parameters.tex + figures), then pdflatex twice

Notebooks are executed in place and committed with their outputs; `results` saves its executed copy
as results/<model>/analysis.ipynb. --models applies to embed and results.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = ROOT / "notebooks"
MODELS = ["st5-base", "st5-large", "st5-xl", "st5-xxl"]


def py(*args: str) -> None:
    """Run a script of this repo with this interpreter; fail fast if it does not exit 0."""
    cmd = [sys.executable, *args]
    print(f"$ {' '.join(cmd)}", flush=True)
    code = subprocess.call(cmd, cwd=ROOT)
    if code:
        raise SystemExit(f"failed ({code}): {' '.join(cmd)}")


def run_notebook(name: str, out: Path | None = None) -> None:
    """Execute notebooks/<name> with cwd=notebooks/ and write it back (in place unless `out` is given)."""
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    src = NOTEBOOKS / name
    print(f"$ jupyter execute notebooks/{name}", flush=True)
    nb = nbformat.read(src, as_version=4)
    ExecutePreprocessor(timeout=None).preprocess(nb, {"metadata": {"path": str(NOTEBOOKS)}})
    dest = out or src
    dest.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, dest)
    print(f"wrote {dest.relative_to(ROOT)}", flush=True)


def step_data(_: list[str]) -> None:
    py("scripts/download_data.py", "--data-dir", "data")


def step_prepare(_: list[str]) -> None:
    run_notebook("prepare.ipynb")


def step_embed(models: list[str]) -> None:
    py("scripts/embed.py", "--models", *models, "--texts", "prepared/texts.parquet", "--out-dir", "embeddings")


def step_results(models: list[str]) -> None:
    for model in models:
        print(f"=== {model} ===", flush=True)
        os.environ["HM_EMB_DIR"] = f"../embeddings/{model}"
        run_notebook("analysis.ipynb", ROOT / "results" / model / "analysis.ipynb")


def step_report(_: list[str]) -> None:
    run_notebook("report.ipynb")
    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "fig4c_report.tex"]
    for _pass in range(2):   # twice, so cross-references settle
        print(f"$ {' '.join(cmd)}", flush=True)
        code = subprocess.call(cmd, cwd=ROOT / "report", stdout=subprocess.DEVNULL)
        if code:
            raise SystemExit(f"failed ({code}): {' '.join(cmd)} -- see report/fig4c_report.log")
    print("wrote report/fig4c_report.pdf", flush=True)


STEPS = {"data": step_data, "prepare": step_prepare, "embed": step_embed,
         "results": step_results, "report": step_report}
ALL = ["data", "prepare", "embed", "results", "report"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="+", choices=[*STEPS, "all"], metavar="step",
                    help=f"one or more of: {', '.join([*STEPS, 'all'])}")
    ap.add_argument("--models", nargs="+", choices=MODELS, default=MODELS,
                    help="embedding models for embed and results (default: all four)")
    a = ap.parse_args()
    for step in (ALL if "all" in a.steps else a.steps):
        print(f"--- {step} ---", flush=True)
        STEPS[step](a.models)


if __name__ == "__main__":
    main()
