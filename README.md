# Habermas Machine, Figure 4C — replication report

This repository produces one thing: `report/fig4c_report.pdf`, a replication of Figure 4C of
Tessler et al. (2024), *"AI can help humans find common ground in democratic deliberation"*,
Science 386, adq2852.

## Requirements

- [uv](https://docs.astral.sh/uv/) — it installs Python 3.12 and the pinned dependencies from `uv.lock`.
- `pdflatex` with the packages the report loads (TeX Live's `texlive-latex-recommended` and
  `texlive-latex-extra` are enough).

## Reproduce

```bash
uv sync
uv run python reproduce.py all
```

`all` runs the whole chain from the public dataset to the PDF. The steps also run individually,
in this order:

```bash
uv run python reproduce.py data        # downloads the three dataset files into data/
uv run python reproduce.py prepare     # writes prepared/{opinions,statements,questions,candidates,texts}.parquet
uv run python reproduce.py embed       # writes embeddings/<model>/chunk_*.parquet, one directory per Sentence-T5 size
uv run python reproduce.py results     # executes notebooks/fig4c.ipynb once per model, writing results/<model>/{fig4c_primary.json,sensitivity.csv,summary.json}
uv run python reproduce.py values      # writes report/values.json
uv run python reproduce.py parameters  # writes report/parameters.tex
uv run python reproduce.py figures     # writes report/figures/*.pdf
uv run python reproduce.py codelinks   # writes report/codelinks.tex and report/codelinks.json
uv run python reproduce.py pdf         # writes report/fig4c_report.pdf
```

`report` is short for the last five steps. `notebook` regenerates `notebooks/fig4c.ipynb` from
`notebooks/make_notebook.py`; `results` does that for you when the generator is the newer of the two.
`--models st5-base st5-large` restricts `embed`, `results` and `values` to a subset of the four
Sentence-T5 sizes; the default is all four.

`embed` is the slow step. Its cache is resumable: it writes one parquet file per chunk and re-running
it embeds only the texts that are not in the cache yet, so an interrupted run continues where it
stopped.

## Data

`data/` holds three parquet files from the public Habermas Machine release accompanying the paper
(Google DeepMind's GCS bucket `habermas_machine`, prefix `datasets/`):

- `hm_all_candidate_comparisons.parquet`
- `hm_all_position_statement_ratings.parquet`
- `hm_all_round_survey_responses.parquet`

`scripts/download_data.py` fetches them and verifies each one's MD5 against the checksum of the copy
this report was built from; a file that is already present with the right checksum is left alone.
Neither `data/`, `prepared/` nor `embeddings/` is committed.

## No number is typed into the report

`report/compute_values.py` writes every quantity the report quotes to `report/values.json`;
`report/build_parameters.py` renders that as `\newcommand` macros in `report/parameters.tex`, which
`report/fig4c_report.tex` reads. The `.tex` contains no literal numbers, so a number can only change
by re-running the analysis. `uv run python reproduce.py check` fails if `parameters.tex` has drifted
from `values.json`. `report/codelinks.py` resolves the report's code references by symbol name and
pins them to the current commit.

## Repository layout

- `reproduce.py` — the steps above; run it from the repository root.
- `scripts/` — `download_data.py` (fetch the dataset) and `embed.py` (embed the texts the analysis needs).
- `hm_fig4c/` — the library: dataset loading, preprocessing, embedding, scoring and the analysis pipeline.
- `notebooks/` — `make_notebook.py`, which generates the analysis notebook `fig4c.ipynb`.
- `results/` — the committed per-model outputs of the notebook, the only results the report reads.
- `report/` — the LaTeX source, the scripts that generate its numbers, figures and code references, and the PDF.
- `pyproject.toml`, `uv.lock`, `.python-version` — the environment.
- `data/`, `prepared/`, `embeddings/` — inputs and intermediates, written by the steps above and not committed.
