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

`all` runs the whole chain from the public dataset to the PDF. The five steps also run
individually, in this order:

```bash
uv run python reproduce.py data      # downloads the three dataset files into data/
uv run python reproduce.py prepare   # notebooks/prepare.ipynb: data/ -> prepared/*.parquet
uv run python reproduce.py embed     # embeddings/<model>/, one directory per Sentence-T5 size
uv run python reproduce.py results   # notebooks/analysis.ipynb once per model -> results/<model>/
uv run python reproduce.py report    # notebooks/report.ipynb, then pdflatex -> report/fig4c_report.pdf
```

`--models st5-base st5-large` restricts `embed` and `results` to a subset of the four Sentence-T5
sizes; the default is all four.

The three notebooks are the analysis. They are committed with their outputs, so every table, plot
and figure can be read on GitHub without running anything: `notebooks/prepare.ipynb` shows the raw
dataset and the tables it builds from it, `notebooks/analysis.ipynb` computes the minority weights
for one embedding model and is committed showing the sentence-t5-large run (`results/` keeps the
executed copy of every run), and
`notebooks/report.ipynb` collects the four runs, draws the three figures into `report/figures/` and
writes `report/parameters.tex`, after which `report` builds the PDF.

`embed` is the slow step. Its cache is resumable: it writes one parquet file per chunk and
re-running it embeds only the texts that are not in the cache yet, so an interrupted run continues
where it stopped. All embeddings are computed in 32-bit floating point, so the results do not
depend on which machine ran this step.

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

`notebooks/report.ipynb` writes every quantity the report quotes as a named macro into
`report/parameters.tex`. `report/fig4c_report.tex` contains no literal result, so a number can only
change by re-running the analysis.

## Repository layout

- `reproduce.py` — the five steps above; run it from the repository root.
- `scripts/` — `download_data.py` (fetch the dataset) and `embed.py` (embed the texts the analysis needs).
- `hm_fig4c/` — the library: `data.py`, `preprocess.py`, `embed.py`, `analysis.py`, `pipeline.py`.
- `notebooks/` — `prepare.ipynb`, `analysis.ipynb` and `report.ipynb`, committed with their outputs.
- `results/` — the committed per-model outputs of the analysis notebook and the executed notebook itself.
- `report/` — `fig4c_report.tex`, the generated `parameters.tex` and `figures/`, and the PDF.
- `pyproject.toml`, `uv.lock`, `.python-version` — the environment.
- `data/`, `prepared/`, `embeddings/` — inputs and intermediates, written by the steps above and not committed.
