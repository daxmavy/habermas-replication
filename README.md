# Habermas Machine, Figure 4C — replication report

This repository produces one thing: `report/fig4c_report.pdf`, a replication of Figure 4C of
Tessler et al. (2024), *"AI can help humans find common ground in democratic deliberation"*,
Science 386, adq2852 — the claim that the Habermas Machine does not under-represent minority
opinions in the group statements it writes.

## Reproduce

```bash
uv sync
make report        # values.json -> parameters.tex -> figures -> codelinks -> PDF
```

`make report` rebuilds the PDF from the committed `results/<model>/{fig4c_primary.json,
sensitivity.csv, summary.json}`, the prepared tables and the embedding caches of all four
Sentence-T5 sizes (`report/compute_values.py` refits the position-axis regression on each). So the
data, prepare and embed steps below must have run once, including the GPU sizes. To rebuild
everything from the public dataset instead:

```bash
make all           # data -> prepare -> embed (CPU models) -> results -> report
```

`make check` verifies that `report/parameters.tex` is exactly what `report/values.json` implies.

## Data

The three parquet files in `data/` are the public Habermas Machine release from Google DeepMind
(GCS bucket `habermas_machine`, prefix `datasets/`). `make data` downloads them and checks their
MD5s. They are not committed (451 MB). `make prepare` turns them into `prepared/*.parquet`, and
`make embed` writes Sentence-T5 embeddings of the texts the analysis needs into
`embeddings/<model>/`; neither directory is committed.

`sentence-t5-base` and `sentence-t5-large` embed on CPU in one to two hours each on two cores
(`scripts/embed_cpu.sh`; 16,741 texts per model). `sentence-t5-xl` and `sentence-t5-xxl` need a GPU: `isambard/` holds the
Slurm job (`embed_st5.sbatch`), the environment setup and the sync script used to run them on
Isambard-AI GH200 nodes.

## No number is typed into the report

`report/compute_values.py` writes every quantity the report quotes to `report/values.json`;
`report/build_parameters.py` renders that as `\newcommand` macros in `report/parameters.tex`, which
`report/fig4c_report.tex` reads. The `.tex` contains no literal results, so a number can only change
by re-running the analysis. `make check` fails if `parameters.tex` has drifted from `values.json`.
`report/codelinks.py` resolves the report's code references by symbol name and pins them to the
current commit.
