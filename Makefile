# Fig. 4C replication report -- the whole chain from the public dataset to report/fig4c_report.pdf.
#
#   make report     # values.json -> parameters.tex -> figures -> codelinks -> PDF  (from committed results/)
#   make all        # everything, including the dataset download, prepare, CPU embeddings and the notebooks
#
# st5-xl and st5-xxl embeddings need a GPU: see isambard/embed_st5.sbatch.

MODELS ?= st5-base st5-large st5-xl st5-xxl
CPU_MODELS ?= st5-base st5-large
PY := uv run python

.PHONY: all data prepare notebook embed results values parameters figures codelinks pdf report check clean

report: values parameters figures codelinks pdf

all: data prepare embed results report

# ------------------------------------------------------------------ inputs
data:
	bash scripts/download_data.sh

prepare:
	$(PY) -m hm_fig4c.data --data-dir data --out-dir prepared

embed:
	HM_MODELS="$(CPU_MODELS)" bash scripts/embed_cpu.sh

# ------------------------------------------------------------------ analysis
notebooks/fig4c.ipynb: notebooks/make_notebook.py
	$(PY) notebooks/make_notebook.py

notebook: notebooks/fig4c.ipynb

results: notebooks/fig4c.ipynb
	@for m in $(MODELS); do \
	  echo "=== $$m ==="; \
	  ( cd notebooks && HM_EMB_DIR=../embeddings/$$m \
	      uv run --project .. jupyter nbconvert --to notebook --execute fig4c.ipynb \
	        --output fig4c_$${m}_out.ipynb ) || exit 1; \
	done

# ------------------------------------------------------------------ report
values:
	$(PY) report/compute_values.py --models $(MODELS)

parameters:
	$(PY) report/build_parameters.py

figures:
	$(PY) report/figures.py

codelinks:
	$(PY) report/codelinks.py

pdf:
	cd report && pdflatex -interaction=nonstopmode -halt-on-error fig4c_report.tex >/dev/null \
	  && pdflatex -interaction=nonstopmode -halt-on-error fig4c_report.tex >/dev/null
	@echo "wrote report/fig4c_report.pdf"

check:
	$(PY) report/build_parameters.py --check

clean:
	rm -f report/fig4c_report.aux report/fig4c_report.log report/fig4c_report.out
