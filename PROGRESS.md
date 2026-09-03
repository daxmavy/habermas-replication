# Fig 4C replication — progress log (habermas-replication VM)

Goal: replicate Tessler et al. (2024) Science Fig. 4C — minority weight in HM group statements (initial vs revised),
via Sentence-T5 embeddings, position-axis projection and convex regression.

## Status
- [x] Data downloaded from GCS (4 parquet files, 450 MB) -> `data/`
- [x] SM PDF: NOT available (not on VM, not in Zotero; science.org blocks curl/headless Chromium; Oxford ssh has no key).
      Method reconstructed from main-paper Methods ("Embedding geometry") + Procaccia et al. footnote (arXiv 2603.16751).
- [x] Prepared tables -> `prepared/` (opinions, statements, questions, texts)
- [x] Minority definition calibrated: cohorts 1-3, neutral participants dropped, groups kept -> mean minority share 0.291 (paper: 29%)
- [ ] Embeddings -> `embeddings/st5-base`, `embeddings/st5-large` (driver: `run_embed_all.sh`, log `embeddings/embed.log`)
- [ ] Analysis notebook `notebooks/fig4c.ipynb`

## Key decisions (reversible)
- Initial statement = critiqued top candidate (iteration-1 rows); revised = candidate in end-of-round survey.
- Primary cohort = cohorts 1-3 pooled (matches 29%); sensitivity: per cohort, cohort 4, training, VCA.
- Embedding: sentence-transformers/sentence-t5-{base,large} (HF ports of ST5), max_seq_length 512.
- Convex regression pooled per division level (n,k); minority weight = sum of minority coefs; levels averaged by #rounds.
