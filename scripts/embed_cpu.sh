#!/bin/bash
# Embed the texts the report needs with the two Sentence-T5 sizes that fit on a CPU.
# The report only uses the pre-registered rounds of cohorts 1-3 plus the position-statement endpoints,
# which is what `--prereg-only --max-priority 3` selects (see hm_fig4c/data.build_text_table).
# The cache is one parquet per chunk and is resumable: re-running skips what is already embedded.
# sentence-t5-xl and -xxl need a GPU; see isambard/embed_st5.sbatch.
#
#   bash scripts/embed_cpu.sh                       # st5-base and st5-large
#   HM_MODELS="st5-large" bash scripts/embed_cpu.sh # one model
set -eu
cd "$(dirname "$0")/.."
MODELS=${HM_MODELS:-"st5-base st5-large"}
MAX_PRIORITY=${HM_MAX_PRIORITY:-3}
for tag in $MODELS; do
  echo "=== $(date) $tag (max_priority=$MAX_PRIORITY, pre-registered rounds only) ==="
  uv run python -m hm_fig4c.embed \
    --model "sentence-transformers/sentence-t5-${tag#st5-}" \
    --cache-dir "embeddings/$tag" \
    --max-priority "$MAX_PRIORITY" --prereg-only \
    --batch-size 16 --chunk-size 256
done
echo "ALL_EMBED_DONE $(date)"
