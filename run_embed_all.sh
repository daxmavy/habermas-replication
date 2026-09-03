#!/bin/bash
# Sequential embedding driver (2 CPU cores -> run one model at a time). Cohorts 1-3 first for both models, then the rest.
cd /home/exedev/work/habermas-fig4c
for stage in "3 st5-base sentence-transformers/sentence-t5-base" "3 st5-large sentence-transformers/sentence-t5-large" "99 st5-base sentence-transformers/sentence-t5-base" "99 st5-large sentence-transformers/sentence-t5-large"; do
  set -- $stage
  echo "=== $(date) stage: max_priority=$1 model=$3 ==="
  uv run python -m hm_fig4c.embed --model "$3" --cache-dir "embeddings/$2" --max-priority "$1" --batch-size 16 --chunk-size 256 2>&1 | grep --line-buffered -v -i "warning\|Loading weights"
done
echo "ALL_EMBED_DONE $(date)"
