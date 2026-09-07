#!/bin/bash
#SBATCH --job-name=st5-embed
#SBATCH --output=isambard/logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
# Embed prepared/texts.parquet with Sentence-T5 models on GH200s, in fp32 like the base/large CPU runs.
# Models run one after another; each model's texts are split into one shard per allocated GPU, run as concurrent
# srun steps (one GPU each) writing to the same cache directory. Per-shard logs: isambard/logs/st5-embed_<job>_<model>_shard<i>.out
# Default is 1 GPU (starts on any node with a free GPU); `sbatch --gpus=4 --time=01:00:00 ...` uses a whole node, ~4x faster.
# Submit from ~/habermas-fig4c after isambard/setup_env.sh:
#   sbatch isambard/embed_st5.sbatch                                   # xl then xxl, all 51,764 texts
#   HM_MODELS=xl HM_EXTRA="--max-priority -2 --prereg-only" sbatch --time=00:15:00 isambard/embed_st5.sbatch   # smoke test: 1,394 texts
# The cache is resumable (one parquet per chunk), so resubmitting after a timeout continues where it stopped.
# Env: HM_MODELS (default "xl xxl"), HM_DTYPE (fp32 | bf16), HM_EXTRA (extra flags for hm_fig4c.embed).
set -eu
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export HF_HOME="$SCRATCHDIR/hf" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
OUT="$SCRATCHDIR/habermas-fig4c/embeddings"
N=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-1}}
echo "gpus: $N"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
for m in ${HM_MODELS:-xl xxl}; do
  echo "=== $(date) sentence-t5-$m -> $OUT/st5-$m, $N shards ==="
  pids=()
  for ((i = 0; i < N; i++)); do
    srun --ntasks=1 --gpus=1 --cpus-per-task=8 --exclusive --output="isambard/logs/%x_%j_${m}_shard$i.out" \
      python -m hm_fig4c.embed --model "sentence-transformers/sentence-t5-$m" --cache-dir "$OUT/st5-$m" \
        --device cuda --dtype "${HM_DTYPE:-fp32}" --batch-size 64 --chunk-size 2048 --shard "$i/$N" ${HM_EXTRA:-} &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done   # a failed shard fails the job (bare `wait` would not)
  echo "=== $(date) sentence-t5-$m done ==="
done
echo "ALL_EMBED_DONE $(date)"
