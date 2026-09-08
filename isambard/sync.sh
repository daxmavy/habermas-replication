#!/bin/bash
# Move code to Isambard-AI, or finished embedding caches back. Run on the VM from anywhere in the repo.
#   isambard/sync.sh to                 # hm_fig4c/, prepared/texts.parquet, lockfile, isambard/ -> ~/habermas-fig4c-report on the login node
#   isambard/sync.sh from [xl xxl ...]  # $SCRATCHDIR/habermas-fig4c-report/embeddings/st5-<m> -> embeddings/st5-<m>, plus the job logs
set -eu
cd "$(dirname "$0")/.."
REMOTE=u6oz.aip2.isambard
REMOTE_SCRATCH=/scratch/u6oz/daxmavy.u6oz
case "${1:-}" in
  to)
    rsync -avR --exclude __pycache__ hm_fig4c prepared/texts.parquet pyproject.toml uv.lock .python-version isambard "$REMOTE:habermas-fig4c-report/" ;;
  from)
    shift; models=("$@"); [ ${#models[@]} -eq 0 ] && models=(xl xxl)
    for m in "${models[@]}"; do
      mkdir -p "embeddings/st5-$m"
      rsync -av "$REMOTE:$REMOTE_SCRATCH/habermas-fig4c-report/embeddings/st5-$m/" "embeddings/st5-$m/"
    done
    rsync -av "$REMOTE:habermas-fig4c-report/isambard/logs/" isambard/logs/ ;;
  *) echo "usage: $0 to | from [model ...]" >&2; exit 2 ;;
esac
