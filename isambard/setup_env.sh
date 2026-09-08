#!/bin/bash
# One-time environment setup on an Isambard-AI login node (no GPU there; nothing heavy runs here).
# From ~/habermas-fig4c-report, after `isambard/sync.sh to` on the VM:
#   HM_MODELS="xl xxl" bash isambard/setup_env.sh
# Installs uv, builds .venv from uv.lock (managed CPython 3.12 for aarch64, torch from the cu126 index; the
# project is not a package -- hm_fig4c is imported from the repo root), and downloads the model weights into
# $SCRATCHDIR/hf so the job can run with HF_HUB_OFFLINE=1.
set -eu
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --frozen
.venv/bin/python -c "import platform, torch; print(platform.machine(), 'torch', torch.__version__, '| cuda build', torch.version.cuda)"
export HF_HOME="$SCRATCHDIR/hf"
mkdir -p "$HF_HOME" isambard/logs
.venv/bin/python - <<PY
from huggingface_hub import snapshot_download
for m in "${HM_MODELS:-xl xxl}".split():
    path = snapshot_download(f"sentence-transformers/sentence-t5-{m}",
                             ignore_patterns=["pytorch_model.bin", "*.h5", "*.msgpack", "*.ot", "onnx/*", "openvino/*"])
    print(m, "->", path, flush=True)
PY
echo "SETUP_DONE $(date)"
