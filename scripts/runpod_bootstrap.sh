#!/usr/bin/env bash
# Pod bootstrap: deps + lens, then run the chain script named in $1.
# Container-local installs only; the pod is disposable.
set -eu
cd /workspace/hw0
pip -q uninstall -y torchvision torchaudio 2>&1 | tail -1
pip -q install -U "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1
pip -q install "transformers>=5.5" huggingface_hub numpy datasets math-verify \
  sympy pandas scipy pyyaml accelerate 2>&1 | tail -1
pip -q install "jlens @ git+https://github.com/anthropics/jacobian-lens@581d398613e5602a5af361e1c34d3a92ea82ba8e" 2>&1 | tail -1
python scripts/fetch_lens.py --model "${HW0_LENS_ID:-qwen3-4b}"
echo "BOOTSTRAP_DONE $(date -u +%H:%M:%S)"
bash "$@"
