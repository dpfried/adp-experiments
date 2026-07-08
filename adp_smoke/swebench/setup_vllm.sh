#!/usr/bin/env bash
set -euxo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /home/dfried/exp/adp-smoke/swebench
uv venv --python 3.12 .venv_vllm
uv pip install --python .venv_vllm/bin/python vllm 2>&1 | tail -3
.venv_vllm/bin/python -c "import vllm; print('vllm', vllm.__version__)"
echo VLLM_SETUP_DONE
