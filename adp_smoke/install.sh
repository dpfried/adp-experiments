#!/usr/bin/env bash
set -euxo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /home/dfried/exp/adp-smoke

# CUDA torch (default index has CUDA wheels), plus deps
uv venv --python 3.12 .venv
UVP=".venv/bin/python"

uv pip install --python "$UVP" \
  "huggingface-hub>=1.0" "datasets>=4.0" "wandb>=0.27" "pyyaml>=6.0"

# CUDA build of torch (NOT the ROCm one from the repo's setup script)
uv pip install --python "$UVP" torch torchvision torchaudio

# LLaMA-Factory from GitHub main (contains Qwen3.5 templates)
mkdir -p .cache
if [ ! -d .cache/LLaMA-Factory ]; then
  git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git .cache/LLaMA-Factory
fi
uv pip install --python "$UVP" -e ".cache/LLaMA-Factory[torch,metrics]"

# sanity
"$UVP" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("dev0:", torch.cuda.get_device_name(0))
PY
echo "INSTALL_DONE_OK"
