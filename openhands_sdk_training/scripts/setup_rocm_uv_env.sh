#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
LLAMA_FACTORY_REF="${LLAMA_FACTORY_REF:-main}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it first: https://docs.astral.sh/uv/"
  exit 1
fi

uv venv --python "$PYTHON_VERSION" .venv

UV_PYTHON=".venv/bin/python"

uv pip install --python "$UV_PYTHON" \
  "huggingface-hub>=1.0" \
  "datasets>=4.0" \
  "wandb>=0.27" \
  "pyyaml>=6.0"

# The verified local run used:
# torch==2.10.0+rocm7.13.0
# torchvision==0.25.0+rocm7.13.0
# torchaudio==2.10.0+rocm7.13.0
# rocm-sdk-libraries-gfx1151==7.13.0
#
# ROCm wheel availability changes quickly. If this install command stops
# resolving, install the matching ROCm PyTorch stack for your machine and keep
# the LLaMA-Factory commands below unchanged.
uv pip install --python "$UV_PYTHON" --pre \
  "torch" "torchvision" "torchaudio"

mkdir -p .cache
if [ ! -d .cache/LLaMA-Factory ]; then
  git clone https://github.com/hiyouga/LLaMA-Factory.git .cache/LLaMA-Factory
fi

git -C .cache/LLaMA-Factory fetch --all --tags
git -C .cache/LLaMA-Factory checkout "$LLAMA_FACTORY_REF"
uv pip install --python "$UV_PYTHON" -e ".cache/LLaMA-Factory[torch,metrics]"

"$UV_PYTHON" - <<'PY'
import torch

print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("cuda_is_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("device_gib:", torch.cuda.get_device_properties(0).total_memory / 1024**3)
PY

