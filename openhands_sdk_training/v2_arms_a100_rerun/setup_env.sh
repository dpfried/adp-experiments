#!/usr/bin/env bash
# Bootstrap the pinned adp-v2 training environment (see README.md §4, §8).
# Idempotent. Run on a node with GPUs visible (flash-attn build + sanity check)
# and internet access (or pre-stage wheels/repos and re-run).
#
#   export ADP_ENV_ROOT=/path/to/bulk/adp-env   # NOT a small home quota
#   bash setup_env.sh
#
# Optional overrides:
#   TORCH_INDEX_URL   (default cu128 index; match your driver, see nvidia-smi)
#   MAX_JOBS          (flash-attn build parallelism, default 8)
set -euxo pipefail

: "${ADP_ENV_ROOT:?set ADP_ENV_ROOT to the env install root (bulk storage)}"
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}
LLAMAFACTORY_COMMIT=a61cfa692a70fcced4ba32a846d1e2de95f2865e
HERE=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$ADP_ENV_ROOT"
cd "$ADP_ENV_ROOT"

# --- uv + venv ---------------------------------------------------------------
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d .venv ] || uv venv --python 3.12 .venv
UVP=.venv/bin/python

# --- torch (pinned, CUDA flavor must match driver) ---------------------------
uv pip install --python "$UVP" --index-url "$TORCH_INDEX_URL" \
  "torch==2.11.0" torchvision torchaudio

# --- LLaMA-Factory at the validated commit ------------------------------------
if [ ! -d LLaMA-Factory ]; then
  git clone https://github.com/hiyouga/LLaMA-Factory.git
fi
git -C LLaMA-Factory fetch --depth 1 origin "$LLAMAFACTORY_COMMIT" || true
git -C LLaMA-Factory checkout "$LLAMAFACTORY_COMMIT"
uv pip install --python "$UVP" -e "LLaMA-Factory[torch,metrics]"

# --- pinned core stack (order matters: after LF so pins win) ------------------
uv pip install --python "$UVP" \
  "transformers==5.6.0" "deepspeed==0.19.2" "accelerate==1.11.0" \
  "datasets==4.0.0" "liger-kernel==0.8.0" "wandb==0.28.0" \
  "huggingface-hub>=1.0" "pyyaml>=6.0" ninja packaging

# --- Qwen3.5 linear-attention deps (MANDATORY, see CLAUDE.md) ------------------
uv pip install --python "$UVP" \
  "flash-linear-attention==0.5.1" "causal-conv1d==1.6.2.post1"

# --- FlashAttention-2 (source build for sm80; ~20-40 min) ---------------------
"$UVP" -c "import flash_attn" 2>/dev/null || \
  MAX_JOBS=${MAX_JOBS:-8} uv pip install --python "$UVP" --no-build-isolation \
    "flash-attn==2.8.3.post1"

# --- transformers FA2 s_aux None-guard patch (idempotent) ---------------------
"$UVP" "$HERE/patch_transformers_fa2_s_aux.py"

# --- prefetch model weights (training runs with HF_HUB_OFFLINE=1) --------------
export HF_HOME="$ADP_ENV_ROOT/hf_cache"
"$UVP" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3.5-4B")
print("MODEL_CACHED_OK")
PY

# --- sanity -------------------------------------------------------------------
"$UVP" - <<'PY'
import torch, transformers, importlib.metadata as m
import fla, causal_conv1d, flash_attn  # hard requirement, ImportError = broken env
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(),
      "ndev:", torch.cuda.device_count())
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print("dev0:", torch.cuda.get_device_name(0), "sm:", cap)
    assert cap >= (8, 0), "flash-attn built for sm80+; older GPU detected"
for p in ("transformers", "deepspeed", "liger-kernel", "flash-attn",
          "flash-linear-attention", "causal-conv1d", "llamafactory"):
    print(p, m.version(p))
PY
echo "SETUP_ENV_DONE_OK root=$ADP_ENV_ROOT"
