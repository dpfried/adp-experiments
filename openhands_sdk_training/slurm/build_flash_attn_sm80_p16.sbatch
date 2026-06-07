#!/usr/bin/env bash
#SBATCH --job-name=build-flash-attn-p16
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --time=12:00:00
#SBATCH --output=/home/gneubig/exp/adp/runs/openhands_sdk_training/full_condenser_24k_all_records_adapted/build-flash-attn-p16-%j.out
#SBATCH --error=/home/gneubig/exp/adp/runs/openhands_sdk_training/full_condenser_24k_all_records_adapted/build-flash-attn-p16-%j.err

set -x
hostname
date -Is
free -h || true
df -h /tmp /data/user_data/gneubig/tir4/tmp || true

export VENV=/home/gneubig/exp/adp-smoke/.venv
export WRAP=/data/user_data/gneubig/tir4/compiler-wrap
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$WRAP:$CUDA_HOME/bin:$VENV/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TMPDIR=/data/user_data/gneubig/tir4/tmp/flash-attn-slurm-${SLURM_JOB_ID}
export PIP_CACHE_DIR=/data/user_data/gneubig/tir4/pip-cache
export MAX_JOBS=16
export NVCC_THREADS=2
export FLASH_ATTN_CUDA_ARCHS=80
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export CC="$WRAP/gcc"
export CXX="$WRAP/g++"
export TORCH_CUDA_ARCH_LIST="8.0"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

python - <<'PY2'
import os, torch
print('python env ok')
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'abi', torch._C._GLIBCXX_USE_CXX11_ABI)
print('MAX_JOBS', os.getenv('MAX_JOBS'), 'NVCC_THREADS', os.getenv('NVCC_THREADS'), 'ARCHS', os.getenv('FLASH_ATTN_CUDA_ARCHS'))
PY2

pip uninstall -y flash-attn || true
pip install -vvv flash-attn --no-build-isolation --no-cache-dir --no-binary=:all:
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
  echo "pip install failed with $STATUS"
  date -Is
  exit "$STATUS"
fi

python - <<'PY2'
import flash_attn
from transformers.utils import is_flash_attn_2_available
print('flash_attn', flash_attn.__version__)
print('is_flash_attn_2_available', is_flash_attn_2_available())
PY2
STATUS=$?
date -Is
exit "$STATUS"
