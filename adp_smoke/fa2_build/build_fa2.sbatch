#!/usr/bin/env bash
#SBATCH --job-name=fa2-build-sm89
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=11:00:00
#SBATCH --output=/home/dfried/exp/adp-smoke/fa2_build/logs/%x-%j.out
#SBATCH --error=/home/dfried/exp/adp-smoke/fa2_build/logs/%x-%j.err

set -euxo pipefail
B=/home/dfried/exp/adp-smoke/fa2_build
mkdir -p "$B/logs" "$B/dist"

source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda-12.9
export CUDA_HOME=/usr/local/cuda-12.9
nvcc --version

VENV_PY=/home/dfried/exp/adp-smoke/.venv/bin/python
# Build for L40S/L40 (sm89) and A100 (sm80, future 9B runs)
export TORCH_CUDA_ARCH_LIST="8.9"
export FLASH_ATTN_CUDA_ARCHS="89"
export MAX_JOBS=10
export NVCC_THREADS=2

cd "$B"
if [ ! -d flash-attention ]; then
  git clone --depth 1 --branch v2.8.3.post1 https://github.com/Dao-AILab/flash-attention.git
fi
cd flash-attention

# build wheel only; do NOT install into the live venv
"$VENV_PY" -m pip wheel --no-build-isolation --no-deps -w "$B/dist" .

ls -lh "$B/dist"
echo "FA2_BUILD_DONE"
