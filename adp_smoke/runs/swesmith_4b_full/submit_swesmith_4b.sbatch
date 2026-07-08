#!/usr/bin/env bash
#SBATCH --job-name=adp-swesmith-4b-32k
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-o5-28,babel-n5-32,babel-p5-28
#SBATCH --open-mode=append
#SBATCH --output=/home/dfried/exp/adp-smoke/runs/swesmith_4b_full/logs/%x-%j.out
#SBATCH --error=/home/dfried/exp/adp-smoke/runs/swesmith_4b_full/logs/%x-%j.err

set -uo pipefail
RUN=/home/dfried/exp/adp-smoke/runs/swesmith_4b_full
PRETOK_CFG=$RUN/pretokenize_swesmith_4b_seq32768.yaml
TRAIN_CFG=$RUN/swesmith_4b_seq32768_train.yaml
mkdir -p "$RUN/logs"

export PATH="/home/dfried/exp/adp-smoke/.venv/bin:$PATH"
export HF_HOME=/home/dfried/exp/adp-smoke/hf_cache
export HF_HUB_OFFLINE=1                 # 4B weights + tokenizer already cached; avoid network stalls
export WANDB_PROJECT=adp-smoke
export WANDB_DIR=$RUN
export TOKENIZERS_PARALLELISM=false     # prevents fast-tokenizer fork deadlock
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

echo "== node=$SLURMD_NODENAME job=$SLURM_JOB_ID started=$(date -Is) =="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

# Fail fast if the linear-attention fast path is unavailable (naive fallback = OOM + 28min steps)
python -c "import fla, causal_conv1d" || { echo "FATAL: flash-linear-attention/causal-conv1d missing"; exit 1; }

# --- checkpoint output dir: tir1 bulk storage (home holds scripts/logs/wandb only; never checkpoints) ---
OUT=/data/tir/projects/tir1/users/dfried/exp/adp-smoke/runs/swesmith_4b_full/output
mkdir -p "$OUT" || { echo "FATAL: tir1 bulk storage unavailable at $OUT (mounted on compute nodes only)"; exit 1; }
echo "== output_dir=$OUT =="

# --- Phase 1: single-RANK pre-tokenization (no model load, no cross-rank contention) ---
# torchrun -nproc 1 satisfies LLaMA-Factory's "must be distributed" check while staying
# single-process. Idempotent: if the cache exists, get_dataset() loads it and returns fast.
echo "== Phase 1: pre-tokenize (single rank) =="
torchrun --nnodes 1 --nproc_per_node 1 --master_port $((29500 + SLURM_JOB_ID % 400)) \
  /home/dfried/exp/adp-smoke/pretokenize.py "$PRETOK_CFG"
RC=$?
if [ $RC -ne 0 ]; then echo "pretokenize FAILED rc=$RC"; exit $RC; fi

# --- Phase 2: distributed training (4 GPUs); loads tokenized cache via load_from_disk ---
echo "== Phase 2: training on 4 GPUs =="
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=8
export NCCL_NVLS_ENABLE=0   # NCCL 2.27-2.28 NVLS partial-alloc fallback can hang (fixed in 2.29); no NVLS hw on L40S anyway
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=$RUN/logs/nccl_${SLURM_JOB_ID}_%h_r%p.log   # forensics if init hangs again
export TRITON_CACHE_DIR=/scratch/dfried/triton-cache-${SLURM_JOB_ID}   # FLA Triton kernels; off NFS
mkdir -p "$TRITON_CACHE_DIR"
export MASTER_PORT=$((30000 + SLURM_JOB_ID % 20000))

( while true; do date -Is; nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits; sleep 30; done ) \
  > "$RUN/logs/gpu_monitor_${SLURM_JOB_ID}.log" 2>&1 &
MON=$!
trap 'kill $MON 2>/dev/null || true' EXIT

llamafactory-cli train "$TRAIN_CFG" output_dir="$OUT"
RC=$?
echo "== finished=$(date -Is) exit=$RC =="
exit $RC
