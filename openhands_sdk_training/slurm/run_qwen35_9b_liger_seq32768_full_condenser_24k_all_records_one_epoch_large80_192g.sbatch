#!/usr/bin/env bash
#SBATCH --job-name=adp-qwen35-9b-32k-cond24k-fa2-a100
#SBATCH --partition=general
#SBATCH --nodelist=babel-v5-20,babel-v5-28,babel-v5-32
#SBATCH --gres=gpu:A100_80GB:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/gneubig/exp/adp/runs/openhands_sdk_training/full_condenser_24k_all_records_adapted/%x-%j.out
#SBATCH --error=/home/gneubig/exp/adp/runs/openhands_sdk_training/full_condenser_24k_all_records_adapted/%x-%j.err

CONFIG=/home/gneubig/exp/adp/runs/openhands_sdk_training/full_condenser_24k_all_records_adapted/qwen35_9b_full_condenser_24k_all_records_seq32768_zero3_one_epoch_a100.yaml
LOG_DIR=/home/gneubig/exp/adp/runs/openhands_sdk_training/full_condenser_24k_all_records_adapted/logs
GPU_LOG=${LOG_DIR}/gpu_monitor_qwen35_9b_liger_seq32768_cond24k_all_1ep_fa2_a100_192g_${SLURM_JOB_ID}.log
mkdir -p "$LOG_DIR"
cd /home/gneubig/work/adp/adp-experiments/openhands_sdk_training

echo "job_id=${SLURM_JOB_ID}"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "started_at=$(date -Is)"
echo "config=${CONFIG}"
echo "gpu_log=${GPU_LOG}"
nvidia-smi || true
nvidia-smi topo -m || true
(
  while true; do
    date -Is
    nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
    sleep 30
  done
) > "$GPU_LOG" 2>&1 &
MONITOR_PID=$!
cleanup() { kill "$MONITOR_PID" 2>/dev/null || true; }
trap cleanup EXIT

export PATH="$HOME/exp/adp-smoke/.venv/bin:$PATH"
export HF_HOME="$HOME/exp/adp-smoke/hf_cache"
export WANDB_ENTITY=gneubig
export WANDB_PROJECT=adp-experiments
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=4
export MASTER_PORT=$((30500 + SLURM_JOB_ID % 1000))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS=8
export TRITON_CACHE_DIR=/tmp/triton-cache-${SLURM_JOB_ID}
mkdir -p "$TRITON_CACHE_DIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
unset CC CXX

python - <<'PY2'
import flash_attn, torch
from transformers.utils import is_flash_attn_2_available
print('flash_attn', flash_attn.__version__)
print('is_flash_attn_2_available', is_flash_attn_2_available())
print('torch_cuda_arch_list', torch.cuda.get_arch_list() if torch.cuda.is_available() else 'cuda_unavailable')
PY2

llamafactory-cli train "$CONFIG"
STATUS=$?
echo "finished_at=$(date -Is)"
echo "exit_status=${STATUS}"
exit "$STATUS"
