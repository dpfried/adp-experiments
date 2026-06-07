#!/usr/bin/env bash
#SBATCH --job-name=adp-qwen35-4b-32k-fa2-l40s
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/gneubig/exp/adp/runs/openhands_sdk_training/full_condenser_24k_all_records_adapted/%x-%j.out
#SBATCH --error=/home/gneubig/exp/adp/runs/openhands_sdk_training/full_condenser_24k_all_records_adapted/%x-%j.err

CONFIG=/home/gneubig/exp/adp/runs/openhands_sdk_training/full_condenser_24k_all_records_adapted/qwen35_4b_full_condenser_24k_all_records_seq32768_zero3_one_epoch_flashattn.yaml
LOG_DIR=/home/gneubig/exp/adp/runs/openhands_sdk_training/full_condenser_24k_all_records_adapted/logs
GPU_LOG=${LOG_DIR}/gpu_monitor_qwen35_4b_liger_seq32768_cond24k_all_1ep_flashattn_l40s_${SLURM_JOB_ID}.log
mkdir -p "$LOG_DIR"
cd /home/gneubig/work/adp/adp-experiments/openhands_sdk_training

echo "job_id=${SLURM_JOB_ID}"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "started_at=$(date -Is)"
echo "config=${CONFIG}"
echo "gpu_log=${GPU_LOG}"
nvidia-smi || true
nvidia-smi topo -m || true

export PATH="$HOME/exp/adp-smoke/.venv/bin:$PATH"
export HF_HOME="$HOME/exp/adp-smoke/hf_cache"
export WANDB_ENTITY=gneubig
export WANDB_PROJECT=adp-experiments
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=4
export MASTER_PORT=$((32500 + SLURM_JOB_ID % 1000))
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

llamafactory-cli train "$CONFIG"
STATUS=$?
echo "finished_at=$(date -Is)"
echo "exit_status=${STATUS}"
exit "$STATUS"
