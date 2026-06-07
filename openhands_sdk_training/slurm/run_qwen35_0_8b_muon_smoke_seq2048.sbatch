#!/usr/bin/env bash
#SBATCH --job-name=adp-muon-smoke-0p8b
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=/home/gneubig/exp/adp/runs/openhands_sdk_training/smoke_muon_qwen35_0_8b/%x-%j.out
#SBATCH --error=/home/gneubig/exp/adp/runs/openhands_sdk_training/smoke_muon_qwen35_0_8b/%x-%j.err

CONFIG=/home/gneubig/exp/adp/runs/openhands_sdk_training/smoke_muon_qwen35_0_8b/smoke_qwen35_0_8b_muon_seq2048.yaml
LOG_DIR=/home/gneubig/exp/adp/runs/openhands_sdk_training/smoke_muon_qwen35_0_8b/logs
GPU_LOG=${LOG_DIR}/gpu_monitor_muon_smoke_${SLURM_JOB_ID}.log
mkdir -p "$LOG_DIR"
cd /home/gneubig/work/adp/adp-experiments/openhands_sdk_training

echo "job_id=${SLURM_JOB_ID}"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "started_at=$(date -Is)"
echo "config=${CONFIG}"
echo "gpu_log=${GPU_LOG}"
nvidia-smi || true
(
  while true; do
    date -Is
    nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
    sleep 20
  done
) > "$GPU_LOG" 2>&1 &
MONITOR_PID=$!
cleanup() { kill "$MONITOR_PID" 2>/dev/null || true; }
trap cleanup EXIT

export PATH="$HOME/exp/adp-smoke/.venv/bin:$PATH"
export HF_HOME="$HOME/exp/adp-smoke/hf_cache"
unset FORCE_TORCHRUN NPROC_PER_NODE NNODES NODE_RANK MASTER_ADDR MASTER_PORT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton-cache-${SLURM_JOB_ID}
mkdir -p "$TRITON_CACHE_DIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
unset CC CXX

llamafactory-cli train "$CONFIG"
STATUS=$?
echo "finished_at=$(date -Is)"
echo "exit_status=${STATUS}"
exit "$STATUS"
