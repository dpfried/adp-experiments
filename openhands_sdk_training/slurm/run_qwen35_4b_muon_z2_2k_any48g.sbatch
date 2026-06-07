#!/usr/bin/env bash
#SBATCH --job-name=adp-muon-4b-2k-z2-48g
#SBATCH --partition=general
#SBATCH --constraint=VRAM_48GB
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=3:00:00
#SBATCH --output=/home/gneubig/exp/adp/runs/openhands_sdk_training/smoke_muon_qwen35_4b_z2/%x-%j.out
#SBATCH --error=/home/gneubig/exp/adp/runs/openhands_sdk_training/smoke_muon_qwen35_4b_z2/%x-%j.err

CONFIG=/home/gneubig/exp/adp/runs/openhands_sdk_training/smoke_muon_qwen35_4b_z2/smoke_qwen35_4b_muon_z2_seq2048.yaml
LOG_DIR=/home/gneubig/exp/adp/runs/openhands_sdk_training/smoke_muon_qwen35_4b_z2/logs
GPU_LOG=${LOG_DIR}/gpu_monitor_muon_4b_2k_z2_48g_${SLURM_JOB_ID}.log
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
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=4
export MASTER_PORT=$((33600 + SLURM_JOB_ID % 1000))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS=8
export TRITON_CACHE_DIR=/tmp/triton-cache-${SLURM_JOB_ID}
mkdir -p "$TRITON_CACHE_DIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
unset CC CXX

llamafactory-cli train "$CONFIG"
STATUS=$?
echo "finished_at=$(date -Is)"
echo "exit_status=${STATUS}"
exit "$STATUS"
