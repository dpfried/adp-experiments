#!/usr/bin/env bash
set -uxo pipefail
RUN=/home/dfried/exp/adp-smoke/runs/swesmith_4b_probe
CONFIG=$RUN/swesmith_4b_probe_2xl40.yaml
mkdir -p "$RUN/logs"
export PATH="/home/dfried/exp/adp-smoke/.venv/bin:$PATH"
export HF_HOME=/home/dfried/exp/adp-smoke/hf_cache
export WANDB_PROJECT=adp-smoke
export WANDB_DIR=$RUN
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=2
export MASTER_PORT=$((33600 + RANDOM % 1000))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
GPU_LOG=$RUN/logs/gpu_monitor.log
( while true; do date -Is; nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits; sleep 15; done ) > "$GPU_LOG" 2>&1 &
MON=$!
trap 'kill $MON 2>/dev/null || true' EXIT
echo "started_at=$(date -Is) master_port=$MASTER_PORT"
llamafactory-cli train "$CONFIG"
echo "exit_status=$?  finished_at=$(date -Is)"
