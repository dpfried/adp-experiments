#!/usr/bin/env bash
#SBATCH --job-name=nccl-p5-28-nvls
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:8
#SBATCH --nodelist=babel-p5-28
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/dfried/exp/adp-smoke/nccl8_test/logs/%x-%j.out
#SBATCH --error=/home/dfried/exp/adp-smoke/nccl8_test/logs/%x-%j.err
echo "node=$SLURMD_NODENAME start=$(date -Is)"
nvidia-smi topo -m 2>/dev/null | head -12
export PATH="/home/dfried/exp/adp-smoke/.venv/bin:$PATH"
T=/home/dfried/exp/adp-smoke/node_o5_28_repro/nccl_smoke.py
run() { echo "=== $1 (nproc=$2) ==="; shift 1; n=$1; shift 1; env "$@" timeout 300 torchrun --nproc_per_node $n --master_port $((29650+RANDOM%50)) "$T" >/dev/null 2>&1; echo "exit=$?"; }
run baseline8 8
run nvls_off8 8 NCCL_NVLS_ENABLE=0
echo "end=$(date -Is)"
