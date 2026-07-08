#!/usr/bin/env bash
#SBATCH --job-name=nccl8-cumem-ab
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:8
#SBATCH --exclude=babel-o5-28
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/home/dfried/exp/adp-smoke/nccl8_test/logs/%x-%j.out
#SBATCH --error=/home/dfried/exp/adp-smoke/nccl8_test/logs/%x-%j.err
set -x
echo "node=$SLURMD_NODENAME start=$(date -Is)"
df -h /dev/shm
export PATH="/home/dfried/exp/adp-smoke/.venv/bin:$PATH"
T=/home/dfried/exp/adp-smoke/node_o5_28_repro/nccl_smoke.py   # 8-rank init+barrier+allreduce, 120s timeout
echo "=== A: default (cuMem on) ==="
timeout 240 torchrun --nproc_per_node 8 --master_port 29644 "$T"; echo "A_exit=$?"
echo "=== B: NCCL_CUMEM_ENABLE=0 ==="
NCCL_CUMEM_ENABLE=0 timeout 240 torchrun --nproc_per_node 8 --master_port 29645 "$T"; echo "B_exit=$?"
echo "end=$(date -Is)"
