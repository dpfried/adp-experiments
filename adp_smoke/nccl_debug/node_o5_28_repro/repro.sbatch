#!/usr/bin/env bash
#SBATCH --job-name=nccl-check-o5-28
#SBATCH --partition=general
#SBATCH --nodelist=babel-o5-28
#SBATCH --gres=gpu:L40S:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --open-mode=append
#SBATCH --output=/home/dfried/exp/adp-smoke/node_o5_28_repro/logs/%x-%j.out
#SBATCH --error=/home/dfried/exp/adp-smoke/node_o5_28_repro/logs/%x-%j.err
set -x
echo "script_start=$(date -Is) node=$SLURMD_NODENAME"   # vs sacct Start = prolog-delay evidence
export PATH="/home/dfried/exp/adp-smoke/.venv/bin:$PATH"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
timeout 400 torchrun --nproc_per_node 8 --master_port 29631 /home/dfried/exp/adp-smoke/node_o5_28_repro/nccl_smoke.py
echo "exit=$? end=$(date -Is)"
