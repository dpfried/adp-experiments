#!/usr/bin/env bash
#SBATCH --job-name=val-loss-follower
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --exclude=babel-o5-28,babel-n5-32,babel-p5-28
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --open-mode=append
#SBATCH --output=/home/dfried/exp/adp-smoke/logs/%x-%j.out
#SBATCH --error=/home/dfried/exp/adp-smoke/logs/%x-%j.err

# Follows training runs: computes offline validation loss for each new checkpoint
# before rolling save_total_limit deletes it. One (output_dir, tokenized, csv) target per line.
set -uo pipefail
export HF_HOME=/home/dfried/exp/adp-smoke/hf_cache
export HF_HUB_OFFLINE=1
export WANDB_PROJECT=adp-smoke
export WANDB_DIR=/home/dfried/exp/adp-smoke/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/dfried/exp/adp-smoke/.venv/bin/python
EV=/home/dfried/exp/adp-smoke/eval_checkpoint_loss.py

# trigger autofs for /data targets (tir1 bulk storage, compute-node-only mount)
ls /data/tir/projects/tir1/users/dfried >/dev/null 2>&1 || true

TARGETS=(
  "/data/tir/projects/tir1/users/dfried/exp/adp-smoke/runs/swesmith_4b_full/output|/home/dfried/exp/adp-smoke/datasets/swesmith_full/tokenized_qwen35_4b_seq32768|/home/dfried/exp/adp-smoke/runs/swesmith_4b_full/val_loss.csv|val-swesmith-4b"
  "/data/tir/projects/tir1/users/dfried/exp/adp-smoke/runs/paper_nonweb_4b/output|/home/dfried/exp/adp-smoke/datasets/paper_nonweb/tokenized_qwen35_4b_seq32768|/home/dfried/exp/adp-smoke/runs/paper_nonweb_4b/val_loss.csv|val-paper-nonweb-4b"
)

echo "follower started $(date -Is) on $SLURMD_NODENAME"
while true; do
  for t in "${TARGETS[@]}"; do
    IFS='|' read -r outdir tok csv wrun <<< "$t"
    [ -d "$outdir" ] || continue
    for ck in $(ls -d "$outdir"/checkpoint-* 2>/dev/null | sort -V); do
      step=$(basename "$ck" | sed 's/checkpoint-//')
      # skip if already in csv
      grep -q ",\"*${step}\"*," "$csv" 2>/dev/null && continue
      grep -q ",${step}," "$csv" 2>/dev/null && continue
      # skip checkpoints still being written (newest file modified <120s ago)
      newest=$(find "$ck" -maxdepth 1 -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1)
      [ -n "$newest" ] && [ $(( $(date +%s) - newest )) -lt 120 ] && continue
      # must have HF weights
      ls "$ck"/model*.safetensors* >/dev/null 2>&1 || continue
      echo "$(date -Is) evaluating $ck"
      "$PY" "$EV" --checkpoint "$ck" --tokenized "$tok" --out "$csv" --wandb-run "$wrun" 2>&1 | grep -E "^RESULT|^Traceback" || echo "eval failed for $ck"
    done
  done
  sleep 300
done
