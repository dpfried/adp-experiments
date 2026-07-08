#!/usr/bin/env bash
#SBATCH --job-name=adp-paper-nonweb-4b
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-o5-28,babel-n5-32,babel-p5-28
#SBATCH --open-mode=append
#SBATCH --output=/home/dfried/exp/adp-smoke/runs/paper_nonweb_4b/logs/%x-%j.out
#SBATCH --error=/home/dfried/exp/adp-smoke/runs/paper_nonweb_4b/logs/%x-%j.err

set -uo pipefail
RUN=/home/dfried/exp/adp-smoke/runs/paper_nonweb_4b
TRAIN_CFG=$RUN/paper_nonweb_4b_seq32768_4gpu_fa2.yaml
mkdir -p "$RUN/logs"

export PATH="/home/dfried/exp/adp-smoke/.venv/bin:$PATH"
export HF_HOME=/home/dfried/exp/adp-smoke/hf_cache
export HF_HUB_OFFLINE=1
export WANDB_PROJECT=adp-smoke
export WANDB_DIR=$RUN
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

echo "== node=$SLURMD_NODENAME job=$SLURM_JOB_ID started=$(date -Is) =="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

# fail fast if any fast path is missing; re-apply transformers FA2 s_aux patch (idempotent)
python -c "import fla, causal_conv1d, flash_attn" || { echo "FATAL: fla/causal_conv1d/flash_attn missing"; exit 1; }
python /home/dfried/exp/adp-smoke/patch_transformers_fa2_s_aux.py || { echo "FATAL: s_aux patch failed"; exit 1; }
# in-training eval at 32k needs Liger skip_logits on loss-only eval steps (idempotent, marker-guarded)
python /home/dfried/projects/adp-experiments/openhands_sdk_training/scripts/patch_llamafactory_liger_eval_skip_logits.py \
  || { echo "FATAL: liger eval skip_logits patch failed"; exit 1; }

# checkpoint dir: tir1 bulk storage (home holds scripts/logs/wandb only; never checkpoints)
OUT=/data/tir/projects/tir1/users/dfried/exp/adp-smoke/runs/paper_nonweb_4b/output
mkdir -p "$OUT" || { echo "FATAL: tir1 bulk storage unavailable at $OUT (mounted on compute nodes only)"; exit 1; }
echo "== output_dir=$OUT =="

# tokenized cache must exist (built offline; keeps tokenization out of torchrun)
python - <<EOF || { echo "FATAL: tokenized cache missing/incomplete"; exit 1; }
from llamafactory.data.loader import has_tokenized_data
assert has_tokenized_data("/home/dfried/exp/adp-smoke/datasets/paper_nonweb/tokenized_qwen35_4b_seq32768")
EOF

# auto-resume: if a full checkpoint exists in OUT, continue from it (job 2 of the chain, or requeue)
RESUME_ARG=""
if ls "$OUT"/checkpoint-*/trainer_state.json >/dev/null 2>&1; then
  RESUME_ARG="resume_from_checkpoint=true"
  echo "== resuming from latest checkpoint in $OUT =="
fi

export FORCE_TORCHRUN=1
export NPROC_PER_NODE=4
export NCCL_NVLS_ENABLE=0   # NCCL 2.28 NVLS partial-alloc fallback hang workaround
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=$RUN/logs/nccl_${SLURM_JOB_ID}_%h_r%p.log
export TRITON_CACHE_DIR=/scratch/dfried/triton-cache-${SLURM_JOB_ID}
mkdir -p "$TRITON_CACHE_DIR"
export MASTER_PORT=$((30000 + SLURM_JOB_ID % 20000))

( while true; do date -Is; nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits; sleep 30; done ) \
  > "$RUN/logs/gpu_monitor_${SLURM_JOB_ID}.log" 2>&1 &
MON=$!
trap 'kill $MON 2>/dev/null || true' EXIT

llamafactory-cli train "$TRAIN_CFG" output_dir="$OUT" $RESUME_ARG
RC=$?
echo "== finished=$(date -Is) exit=$RC =="
exit $RC
