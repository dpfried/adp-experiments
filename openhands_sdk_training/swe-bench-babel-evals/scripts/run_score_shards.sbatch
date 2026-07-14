#!/usr/bin/env bash
#SBATCH --job-name=swebench-score
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1   # preempt_qos requires >=1 GPU; unused by the build
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --output=/home/dfried/exp/adp-smoke/swebench/logs/%x-%A_%a.out
#SBATCH --error=/home/dfried/exp/adp-smoke/swebench/logs/%x-%A_%a.err

# Sharded Apptainer scoring of a completed inference run. CPU-only; no GPUs.
#   sbatch --array=0-7 run_score_shards.sbatch <OUTPUT_JSONL> <TAG> [TOTAL_SHARDS]
# Each array task scores lines where (line_index % array_count == task_id).
# TOTAL_SHARDS overrides the modulus for partial resubmits (e.g. rerun only
# failed shards 4-7 of an 8-way split: --array=4-7 ... 8) — without it,
# SLURM_ARRAY_TASK_COUNT of the partial array would mis-shard.
# ~4 min/instance sequential per shard; 500/8 ≈ 63 instances ≈ 4-5h per task.
# Merge results afterwards: python merge_shard_reports.py <TAG>

set -uo pipefail
OUTJSON=${1:?usage: sbatch --array=0-N run_score_shards.sbatch OUTPUT_JSONL TAG}
TAG=${2:?usage: sbatch --array=0-N run_score_shards.sbatch OUTPUT_JSONL TAG}
I=${SLURM_ARRAY_TASK_ID:?run with --array}
N=${3:-${SLURM_ARRAY_TASK_COUNT:?run with --array}}

SB=/home/dfried/exp/adp-smoke/swebench
echo "== node=$SLURMD_NODENAME shard=$I/$N tag=$TAG started=$(date -Is) =="

DATA=/data/tir/projects/tir1/users/dfried/exp/adp-smoke/openhands
mkdir -p "$DATA" || { echo "FATAL: tir1 bulk storage unavailable at $DATA (compute nodes only)"; exit 1; }
export APPTAINER_CACHEDIR=$DATA/apptainer-cache
export APPTAINER_TMPDIR=/scratch/dfried/apptainer-tmp-${SLURM_JOB_ID}_$I
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
export OPENHANDS_SWEBENCH_IMAGE_TEMPLATE="ghcr.io/epoch-research/swe-bench.eval.{arch}.{instance_id}:latest"
export HF_HOME=/home/dfried/exp/adp-smoke/hf_cache
unset HF_HUB_OFFLINE

# --- write this task's shard file (deterministic round-robin by line index) ---
# include output_errors.jsonl: PR #751 captures git patches from failed/stuck
# rows, and Graham's scoring counted those instead of silently dropping them
SHARD_DIR=$SB/full/score_$TAG
mkdir -p "$SHARD_DIR"
SHARD=$SHARD_DIR/shard_${I}of${N}.jsonl
ERRJSON=$(dirname "$OUTJSON")/output_errors.jsonl
cat "$OUTJSON" $([ -s "$ERRJSON" ] && echo "$ERRJSON") | awk -v i="$I" -v n="$N" '(NR-1) % n == i' > "$SHARD"
echo "== shard has $(wc -l < "$SHARD") instances =="
[ -s "$SHARD" ] || { echo "empty shard, nothing to do"; exit 0; }

cd "$SB/benchmarks"
.venv/bin/swebench-eval "$SHARD" \
  --apptainer --dataset princeton-nlp/SWE-bench_Verified --split test \
  --run-id "score_${TAG}_${I}of${N}" \
  --timeout 1800 \
  --apptainer-score-dir "$SHARD_DIR/reports_${I}of${N}" \
  --apptainer-sandbox-root "$DATA/swebench-sandboxes" \
  --apptainer-cache "$APPTAINER_CACHEDIR"
RC=$?
echo "== score exit=$RC finished=$(date -Is) =="

# sandboxes are one-shot for a full eval; prune this shard's to save ~3GB/instance
if [ $RC -eq 0 ]; then
  while read -r line; do
    iid=$(printf '%s' "$line" | "$SB/benchmarks/.venv/bin/python" -c 'import json,sys; print(json.load(sys.stdin)["instance_id"])' 2>/dev/null) || continue
    rm -rf "$DATA/swebench-sandboxes/$iid"
  done < "$SHARD"
  echo "== pruned shard sandboxes =="
fi
exit $RC
