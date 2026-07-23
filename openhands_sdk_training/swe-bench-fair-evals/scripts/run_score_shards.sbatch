#!/usr/bin/env bash
#SBATCH --job-name=swebench-score
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --output=/checkpoint/dpf/swebench-eval/logs/%x-%A_%a.out
#SBATCH --error=/checkpoint/dpf/swebench-eval/logs/%x-%A_%a.err

# FAIR port of ../../swe-bench-babel-evals/scripts/run_score_shards.sbatch.
# Sharded Apptainer scoring of a completed inference run. CPU-only → scavenge.
#
#   sbatch --array=0-3 run_score_shards.sbatch <OUTPUT_JSONL> <TAG> [TOTAL_SHARDS]
# Each array task scores lines where (line_index % TOTAL == task_id).
# TOTAL_SHARDS overrides the modulus for partial resubmits (rerun only failed
# shards, e.g. --array=2-3 ... 4) — without it, SLURM_ARRAY_TASK_COUNT of the
# partial array would mis-shard. ~4 min/instance sequential per shard.
# Merge afterward:  $SB_VENV/bin/python scripts/merge_shard_reports.py <TAG>

set -uo pipefail
OUTJSON=${1:?usage: sbatch --array=0-N run_score_shards.sbatch OUTPUT_JSONL TAG [TOTAL_SHARDS]}
TAG=${2:?usage: sbatch --array=0-N run_score_shards.sbatch OUTPUT_JSONL TAG [TOTAL_SHARDS]}
I=${SLURM_ARRAY_TASK_ID:?run with --array}
N=${3:-${SLURM_ARRAY_TASK_COUNT:?run with --array}}

source "${SWEBENCH_ROOT:-/checkpoint/dpf/swebench-eval}/env.sh"
echo "== node=$SLURMD_NODENAME shard=$I/$N tag=$TAG started=$(date -Is) =="

# --- this task's shard file (deterministic round-robin by line index) --------
# Include output_errors.jsonl: PR #751 captures git patches from failed/stuck
# rows; Babel scored those instead of silently dropping them.
SHARD_DIR=$SWEBENCH_ROOT/runs/score_$TAG
mkdir -p "$SHARD_DIR"
SHARD=$SHARD_DIR/shard_${I}of${N}.jsonl
ERRJSON=$(dirname "$OUTJSON")/output_errors.jsonl
cat "$OUTJSON" $([ -s "$ERRJSON" ] && echo "$ERRJSON") | awk -v i="$I" -v n="$N" '(NR-1) % n == i' > "$SHARD"
echo "== shard has $(wc -l < "$SHARD") instances =="
[ -s "$SHARD" ] || { echo "empty shard, nothing to do"; exit 0; }

cd "$BENCHMARKS_DIR"
"$SB_VENV/bin/swebench-eval" "$SHARD" \
  --apptainer --dataset princeton-nlp/SWE-bench_Verified --split test \
  --run-id "score_${TAG}_${I}of${N}" \
  --timeout 1800 \
  --apptainer-score-dir "$SHARD_DIR/reports_${I}of${N}" \
  --apptainer-sandbox-root "$OH_SANDBOX_ROOT" \
  --apptainer-cache "$APPTAINER_CACHEDIR"
RC=$?
echo "== score exit=$RC finished=$(date -Is) =="

# sandboxes are one-shot; prune this shard's to reclaim ~3GB/instance
if [ $RC -eq 0 ]; then
  while read -r line; do
    iid=$(printf '%s' "$line" | "$SB_VENV/bin/python" -c 'import json,sys; print(json.load(sys.stdin)["instance_id"])' 2>/dev/null) || continue
    rm -rf "$OH_SANDBOX_ROOT/$iid"
  done < "$SHARD"
  echo "== pruned shard sandboxes =="
fi
exit $RC
