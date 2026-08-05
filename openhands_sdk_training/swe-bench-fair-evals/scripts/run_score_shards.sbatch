#!/usr/bin/env bash
#SBATCH --job-name=swebench-score
#SBATCH --partition=scavenge
#SBATCH --requeue
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

# --- PER-JOB sandbox root. This is a correctness fix, not tidiness ------------
# Sandboxes used to be created under one SHARED root and pruned by instance_id
# alone (`rm -rf $OH_SANDBOX_ROOT/$iid`). Every cell of a comparison scores the
# SAME instance set, so two concurrent scoring jobs shared one sandbox directory
# per instance and then deleted it out from under each other mid-run. The failure
# is silent and one-directional: a vanished sandbox scores as UNRESOLVED.
#
# Measured, not theorised: cell B's attempt-1 subset scored 6/100 when 20 tasks
# ran concurrently over the shared instance set, and 13/100 when the identical
# bytes were re-scored alone -- 9 instances flipping, all toward resolved. That
# ~7/100 is larger than every effect the parity ladder was built to detect, so
# any concurrent scoring done before this fix is depressed by an unknown amount.
#
# Isolating by TAG and shard makes concurrent scoring of overlapping instance
# sets safe. Sandboxes are one-shot, so nothing is lost by not sharing them.
# (env.sh's ADP_WORKSPACE_LOCAL=1 also yields a per-job root, on node-local
# scratch. Correctness must not depend on that being switched on.)
SANDBOX_ROOT="$OH_SANDBOX_ROOT/job_${TAG}_${I}of${N}"
mkdir -p "$SANDBOX_ROOT"
echo "== sandbox root=$SANDBOX_ROOT =="

cd "$BENCHMARKS_DIR"
"$SB_VENV/bin/swebench-eval" "$SHARD" \
  --apptainer --dataset princeton-nlp/SWE-bench_Verified --split test \
  --run-id "score_${TAG}_${I}of${N}" \
  --timeout 1800 \
  --apptainer-score-dir "$SHARD_DIR/reports_${I}of${N}" \
  --apptainer-sandbox-root "$SANDBOX_ROOT" \
  --apptainer-cache "$APPTAINER_CACHEDIR"
RC=$?
echo "== score exit=$RC finished=$(date -Is) =="

# sandboxes are one-shot; prune this job's whole subtree (~3GB/instance).
# Pruning the subtree rather than per-instance paths under a shared root is what
# makes this safe to do while another job is still running.
if [ $RC -eq 0 ]; then
  rm -rf "$SANDBOX_ROOT"
  echo "== pruned shard sandboxes =="
fi
exit $RC
