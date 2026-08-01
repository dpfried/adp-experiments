#!/usr/bin/env bash
#SBATCH --job-name=swebench-prebuild
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --output=/checkpoint/dpf/swebench-eval/logs/%x-%A_%a.out
#SBATCH --error=/checkpoint/dpf/swebench-eval/logs/%x-%A_%a.err

# FAIR port of ../../swe-bench-babel-evals/scripts/run_prebuild.sbatch.
# Prebuild per-instance Apptainer agent-server SIFs so GPU inference never
# builds lazily (lazy builds under N workers OOM the node). CPU-only → scavenge.
#
#   sbatch --array=0-7 run_prebuild.sbatch
# Idempotent (ensure_* reuses existing SIFs); rerun to fill gaps. --requeue so a
# scavenge preemption auto-restarts the task (cached SIFs skip). For a one-shot
# smoke you can instead submit on --partition=learnfair to avoid preemption.
# ~500/8 ≈ 63 builds/task; SIFs ~1.5GB each → ~750GB total under
# OPENHANDS_APPTAINER_BUILD_ROOT on /checkpoint. 64G + serialized uv builds
# (Babel measured ~59GB peak RSS/build).
#
# For the proof-of-port smoke you only need the SELECTED instances built; build
# just those first, e.g.:  SELECT_IDS=$SWEBENCH_ROOT/select/smoke10_ids.txt
# and run a 1-task array — the python helper honors --select-ids if provided.

set -uo pipefail
I=${SLURM_ARRAY_TASK_ID:?run with --array}
N=${SLURM_ARRAY_TASK_COUNT:?run with --array}

source "${SWEBENCH_ROOT:-/checkpoint/dpf/swebench-eval}/env.sh"
echo "== node=$SLURMD_NODENAME shard=$I/$N started=$(date -Is) =="

cd "$BENCHMARKS_DIR"
PREBUILD_ARGS=(--shard-index "$I" --num-shards "$N")
[ -n "${SELECT_IDS:-}" ] && PREBUILD_ARGS+=(--select-ids "$SELECT_IDS")
"$SB_VENV/bin/python" "$SWEBENCH_ROOT/scripts/prebuild_agent_images.py" "${PREBUILD_ARGS[@]}"
RC=$?
echo "== prebuild shard exit=$RC finished=$(date -Is) =="
exit $RC
