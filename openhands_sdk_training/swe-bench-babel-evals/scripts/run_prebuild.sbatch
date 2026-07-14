#!/usr/bin/env bash
#SBATCH --job-name=swebench-prebuild
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1   # preempt_qos requires >=1 GPU; unused by the build
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --output=/home/dfried/exp/adp-smoke/swebench/logs/%x-%A_%a.out
#SBATCH --error=/home/dfried/exp/adp-smoke/swebench/logs/%x-%A_%a.err

# Prebuild all 500 per-instance agent SIFs before full inference:
#   sbatch --array=0-7 run_prebuild.sbatch
# Idempotent (ensure_* reuses existing SIFs); rerun to fill gaps.
# 64G + serialized uv builds per Graham's measurements (~59GB peak RSS/build).
# Expect ~500/8 ≈ 63 builds/task; SIFs ~1.5GB each → ~750GB total on tir1.

set -uo pipefail
I=${SLURM_ARRAY_TASK_ID:?run with --array}
N=${SLURM_ARRAY_TASK_COUNT:?run with --array}
SB=/home/dfried/exp/adp-smoke/swebench
echo "== node=$SLURMD_NODENAME shard=$I/$N started=$(date -Is) =="

DATA=/data/tir/projects/tir1/users/dfried/exp/adp-smoke/openhands
mkdir -p "$DATA" || { echo "FATAL: tir1 bulk storage unavailable at $DATA (compute nodes only)"; exit 1; }
export OPENHANDS_APPTAINER_BUILD_ROOT=$DATA/agent-images
export APPTAINER_CACHEDIR=$DATA/apptainer-cache
export APPTAINER_TMPDIR=/scratch/dfried/apptainer-tmp-${SLURM_JOB_ID}_$I
mkdir -p "$APPTAINER_TMPDIR" "$OPENHANDS_APPTAINER_BUILD_ROOT" "$APPTAINER_CACHEDIR"
export OPENHANDS_SWEBENCH_IMAGE_TEMPLATE="ghcr.io/epoch-research/swe-bench.eval.{arch}.{instance_id}:latest"
export HF_HOME=/home/dfried/exp/adp-smoke/hf_cache
unset HF_HUB_OFFLINE
# serialize uv inside image builds; unserialized builds peaked ~59GB RSS
export OPENHANDS_APPTAINER_UV_CONCURRENT_DOWNLOADS=4
export OPENHANDS_APPTAINER_UV_CONCURRENT_BUILDS=1
export OPENHANDS_APPTAINER_UV_CONCURRENT_INSTALLS=1

cd "$SB/benchmarks"
.venv/bin/python "$SB/full/prebuild_agent_images.py" --shard-index "$I" --num-shards "$N"
RC=$?
echo "== prebuild shard exit=$RC finished=$(date -Is) =="
exit $RC
