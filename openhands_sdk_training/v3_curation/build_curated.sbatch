#!/usr/bin/env bash
#SBATCH --job-name=adp-v3-build-curated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --open-mode=append
#
# CPU-only job that runs build_curated_subset.py over a 25GB source subset.
# Submit with --partition/--output/--error and the env below, e.g.:
#
#   sbatch --partition=learnfair \
#     --output=/checkpoint/dpf/adp-analysis/v3_scoping/logs/build-%j.out \
#     --error=/checkpoint/dpf/adp-analysis/v3_scoping/logs/build-%j.err \
#     --export=ALL,SRC=...,OUT_DIR=...,MAX_RECORDS=55000,ENV_ROOT=/checkpoint/dpf/adp-env \
#     build_curated.sbatch
#
# NOTE: this is a real sbatch file rather than `sbatch --wrap "source ... && ..."`
# on purpose. --wrap runs the command under dash, where `source` does not exist, which
# is the exit-127 failure that silently ate a merge step in the v2 eval kit.

set -uo pipefail

: "${SRC:?SRC (source train.llamafactory.jsonl) required}"
: "${OUT_DIR:?OUT_DIR (destination dir) required}"
: "${ENV_ROOT:=/checkpoint/dpf/adp-env}"
: "${MAX_RECORDS:=}"
: "${EXTRA_ARGS:=}"

KIT_DIR="${KIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

if [ -f "$ENV_ROOT/env.sh" ]; then
  # shellcheck disable=SC1091
  . "$ENV_ROOT/env.sh"
fi
export PATH="$ENV_ROOT/.venv/bin:$PATH"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "== node=$(hostname) job=${SLURM_JOB_ID:-manual} started=$(date -Is) =="
echo "== SRC=$SRC"
echo "== OUT_DIR=$OUT_DIR"
echo "== MAX_RECORDS=${MAX_RECORDS:-<all>}  EXTRA_ARGS=${EXTRA_ARGS:-<none>}"
df -h "$(dirname "$OUT_DIR")" 2>/dev/null | tail -2 || true

ARGS=(--src "$SRC" --out-dir "$OUT_DIR")
[ -n "$MAX_RECORDS" ] && ARGS+=(--max-records "$MAX_RECORDS")
# shellcheck disable=SC2206
[ -n "$EXTRA_ARGS" ] && ARGS+=($EXTRA_ARGS)

python "$KIT_DIR/build_curated_subset.py" "${ARGS[@]}"
RC=$?
echo "== finished=$(date -Is) exit=$RC =="
exit $RC
