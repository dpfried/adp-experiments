#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 CONFIG_YAML LOG_PREFIX"
  echo "Example: $0 ~/data/adp_openhands_sdk/balanced_splits/qwen35_0_8b_openhands_nonweb_full_10k_bs1_seq2048_mm_safe.yaml ~/data/adp_openhands_sdk/balanced_splits/logs/train_10k_mm_safe"
  exit 1
fi

CONFIG_YAML="$1"
LOG_PREFIX="$2"
LOG_DIR="$(dirname "$LOG_PREFIX")"

mkdir -p "$LOG_DIR"
rm -f "${LOG_PREFIX}.exit"

(
  set +e
  llamafactory-cli train "$CONFIG_YAML" >"${LOG_PREFIX}.log" 2>&1
  echo "$?" >"${LOG_PREFIX}.exit"
) &

echo "$!" >"${LOG_PREFIX}.pid"
echo "Started wrapper PID $(cat "${LOG_PREFIX}.pid")"
echo "Log: ${LOG_PREFIX}.log"
echo "Exit status file: ${LOG_PREFIX}.exit"

