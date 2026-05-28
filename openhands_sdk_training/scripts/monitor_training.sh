#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 LOG_PREFIX"
  exit 1
fi

LOG_PREFIX="$1"
PID_FILE="${LOG_PREFIX}.pid"
LOG_FILE="${LOG_PREFIX}.log"
EXIT_FILE="${LOG_PREFIX}.exit"

if [ -f "$PID_FILE" ]; then
  WRAPPER_PID="$(cat "$PID_FILE")"
  echo "wrapper_pid=$WRAPPER_PID"
  echo "children:"
  pgrep -P "$WRAPPER_PID" -a || true
else
  echo "No pid file: $PID_FILE"
fi

if [ -f "$EXIT_FILE" ]; then
  echo "exit=$(cat "$EXIT_FILE")"
else
  echo "exit_file=not_present"
fi

if [ -d /sys/class/drm/card1/device ]; then
  echo -n "vram_total="
  cat /sys/class/drm/card1/device/mem_info_vram_total 2>/dev/null || true
  echo -n "vram_used="
  cat /sys/class/drm/card1/device/mem_info_vram_used 2>/dev/null || true
  echo -n "gtt_total="
  cat /sys/class/drm/card1/device/mem_info_gtt_total 2>/dev/null || true
  echo -n "gtt_used="
  cat /sys/class/drm/card1/device/mem_info_gtt_used 2>/dev/null || true
fi

free -h || true

if [ -f "$LOG_FILE" ]; then
  tail -120 "$LOG_FILE" | sed -r 's/\x1b\[[0-9;]*[A-Za-z]//g'
else
  echo "No log file: $LOG_FILE"
fi

