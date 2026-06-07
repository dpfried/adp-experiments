#!/usr/bin/env bash

DATASET=$1
if [ -z "$DATASET" ]; then
  echo "Usage: $0 DATASET_NAME" >&2
  exit 2
fi

REPO=${ADP_REPO:-/home/gneubig/work/adp/agent-data-protocol-pr244}
EXP_ROOT=${ADP_EXP_ROOT:-/home/gneubig/exp/adp}
PYTHON=${ADP_PYTHON:-/home/gneubig/work/adp/.venvs/openhands_sdk_training/bin/python}
OUT_ROOT=${ADP_OUT_ROOT:-$EXP_ROOT/datasets/software_agent_pipeline}
OUT_DIR=$OUT_ROOT/$DATASET
LOG_DIR=$OUT_ROOT/logs
TMP_DIR=$OUT_DIR/tmp
FULL_SFT_DIR=$OUT_DIR/full_sft

mkdir -p "$OUT_DIR" "$LOG_DIR" "$TMP_DIR" "$FULL_SFT_DIR"

RAW_JSONL=$OUT_DIR/full_raw.jsonl
STD_JSONL=$OUT_DIR/full_std.jsonl
OPENHANDS_JSONL=$FULL_SFT_DIR/full_sft_openhands_v0.jsonl
SWEAGENT_JSONL=$FULL_SFT_DIR/full_sft_sweagent.jsonl
MANIFEST=$OUT_DIR/manifest.json

started_at=$(date -Is)
echo "dataset=$DATASET"
echo "repo=$REPO"
echo "out_dir=$OUT_DIR"
echo "started_at=$started_at"

if [ ! -d "$REPO/datasets/$DATASET" ]; then
  echo "Dataset directory not found: $REPO/datasets/$DATASET" >&2
  exit 1
fi

(
  cd "$REPO/datasets/$DATASET" || exit 1
  env -u PYTHONPATH "$PYTHON" extract_raw.py
) > "$RAW_JSONL.tmp" 2> "$LOG_DIR/${DATASET}.extract_raw.stderr"
extract_status=$?
raw_lines=$(wc -l < "$RAW_JSONL.tmp" 2>/dev/null || echo 0)
echo "extract_status=$extract_status raw_lines=$raw_lines"
if [ "$extract_status" -ne 0 ] || [ "$raw_lines" -eq 0 ]; then
  echo "extract_raw failed or produced no rows" >&2
  exit 1
fi
mv "$RAW_JSONL.tmp" "$RAW_JSONL"

(
  cd "$REPO/datasets/$DATASET" || exit 1
  PYTHONPATH="$REPO:${PYTHONPATH:-}" "$PYTHON" raw_to_standardized.py < "$RAW_JSONL"
) > "$STD_JSONL.tmp" 2> "$LOG_DIR/${DATASET}.raw_to_standardized.stderr"
std_status=$?
std_lines=$(wc -l < "$STD_JSONL.tmp" 2>/dev/null || echo 0)
echo "std_status=$std_status std_lines=$std_lines"
if [ "$std_status" -ne 0 ] || [ "$std_lines" -eq 0 ]; then
  echo "raw_to_standardized failed or produced no rows" >&2
  exit 1
fi
mv "$STD_JSONL.tmp" "$STD_JSONL"

(
  cd "$REPO" || exit 1
  MY_DATASET="$DATASET" PYTHONPATH="$REPO:${PYTHONPATH:-}" "$PYTHON" \
    agents/openhands_v0/std_to_sft.py --is_web=no --api_env=execute_bash < "$STD_JSONL"
) > "$OPENHANDS_JSONL.tmp" 2> "$LOG_DIR/${DATASET}.openhands_v0.stderr"
openhands_status=$?
openhands_lines=$(wc -l < "$OPENHANDS_JSONL.tmp" 2>/dev/null || echo 0)
echo "openhands_status=$openhands_status openhands_lines=$openhands_lines"
if [ "$openhands_status" -eq 0 ] && [ "$openhands_lines" -gt 0 ]; then
  mv "$OPENHANDS_JSONL.tmp" "$OPENHANDS_JSONL"
else
  rm -f "$OPENHANDS_JSONL.tmp"
fi

(
  cd "$REPO" || exit 1
  MY_DATASET="$DATASET" PYTHONPATH="$REPO:${PYTHONPATH:-}" "$PYTHON" \
    agents/sweagent/std_to_sft.py < "$STD_JSONL"
) > "$SWEAGENT_JSONL.tmp" 2> "$LOG_DIR/${DATASET}.sweagent.stderr"
sweagent_status=$?
sweagent_lines=$(wc -l < "$SWEAGENT_JSONL.tmp" 2>/dev/null || echo 0)
echo "sweagent_status=$sweagent_status sweagent_lines=$sweagent_lines"
if [ "$sweagent_status" -eq 0 ] && [ "$sweagent_lines" -gt 0 ]; then
  mv "$SWEAGENT_JSONL.tmp" "$SWEAGENT_JSONL"
else
  rm -f "$SWEAGENT_JSONL.tmp"
fi

finished_at=$(date -Is)
cat > "$MANIFEST" <<JSON
{
  "dataset": "$DATASET",
  "started_at": "$started_at",
  "finished_at": "$finished_at",
  "raw_lines": $raw_lines,
  "std_lines": $std_lines,
  "openhands_status": $openhands_status,
  "openhands_lines": $openhands_lines,
  "sweagent_status": $sweagent_status,
  "sweagent_lines": $sweagent_lines,
  "raw_jsonl": "$RAW_JSONL",
  "std_jsonl": "$STD_JSONL",
  "openhands_jsonl": "$OPENHANDS_JSONL",
  "sweagent_jsonl": "$SWEAGENT_JSONL"
}
JSON

echo "finished_at=$finished_at"
