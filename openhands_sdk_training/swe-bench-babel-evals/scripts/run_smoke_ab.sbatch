#!/usr/bin/env bash
#SBATCH --job-name=swebench-smoke-ab
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:2
#SBATCH --exclude=babel-o5-28,babel-n5-32,babel-p5-28,babel-s5-24,babel-n5-24
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --open-mode=append
#SBATCH --output=/home/dfried/exp/adp-smoke/swebench/logs/%x-%j.out
#SBATCH --error=/home/dfried/exp/adp-smoke/swebench/logs/%x-%j.err

# 20-instance A/B smoke test: vLLM hybrid (GDN) prefix caching + sampling temp.
#   sbatch run_smoke_ab.sbatch <MODEL_PATH> <TAG> <TEMP> <TOP_P> [CACHE=on|off]
# Fixed instance set: full/smoke20_ids.txt (stride-25 over the Verified split
# for repo diversity; identical across all A/B runs and present in the full
# temp-0 papernonweb1154 baseline).
# Serving deltas vs run_full_infer.sbatch: --enable-prefix-caching
# --mamba-cache-mode align (vLLM PR #30877, merged 2026-01-23, in v0.24.0).

set -uo pipefail
MODEL=${1:?usage: sbatch run_smoke_ab.sbatch MODEL_PATH TAG TEMP TOP_P}
TAG=${2:?usage: sbatch run_smoke_ab.sbatch MODEL_PATH TAG TEMP TOP_P}
TEMP=${3:?usage: sbatch run_smoke_ab.sbatch MODEL_PATH TAG TEMP TOP_P}
TOPP=${4:?usage: sbatch run_smoke_ab.sbatch MODEL_PATH TAG TEMP TOP_P}
CACHE=${5:-on}

SB=/home/dfried/exp/adp-smoke/swebench
export PATH="$SB/.venv_vllm/bin:$HOME/.local/bin:$PATH"
echo "== node=$SLURMD_NODENAME job=$SLURM_JOB_ID model=$MODEL tag=$TAG temp=$TEMP top_p=$TOPP started=$(date -Is) =="

DATA=/data/tir/projects/tir1/users/dfried/exp/adp-smoke/openhands
mkdir -p "$DATA" || { echo "FATAL: tir1 bulk storage unavailable at $DATA (compute nodes only)"; exit 1; }
export OPENHANDS_APPTAINER_BUILD_ROOT=$DATA/agent-images
export APPTAINER_CACHEDIR=$DATA/apptainer-cache
export OPENHANDS_APPTAINER_WORKSPACE_ROOT=$DATA/workspaces
export APPTAINER_TMPDIR=/scratch/dfried/apptainer-tmp-$SLURM_JOB_ID
mkdir -p "$APPTAINER_TMPDIR" "$OPENHANDS_APPTAINER_BUILD_ROOT" "$APPTAINER_CACHEDIR" "$OPENHANDS_APPTAINER_WORKSPACE_ROOT"
export OPENHANDS_SWEBENCH_IMAGE_TEMPLATE="ghcr.io/epoch-research/swe-bench.eval.{arch}.{instance_id}:latest"
export HF_HOME=/home/dfried/exp/adp-smoke/hf_cache
unset HF_HUB_OFFLINE

# --- phase 1: vLLM server (prefix caching ON — the thing under test) ---
export NCCL_NVLS_ENABLE=0
PORT=8012
VLLM_ARGS=(--model "$MODEL" --served-model-name "adp-eval-$TAG"
  --tensor-parallel-size 2 --max-model-len 32768 --dtype bfloat16
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
  --gpu-memory-utilization 0.85 --port $PORT)
[ "$CACHE" = "on" ] && VLLM_ARGS+=(--enable-prefix-caching --mamba-cache-mode align)
echo "== prefix cache: $CACHE =="
nohup "$SB/.venv_vllm/bin/python" -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" \
  > "$SB/logs/vllm_smoke_${TAG}_$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:$PORT/health >/dev/null && break
  kill -0 $VLLM_PID 2>/dev/null || { echo "FATAL: vLLM died during startup"; tail -30 "$SB/logs/vllm_smoke_${TAG}_$SLURM_JOB_ID.log"; exit 1; }
  sleep 10
done
curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo "FATAL: vLLM never became healthy"; exit 1; }
echo "== vLLM healthy =="

# --- phase 2: llm config (temp/top_p are the B-test variables) ---
OUT=$SB/full/ab/out_$TAG
mkdir -p "$OUT"
cat > "$OUT/llm_config.json" <<EOF
{
  "model": "hosted_vllm/adp-eval-$TAG",
  "base_url": "http://127.0.0.1:$PORT/v1",
  "api_key": "EMPTY",
  "temperature": $TEMP,
  "top_p": $TOPP,
  "max_input_tokens": 28000,
  "max_output_tokens": 2047,
  "timeout": 600,
  "num_retries": 3,
  "native_tool_calling": true,
  "custom_tokenizer": "$MODEL",
  "reasoning_effort": "none",
  "litellm_extra_body": {
    "stop_token_ids": [248046],
    "chat_template_kwargs": {"enable_thinking": false}
  }
}
EOF
"$SB/benchmarks/.venv/bin/validate-cfg" "$OUT/llm_config.json" || echo "WARN: validate-cfg failed, continuing"

# --- phase 3: rollouts over the fixed 20-instance set ---
cd "$SB/benchmarks"
T0=$(date +%s)
.venv/bin/swebench-infer "$OUT/llm_config.json" \
  --dataset princeton-nlp/SWE-bench_Verified --split test \
  --workspace apptainer \
  --num-workers 8 \
  --select "$SB/full/smoke20_ids.txt" \
  --enable-condenser \
  --condenser-max-tokens 28000 \
  --condenser-max-output-tokens 1024 \
  --condenser-max-size 240 \
  --condenser-keep-first 2 \
  --output-dir "$OUT"
RC=$?
T1=$(date +%s)
echo "== infer exit=$RC wallclock=$((T1-T0))s finished=$(date -Is) =="
grep -o "Prefix cache hit rate: [0-9.]*%" "$SB/logs/vllm_smoke_${TAG}_$SLURM_JOB_ID.log" | tail -5
OUTJSON=$(find "$OUT" -name output.jsonl | head -1)
[ -n "$OUTJSON" ] && echo "== output: $OUTJSON ($(wc -l < "$OUTJSON") instances) =="
exit $RC
