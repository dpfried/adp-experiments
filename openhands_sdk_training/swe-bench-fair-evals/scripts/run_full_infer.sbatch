#!/usr/bin/env bash
#SBATCH --job-name=swebench-infer
#SBATCH --partition=learnfair
#SBATCH --constraint=ampere80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --output=/checkpoint/dpf/swebench-eval/logs/%x-%j.out
#SBATCH --error=/checkpoint/dpf/swebench-eval/logs/%x-%j.err

# FAIR port of ../../swe-bench-babel-evals/scripts/run_full_infer.sbatch.
# SWE-bench Verified agent rollouts for one model on a single A100 (learnfair).
#
#   sbatch run_full_infer.sbatch <MODEL> <TAG> [SELECT_IDS_FILE] [CHAT_TEMPLATE]
#     MODEL           HF id or local checkpoint dir. Proof-of-port default target
#                     is the instruct model Qwen/Qwen3.5-4B (self-contained chat
#                     template — no CHAT_TEMPLATE needed).
#     TAG             short label; names the served model + output dir.
#     SELECT_IDS_FILE optional file of instance_ids (one per line) to restrict
#                     the run (use for the proof-of-port smoke). Omit / "" = full
#                     500-instance split.
#     CHAT_TEMPLATE   optional --chat-template file; REQUIRED only for a raw
#                     *base* model (which ships no template). For the instruct
#                     4B leave it empty.
#   env knobs: TP (tensor-parallel, default 1 — a 4B fits on one A100-80GB),
#              NUM_WORKERS (concurrent agents, default 8).
#
# Scoring is a SEPARATE, CPU-only step — do NOT hold GPUs for it. After this
# finishes, submit run_score_shards.sbatch on the scavenge partition.
# PREREQUISITE: run_prebuild.sbatch has cached the per-instance agent SIFs.

set -uo pipefail
MODEL=${1:?usage: sbatch run_full_infer.sbatch MODEL TAG [SELECT_IDS_FILE] [CHAT_TEMPLATE]}
TAG=${2:?usage: sbatch run_full_infer.sbatch MODEL TAG [SELECT_IDS_FILE] [CHAT_TEMPLATE]}
SELECT=${3:-}
CHAT_TEMPLATE=${4:-}
TP=${TP:-1}

source "${SWEBENCH_ROOT:-/checkpoint/dpf/swebench-eval}/env.sh"
export PATH="$SB_VENV_VLLM/bin:$PATH"   # vLLM venv first (flashinfer JIT needs `ninja`)
echo "== node=$SLURMD_NODENAME job=$SLURM_JOB_ID model=$MODEL tag=$TAG tp=$TP select=${SELECT:-<full-500>} started=$(date -Is) =="

# --- phase 1: vLLM server ----------------------------------------------------
export NCCL_NVLS_ENABLE=0   # NCCL NVLS partial-alloc hang workaround (Babel saw a 13min silent TP2 init hang)
PORT=8012
VLLM_ARGS=(--model "$MODEL" --served-model-name "adp-eval-$TAG"
  --tensor-parallel-size "$TP" --max-model-len 32768 --dtype bfloat16
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
  --gpu-memory-utilization 0.85 --port "$PORT")
[ -n "$CHAT_TEMPLATE" ] && VLLM_ARGS+=(--chat-template "$CHAT_TEMPLATE")
VLLM_LOG="$SWEBENCH_ROOT/logs/vllm_${TAG}_$SLURM_JOB_ID.log"
nohup "$SB_VENV_VLLM/bin/python" -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 180); do   # up to 30 min: JIT + cold NFS model load
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
  kill -0 $VLLM_PID 2>/dev/null || { echo "FATAL: vLLM died during startup"; tail -30 "$VLLM_LOG"; exit 1; }
  sleep 10
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "FATAL: vLLM never became healthy"; tail -30 "$VLLM_LOG"; exit 1; }
echo "== vLLM healthy =="

# --- phase 2: llm config (Babel / Graham parity) -----------------------------
OUT=$SWEBENCH_ROOT/runs/out_$TAG
mkdir -p "$OUT"
cat > "$OUT/llm_config.json" <<EOF
{
  "model": "hosted_vllm/adp-eval-$TAG",
  "base_url": "http://127.0.0.1:$PORT/v1",
  "api_key": "EMPTY",
  "temperature": 0.0,
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
"$SB_VENV/bin/validate-cfg" "$OUT/llm_config.json" || echo "WARN: validate-cfg failed, continuing"

# --- phase 3: agent rollouts -------------------------------------------------
cd "$BENCHMARKS_DIR"
INFER_ARGS=("$OUT/llm_config.json"
  --dataset princeton-nlp/SWE-bench_Verified --split test
  --workspace apptainer
  --num-workers "${NUM_WORKERS:-8}"
  --enable-condenser
  --condenser-max-tokens 28000
  --condenser-max-output-tokens 1024
  --condenser-max-size 240
  --condenser-keep-first 2
  --output-dir "$OUT")
[ -n "$SELECT" ] && INFER_ARGS+=(--select "$SELECT")
"$SB_VENV/bin/swebench-infer" "${INFER_ARGS[@]}"
RC=$?
echo "== infer exit=$RC finished=$(date -Is) =="
OUTJSON=$(find "$OUT" -name output.jsonl | head -1)
[ -z "$OUTJSON" ] && { echo "FATAL: no output.jsonl produced"; exit 1; }
echo "== output: $OUTJSON ($(wc -l < "$OUTJSON") instances) =="
echo "== next: sbatch --array=0-3 run_score_shards.sbatch $OUTJSON $TAG =="
exit $RC
