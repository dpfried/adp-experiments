#!/usr/bin/env bash
#SBATCH --job-name=swebench-full-infer
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:2
#SBATCH --exclude=babel-o5-28,babel-n5-32,babel-p5-28,babel-s5-24
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --open-mode=append
#SBATCH --output=/home/dfried/exp/adp-smoke/swebench/logs/%x-%j.out
#SBATCH --error=/home/dfried/exp/adp-smoke/swebench/logs/%x-%j.err

# Full 500-instance SWE-bench Verified inference for one model.
#   sbatch run_full_infer.sbatch <MODEL_PATH> <TAG> [CHAT_TEMPLATE]
#     MODEL_PATH    HF id or local checkpoint dir (e.g. .../checkpoint-540)
#     TAG           short label; names the served model, output dir, wandb-free
#     CHAT_TEMPLATE optional --chat-template file for vLLM. REQUIRED for the
#                   raw base model (no template of its own): pass the trained
#                   checkpoint's chat_template.jinja so all three models see
#                   identical conversation formatting.
# Scoring is NOT run here — submit run_score_shards.sbatch on the cpu
# partition afterwards (scoring is CPU-only; don't hold GPUs for it).

set -uo pipefail
MODEL=${1:?usage: sbatch run_full_infer.sbatch MODEL_PATH TAG [CHAT_TEMPLATE]}
TAG=${2:?usage: sbatch run_full_infer.sbatch MODEL_PATH TAG [CHAT_TEMPLATE]}
CHAT_TEMPLATE=${3:-}

SB=/home/dfried/exp/adp-smoke/swebench
export PATH="$SB/.venv_vllm/bin:$HOME/.local/bin:$PATH"   # venv bin first: flashinfer JIT needs `ninja`
echo "== node=$SLURMD_NODENAME job=$SLURM_JOB_ID model=$MODEL tag=$TAG started=$(date -Is) =="

# --- storage: caches on tir1 (persistent), tmp on /scratch (fast, disposable) ---
DATA=/data/tir/projects/tir1/users/dfried/exp/adp-smoke/openhands
mkdir -p "$DATA" || { echo "FATAL: tir1 bulk storage unavailable at $DATA (compute nodes only)"; exit 1; }
export OPENHANDS_APPTAINER_BUILD_ROOT=$DATA/agent-images
export APPTAINER_CACHEDIR=$DATA/apptainer-cache
export OPENHANDS_APPTAINER_WORKSPACE_ROOT=$DATA/workspaces
export APPTAINER_TMPDIR=/scratch/dfried/apptainer-tmp-$SLURM_JOB_ID
mkdir -p "$APPTAINER_TMPDIR" "$OPENHANDS_APPTAINER_BUILD_ROOT" "$APPTAINER_CACHEDIR" "$OPENHANDS_APPTAINER_WORKSPACE_ROOT"
export OPENHANDS_SWEBENCH_IMAGE_TEMPLATE="ghcr.io/epoch-research/swe-bench.eval.{arch}.{instance_id}:latest"
export HF_HOME=/home/dfried/exp/adp-smoke/hf_cache
unset HF_HUB_OFFLINE   # needs princeton-nlp/SWE-bench_Verified

# --- phase 1: vLLM server ---
export NCCL_NVLS_ENABLE=0   # NCCL 2.28 NVLS partial-alloc hang workaround (job 9131036 hung 13 min silent at TP2 init on s5-24)
PORT=8012
VLLM_ARGS=(--model "$MODEL" --served-model-name "adp-eval-$TAG"
  --tensor-parallel-size 2 --max-model-len 32768 --dtype bfloat16
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
  --gpu-memory-utilization 0.85 --port $PORT)
[ -n "$CHAT_TEMPLATE" ] && VLLM_ARGS+=(--chat-template "$CHAT_TEMPLATE")
nohup "$SB/.venv_vllm/bin/python" -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" \
  > "$SB/logs/vllm_full_${TAG}_$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
for i in $(seq 1 180); do   # 30 min: JIT + tir1-NFS model load can be slow on a cold node
  curl -sf http://127.0.0.1:$PORT/health >/dev/null && break
  kill -0 $VLLM_PID 2>/dev/null || { echo "FATAL: vLLM died during startup"; tail -30 "$SB/logs/vllm_full_${TAG}_$SLURM_JOB_ID.log"; exit 1; }
  sleep 10
done
curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo "FATAL: vLLM never became healthy"; exit 1; }
echo "== vLLM healthy =="

# --- phase 2: llm config ---
OUT=$SB/full/out_$TAG
mkdir -p "$OUT"
# Graham-parity settings (swe-bench-full-4b/README.md): trained boundary
# 32768 total / 28000 input, output 2047, thinking disabled, temp 0.
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
"$SB/benchmarks/.venv/bin/validate-cfg" "$OUT/llm_config.json" || echo "WARN: validate-cfg failed, continuing"

# --- phase 3: agent rollouts over the full split ---
# Graham-parity: default.j2 prompt, default max-iterations (500), condenser at
# the 28000-token threshold. PREREQUISITE: run_prebuild.sbatch array completed
# (500 agent SIFs cached) — lazy builds under 8 workers would OOM the node.
cd "$SB/benchmarks"
.venv/bin/swebench-infer "$OUT/llm_config.json" \
  --dataset princeton-nlp/SWE-bench_Verified --split test \
  --workspace apptainer \
  --num-workers 8 \
  --enable-condenser \
  --condenser-max-tokens 28000 \
  --condenser-max-output-tokens 1024 \
  --condenser-max-size 240 \
  --condenser-keep-first 2 \
  --output-dir "$OUT"
RC=$?
echo "== infer exit=$RC finished=$(date -Is) =="
OUTJSON=$(find "$OUT" -name output.jsonl | head -1)
[ -z "$OUTJSON" ] && { echo "FATAL: no output.jsonl produced"; exit 1; }
echo "== output: $OUTJSON ($(wc -l < "$OUTJSON") instances) =="
echo "== next: sbatch --array=0-7 $SB/full/run_score_shards.sbatch $OUTJSON $TAG =="
exit $RC
