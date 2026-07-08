#!/usr/bin/env bash
#SBATCH --job-name=swebench-smoke-swesmith
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:2
#SBATCH --exclude=babel-o5-28,babel-n5-32,babel-p5-28
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --output=/home/dfried/exp/adp-smoke/swebench/logs/%x-%j.out
#SBATCH --error=/home/dfried/exp/adp-smoke/swebench/logs/%x-%j.err

set -uo pipefail
SB=/home/dfried/exp/adp-smoke/swebench
export PATH="$SB/.venv_vllm/bin:$HOME/.local/bin:$PATH"   # venv bin first: flashinfer JIT shells out to `ninja`; also uv
echo "== node=$SLURMD_NODENAME job=$SLURM_JOB_ID started=$(date -Is) =="

# --- storage: caches on /data (persistent), tmp on /scratch (fast, disposable) ---
DATA=/data/tir/projects/tir1/users/dfried/exp/adp-smoke/openhands
mkdir -p "$DATA" || { echo "FATAL: tir1 bulk storage unavailable at $DATA (mounted on compute nodes only)"; exit 1; }
export OPENHANDS_APPTAINER_BUILD_ROOT=$DATA/agent-images
export APPTAINER_CACHEDIR=$DATA/apptainer-cache
export OPENHANDS_APPTAINER_WORKSPACE_ROOT=$DATA/workspaces
export APPTAINER_TMPDIR=/scratch/dfried/apptainer-tmp-$SLURM_JOB_ID
mkdir -p "$APPTAINER_TMPDIR" "$OPENHANDS_APPTAINER_BUILD_ROOT" "$APPTAINER_CACHEDIR" "$OPENHANDS_APPTAINER_WORKSPACE_ROOT"
# avoid Docker Hub rate limits: use the Epoch GHCR mirror (template pattern from repo tests)
export OPENHANDS_SWEBENCH_IMAGE_TEMPLATE="ghcr.io/epoch-research/swe-bench.eval.{arch}.{instance_id}:latest"
export HF_HOME=/home/dfried/exp/adp-smoke/hf_cache
unset HF_HUB_OFFLINE   # needs princeton-nlp/SWE-bench_Verified

# --- resolve checkpoint (latest swe-smith; checkpoints live on tir1 bulk storage) ---
CKPT=$(ls -d /data/tir/projects/tir1/users/dfried/exp/adp-smoke/runs/swesmith_4b_full/output/checkpoint-* 2>/dev/null | sort -V | tail -1)
[ -z "$CKPT" ] && { echo "FATAL: no swe-smith checkpoint found on tir1"; exit 1; }
echo "== serving checkpoint: $CKPT =="

# --- phase 1: vLLM server on this node ---
PORT=8012
nohup $SB/.venv_vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$CKPT" --served-model-name adp-swesmith-4b \
  --tensor-parallel-size 2 --max-model-len 32768 --dtype bfloat16 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --gpu-memory-utilization 0.85 --port $PORT > $SB/logs/vllm_$SLURM_JOB_ID.log 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
for i in $(seq 1 90); do   # first boot JIT-compiles flashinfer kernels; ~7 min observed on TP1
  curl -sf http://127.0.0.1:$PORT/health >/dev/null && break
  kill -0 $VLLM_PID 2>/dev/null || { echo "FATAL: vLLM died during startup"; tail -30 $SB/logs/vllm_$SLURM_JOB_ID.log; exit 1; }
  sleep 10
done
curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo "FATAL: vLLM never became healthy"; exit 1; }
echo "== vLLM healthy =="

# --- phase 2: llm config ---
cat > $SB/smoke/llm_config_vllm.json <<EOF
{
  "model": "hosted_vllm/adp-swesmith-4b",
  "base_url": "http://127.0.0.1:$PORT/v1",
  "api_key": "EMPTY",
  "temperature": 0.0,
  "max_output_tokens": 4096,
  "timeout": 600,
  "num_retries": 3,
  "native_tool_calling": true,
  "custom_tokenizer": "$CKPT"
}
EOF
$SB/benchmarks/.venv/bin/validate-cfg $SB/smoke/llm_config_vllm.json || echo "WARN: validate-cfg failed, continuing"

# --- phase 3: agent rollouts (builds per-instance images lazily on first use) ---
cd $SB/benchmarks
OUT=$SB/smoke/out_$SLURM_JOB_ID
.venv/bin/swebench-infer $SB/smoke/llm_config_vllm.json \
  --dataset princeton-nlp/SWE-bench_Verified --split test \
  --workspace apptainer \
  --select $SB/smoke/instances10.txt \
  --max-iterations 50 --num-workers 2 \
  --prompt-path $SB/smoke/smoke_native_repo_root.j2 \
  --output-dir "$OUT"
RC=$?
echo "== infer exit=$RC =="
OUTJSON=$(find "$OUT" -name output.jsonl | head -1)
[ -z "$OUTJSON" ] && { echo "FATAL: no output.jsonl produced"; exit 1; }

# --- phase 4: apptainer scoring (sequential by design) ---
.venv/bin/swebench-eval "$OUTJSON" \
  --apptainer --dataset princeton-nlp/SWE-bench_Verified --split test \
  --run-id smoke_$SLURM_JOB_ID \
  --timeout 1800 \
  --apptainer-score-dir $SB/smoke/score_$SLURM_JOB_ID \
  --apptainer-sandbox-root $DATA/swebench-sandboxes \
  --apptainer-cache $APPTAINER_CACHEDIR
RC=$?
echo "== score exit=$RC finished=$(date -Is) =="
grep -h "" $SB/smoke/score_$SLURM_JOB_ID/*.json 2>/dev/null | head -5 || true
exit $RC
