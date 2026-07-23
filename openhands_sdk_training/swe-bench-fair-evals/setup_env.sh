#!/usr/bin/env bash
# One-time setup of the FAIR SWE-bench Verified eval environment.
# Run on a FAIR *compute* node (needs internet + /checkpoint + /scratch), e.g.:
#   srun --partition=scavenge --cpus-per-task=8 --mem=32G --time=02:00:00 \
#     --pty bash setup_env.sh
#
# Builds two venvs under $SWEBENCH_ROOT:
#   .venv       benchmarks + software-agent-sdk  (swebench-infer / -eval / validate-cfg)
#   .venv_vllm  vLLM OpenAI server               (kept separate from the training env)
# and installs the apptainer shim + this kit's scripts + env.sh into $SWEBENCH_ROOT.
#
# REQUIRED inputs (the babel-scoring-fixes checkout, pushed to GitHub per the
# transfer decision) — export these before running, or edit the defaults:
#   BENCHMARKS_GIT   git URL of the OpenHands/benchmarks fork (branch babel-scoring-fixes)
#   BENCHMARKS_REF   branch/tag/sha to check out
#   AGENT_SDK_GIT    git URL of the software-agent-sdk fork (PR #3641 changes)
#   AGENT_SDK_REF    branch/tag/sha to check out
#
# NOTE: the exact editable-install layout (whether the SDK is a monorepo of
# sub-packages, whether benchmarks pins the SDK as a path dep) is finalized
# against the actual checkout — the install block below is the common case and
# is marked ADJUST. Everything else (dirs, shim, vLLM venv, prefetch) is fixed.
set -euo pipefail

export SWEBENCH_ROOT="${SWEBENCH_ROOT:-/checkpoint/dpf/swebench-eval}"
KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SINGULARITY_BIN="${SINGULARITY_BIN:-/public/apps/singularity/4.0.2/bin/singularity}"
UV="${UV:-uv}"; command -v "$UV" >/dev/null || UV="/checkpoint/dpf/adp-env/.venv/bin/uv"

BENCHMARKS_GIT="${BENCHMARKS_GIT:-}"
BENCHMARKS_REF="${BENCHMARKS_REF:-babel-scoring-fixes}"
AGENT_SDK_GIT="${AGENT_SDK_GIT:-}"
AGENT_SDK_REF="${AGENT_SDK_REF:-}"
[ -n "$BENCHMARKS_GIT" ] || { echo "SET BENCHMARKS_GIT (see header)"; exit 2; }
[ -n "$AGENT_SDK_GIT" ]  || { echo "SET AGENT_SDK_GIT (see header)";  exit 2; }

echo "== SWEBENCH_ROOT=$SWEBENCH_ROOT =="
mkdir -p "$SWEBENCH_ROOT"/{bin,logs,runs,select,hf_cache,oh}

# --- 1. deploy env.sh + scripts into $SWEBENCH_ROOT (sbatch source them there) --
cp "$KIT_DIR/env.sh" "$SWEBENCH_ROOT/env.sh"
mkdir -p "$SWEBENCH_ROOT/scripts"
cp "$KIT_DIR"/scripts/*.py "$SWEBENCH_ROOT/scripts/"
echo "deployed env.sh + scripts to $SWEBENCH_ROOT"

# --- 2. singularity -> apptainer shim ---------------------------------------
ln -sf "$SINGULARITY_BIN" "$SWEBENCH_ROOT/bin/apptainer"
"$SWEBENCH_ROOT/bin/apptainer" --version
echo "apptainer shim -> $SINGULARITY_BIN"

# --- 3. clone the code -------------------------------------------------------
clone() { # url ref dest
  local url=$1 ref=$2 dest=$3
  if [ -d "$dest/.git" ]; then
    git -C "$dest" fetch --all --tags && git -C "$dest" checkout "$ref" && git -C "$dest" pull --ff-only || true
  else
    git clone "$url" "$dest" && git -C "$dest" checkout "$ref"
  fi
}
clone "$BENCHMARKS_GIT" "$BENCHMARKS_REF" "$SWEBENCH_ROOT/benchmarks"
clone "$AGENT_SDK_GIT"  "$AGENT_SDK_REF"  "$SWEBENCH_ROOT/software-agent-sdk"

# --- 4. .venv: benchmarks + SDK ---------------------------------------------
# ADJUST to the actual checkout layout once it lands on FAIR.
"$UV" venv "$SWEBENCH_ROOT/.venv" --python 3.12
VENV_PY="$SWEBENCH_ROOT/.venv/bin/python"
"$UV" pip install --python "$VENV_PY" -e "$SWEBENCH_ROOT/software-agent-sdk"
"$UV" pip install --python "$VENV_PY" -e "$SWEBENCH_ROOT/benchmarks"
# sanity: entry points must exist
for ep in swebench-infer swebench-eval validate-cfg; do
  [ -x "$SWEBENCH_ROOT/.venv/bin/$ep" ] || echo "WARN: missing entry point $ep — check install layout"
done

# --- 5. .venv_vllm: serving --------------------------------------------------
# vLLM >=0.24 for the GDN hybrid prefix-caching flag; pin CUDA wheels for the
# FAIR driver (cu12x). Confirm the version that ships the qwen3_coder tool parser.
"$UV" venv "$SWEBENCH_ROOT/.venv_vllm" --python 3.12
"$UV" pip install --python "$SWEBENCH_ROOT/.venv_vllm/bin/python" 'vllm>=0.24' ninja

# --- 6. prefetch dataset + smoke apptainer ----------------------------------
export HF_HOME="$SWEBENCH_ROOT/hf_cache"; unset HF_HUB_OFFLINE
"$VENV_PY" - <<'PY'
from datasets import load_dataset
ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
print("SWE-bench Verified:", len(ds), "instances cached")
PY
"$SWEBENCH_ROOT/bin/apptainer" pull --force "$SWEBENCH_ROOT/oh/_smoke.sif" \
  docker://ghcr.io/linuxcontainers/alpine:latest && \
  "$SWEBENCH_ROOT/bin/apptainer" exec "$SWEBENCH_ROOT/oh/_smoke.sif" true && \
  rm -f "$SWEBENCH_ROOT/oh/_smoke.sif" && echo "apptainer docker:// pull+exec OK"

echo "== setup complete. Next: make a smoke id list, prebuild those SIFs, infer, score. See README.md =="
