#!/usr/bin/env bash
# One-time setup of the FAIR SWE-bench Verified eval environment.
# Run on a FAIR *compute* node (needs internet + /checkpoint + /scratch), e.g.:
#   srun --partition=scavenge --cpus-per-task=8 --mem=48G --time=03:00:00 \
#     --pty bash setup_env.sh
#
# Builds:
#   $BENCHMARKS_DIR/.venv   benchmarks + software-agent-sdk (uv workspace) —
#                           swebench-infer / swebench-eval / validate-cfg
#   $SWEBENCH_ROOT/.venv_vllm  vLLM OpenAI server (separate from the training env)
# and installs the apptainer shim + this kit's helper scripts + env.sh into
# $SWEBENCH_ROOT.
#
# benchmarks (dpfried/benchmarks @ babel-scoring-fixes) is a uv workspace whose
# SDK lives at vendor/software-agent-sdk (a submodule pinned to UPSTREAM). We
# replace that member with the fork branch that carries PR #3641
# (dpfried/software-agent-sdk @ fix-apptainer-tokenizer-condenser), then run
# `uv sync`, which builds the venv inside the checkout.
set -euo pipefail

export SWEBENCH_ROOT="${SWEBENCH_ROOT:-/checkpoint/dpf/swebench-eval}"
export BENCHMARKS_DIR="$SWEBENCH_ROOT/benchmarks"
KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SINGULARITY_BIN="${SINGULARITY_BIN:-/public/apps/singularity/4.0.2/bin/singularity}"
UV="${UV:-uv}"; command -v "$UV" >/dev/null 2>&1 || UV="/checkpoint/dpf/adp-env/.venv/bin/uv"
command -v "$UV" >/dev/null 2>&1 || { echo "FATAL: no uv found (set UV=...)"; exit 3; }

BENCHMARKS_GIT="${BENCHMARKS_GIT:-https://github.com/dpfried/benchmarks.git}"
BENCHMARKS_REF="${BENCHMARKS_REF:-babel-scoring-fixes}"
AGENT_SDK_GIT="${AGENT_SDK_GIT:-https://github.com/dpfried/software-agent-sdk.git}"
AGENT_SDK_REF="${AGENT_SDK_REF:-fix-apptainer-tokenizer-condenser}"

echo "== SWEBENCH_ROOT=$SWEBENCH_ROOT  uv=$UV =="
mkdir -p "$SWEBENCH_ROOT"/{bin,logs,runs,select,hf_cache,oh,scripts}

# --- 1. deploy env.sh + helper scripts into $SWEBENCH_ROOT -------------------
# (no-op when the kit was already extracted in-place at $SWEBENCH_ROOT)
if [ "$KIT_DIR" != "$SWEBENCH_ROOT" ]; then
  cp "$KIT_DIR/env.sh" "$SWEBENCH_ROOT/env.sh"
  cp "$KIT_DIR"/scripts/*.py "$SWEBENCH_ROOT/scripts/"
  echo "deployed env.sh + scripts to $SWEBENCH_ROOT"
else
  echo "kit already in place at $SWEBENCH_ROOT (skip deploy)"
fi

# --- 2. singularity -> apptainer shim + proot -------------------------------
ln -sf "$SINGULARITY_BIN" "$SWEBENCH_ROOT/bin/apptainer"
"$SWEBENCH_ROOT/bin/apptainer" --version
echo "apptainer shim -> $SINGULARITY_BIN"
# FAIR's SingularityCE is a non-setuid user install and dpf has no /etc/subuid
# mapping, so `apptainer build` of a %post def (apt/uv install) fails for a
# normal user unless `proot` is on PATH — SingularityCE auto-uses it for
# unprivileged builds. Drop a static proot into the shim dir (already on PATH
# via env.sh). Without this every agent-SIF prebuild fails with exit 255.
if [ ! -x "$SWEBENCH_ROOT/bin/proot" ]; then
  curl -fsSL -o "$SWEBENCH_ROOT/bin/proot" https://proot.gitlab.io/proot/bin/proot
  chmod +x "$SWEBENCH_ROOT/bin/proot"
fi
"$SWEBENCH_ROOT/bin/proot" --version >/dev/null 2>&1 && echo "proot installed" || echo "WARN: proot not runnable"

# --- 3. clone benchmarks + populate the SDK workspace member -----------------
git_checkout() { # url ref dest
  local url=$1 ref=$2 dest=$3
  if [ -d "$dest/.git" ]; then
    git -C "$dest" fetch origin "$ref" && git -C "$dest" checkout "$ref" \
      && git -C "$dest" reset --hard "origin/$ref" 2>/dev/null || true
  else
    git clone -b "$ref" "$url" "$dest"
  fi
}
git_checkout "$BENCHMARKS_GIT" "$BENCHMARKS_REF" "$BENCHMARKS_DIR"
# vendor/software-agent-sdk is a submodule pinned upstream; swap in the fork branch.
SDK_DIR="$BENCHMARKS_DIR/vendor/software-agent-sdk"
[ -d "$SDK_DIR/.git" ] || rm -rf "$SDK_DIR"
git_checkout "$AGENT_SDK_GIT" "$AGENT_SDK_REF" "$SDK_DIR"
echo "== benchmarks @ $(git -C "$BENCHMARKS_DIR" rev-parse --short HEAD)  sdk @ $(git -C "$SDK_DIR" rev-parse --short HEAD) =="

# --- 3b. proot-safe agent-SIF builds ----------------------------------------
# The generated agent-server def creates an `openhands` user with groupadd/
# useradd, then `su openhands -c 'uv sync'`. Under proot (our only unprivileged
# build path, see step 2) groupadd/useradd fail on locked /etc writes → build
# aborts exit 255. Replace them with direct /etc/{group,passwd,shadow} appends,
# which proot handles fine (su + uv sync then succeed). Idempotent.
python3 - "$BENCHMARKS_DIR/benchmarks/swebench/apptainer_build.py" <<'PY'
import sys, pathlib
f = pathlib.Path(sys.argv[1]); s = f.read_text()
if "proot-safe user creation" in s:
    print("apptainer_build.py already proot-patched"); raise SystemExit(0)
old = ('''    grep -Eq "^[^:]*:[^:]*:${{GID}}:" /etc/group || groupadd -g "${{GID}}" "${{USERNAME}}"\n'''
       '''    grep -Eq "^${{USERNAME}}:" /etc/passwd || useradd -m -u "${{UID}}" -g "${{GID}}" -s /bin/bash "${{USERNAME}}"''')
new = ('''    # proot-safe user creation (FAIR unprivileged apptainer build via proot;\n'''
       '''    # groupadd/useradd fail on locked /etc writes under proot)\n'''
       '''    grep -Eq "^[^:]*:[^:]*:${{GID}}:" /etc/group || echo "${{USERNAME}}:x:${{GID}}:" >> /etc/group\n'''
       '''    if ! grep -Eq "^${{USERNAME}}:" /etc/passwd; then\n'''
       '''        echo "${{USERNAME}}:x:${{UID}}:${{GID}}::/home/${{USERNAME}}:/bin/bash" >> /etc/passwd\n'''
       '''        echo "${{USERNAME}}:!::0:99999:7:::" >> /etc/shadow 2>/dev/null || true\n'''
       '''        mkdir -p /home/${{USERNAME}} && chown "${{UID}}:${{GID}}" /home/${{USERNAME}}\n'''
       '''    fi''')
if old not in s:
    print("WARN: proot patch target not found (upstream changed?) — build may fail under proot"); raise SystemExit(0)
f.write_text(s.replace(old, new)); print("patched apptainer_build.py for proot-safe builds")
PY

# --- 4. build benchmarks/.venv via uv sync (workspace: benchmarks + SDK) -----
( cd "$BENCHMARKS_DIR" && "$UV" sync )
MISSING=0
for ep in swebench-infer swebench-eval validate-cfg; do
  [ -x "$BENCHMARKS_DIR/.venv/bin/$ep" ] || { echo "WARN: missing entry point $ep after uv sync"; MISSING=1; }
done
[ $MISSING -eq 0 ] && echo "== benchmarks venv OK: $BENCHMARKS_DIR/.venv =="

# --- 5. .venv_vllm: serving --------------------------------------------------
# Needs a vLLM new enough to serve Qwen3.5 (GDN hybrid) with the qwen3_coder
# tool parser; >=0.24 also has the --mamba-cache-mode align prefix-caching flag.
#
# CRITICAL — CUDA 12.x only: the FAIR A100 driver is 550.144 (CUDA 12.4-era). It
# runs any CUDA 12.x build via minor-version compat (like the training env's
# cu128 torch) but CANNOT run CUDA 13 (needs driver >=580). vLLM's *PyPI* wheels
# are now cu13-linked (_C -> libcudart.so.13) and will NOT load here. vLLM
# publishes CUDA-12.x wheels only on GitHub releases, tagged +cu129 (libcudart
# .so.12, same SONAME as cu128 → compatible). Install that wheel + cu128 torch.
VLLM_VERSION="${VLLM_VERSION:-0.25.1}"
VLLM_CUDA="${VLLM_CUDA:-cu129}"
VLLM_WHL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+${VLLM_CUDA}-cp38-abi3-manylinux_2_28_x86_64.whl"
"$UV" venv "$SWEBENCH_ROOT/.venv_vllm" --python 3.12
"$UV" pip install --python "$SWEBENCH_ROOT/.venv_vllm/bin/python" --torch-backend=cu128 "vllm @ $VLLM_WHL" ninja

# --- 6. prefetch dataset + smoke apptainer ----------------------------------
export HF_HOME="$SWEBENCH_ROOT/hf_cache"; unset HF_HUB_OFFLINE
"$BENCHMARKS_DIR/.venv/bin/python" - <<'PY'
from datasets import load_dataset
ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
print("SWE-bench Verified:", len(ds), "instances cached")
PY
SM="$SWEBENCH_ROOT/oh/_smoke.sif"
"$SWEBENCH_ROOT/bin/apptainer" pull --force "$SM" docker://ghcr.io/linuxcontainers/alpine:latest \
  && "$SWEBENCH_ROOT/bin/apptainer" exec "$SM" true \
  && rm -f "$SM" && echo "apptainer docker:// pull+exec OK"

echo "== setup complete. Next: make a smoke id list, prebuild those SIFs, infer, score. See README.md =="
