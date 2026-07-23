#!/usr/bin/env bash
# Shared environment for the FAIR SWE-bench Verified eval kit.
# Sourced by every sbatch script here. Port of ../swe-bench-babel-evals to the
# on-prem FAIR cluster (learnfair A100 + scavenge, classic H2 SLURM).
#
# Verified 2026-07-23 on learnfair compute:
#   - Docker daemon is NOT accessible (permission denied) — same as Babel.
#   - SingularityCE 4.0.2 (/public/apps/singularity/4.0.2) pulls docker:// GHCR
#     images and runs them under unprivileged user namespaces
#     (kernel.unprivileged_userns_clone=1). This is what makes the Babel
#     Apptainer eval path portable to FAIR unchanged.
#
# Everything bulk (venvs, SIF build cache, sandboxes, HF cache) lives on
# /checkpoint (4PB shared), never on the home quota.

# --- root ---------------------------------------------------------------------
export SWEBENCH_ROOT="${SWEBENCH_ROOT:-/checkpoint/dpf/swebench-eval}"

# --- singularity -> apptainer shim -------------------------------------------
# FAIR ships SingularityCE (an Apptainer fork), binary named `singularity`;
# OpenHands invokes `apptainer`. setup_env.sh drops a symlink
# $SWEBENCH_ROOT/bin/apptainer -> the singularity binary, put first on PATH.
export SINGULARITY_BIN="${SINGULARITY_BIN:-/public/apps/singularity/4.0.2/bin/singularity}"
export PATH="$SWEBENCH_ROOT/bin:$PATH"

# --- venvs / code -------------------------------------------------------------
# benchmarks is a uv workspace (SDK is a submodule member at vendor/software-agent-sdk);
# `uv sync` builds the venv INSIDE the checkout — hence benchmarks/.venv, matching Babel.
export BENCHMARKS_DIR="$SWEBENCH_ROOT/benchmarks"
export SB_VENV="$BENCHMARKS_DIR/.venv"           # swebench-infer / swebench-eval / validate-cfg / python
export SB_VENV_VLLM="$SWEBENCH_ROOT/.venv_vllm"  # vLLM OpenAI server (separate: different torch/deps)

# --- HuggingFace: eval needs princeton-nlp/SWE-bench_Verified, so NOT offline -
export HF_HOME="${HF_HOME:-$SWEBENCH_ROOT/hf_cache}"
unset HF_HUB_OFFLINE

# --- container image cache / build roots (bulk on /checkpoint) ---------------
# Epoch GHCR mirror of the 500 per-instance SWE-bench eval images.
export OPENHANDS_SWEBENCH_IMAGE_TEMPLATE="ghcr.io/epoch-research/swe-bench.eval.{arch}.{instance_id}:latest"
export OPENHANDS_APPTAINER_BUILD_ROOT="$SWEBENCH_ROOT/oh/agent-images"
export OPENHANDS_APPTAINER_WORKSPACE_ROOT="$SWEBENCH_ROOT/oh/workspaces"
export OH_SANDBOX_ROOT="$SWEBENCH_ROOT/oh/swebench-sandboxes"
# Cache dir: set under BOTH prefixes — SingularityCE reads SINGULARITY_*, the
# OpenHands build code sets/needs APPTAINER_*. (SingularityCE 4.x honors both,
# but be explicit so neither path guesses.)
export APPTAINER_CACHEDIR="$SWEBENCH_ROOT/oh/apptainer-cache"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
# Build tmp on node-local /scratch (fast, disposable), per-job to avoid clashes.
export APPTAINER_TMPDIR="/scratch/${USER}/apptainer-tmp-${SLURM_JOB_ID:-$$}"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
# Serialize uv inside image builds — Babel measured ~59GB peak RSS/build
# unserialized; serialized fits comfortably in a 64G job.
export OPENHANDS_APPTAINER_UV_CONCURRENT_DOWNLOADS=4
export OPENHANDS_APPTAINER_UV_CONCURRENT_BUILDS=1
export OPENHANDS_APPTAINER_UV_CONCURRENT_INSTALLS=1
# Public skills are part of the standard OpenHands harness (Babel parity).
export OPENHANDS_DISABLE_PUBLIC_SKILLS=0

mkdir -p "$OPENHANDS_APPTAINER_BUILD_ROOT" "$OPENHANDS_APPTAINER_WORKSPACE_ROOT" \
  "$OH_SANDBOX_ROOT" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$HF_HOME" \
  "$SWEBENCH_ROOT/logs" "$SWEBENCH_ROOT/runs" 2>/dev/null || true
