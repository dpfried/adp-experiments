# SWE-bench Verified eval on FAIR — proof-of-port

Port of the Babel eval kit (`../swe-bench-babel-evals`, `../swe-bench-full-4b`) to
the on-prem FAIR cluster. Goal of this first pass **(a)**: score **one** model
end-to-end on FAIR to prove the pipeline works, before building the 150-instance
souping harness.

First target: **`Qwen/Qwen3.5-4B`** (instruct — self-contained chat template,
already cached in the training env). No Babel resolve-rate anchor for instruct,
but it proves serve → agent-rollout → containerized scoring works on FAIR.

## Why this ports cleanly (verified 2026-07-23 on `learnfair` compute)

- Docker daemon is **not** accessible on FAIR (`permission denied` on the socket,
  login + compute) — same constraint as Babel.
- **SingularityCE 4.0.2** (`/public/apps/singularity/4.0.2/bin/singularity`)
  **pulls `docker://ghcr.io/...` images and runs them under unprivileged user
  namespaces** (`kernel.unprivileged_userns_clone=1`). The Babel Apptainer path
  (prebuild SIFs → vLLM serve → `swebench-infer` → sharded Apptainer scoring)
  therefore transfers unchanged.
- GHCR egress works from compute nodes (needed for the Epoch mirror of the 500
  per-instance eval images).
- A100 80GB (`learnfair[6000-6039]`, `--constraint=ampere80gb`) — easier serving
  than Babel's 2×L40S; a 4B runs at TP1.

## FAIR-specific changes vs. Babel

| Concern | Babel | FAIR (this kit) |
|---|---|---|
| Container runtime | `apptainer` binary | SingularityCE 4.0.2 via a `bin/apptainer` symlink shim on PATH; cache/tmp set under **both** `APPTAINER_*` and `SINGULARITY_*` |
| GPU inference | `general`, 2×L40S, TP2 | `learnfair` + `--constraint=ampere80gb`, 1×A100, TP1 (env `TP=2` for bigger) |
| CPU prebuild/scoring | `preempt` + dummy `gpu:1` | `scavenge`, no GPU |
| Bulk storage | `/data/tir/.../dfried` | `/checkpoint/dpf/swebench-eval` (`$SWEBENCH_ROOT`), tmp on node-local `/scratch` |
| Model | base/ft 4B, TP2 | instruct `Qwen/Qwen3.5-4B`, TP1 (proof-of-port) |

All paths/knobs live in **`env.sh`** (sourced by every sbatch from
`$SWEBENCH_ROOT/env.sh`, deployed there by `setup_env.sh`).

## Step 0 — code source (on GitHub, reachable from FAIR compute)

Transfer decision: **git push → GitHub** (done). Both are public forks:

- **benchmarks**: `dpfried/benchmarks` @ `babel-scoring-fixes` — a **uv
  workspace**; carries PRs #745 / #743 / #751 + the 3 local infra fixes
  (`../swe-bench-babel-evals/RESULTS.md`). Its SDK is a submodule at
  `vendor/software-agent-sdk` pinned to **upstream**.
- **software-agent-sdk**: `dpfried/software-agent-sdk` @
  `fix-apptainer-tokenizer-condenser` — PR #3641 apptainer tokenizer/condenser
  binds. `setup_env.sh` swaps this fork branch into the workspace member so #3641
  is present, then `uv sync`.

`setup_env.sh` has these URLs/refs as defaults (override via `BENCHMARKS_GIT/REF`,
`AGENT_SDK_GIT/REF`).

## Step 1 — build the env (one-time)

```bash
# on a scavenge compute node (needs internet + /checkpoint + /scratch)
srun --partition=scavenge --cpus-per-task=8 --mem=48G --time=03:00:00 \
  bash /checkpoint/dpf/swebench-eval/setup_env.sh
```

Clones `dpfried/benchmarks`, replaces the `vendor/software-agent-sdk` submodule
with the fork branch, runs `uv sync` → `$BENCHMARKS_DIR/.venv` (with
`swebench-infer`/`-eval`/`validate-cfg`), builds `$SWEBENCH_ROOT/.venv_vllm`,
installs the apptainer shim, deploys `env.sh` + helper `scripts/`, prefetches the
Verified dataset, and smoke-tests a `docker://` pull.

## Step 2 — proof-of-port smoke (a handful of instances first)

```bash
source $SWEBENCH_ROOT/env.sh
# 1. pick a small diverse subset
$SB_VENV/bin/python $SWEBENCH_ROOT/scripts/make_select_ids.py \
  --n 10 --out $SWEBENCH_ROOT/select/smoke10_ids.txt
# 2. prebuild ONLY those SIFs (CPU, scavenge)
SELECT_IDS=$SWEBENCH_ROOT/select/smoke10_ids.txt \
  fair 'cd <kit> && sbatch --array=0-0 scripts/run_prebuild.sbatch'
# 3. serve + rollouts on one A100
fair 'cd <kit> && sbatch scripts/run_full_infer.sbatch Qwen/Qwen3.5-4B smoke10 \
  '"$SWEBENCH_ROOT"'/select/smoke10_ids.txt'
# 4. score (CPU, scavenge) then merge
fair 'cd <kit> && sbatch --array=0-1 scripts/run_score_shards.sbatch \
  '"$SWEBENCH_ROOT"'/runs/out_smoke10/.../output.jsonl smoke10'
$SB_VENV/bin/python $SWEBENCH_ROOT/scripts/merge_shard_reports.py smoke10
```

Green smoke → drop `--select` / build all 500 SIFs for a full run.

## Open items to validate at first run (can't confirm without the checkout / a live run)

1. **Shim completeness** — SingularityCE 4.x honors `SINGULARITY_*` env and most
   `apptainer` CLI verbs, but confirm the OpenHands build code doesn't shell out
   to `apptainer`-only flags. If it reads a binary path from config, point it at
   the shim.
2. **vLLM version** — pin the version that ships the `qwen3_coder` tool parser
   and the `--mamba-cache-mode align` GDN prefix-caching flag (vLLM ≥0.24), built
   against the FAIR cu12x driver.
3. **GHCR rate limits** — anonymous pulls get 429'd from shared cluster IPs
   (`RESULTS.md`). Prebuild serializes and caches once; if throttled, authenticate
   to GHCR or stagger the array. Serve **local snapshot paths, never hub ids**.
4. **uv.lock vs. the SDK fork** — `uv sync` may re-lock because the workspace
   member (`vendor/software-agent-sdk`) is the fork branch, not the upstream
   submodule pin the committed `uv.lock` was generated against. Network is
   available on FAIR compute, so a re-lock is fine; watch for resolution drift.
