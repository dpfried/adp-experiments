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
| Unprivileged image build | setuid apptainer / configured fakeroot | non-setuid SingularityCE + no `/etc/subuid` mapping → `%post` builds need **`proot`** on PATH (static binary in `bin/`; auto-used). Without it every SIF prebuild fails `exit 255`. |
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

## FAIR bring-up gotchas — all resolved in this kit (2026-07-23)

These were found and fixed while standing the pipeline up end-to-end on FAIR; the
fixes live in `env.sh` / `setup_env.sh` and the two source patches setup_env.sh
applies to the checkout.

1. **Unprivileged SIF builds → `proot`.** SingularityCE is non-setuid and dpf has
   no `/etc/subuid` mapping, so `apptainer build` of a `%post` def fails
   `exit 255`. A static `proot` on PATH (installed by setup_env.sh) is auto-used.
2. **`groupadd`/`useradd` fail under proot.** setup_env.sh patches the agent-def
   generator (`apptainer_build.py`) to append to `/etc/{group,passwd,shadow}`.
3. **Runtime containers → `use_fakeroot=False`.** The workspace runs
   `apptainer run --fakeroot`, which also needs subuid → exit 255. setup_env.sh
   patches `run_infer.py`'s `ApptainerWorkspace(...)` to pass `use_fakeroot=False`
   (works via `--compat` writable-tmpfs as the host user).
4. **vLLM must be CUDA-12.x.** A100 driver is **550.144** (cu12.x; cu13 needs
   ≥580). PyPI vLLM wheels are cu13-linked and won't load — install the `+cu129`
   GitHub-release wheel with `--torch-backend=cu129` (`libcudart.so.12`).
5. **`CUDA_HOME=/public/apps/cuda/12.4`.** vLLM/flashinfer JIT-compile Qwen3.5
   GDN kernels at serve time and need nvcc; FAIR compute has none. Set in env.sh.
6. **Node-local scratch is inconsistent** (`/scratch`→`/raid`, mkdir-fails on some
   nodes) → env.sh probes writable candidates for `APPTAINER_TMPDIR`.
7. **scavenge preempts** long prebuilds → `--requeue` (idempotent) or run the
   prebuild on `learnfair`. GHCR pulls can 429 on shared IPs — prebuild once.

## Still to watch at larger scale

- **GHCR 429s** when prebuilding all 500 base images back-to-back; stagger/auth if
  throttled. Serve **local snapshot paths, never hub ids**.
- **uv.lock vs. the SDK fork** — `uv sync` may re-lock (workspace member is the
  fork branch, not the submodule pin); fine with network, watch for drift.

## Eval comparability spec (the "eval card")

**Why this exists.** SWE-bench agentic resolve-rate is a property of *model × harness*, not the
model alone: on **identical** Qwen3.5-4B-Base weights, one harness scored **5.0%** and another
**≥10.6–16%** — a ~2–3× swing driven almost entirely by **patch-production rate** (non-empty patches
19% vs 70%), not capability. So a resolve-rate delta between two runs is a *model* signal only if
**every knob below matches**; otherwise it is a harness artifact. (In the ADP-v2 campaign a headline
was inverted twice by a single mismatched setting — an unmeasured / wrong-init baseline.) Treat this
section as the invariant any future eval must satisfy to be comparable to the reference runs at the end.

### Scaffold — must match exactly
- **Code**: `dpfried/benchmarks @ babel-scoring-fixes` (PRs #745/#743/#751 + the 3 infra fixes) + SDK
  `dpfried/software-agent-sdk @ fix-apptainer-tokenizer-condenser` (PR #3641). OpenHands SDK **1.27**.
- **Agent loop**: `default.j2` prompt, `max_iterations 500`, **temp 0 (single greedy)**, thinking **off**.
- **Token budget**: `max_input_tokens 28000`, `max_output_tokens 2047`.
- **Condenser**: **on**, 28k-token threshold.
- **Tool calling**: vLLM **native** parsing, `qwen3_coder` parser, `native_tool_calling: true`.
- **Serving**: local snapshot path (**never** a hub id → 429s), **prefix-caching on**, per-job vLLM
  port (2 shards/node collide otherwise).
- **Sampling**: **single-run pass@1**. Dedup `instance_id` before counting — a pass@k / retry *union*
  silently inflates (a θ₀ eval once read 145/500 as a pass@3-union artifact vs a true 119 single-run).

### Infra fixes that MOVE the score — must be present
- **`workspace.execute_command` timeout 30s→600s** (`cp_testbed_repo`) / 300s (`git reset`). Full-repo
  copies over loaded NFS blow through 30s → the agent loses its workspace → **empty patch**. Missing
  this is the single biggest known score-swinger (it explains the 5% vs ≥10.6% base gap).
- **Scoring `/private` bind** (`apptainer_eval.py`: `--no-mount hostfs`→`hostfs,home,cwd`). Missing →
  every instance false `patch_successfully_applied:false` → uniform 0-resolved. **Always spot-check one
  shard `report.json` for `scoring_error` before trusting counts — a uniform 0 across models is a bug,
  not model perf.**
- SDK apptainer tokenizer/condenser binds (PR #3641).

### Scoring board
- Official SWE-bench pytest parser, **but** `sphinx-doc__sphinx` (44 instances) is mis-scored **0/44**
  — its `tox` output has no per-test `PASSED` markers, so genuine passes read as failures for *every*
  model. **Report the 456 non-sphinx board as primary**; full-500 secondary with a footnote. (Uniform
  across models ⇒ doesn't bias a *comparison*, but deflates absolute totals.)

### Baseline & stats discipline
- **Baseline = the model's EXACT init checkpoint**, evaluated under this identical harness (base ≠
  instruct — a different model; the "~5% lift" saga was a base-vs-instruct mix-up). No borrowed /
  cross-harness anchors.
- Compare models on the **same instance set**, single-run, same seed. Significance = **paired McNemar**
  (+ TOST for equivalence), not a gap to a round number. **Noise floor ≈ ±15/500** for one temp-0
  rollout — deltas under that are "unresolved," not "tied." Report per-board (500 / 456 / clean).

### Reference runs (this FAIR A100 harness, SWE-bench Verified 500, single-run pass@1)
A re-run under the spec above should reproduce these ±~15/500:
- **θ₀ = `Qwen/Qwen3.5-4B` (instruct) = 119/500 (~24%)** — the load-bearing baseline.
- adp-v2 SFT arms: swezero 77, rebench 70, coderforge 48, scale 35; pooled55k (joint-train) 46;
  best soup (top-2) 62.
- **NOT comparable**: the Babel `../swe-bench-babel-evals/RESULTS.md` numbers (base-init, L40S / Graham
  harness) — different harness *and* init. Cross-harness absolute comparison is exactly what this card
  exists to prevent (see the cross-campaign section of the adp-v2 findings report).
