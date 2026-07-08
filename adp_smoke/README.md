# adp_smoke: Qwen3.5-4B fine-tunes on ADP subsets (Babel)

Training + eval campaign comparing ADP data recipes by fine-tuning
**Qwen3.5-4B-Base** on subsets of `neulab/agent-data-collection` and evaluating
on SWE-bench Verified. Runs on CMU's Babel cluster (Slurm, L40S nodes).
Complements `openhands_sdk_training/` (Graham's condenser experiments); this
directory is dfried's environment, motivated by the finding that the condenser
fine-tune regressed vs base (14 vs 25 resolved / 500 at temp 0, with the caveat
that it used a partial checkpoint at step 2000/5322).

wandb project: `adp-smoke` (runs `<name>` for train loss, `val-<name>` for
offline val loss on the same step axis).

## Layout

```
install.sh                  bootstrap the training venv (see "Environment" for corrections)
pretokenize.py              single-rank pre-tokenization into tokenized_path (see gotchas)
eval_checkpoint_loss.py     offline val loss: chunked bf16 cross-entropy, 1 GPU, ~5 min/ckpt
val_loss_follower.sbatch    follower job: evals each new checkpoint before rolling deletion
patch_transformers_fa2_s_aux.py  None-guard for transformers 5.6.0 qwen3_5 FA2 crash
launch_smoke.sh, launch_swesmith_probe.sh  early interactive smoke/probe launchers
DEBUG-swesmith-4b-oom.md    narrative of the debugging that produced this setup
env/                        pip freezes of both venvs + pinned upstream commits
runs/<name>/                per-run sbatch + LLaMA-Factory yaml + DeepSpeed json + val_loss.csv
  swesmith_4b_full/         swe-smith subset (17,380 trajectories), 540 steps/epoch — COMPLETED
  paper_nonweb_4b/          paper openhands_nonweb subset, 1154 steps, 48h-chained
  oom_debug/                16k probe configs from the OOM investigation
swebench/                   SWE-bench Verified eval harness (vLLM serve + OpenHands/benchmarks)
nccl_debug/                 minimal repro + test matrix for the 8-rank NCCL init hang
fa2_build/                  sbatch scripts that built the FlashAttention-2 wheels (sm89 + sm80)
```

## Environment

Live workspace: `~/exp/adp-smoke/` (venv, `hf_cache/`, `datasets/`, run logs,
wandb). Checkpoints go to
`/data/tir/projects/tir1/users/dfried/exp/adp-smoke/runs/<name>/output/` —
never to home (a full home quota breaks the whole cluster account; the sbatch
scripts fail loudly if tir1 is not mounted).

Exact package versions: `env/train_venv_freeze.txt` and
`env/vllm_venv_freeze.txt`. Upstream commits: `env/pinned_commits.txt`
(LLaMA-Factory `main` @ a61cfa6, OpenHands/benchmarks @ 4e5469e).

`install.sh` and `swebench/setup_vllm.sh` are checked in **as originally run**,
and both needed post-hoc CUDA fixes (Babel's driver is CUDA 12.9; default
indexes serve cu130 builds that fail with "driver too old"):

- **torch**: install from `--index-url https://download.pytorch.org/whl/cu128`
  → `torch==2.11.0+cu128` (both venvs).
- **vLLM**: the cu129 variant exists **only** on GitHub releases (PyPI and
  wheels.vllm.ai carry cu130 only):
  `https://github.com/vllm-project/vllm/releases/download/v0.24.0/vllm-0.24.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl`
- **Qwen3.5 requires `flash-linear-attention` + `causal-conv1d`** (24/32 layers
  are gated-delta-net). Without them transformers silently uses a naive fp32
  fallback: 28-minute steps and OOM in backward. The sbatch scripts fail fast
  on `import fla`.
- **FlashAttention-2** had no wheels for torch 2.11 — built from source
  (`fa2_build/`, v2.8.3.post1, sm89 for L40S/L40 and sm80 for A100). The
  transformers 5.6.0 qwen3_5 FA2 path then crashes on `s_aux=None`;
  `patch_transformers_fa2_s_aux.py` adds the guard (idempotent, re-applied by
  each sbatch preamble).

## Reproducing a training run

1. Pre-tokenize once, single rank (HF `datasets` `map(num_proc>1)` deadlocks
   under multi-rank torchrun): the sbatch scripts do this automatically via
   `torchrun --nproc_per_node 1` + the run's `pretok*.yaml`, writing to
   `tokenized_path`. Training then only does `load_from_disk`.
2. `sbatch runs/<name>/submit_*.sbatch`. Scripts are self-contained: fail-fast
   env checks, tir1 output dir, auto-resume from the latest checkpoint, NCCL
   mitigations. For runs longer than the 48h `general` limit, chain:
   `sbatch --dependency=afterany:<jobid> same_file.sbatch`.

Proven recipe (job 9075689, swe-smith): 8× L40S, ZeRO-3, Liger, FLA, sdpa
attention, seq 32768, per-device bs 1 × ga 4 → ~99 s/step, ~15 h/epoch,
~27 GB/GPU. The 4-GPU + FA2 variant (`submit_swesmith_4b_4gpu_fa2.sbatch`) is
preferred for queueability (the `normal` QoS caps at 8 GPUs total).

Validation loss: completed/queued runs used `eval_strategy: "no"` (Liger 32k
eval materializes full logits and HF upcasts them to fp32 — 27 GiB at
32k × 248k vocab) plus the offline follower. **Future runs should instead use
in-training eval**: apply
`../openhands_sdk_training/scripts/patch_llamafactory_liger_eval_skip_logits.py`
and set `eval_strategy: steps`, `per_device_eval_batch_size: 1`,
`prediction_loss_only: true` (already done in
`runs/paper_nonweb_4b/paper_nonweb_4b_seq32768_4gpu_fa2.yaml`).

## SWE-bench eval (`swebench/`)

`setup_benchmarks.sh` clones + `uv sync`s OpenHands/benchmarks;
`setup_vllm.sh` builds the vLLM venv (apply the cu128/cu129 corrections above).
`smoke/run_smoke_eval.sbatch` serves a checkpoint with vLLM (port 8012,
`qwen3_coder` tool parser) and runs the 10-instance smoke set
(`smoke/instances10.txt`, config `smoke/llm_config_vllm.json`). Notes:
flashinfer JIT-compiles at first use and needs `ninja` on PATH (venv `bin/`
first); first boot ≈ 7 min. Scoring uses Apptainer (compute nodes only).

## Cluster gotchas (Babel-specific)

- **8-rank NCCL init hangs on some L40S nodes** (babel-o5-28, n5-32, p5-28):
  all ranks connect rings over SHM, then the first broadcast wedges until the
  ~24 min watchdog SIGABRT. Suspected NCCL 2.28 NVLS partial-alloc bug (fixed
  upstream in 2.29; torch 2.11 ships 2.28.9). Mitigations (in the sbatch
  scripts): `NCCL_NVLS_ENABLE=0` + `--exclude` of those nodes. Minimal repro:
  `nccl_debug/node_o5_28_repro/`.
- Interactive `debug` jobs default to ~8 GB RAM — loading the 4B model needs
  `--cpus-per-task>=16 --mem>=200G`.
- Always `TOKENIZERS_PARALLELISM=false`.
- `flash_attn: sdpa` in the train yaml unless the s_aux patch is applied.
