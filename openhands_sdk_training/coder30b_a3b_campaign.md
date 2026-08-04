# Qwen3-Coder-30B-A3B SWE SFT campaign (FAIR A100)

A separate campaign from the adp-v2 souping study: full-parameter SFT of a ~30B-class MoE for
stronger SWE-bench performance, run on the legacy FAIR on-prem cluster (learnfair, A100-80GB).
This doc captures the setup, the (substantial) infra findings, and the eval outcome so the
hard-won details aren't lost. Dates 2026-07 → 2026-08.

## TL;DR

- **Model trained and saved.** `Qwen/Qwen3-Coder-30B-A3B-Instruct` (text MoE, `Qwen3MoeForCausalLM`,
  128 experts / 8 active, ~30B total / ~3B active), full-param SFT, 1 epoch on the pooled220k SWE
  union (220K records, 32k ctx). Final model + a preserved mid-checkpoint on `/checkpoint`.
- **Chosen over Qwen3.5-35B-A3B** (which is a multimodal MoE w/ linear attention + MTP head that
  Graham smoked ~26× without a production run). Qwen3-Coder-30B-A3B is plain text MoE, mature,
  code-specialized + instruct — easiest reliable path and best SWE bet at MoE scale.
- **Multi-node is blocked** by a FAIR RoCE fabric issue (below) → trained **single-node**.
- **No clean SBV-500 number**: the 30B Coder "gets stuck" (no-progress loops) on a large fraction of
  instances on the 4B-tuned OpenHands eval scaffold. Ruled out concurrency and turn-timeout; the
  **base model loops too** → base-30B × scaffold mismatch, not a clean SFT regression. Eval parked.

## Model / data / recipe

- Init: `Qwen/Qwen3-Coder-30B-A3B-Instruct` (fetched to the HF cache; ~57 GB, 16 shards).
- Data: `pooled220k` SWE union (`/checkpoint/dpf/adp-data/v2_swe_subsets/pooled220k`), template
  `qwen3_nothink`, cutoff_len 32768, no packing.
- Recipe: full SFT, **ZeRO-3 no offload**, FA2 + Liger (`apply_liger_kernel_to_qwen3_moe`), bf16,
  gradient-checkpointing, **LR 1e-5** cosine / warmup 0.03, global batch 32 (8×A100 × micro 1 ×
  accum 4), 1 epoch = 6875 steps. LR validated in-run (loss ↓, grad_norm ~0.4, no spikes).
- **Loss is NOT comparable to the Qwen3.5-4B arms** — different tokenizer/vocab (151,936 vs 248,320).
  Cross-model comparison must use SBV resolved-rate, not loss.

### Paths (FAIR)
- Run dir / configs / logs: `/private/home/dpf/exp/adp/runs/coder30ba3b_pooled220k_a100/`
  (`train_prod.yaml`, `pretok_prod.yaml`, `submit_prod.sbatch`).
- Output + checkpoints: `/checkpoint/dpf/adp-runs/coder30ba3b_pooled220k_a100/output/`
  (final model saved at the output root at completion).
- Preserved eval-ready snapshot (step-4000, weights-only, rotation-safe):
  `/checkpoint/dpf/adp-runs/coder30ba3b_pooled220k_a100/preserved/checkpoint-4000-eval/`.

## Infra findings (reusable)

### Single-node throughput (8×A100, gb16 smoke, 32k)
- ZeRO-3 **+ optimizer offload**: fits ~16.7 GB/GPU, **~44 s/it** (CPU-Adam over 30B is the bottleneck).
- ZeRO-3 **no offload** (production): **~21 s/it**, peak **74.5 GB/80 GB** (tight, holds at true 32k),
  ~2× faster — optimizer-on-GPU. Full epoch ≈ 55–75 h (vs the 4B's ~48 h). Model load from NFS is slow
  (~9 min: 8 ranks mmap-read the 57 GB checkpoint; silent, GPUs idle — not a hang).

### Multi-node RoCE — BLOCKED (needs FAIR fabric config)
On `ampere80gb` nodes, multi-node NCCL needs (see also the FAIR RoCE ticket at
`~/devfair-documents/fair-roce-ticket.md`):
- `export NCCL_SOCKET_IFNAME=front0` — the nodes expose `lo`/`front0`/`docker0`; **docker0 is
  172.17.0.1 on every node**, so auto-pick makes ranks resolve peers to themselves → hang.
- Let NCCL auto-detect HCAs — only **mlx5_2 + mlx5_5** are live RoCE (200 Gb/s); forcing
  `NCCL_IB_HCA=mlx5` tries 8 dead HCAs → hang.
- **Remaining blocker:** small (16 MB) cross-node all-reduce works, but **≥~1 GB transfers stall**
  (lossless/PFC not configured). `NCCL_IB_TC=106` partially unblocked (2 GB completed once at ~0.1 GB/s,
  then hung) → unusable for ZeRO-3. Needs the switch's lossless DSCP/priority + PFC from FAIR support.
- SLURM launch that works: `--nodes=N --ntasks-per-node=1 --gres=gpu:ampere:8`; `FORCE_TORCHRUN=1
  NNODES=$SLURM_NNODES NPROC_PER_NODE=8 MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST|head -1)`
  then `srun --ntasks-per-node=1 bash -c 'export NODE_RANK=$SLURM_NODEID; llamafactory-cli train …'`.
  Pre-tokenize once on the batch node before the srun. NCCL debug goes to stdout.

### Requeue / resume gotcha
72h-wall USR1 self-requeue + resume-picker (highest checkpoint with `trainer_state.json` + optimizer
state) worked, BUT the first resume FAILED ~16 min in: loading the ~342 GB ZeRO-3 optimizer state from
NFS starved the NCCL heartbeat thread → **false-positive watchdog abort** (`watchdog got stuck 480s`).
**Fix: `export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600`** (only bites on resume). Also: an interrupted
mid-save checkpoint lacks `trainer_state.json`, so the picker correctly skips it and resumes from the
prior complete one (LR schedule intact).

## SBV eval — parked, with a characterized reason

Adapted the shared 4B-tuned kit (`/checkpoint/dpf/swebench-eval`) for the 30B via args/env only (no
edits to shared scripts): `TP=2` + `--gres=gpu:2` (~60 GB weights), `PREFIX_CACHE=off` (drops a
mamba-only flag), `qwen3_coder` tool parser (native), distinct tags. vLLM 0.25.1 serves
`Qwen3MoeForCausalLM`. A 1-instance smoke passed (patch applied + scored).

**At scale it doesn't hold up:** ~74% of instances error with OpenHands' "**Remote conversation got
stuck**" (no-progress loop); only ~26% complete (those DO produce valid patches). Diagnosis:
- NOT concurrency — NW=1 also stuck.
- NOT the turn timeout — 600 → 2400 s unchanged (~70% stuck).
- **Base-model control** (`Qwen3-Coder-30B-A3B-Instruct`, same 10-inst smoke): 4/10 clean completions
  then **looped ≥1 h on the rest** → the base loops too, slower-onset.

**Conclusion:** largely a **base-30B-Coder × OpenHands-scaffold mismatch** (the kit was tuned for the
4B); our SFT may worsen it by degree but did not cleanly "break" a fine base. A valid SBV number needs
the agent-scaffold interaction fixed (stuck-detector / max-turns / tool-format / harness for 30B), which
is a separate infra effort. Note: pooled220k has SWE-bench-Verified contamination, so any SBV number
would be inflated (but apples-to-apples vs the arms).

## Status / next steps
- Model: **trained + saved**, ready to use.
- Eval: **parked**; to get a number, adapt the agent harness for 30B (or use a different SWE scaffold).
- Kit throughput note for a future clean run: the 30B agent rollouts are ~10-20× slower than the 4B
  (~1.5 inst/hr/shard) — use MANY more shards (few instances each) so a shard finishes within its 12h
  walltime, or expect catch-up rounds.
