# Debug brief: OOM crash of `adp-swesmith-4b-32k` (Slurm job 9035366)

You are in an interactive debug session on babel-p5-32 (L40S node, debug partition —
max 2 GPUs, 12h). Your task: diagnose and fix the OOM so the run below can be
resubmitted successfully. Cluster-wide context (partitions, GPU inventory, filesystem
rules) is in `~/.claude/CLAUDE.md` — read it if you haven't.

## What happened

Full SFT of Qwen/Qwen3.5-4B-Base at 32k context on 4× L40S (48GB, node babel-s5-28),
LLaMA-Factory + DeepSpeed ZeRO-3, Liger kernels on, `flash_attn: sdpa`, micro-batch 1,
grad-accum 8, gradient checkpointing on.

- Step 1/540 took **28min 27s** (projected 255h — itself a problem; TimeLimit was 2 days).
- At the end of step 1, **rank 3 hit CUDA OOM allocating 514MB** — GPU had 44.39GiB
  capacity, 43.31GiB already allocated by PyTorch, only 346MB free. Other ranks then
  got SIGTERM. wandb shows the run as "crashed".
- GPU monitor log: all 4 GPUs plateaued at ~40.8GB used, 100% util, for the entire
  first step. The job died at the **first optimizer step** — the classic signature of
  Adam states + fp32 master weights materializing on first `step()`.

## Artifacts (all under `/home/dfried/exp/adp-smoke/`)

- Training config: `runs/swesmith_4b_full/swesmith_4b_seq32768_train.yaml`
- Submit script:   `runs/swesmith_4b_full/submit_swesmith_4b.sbatch`
- stderr log (OOM traceback ~line 1088–1166): `runs/swesmith_4b_full/logs/adp-swesmith-4b-32k-9035366.err`
- stdout log:      `runs/swesmith_4b_full/logs/adp-swesmith-4b-32k-9035366.out`
- GPU memory/util timeline: `runs/swesmith_4b_full/logs/gpu_monitor_9035366.log`
  (columns: gpu_index, mem_used_MiB, mem_free_MiB, util%)
- LLaMA-Factory checkout: `.cache/LLaMA-Factory/`; venv: `.venv/` (uv-managed)
- Pre-tokenized data: `datasets/swesmith_full/tokenized_qwen35_4b_seq32768`

## Analysis so far (verify, don't assume)

ZeRO-3 over 4 GPUs should put ~16GB/GPU of param/grad/optimizer state for a 4B model,
yet 40.8GB was in use *before* optimizer states existed → ~25GB unaccounted, likely
activations at 32k + sdpa attention workspace + ZeRO-3 gather buffers. Things to check:

1. Is gradient checkpointing actually active? (HF + Liger patch interactions can
   silently disable it; look for the "use_cache=True is incompatible" warning or
   measure activation memory directly.)
2. DeepSpeed z3 config knobs: `stage3_max_live_parameters`, `stage3_prefetch_bucket_size`,
   `reduce_bucket_size` — defaults hold large gather/reduce buffers.
3. `flash_attn: sdpa` — fa2 is not built in this venv (see comment in the yaml).
   At 32k, FA2 saves real memory and a lot of time. Building it is a candidate fix.
4. The 28min step: profile whether it's PCIe all-gather bound (no NVLink on L40S)
   or dataloader/NFS bound (tokenized data lives on /home NFS).

## Reproduction protocol (this session has only 2 GPUs — don't try the full 4-GPU run)

- Repro the memory profile small: same config but `--nproc_per_node 2`, and/or
  `cutoff_len` 8192/16384, `max_steps: 2`, and log `torch.cuda.max_memory_allocated()`
  per phase. Scaling activations vs. state memory across those runs will separate
  the two. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is worth setting always.
- A 2-GPU/short-context run that survives step 2 proves the first-optimizer-step
  theory; then fixes only need to buy ~2–3GB/GPU at full scale — but prefer fixes
  with margin.

## Candidate fixes, ranked (from prior analysis with the user)

1. **Resubmit on 4× A100_80GB** (`--gres=gpu:A100_80GB:4`, general partition,
   babel-v5 nodes): removes both the memory ceiling and the PCIe bandwidth problem.
   Consider ZeRO-2 there for speed.
2. **8× L40S** (`--gres=gpu:L40S:8`, general): halves ZeRO-3 shards (~8GB/GPU freed);
   activations don't shard, so verify with the scaling measurements first.
3. Stay on 4× L40S + **optimizer CPU offload** (`ds_z3_offload_config.json`):
   fits for sure, but step time is already terrible — measure before committing.
4. Orthogonal wins regardless: build flash-attn 2; expandable_segments; shrink z3
   bucket sizes; consider 8-bit optimizer (halves optimizer shards).

Constraints: `general` is sbatch-only (max 2 days, 8 GPUs); interactive work only in
`debug` (2 GPUs). Long runs should checkpoint and auto-resume. Fastest-throughput
option if queue allows: A100_80GB. When resubmitting, keep `save_steps` low enough
that a 2-day limit can't eat >100 steps of progress.

## Unrelated but FYI

The user's previous interactive session (job 9029980, babel-w5-28) was cancelled at
21:26 by the user's own UID — possibly a stray `scancel` (or one issued by the Claude
session running there) while clearing a pending job stuck on QOSMaxGRESPerUser.
If you are asked to cancel jobs: always `scancel <specific-jobid>`, never
`scancel -u dfried`, and never cancel the job your own session is running inside.

---

## RESOLVED (session 2, babel-p5-32, 2026-07-06 ~22:00)

**Root cause — not what the analysis above assumed.** The OOM traceback bottoms out in
Qwen3.5's `linear_attn` (gated-delta-net) path: `hidden_states * F.silu(gate.to(torch.float32))`
during checkpoint recompute in backward. Qwen3.5-4B is a hybrid: **24 of 32 layers are
linear-attention**, and without `flash-linear-attention` + `causal-conv1d` installed,
transformers uses a naive torch fallback ("The fast path is not available..." warning,
present in the crashed job's log at .err:1167). The fallback materializes large fp32
intermediates (the 514MB alloc) AND is the reason for the 28-min steps. `flash_attn: sdpa`
vs fa2 was a red herring — it only affects the 8 full-attention layers.

**Proof (2× L40S, 16k, ZeRO-3, 3-step A/B, runs/oom_debug/):**
- fallback: 154.7s/step, OOM right after step 1 (optimizer-state alloc pushed past 44GB)
- FLA:      ~45-63s/step steady (3.4×), all 3 steps + optimizer step OK, loss identical
  (0.2859 vs 0.2864); peak 45.0GB — still tight on 2 GPUs, fine on more.
- Gradient checkpointing WAS active; the "first optimizer step" theory was wrong for the
  4-GPU crash (it died in backward), but real as a second cliff (proved by the 2-GPU probe).

**Fix applied:** `uv pip install flash-linear-attention causal-conv1d` into `.venv`
(wheels, no build). Resubmitted as **job 9043143** on **8× L40S** (ga 8→4, effective batch
still 32; per-GPU ZeRO-3 state halved; est. 240-300s/step → 36-45h < 48h limit; 4×L40S
even with FLA projected ~85h, and A100_80GB queue had 18 pending jobs). Also: save_steps
50, TRITON_CACHE_DIR moved off NFS, and the sbatch now fails fast if `import fla` breaks.
