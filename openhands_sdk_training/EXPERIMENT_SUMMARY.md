# Summary of Graham's ADP / OpenHands SDK training experiments

Compiled 2026-07-09 from `README.md`, `CURRENT_EXPERIMENTS.md`, `configs/`, `slurm/`,
`run_history/slurm_adp_jobs_2026-06-01.tsv`, `swe-bench-smoke-2000/`, and `swe-bench-full-4b/`.

## Starting models: all BASE, no instruct

Every run started from a base (non-instruct) checkpoint, fine-tuned with LLaMA-Factory
full-parameter SFT using the `qwen3_5_nothink` template:

- **Qwen3.5-0.8B-Base** — the original proof-of-life line
- **Qwen3.5-4B-Base** — the main production line
- **Qwen3.5-9B** — no "-Instruct" suffix, so also base
- **Qwen3.5-35B-A3B** (MoE, ~4B active params) — local checkpoint on the FLAME cluster, also base-style

## Training data

- **0.8B:** the ADP-paper-style "openhands_nonweb" mixture (AgentTuning, CodeActInstruct,
  SWE-Gym, SWE-Smith, Code Feedback, Orca AgentInstruct, etc.), ~40k examples after cleaning.
- **4B / 9B:** the "full_condenser_24k_all_records" set — OpenHands SDK condenser SFT
  trajectories at 24k context, ~170,458 train records / 500 eval records, drawn from
  ~107k unique source trajectories. The 35B MoE smokes used a "v2" variant of the same family.

## Hardware and duration

- **0.8B:** first on a local AMD ROCm box (Radeon 8060S), then a 2-minute smoke on Babel.
  Proof-of-life only.
- **4B:** Babel `general` partition, 4x L40S single node, seq len 32768, ZeRO-3 +
  flash-attn + Liger, batch 1 x grad-accum 8, lr 1e-5. The production run hit the 2-day
  partition TIMEOUT at step 1697 of 5322 (~32% of one epoch) — a full epoch needs well
  over 2 days on 4x L40S. Checkpoint-1500 (eval_loss 0.230) and checkpoint-2000 are what
  got evaluated.
- **9B:** same recipe on 4x A100_80GB; still running at the doc snapshot (~1d 16h elapsed),
  no completion recorded.
- **35B-A3B:** FLAME cluster, 2 nodes x 8 H100 — only 10–25-minute parallelism smokes
  (DeepSpeed vs Megatron-Core, best ~17.4k tok/s/GPU); no full production run recorded.

## Eval results (4B checkpoint-2000, SWE-bench Verified)

**Smoke (10 instances):** 2/10 resolved. Of the first 5, only 2/5 produced non-empty
patches — both applied and both resolved. Of the next 5, 3 patches applied cleanly but
none resolved.

**Full run (500 instances), fine-tuned vs. base — the headline finding is that
fine-tuning *hurt*:**

| Run (temp 0.0) | Resolved | Non-empty patches | % of non-empty | % of all 500 |
|---|---:|---:|---:|---:|
| Qwen3.5-4B-Base (raw) | **25** | 96 | 26.0% | 5.0% |
| Fine-tuned ckpt-2000 | **14** | 79 | 17.7% | 2.8% |

At temp 1.0 (partial runs) both resolved 13, but the fine-tune's per-patch hit rate was
still lower (8.9% vs 15.5% of non-empty patches).

**Patch application rates:** the dominant failure mode at temp 0.0 was *empty patches* —
84% of fine-tuned outputs and 81% of base outputs produced no patch at all. But patches
that were non-empty applied very reliably: essentially all scored non-empty patches
applied cleanly (0 scoring errors across all four runs; smoke run 5/5 applied). So the
bottleneck isn't malformed diffs — it's the model failing to produce a patch, and
secondarily patches that apply but don't fix the tests.

Two caveats: the fine-tuned checkpoint was only ~38% through one epoch when the timeout
hit, so this isn't a verdict on the full recipe; and the smoke docs note native vLLM tool
parsing (qwen3_coder parser) materially improved patch quality over text-only mode.
