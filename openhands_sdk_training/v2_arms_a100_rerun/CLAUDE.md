# Agent guide: adp-v2 4-arm rerun (A100 cluster)

You are rerunning four SFT training arms whose original runs were confounded by
learning-rate schedule resets. **Read README.md in this directory end-to-end before
acting.** These are the non-negotiable rules; rationale is in the README.

## Hard rules

1. **LR integrity is the entire point of this rerun.**
   - `save_only_model` must stay `false` in every train.yaml.
   - Never resume from a checkpoint that lacks `trainer_state.json` or a
     `global_step*/` (optimizer state) directory. The generated sbatch enforces this —
     do not weaken it.
   - After every resume, verify LR continuity in `trainer_log.jsonl` (a >2× jump/drop
     between consecutive rows = schedule reset). If a schedule reset ever happens,
     restart that arm from scratch; it is not repairable.
2. **Do not change the recipe** (model, template, cutoff 32768, 55k samples, 1 epoch,
   peak LR 1e-5 cosine, warmup_ratio 0.03, seed 42, **global batch 32**). If you change
   GPU count per job, adjust `gradient_accumulation_steps` to keep global batch = 32
   (nproc × bs1 × ga = 32); otherwise step count and schedule stop matching the
   campaign anchors.
3. **Checkpoints go to bulk storage only** — never to a quota-limited home filesystem.
   Verify free space (need ~150 GB per arm headroom) before every launch.
4. Keep the training data in **OpenAI tool-calling format** (as built) — do not
   convert to sharegpt; the eval harness parses the OpenAI format.
5. One smoke run (`--smoke`) before launching the four real arms. Verify the smoke
   checkpoint contains optimizer state.
6. All four arms must run the **same code, same configs (except dataset paths), same
   schedule**. Don't "improve" one arm mid-campaign; note issues and apply fixes to
   all arms or none.

## Environment gotchas (each cost debugging time on Babel)

- `flash-linear-attention` + `causal-conv1d` are mandatory for Qwen3.5 (gated-delta-net
  layers). The sbatch fails fast if missing — if you see transformers' "fast path is
  not available" warning, stop; you're on the 17×-slower fallback.
- transformers 5.6.0 FA2 needs `patch_transformers_fa2_s_aux.py` after every
  transformers (re)install, or training crashes with
  `'NoneType' object has no attribute 'to'`.
- Tokenization must happen in the single-rank pretokenize phase (phase 1 of the
  sbatch). Multi-rank tokenization deadlocks. `TOKENIZERS_PARALLELISM=false` always.
- torch wheel CUDA flavor must match the driver (`nvidia-smi` top line). cu128 build
  is known-good on driver 12.9.
- `HF_HUB_OFFLINE=1` is set during training — prefetch `Qwen/Qwen3.5-4B` (setup_env.sh
  does) or the job dies at model load.

## When something breaks

Diagnose from the arm's `logs/*.err`, `trainer_log.jsonl`, and the GPU monitor log
before resubmitting. Known failure signatures and their fixes are in README §7 and the
gotchas above. Do not delete checkpoints to "clean up" a stuck run — quarantine
(rename) them so the failure remains inspectable.
