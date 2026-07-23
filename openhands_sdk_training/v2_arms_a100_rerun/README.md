# adp-v2 4-arm SFT rerun on the A100 cluster (clean LR schedules)

Self-contained kit to rerun the four adp-v2 SWE data-recipe training arms
(**coderforge / scale / rebench / swezero**) from a fresh checkout of this repo on the
new cluster (32 nodes × 8× A100), fixing the learning-rate integrity problems that
confounded the original Babel (CMU) campaign of 2026-07-13..22.

Read this whole file before running anything. The agent-facing hard rules live in
[CLAUDE.md](CLAUDE.md) in this directory.

## 1. What these experiments are

Four full-SFT arms of `Qwen/Qwen3.5-4B` (instruct), one per adp-v2 SWE data source,
identical recipe, differing ONLY in training data. Purpose: a SWE-bench data-recipe
sweep plus the headline **verification contrast** (coderforge = resolution-verified vs
swezero = unverified, both distilled from the same Qwen3-Coder-480B teacher), feeding a
model-soup coefficient search over the four task vectors.

| arm | adp-v2 config (HF `neulab/adp-v2`) | teacher | verified? |
|---|---|---|---|
| coderforge | `coderforge_preview` | Qwen3-Coder-480B | yes |
| scale | `scale_swe_distilled` | DeepSeek v3.2 | yes |
| rebench | `nebius_SWE-rebench-openhands-trajectories` | Qwen3-Coder-480B | yes (+regr tests) |
| swezero | `nvidia_SWE-Zero-openhands-trajectories` | Qwen3-Coder-480B | **no** |

Recipe constants (do not change — comparability with the Babel campaign's anchors
depends on them): seq/cutoff 32768, LLaMA-Factory `template: qwen3_5_nothink`, OpenAI
tool-calling format (matches the OpenHands SDK eval harness — do NOT convert to
sharegpt), 55,000 records/arm (`max_samples` at pretokenize), 1 epoch, **global batch
32** → 1719 optimizer steps, peak LR 1e-5, cosine to 0, warmup_ratio 0.03, bf16,
ZeRO-3 + FlashAttention-2 + Liger, gradient checkpointing.

Babel reference points: untrained base = 25/500 on SWE-bench Verified;
paper-nonweb 24k run = 52/500; the Babel arm evals were still in flight when this kit
was written (see `analysis/adp_v2_data_analysis_report.md` for the soup plan, §6).

## 2. Why rerun: the LR-schedule problem

The Babel arms ran with `save_only_model: true` (no optimizer/scheduler state in
checkpoints). Every preemption/crash resume therefore restored weights + step counter
only, and HF Trainer silently started a **fresh warmup + full-length cosine from the
resume step**. Consequences observed:

- Per-arm LR trajectories became piecewise fresh cosines keyed to each arm's individual
  preemption history — schedules NOT matched across arms. Only coderforge ran clean.
- swezero's late (step ~1180) restart re-warmed to 1e-5 and never fully annealed —
  directly confounding the headline coderforge-vs-swezero contrast (it was salvaged
  with a hand-built WSD tail, but that's a patch, not a clean run).
- Val loss was shown to be dominated by instantaneous LR state, not data quality
  (same weights 25 steps apart spanned nearly the whole base→best val range across a
  reset). Cross-arm checkpoint comparisons are only valid at matched, properly
  annealed LR.

**Fixes baked into this kit** (already in the generated configs/sbatch — listed so you
know what NOT to undo):

1. `save_only_model: false` from step 0 — checkpoints carry full optimizer+scheduler
   state (~70 GB each at 4B/ZeRO-3; `save_total_limit: 2`, `save_steps: 100`).
2. The sbatch resume picker only accepts checkpoints that have `trainer_state.json`
   (partial saves from mid-save kills crash-looped the Babel runs) **and refuses to
   resume from a model-only checkpoint** (hard exit instead of a silent LR reset).
3. Identical schedule across arms by construction: same global batch (8 GPUs ×
   bs 1 × grad-accum 4 = 32), same steps, same cosine. Explicit `seed: 42`.
4. If an arm's schedule ever gets confounded anyway (e.g. someone resumes it wrong),
   the correct action is **restart that arm from scratch** — optimizer state is not
   reconstructable retroactively.

LR-integrity detector (run it after every resume): an LR drop >2× between consecutive
loss rows of `<output_dir>/trainer_log.jsonl` means a schedule reset happened.

## 3. Cluster adaptation (do this first)

This kit was written on CMU Babel (Slurm) without access to the new cluster; verify
each assumption:

- **Scheduler**: templates are Slurm. Check partition/account/QoS names
  (`sinfo`, `sacctmgr show user $USER withassoc`) and pass them to
  `generate_arm_runs.py` (`--partition`, `--account`, `--gres`, `--time`). If it is not
  Slurm, the sbatch bodies are plain bash after the `#SBATCH` header — port the header.
- **A100 variant**: recipe is sized for 80 GB. On 8-GPU ZeRO-3 + Liger + grad-ckpt a
  4B/32k run used ~27 GB/GPU on Babel, so 40 GB A100s likely also work — verify with a
  short smoke run before launching all arms.
- **Storage**: pick a bulk filesystem for checkpoints (each arm holds ≤ ~140 GB steady
  state with limit 2, plus 8 GB final model) and for datasets (~32 GB total). Do NOT
  put checkpoints on a small home quota — on Babel a full home broke SSH cluster-wide.
- **GPU-hours**: 55k records/arm ≈ 1.7 days on 4×L40S ≈ (rough guess) 12–24 h on one
  8×A100 node. All four arms fit on 4 of the 32 nodes simultaneously.
- **Network on compute nodes**: if none, pre-download the model and datasets from a
  login node first (`HF_HUB_OFFLINE=1` is set in the sbatch); set `WANDB_MODE=offline`
  if wandb can't reach the internet (sync later with `wandb sync`).
- **NCCL**: single-node 8-GPU on NVLink'd A100s needs no special flags. (Babel's
  `NCCL_NVLS_ENABLE=0` workaround was for a NCCL 2.28 bug on PCIe L40S nodes — not
  carried over. If 8-rank init hangs at the first collective, that bug is the suspect;
  set `NCCL_NVLS_ENABLE=0` and/or bump `nvidia-nccl-cu12>=2.29`.)

## 4. Environment setup

```bash
export ADP_ENV_ROOT=/path/to/bulk/adp-env     # venv + LLaMA-Factory + hf_cache live here
bash setup_env.sh                              # idempotent; ~15 min + flash-attn build
```

`setup_env.sh` pins the exact stack the Babel campaign validated (see §8 manifest):
torch 2.11.0+cu128, transformers 5.6.0, LLaMA-Factory @ `a61cfa69` (2026-07-03 main —
has the qwen3_5 templates), deepspeed 0.19.2, liger-kernel 0.8.0, flash-attn
2.8.3.post1 (source build, sm80), flash-linear-attention 0.5.1 + causal-conv1d
1.6.2.post1 (**required** — 24/32 Qwen3.5 layers are gated-delta-net; without `fla` you
get a silent 17× slowdown and fp32-intermediate OOMs), and applies
`patch_transformers_fa2_s_aux.py` (transformers 5.6.0 FA2 crashes on Qwen3.5's
`s_aux=None` without it; re-run after any transformers reinstall).

Torch CUDA flavor: the script installs from the cu128 index (works on driver ≥ 12.8,
which any A100 cluster with recent drivers has). Check `nvidia-smi` first; override
with `TORCH_INDEX_URL` if the driver is older.

## 5. Data

Training data = deterministic 80k-record reservoir sample (seed 0) per config from HF
`neulab/adp-v2`, converted to LLaMA-Factory OpenAI format by the ADP
`sft_to_llamafactory` adapter, with a 200-record eval carve-out. Two ways to get it:

**Option A — transfer from Babel (preferred: bit-identical data).** From Babel, the
four finished subset dirs are at
`/data/tir/projects/tir1/users/dfried/adp-smoke/datasets/v2_swe_subsets/<config>/`
(~8 GB each; tir1 is mounted on compute nodes only). Minimum needed per config:
`train.llamafactory.jsonl`, `eval.llamafactory.jsonl`, `dataset_info.json`. The
`tokenized_qwen35_4b_inst_seq32768/` Arrow caches can come too (skips pretokenize) or
be rebuilt in ~1 h.

```bash
rsync -avP dfried@<babel-compute-path>:/data/tir/.../v2_swe_subsets/ $DATA_ROOT/v2_swe_subsets/
```

**Option B — rebuild from HF.** `../scripts/build_v2_swe_subsets.py` (this repo) does
the whole pipeline. Same seed → same sample **as long as the HF repo's shard listing
hasn't changed since 2026-07-13** — spot-check a few record hashes against Babel if
exactness matters. Before running, edit its two module constants `ADP_REPO` (a clone of
the agent-data-protocol repo, for the adapter) and `PYTHON` (this venv's python), and
pass `--out-root $DATA_ROOT/v2_swe_subsets`. Needs `HF_TOKEN`.

## 6. Generate run dirs and launch

```bash
python generate_arm_runs.py \
  --env-root  $ADP_ENV_ROOT \
  --data-root $DATA_ROOT/v2_swe_subsets \
  --out-root  $CKPT_ROOT/v2_arms_a100 \
  --runs-root $ADP_ENV_ROOT/runs \
  --partition <partition> [--account <acct>] [--gres gpu:a100:8] [--time 2-00:00:00] \
  [--wandb-project adp-v2-a100]
# then, per arm:
sbatch $ADP_ENV_ROOT/runs/v2_<arm>_inst_4b_a100/submit.sbatch
```

Each run dir gets `pretok.yaml` (single-rank pre-tokenization — tokenizing under
multi-rank torchrun deadlocks), `train.yaml`, and `submit.sbatch` (phase 1 pretok →
phase 2 8-GPU train, auto-resume, GPU monitor sidecar). The sbatch is requeue-safe:
preemption/timeout requeues resume **exactly** (full-state checkpoints), which is the
whole point of this rerun.

**Smoke first**: before the 4 real launches, run one arm with `--smoke` (adds
`max_steps: 30` override and a `_smoke` suffix) and check: it reaches step 30, loss is
finite and ~1.x early, sec/step is sane, `nvidia-smi` shows all 8 GPUs busy, and a
checkpoint written at step 25 contains `trainer_state.json` **and** a `global_step*/`
dir (= optimizer state present).

## 7. Monitoring and integrity checks (per arm)

- Progress: `tail <output_dir>/trainer_log.jsonl` — fields include step, loss, lr,
  ETA. Sidecar GPU log in the run dir's `logs/`.
- After ANY resume: confirm the first post-resume `lr` continues the cosine (compare
  to the row before the kill). >2× LR discontinuity = schedule reset = stop and
  investigate; do not let it train on.
- Completion marker: sbatch logs `== finished=... exit=0 ==`; final checkpoint is
  `checkpoint-1719` with `epoch: 1.0` in `trainer_state.json`.
- If a job dies mid-checkpoint-save it leaves a partial `checkpoint-N` (no
  `trainer_state.json`). The resume picker skips these automatically, but move them to
  a `quarantine/` sibling dir to keep the output clean.
- Watch the first ~50 steps of each arm for loss ~0.9→0.4 trajectory (Babel arms all
  did this); wildly different means a data/template problem, stop early.

## 8. Version manifest (validated on Babel, 2026-07)

```
python 3.12 (uv venv)          llamafactory 0.9.6.dev0 @ a61cfa692a70fcced4ba32a846d1e2de95f2865e
torch 2.11.0+cu128             transformers 5.6.0 (+ s_aux patch)
deepspeed 0.19.2               accelerate 1.11.0
flash-attn 2.8.3.post1 (sm80)  flash-linear-attention 0.5.1
causal-conv1d 1.6.2.post1      liger-kernel 0.8.0
datasets 4.0.0                 triton 3.6.0
wandb 0.28.0                   huggingface-hub 1.22.0
```

## 9. After training

The deliverable is 4 final checkpoints (`checkpoint-1719`) with **matched, clean,
fully-annealed LR schedules**. Downstream (SWE-bench Verified eval via the OpenHands
SDK harness, the model-soup coefficient search, and the confirmation mix train) is
specced in `../analysis/adp_v2_data_analysis_report.md` §6 and currently lives on
Babel (`~dfried/exp/adp-smoke/swebench/`) — port or ship checkpoints back as decided
later. Keep each arm's full `trainer_log.jsonl` and wandb run; they are the evidence
the schedules were clean.
