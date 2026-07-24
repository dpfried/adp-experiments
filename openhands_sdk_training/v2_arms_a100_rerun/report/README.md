# adp-v2 4-arm SFT — A100 rerun report

*Compiled 2026-07-24 from the `v2_arms_a100_rerun` kit and the wandb project
`dfried/adp-v2-a100` (runs created 2026-07-23, all `finished`).*

## 1. What we did

Full-parameter supervised fine-tuning of **`Qwen/Qwen3.5-4B` (the *instruct* model, not
`-Base`)** on four different adp-v2 SWE data recipes — one training arm per data source,
**identical recipe, differing only in the training data**. This is the clean A100 rerun
of the Babel (CMU) campaign, whose original runs were confounded by learning-rate
schedule resets on preemption (fixed here with full-state checkpoints; see §6).

Purpose: a SWE-bench data-recipe sweep plus a **verified-vs-unverified *sources*
comparison** (coderforge = resolution-verified vs swezero = unverified, both distilled
from the same Qwen3-Coder-480B teacher), feeding a downstream model-soup coefficient
search over the four task vectors.

> **Confound caveat on the coderforge/swezero comparison.** These two arms share a
> teacher but are otherwise *entirely different pipelines* — different task/repo
> populations, prompts, and filtering. Resolution-verification is only one of several
> differences, so any downstream gap between them **cannot be attributed to
> verification alone**. A clean, causal verification contrast (same pipeline with
> verification toggled on/off) is listed as future work (§10).

> **Base-model note.** Earlier ADP "condenser" experiments used Qwen3.5 **base**
> checkpoints (`Qwen3.5-4B-Base`, `0.8B-Base`, `9B`, `35B-A3B`). This v2 campaign is
> different: it fine-tunes the **instruct** `Qwen/Qwen3.5-4B`. Confirmed from the run
> config (`model_name_or_path: Qwen/Qwen3.5-4B`, `model_type: qwen3_5`) and the run/dir
> naming (`v2_<arm>_inst_4b_a100`).

## 2. The four arms

All four pull from HF `neulab/adp-v2`, distilled agentic SWE trajectories in the
OpenHands SDK format:

| arm | adp-v2 config | teacher | resolution-verified? |
|---|---|---|---|
| **coderforge** | `coderforge_preview` | Qwen3-Coder-480B | yes |
| **scale** | `scale_swe_distilled` | DeepSeek v3.2 | yes |
| **rebench** | `nebius_SWE-rebench-openhands-trajectories` | Qwen3-Coder-480B | yes (+ regression tests) |
| **swezero** | `nvidia_SWE-Zero-openhands-trajectories` | Qwen3-Coder-480B | **no** |

## 3. Training data

- **Source & sampling:** per config, a deterministic 80k-record reservoir sample
  (seed 0) from `neulab/adp-v2`, converted to LLaMA-Factory **OpenAI tool-calling
  format** by the ADP `sft_to_llamafactory` adapter (kept in OpenAI format — *not*
  sharegpt — so the OpenHands SDK eval harness can parse it).
- **Records actually trained:** capped at **`max_samples: 55,000` per arm** at
  pre-tokenization.
- **Context length:** trajectories tokenized at **`cutoff_len: 32768`**, template
  **`qwen3_5_nothink`**.
- **Eval carve-outs (logged every 50 steps as `eval_<name>_loss`):**
  - **`v2_id_mix`** (**400 records**) — an *in-distribution* validation mixture
    (held-out trajectories pooled across the adp-v2 SWE families the arms train on).
  - **`v2_swegym_ood`** (**200 records**) — an *out-of-distribution* held-out set
    (SWE-Gym), not part of any arm's training source, used as a generalization probe.

  Eval sets are baked into the tokenized cache at pre-tokenize time and are **identical
  across all four arms**, so the curves are directly comparable.

## 4. Training recipe (identical across all four arms)

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen3.5-4B` (instruct), full-parameter SFT |
| Template / format | `qwen3_5_nothink`, OpenAI tool-calling |
| Loss masking | assistant/response tokens only (LLaMA-Factory default: `train_on_prompt` unset → prompt, user, and tool-observation tokens are masked out of the loss). Both train and eval loss use this masking, so the curves measure per-assistant-token NLL. |
| Sequence length (`cutoff_len`) | 32768 |
| Samples / arm | 55,000 |
| Epochs | 1 → **1719 optimizer steps** |
| **Global batch** | **32** = 8 GPUs × `per_device_train_batch_size 1` × `gradient_accumulation_steps 4` |
| Peak LR | **1.0e-5** |
| LR schedule | **cosine** decay to ~0 (final LR ≈ 2.2e-10 — fully annealed) |
| Warmup | **`warmup_ratio: 0.03`** (~52 steps) |
| Precision | bf16 |
| Seed | 42 |
| Attention / kernels | FlashAttention-2, Liger kernel, `flash-linear-attention` + `causal-conv1d` (required — 24/32 Qwen3.5 layers are gated-delta-net) |
| Memory | gradient checkpointing on; DeepSpeed **ZeRO-2** |
| Checkpointing | `save_only_model: false` (full optimizer+scheduler state), `save_steps 100`, `save_total_limit 2` |
| In-training eval | `eval_steps 50`, `per_device_eval_batch_size 1`, `prediction_loss_only: true`, Liger `skip_logits` eval patch (avoids the 32k×~248k-vocab logits OOM) |

## 5. Hardware & wall-clock

- **Per arm:** 1 node × **8× NVIDIA A100 80 GB** (`ampere80gb`, NVLink), single-node
  8-GPU run (no special NCCL flags needed). ZeRO-2 keeps the 4B params replicated;
  peak memory sits comfortably under 80 GB/GPU.
- **All four arms ran in parallel** on 4 of the cluster's nodes.
- **Wall-clock per arm** (1 epoch, 1719 steps, incl. in-training eval):

  | arm | GPUs | wall-clock |
  |---|---|---|
  | coderforge | 8× A100-80GB | **14.6 h** |
  | rebench | 8× A100-80GB | **14.3 h** |
  | swezero | 8× A100-80GB | **13.8 h** |
  | scale | 8× A100-80GB | **13.6 h** |

  Campaign wall-clock ≈ **14.6 h** (arms run concurrently); aggregate ≈ **56 GPU-node-hours**
  ≈ **~448 A100-GPU-hours** (4 arms × ~14 h × 8 GPUs).

### Token counts & truncation at `cutoff_len` 32768

Arms are matched on **records** (55,000 each) but not on tokens — a ~6.6% spread, which
partly explains the wall-clock differences. Crucially, **truncation at 32k is
negligible** in every arm (≤0.03% of records), so the concern that the 32k cap clips
the trajectory's patch/submit turn does **not** materially apply here — median
trajectory is ~15k tokens and p95 ~26k, well inside the window.

| arm | records | total tokens | mean len | median | p95 | max | records truncated (≥32768) |
|---|---:|---:|---:|---:|---:|---:|---:|
| coderforge | 55,000 | 838.3 M | 15,242 | 17,208 | 26,763 | 32,768 | 18 (0.03%) |
| swezero | 55,000 | 852.5 M | 15,500 | 17,296 | 26,328 | 32,768 | 4 (0.01%) |
| rebench | 55,000 | 825.4 M | 15,008 | 16,991 | 26,160 | 32,768 | 6 (0.01%) |
| scale | 55,000 | 799.7 M | 14,541 | 16,303 | 25,775 | 32,768 | 1 (0.00%) |

*(Token lengths are full tokenized sequence lengths from the pretokenize cache; the
loss is masked to assistant tokens only, so supervised-token counts are lower.)*

## 6. LR-schedule integrity (why this is a rerun)

The original Babel arms used `save_only_model: true`, so every preemption resume
silently restarted a fresh warmup + full-length cosine from the resume step —
schedules were never matched across arms, and validation loss ended up dominated by
instantaneous LR state rather than data quality. This rerun bakes in the fix:
full-state checkpoints, a resume picker that hard-fails on model-only checkpoints, and
an identical cosine across arms by construction. **Verified clean here:** LR peaks at
1.0e-5 and anneals monotonically to ~2.2e-10 over exactly 1719 steps, one continuous
schedule per arm.

## 7. Results — training & validation loss

Final losses (step 1719):

| arm | final train loss | final `v2_id_mix` (ID) | final `v2_swegym_ood` (OOD) |
|---|---:|---:|---:|
| coderforge | 0.304 | 0.314 | 0.450 |
| scale | 0.284 | 0.340 | **0.415** |
| rebench | 0.313 | **0.312** | 0.459 |
| swezero | 0.315 | 0.326 | 0.448 |

### In-distribution validation loss (`v2_id_mix`)

![in-distribution val loss](img/v2_id_mix_loss.png)

All four arms decrease smoothly and monotonically after warmup. **rebench** and
**coderforge** reach the lowest ID loss (~0.312–0.314); **scale** is consistently
highest on the ID mixture (~0.340) — expected, since scale is the only arm distilled
from a different teacher (DeepSeek v3.2) and its distribution overlaps the ID mixture
least.

### Out-of-distribution validation loss (`v2_swegym_ood`)

![out-of-distribution val loss](img/v2_swegym_ood_loss.png)

**The OOD result is the headline: the three Qwen3-Coder-480B-distilled arms transfer
essentially *nothing* to SWE-Gym in loss terms.** Net change over the full run is
coderforge ~flat (0.449 → 0.450), swezero −0.005 (0.453 → 0.448), rebench −0.013
(0.472 → 0.459) — all with an early *increase* (hump around steps 250–350) before
drifting back. Only **scale** shows a genuine downward OOD trend (0.439 → 0.415).

Two important caveats on that scale reading. First, both plots start at **step 50**
(the first eval point), and scale is already ~0.01–0.03 lower on OOD there than the
other arms. Since all four arms share the *same* instruct init, the missing **step-0
anchor** is exactly what would distinguish "scale's data teaches transferable
behavior" from "scale's data simply sits closer to SWE-Gym to begin with" — we can't
tell which from these curves. Running the untrained instruct model through both eval
sets (§10, item 1) would resolve this. Second, all arms are a **single seed (42), one
run each**, so small gaps are within plausible seed noise.

So lowest ID loss (rebench/coderforge) does **not** predict best OOD loss (scale) — a
caution against reading the ID curve as a generalization proxy — but the step-0 gap
means we should not yet call scale's data causally "more transferable."

The **coderforge (verified source) vs swezero (unverified source)** comparison is tiny
in loss terms: coderforge is lower (better) on ID by 0.012 (0.314 vs 0.326), and the
two are near-identical on OOD (0.450 vs 0.448). Both gaps (ID 0.012, OOD 0.002) are
small and, at one seed, not clearly outside noise — and per the §1 confound, a
difference here would not isolate verification anyway. Loss alone doesn't separate the
two; the SWE-bench resolution numbers (§8) are the real test. (Standard caveat:
trajectory NLL is only weakly predictive of agentic rollout success.)

## 8. SWE-bench Verified results — *placeholder*

> Downstream eval (500-instance SWE-bench Verified via the OpenHands SDK harness) is
> pending / to be run on the four `checkpoint-1719` models. The correct untrained
> baseline is the **instruct** `Qwen/Qwen3.5-4B` (same starting point as the arms) —
> to be measured. Legacy Babel anchors, for context only, were on *different* models:
> untrained **base** `Qwen3.5-4B-Base` **25/500**; paper-nonweb 24k SFT (base) **52/500**.

| model | resolved / 500 | % resolved | non-empty patch % | notes |
|---|---:|---:|---:|---|
| Qwen/Qwen3.5-4B (instruct, untrained) | _TBD_ | _TBD_ | _TBD_ | **the baseline** — same starting model as the arms |
| coderforge (verified) | _TBD_ | _TBD_ | _TBD_ | |
| scale (verified) | _TBD_ | _TBD_ | _TBD_ | |
| rebench (verified) | _TBD_ | _TBD_ | _TBD_ | |
| swezero (unverified) | _TBD_ | _TBD_ | _TBD_ | verification-contrast counterpart to coderforge |
| uniform soup (equal weights) | _TBD_ | _TBD_ | _TBD_ | baseline soup |
| best single arm | _TBD_ | _TBD_ | _TBD_ | baseline for the soup search |
| model soup (best coeff) | _TBD_ | _TBD_ | _TBD_ | from coefficient search over the 4 task vectors |

**Soup evaluation hygiene (protocol for the pending work):**

- Include the **uniform soup** and the **best single arm** as baselines — a coefficient
  search only earns its keep if it beats both.
- **Do not tune the soup coefficients on the same 500 instances you report.** With 4
  coefficients and a noisy 0/1 per-instance metric, that overfits easily. Search on a
  **disjoint dev set** (a held-out Verified slice or SWE-bench Lite), fix the
  coefficients, then evaluate the chosen soup on the full 500 **once**.
- Fix decoding (temp, tool-parser, max turns) identically across all rows; report
  non-empty-patch rate alongside resolved (empty patches were the dominant failure mode
  in the 4B condenser eval).

## 9. Provenance

- **wandb:** project `dfried/adp-v2-a100`; runs `adp-v2-{coderforge,scale,rebench,swezero}-4b-a100`
  (ids `2ngl029m`, `j99q3gzd`, `r1loirm2`, `bihafz14`), all `finished`, 2026-07-23.
- **Kit:** `openhands_sdk_training/v2_arms_a100_rerun/` (see `README.md` + `CLAUDE.md`
  for the full recipe rationale and integrity rules).
- **Plots:** `img/v2_id_mix_loss.png`, `img/v2_swegym_ood_loss.png`, regenerated from
  wandb history (eval loss vs. training step, 36 eval points/arm at `eval_steps 50`).

## 10. Planned additions & future work

Ordered by cost. Items 1–3 need no retraining; items 4–5 are new training.

1. **Untrained-baseline anchor on the loss plots (eval-only).** Run the untrained
   instruct `Qwen/Qwen3.5-4B` through both eval sets once and draw it as a horizontal
   line on each plot. This is the step-0 reference the OOD reading (§7) currently
   lacks — it separates "scale's data is transferable" from "scale's data starts closer
   to SWE-Gym." Also add `eval_on_start: true` to future run configs so step 0 is
   logged natively.
2. ~~Per-arm token counts and truncation rate at `cutoff_len` 32768.~~ **Done** — see
   the token-count table in §5. Truncation turned out negligible (≤0.03%), so the
   end-of-trajectory clipping concern doesn't materially apply.
3. **4×4 arm-by-family eval matrix (eval-only).** `v2_id_mix` pools all four families,
   so each arm's ID number conflates "how well it learned" with "how much of the pool
   resembles its own training data" (worse if the pool is size-weighted). Evaluate each
   final checkpoint on **each family's** held-out carve-out: diagonal = own-fit,
   off-diagonal = cross-source transfer. Far more informative than the pooled scalar,
   and directly useful for choosing soup coefficients.
4. **Second seed on the coderforge/swezero pair.** ~2 × 14 h × 8 A100s — cheap
   insurance that the (small) headline gaps are real rather than seed noise (§7).
5. **Same-pipeline verification toggle.** The only way to make the verification claim
   *causal*: one pipeline, verification switched on vs off, everything else held fixed
   (§1 confound).
