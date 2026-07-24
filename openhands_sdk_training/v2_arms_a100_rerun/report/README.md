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
  - **`v2_id_mix`** (**400 records**) — an *in-distribution* validation mixture: the
    four training families' held-out carve-outs pooled (100 records each), tagged by
    `source_dataset`. Its teachers therefore **match training** (Qwen3-Coder-480B ×3 +
    DeepSeek v3.2), so it is in-distribution in both *task source* and *teacher*.
  - **`v2_swegym_ood`** (**200 records**) — an *out-of-distribution* held-out set, a
    carve-out of the adp-v2 config `swe-gym_openhands_sampled_trajectories` (~12K v2
    records total), tokenized the same way (OpenHands SDK / OpenAI tool format, 32k).
    It is out-of-distribution in **two** ways: a task source not in any arm's training
    data, **and a different teacher**.

  **Eval-set provenance** (mirrors the §2 training table):

  | eval set | adp-v2 source | carve-out | teacher | scaffold | verified? | license |
  |---|---|---:|---|---|:--:|---|
  | `v2_id_mix` | pooled coderforge/scale/rebench/swezero | 400 (100×4) | Qwen3-Coder-480B (×3) + DeepSeek v3.2 | OpenHands | mixed (see §2) | per-source |
  | `v2_swegym_ood` | `swe-gym_openhands_sampled_trajectories` | 200 | **GPT-4o / Claude-3.5** | OpenHands | ⚠️ ambiguous (possibly unfiltered verifier set) | MIT |

  > **Teacher mismatch (matters for §7).** No training arm shares the SWE-Gym eval
  > teacher: the arms are distilled from Qwen3-Coder-480B / DeepSeek v3.2, while the
  > SWE-Gym trajectories are GPT-4o / Claude-3.5. So the OOD curve measures token-CE
  > against a teacher the model was never trained to imitate — it carries a
  > **teacher-mismatch floor** independent of SWE task capability, and inter-arm OOD
  > differences may partly reflect stylistic proximity to the GPT-4o/Claude teacher
  > rather than better task transfer.

  All eval sets are baked into the tokenized cache at pre-tokenize time and are
  **identical across all four arms**, so the curves are directly comparable.

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
loss is masked to assistant tokens only, so supervised-token counts are lower. Mean <
median in every arm because the length distribution is **bimodal** — a large short mode
(~35–41% of records under 8k tokens, e.g. short single-segment/condensation records)
plus a dense cluster of long multi-turn trajectories near the 32k cap; the short mode
pulls the mean below the median. Spot-checked against per-arm percentiles.)*

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
| _untrained instruct (step 0)_ | _–_ | _0.462_ | _0.573_ |
| coderforge | 0.304 | 0.314 | 0.450 |
| scale | 0.284 | 0.340 | **0.415** |
| rebench | 0.313 | **0.312** | 0.459 |
| swezero | 0.315 | 0.326 | 0.448 |

The **untrained instruct** row is the shared step-0 anchor (all arms start here) — it is
the horizontal line on both plots below. It was measured directly by running
`Qwen/Qwen3.5-4B` through both eval splits with the *same* masking as the curves; the
same harness reproduces all four trained endpoints to ±0.0001 vs. wandb, so it is
faithful. Improvements from baseline: ID 0.462 → 0.31–0.34 (−0.12 to −0.15); OOD 0.573 →
0.415–0.459 (−0.11 to −0.16).

### In-distribution validation loss (`v2_id_mix`)

![in-distribution val loss](img/v2_id_mix_loss.png)

All four arms decrease smoothly and monotonically after warmup, dropping from the shared
untrained baseline (0.462, the horizontal line) to 0.31–0.34. **rebench** and
**coderforge** reach the lowest ID loss (~0.312–0.314); **scale** is consistently
highest on the ID mixture (~0.340) — expected, since scale is the only arm distilled
from a different teacher (DeepSeek v3.2) and its distribution overlaps the ID mixture
least. Unlike OOD, ID loss keeps falling across the whole run rather than plateauing early.

### Out-of-distribution validation loss (`v2_swegym_ood`)

![out-of-distribution val loss](img/v2_swegym_ood_loss.png)

**With the step-0 anchor in place, the OOD story is: all four arms transfer meaningfully
to SWE-Gym, but almost all of the gain lands in the first ~50 steps and then plateaus.**
From the shared untrained baseline (0.573), every arm drops to 0.415–0.459 — a −0.11 to
−0.16 improvement — but by the first eval point (step 50) each is already at ~0.44–0.47,
so the curves *look* flat only because the plots begin after the fast early drop.
(Reading the step-50-onward curves without the anchor invites calling the non-scale arms
"essentially no transfer" — a step-0 artifact the anchor corrects, and the reason the
anchor was worth measuring.)

A caveat on what that early drop *means*: much of the step-0→50 fall is plausibly
**format adaptation** rather than task-capability transfer. The untrained instruct model
has never seen the exact OpenHands harness / tool-call rendering, so NLL on *anything* in
that format drops fast regardless of SWE skill. So the early gain is real but partly
formatting; the more interpretable signal is the **post-step-50 behavior** — scale's
continued decline vs. the other three plateauing.

Beyond step 50 the arms separate: **scale** keeps improving (0.439 → 0.415), while
coderforge/swezero plateau (~0.45) and rebench sits highest (~0.46), all with a small
early hump (~steps 250–350). Because every arm shares the *identical* untrained init,
scale's lower OOD is **earned during training, not a starting-point artifact**. But
*what* it earned is ambiguous: recall (§3) that the SWE-Gym eval was distilled by a
**different teacher** (GPT-4o / Claude-3.5) than any training arm. scale is the only arm
distilled from DeepSeek v3.2 rather than Qwen3-Coder-480B, so its lower OOD token-CE may
reflect **DeepSeek's stylistic proximity to the GPT-4o/Claude eval teacher** as much as
genuinely more transferable SWE behavior — the two are confounded here and this loss
cannot separate them. (Standard single-seed caveat also applies to the small gaps among
the other three.) This is a strong reason to treat OOD *loss* as a sanity signal only
and lean on the SWE-bench resolution numbers (§8).

So lowest ID loss (rebench/coderforge) does **not** predict best OOD loss (scale) — a
caution against reading the ID curve as a generalization proxy — and the shared step-0
anchor lets us attribute scale's OOD edge to its *data* rather than to initialization,
even if data-quality vs. teacher-style remains confounded (above).

The **coderforge (verified source) vs swezero (unverified source)** comparison is tiny
in loss terms: coderforge is lower (better) on ID by 0.012 (0.314 vs 0.326), and the
two are near-identical on OOD (0.450 vs 0.448). Both gaps (ID 0.012, OOD 0.002) are
small and, at one seed, not clearly outside noise — and per the §1 confound, a
difference here would not isolate verification anyway. Loss alone doesn't separate the
two; the SWE-bench resolution numbers (§8) are the real test. (Standard caveat:
trajectory NLL is only weakly predictive of agentic rollout success.)

### Per-family breakdown (arm × family eval matrix)

The pooled `v2_id_mix` number conflates "how well an arm learned" with "how much of the
pool looks like its own training data." Since `v2_id_mix` is exactly the four family
carve-outs pooled (100 records each, tagged by `source_dataset`), we can un-pool it:
each cell is the final checkpoint's eval loss on one family's 100-record held-out set.
Diagonal = own-family fit; off-diagonal = cross-source transfer. (Pooled column
reproduces the §7 `id_mix` numbers exactly, as a consistency check.)

| model ↓ / eval family → | coderforge | scale | rebench | swezero | **pooled (id_mix)** |
|---|---:|---:|---:|---:|---:|
| untrained instruct | 0.481 | 0.423 | 0.462 | 0.483 | 0.462 |
| **coderforge** | **0.284** | 0.324 | 0.310 | 0.339 | 0.314 |
| **scale** | 0.356 | **0.256** | 0.357 | 0.391 | 0.340 |
| **rebench** | 0.305 | 0.325 | **0.288** | 0.330 | 0.312 |
| **swezero** | 0.333 | 0.331 | 0.328 | **0.311** | 0.326 |

Reading it:

- **Every arm fits its own family best** — the diagonal is each row's minimum, confirming
  the arms actually specialized to their data rather than all converging to the same place.
- **scale is a specialist**: it has by far the lowest own-family loss (0.256) yet the
  *worst* cross-family transfer (0.356 / 0.357 / 0.391 on the three Qwen3-Coder families).
  This is exactly why its **pooled** `id_mix` is *highest* (0.340) despite its best
  own-fit — the pool is ¾ non-scale, so scale is judged mostly on families it transfers
  to poorly. The pooled scalar would have hidden this entirely.
- **coderforge and rebench are mutually compatible** (coderforge→rebench 0.310,
  rebench→coderforge 0.305) — unsurprising, as both are Qwen3-Coder-480B-distilled.
- Even **untrained**, the scale family is the easiest (0.423 vs ~0.46–0.48), i.e. the
  DeepSeek-distilled distribution sits closest to the instruct model's priors — which
  also helps explain scale's lower OOD SWE-Gym loss.
- **Implication for the soup:** the arms behave like complementary specialists (strong
  diagonal, weaker off-diagonal), which is the regime where model-souping tends to help;
  scale is the clear outlier and its coefficient is worth examining separately.

*Sample-size caveat:* each family carve-out is only **100 records**, so small
off-diagonal differences (e.g. coderforge→rebench 0.310 vs. rebench→coderforge 0.305)
are within eval noise and shouldn't be over-read. The structural conclusions — diagonal
minima, and scale's ~+0.10 cross-transfer gap — are large enough to be safe.

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

1. ~~Untrained-baseline anchor on the loss plots (eval-only).~~ **Done** — the untrained
   instruct `Qwen/Qwen3.5-4B` was run through both eval splits (ID 0.462, OOD 0.573) and
   is drawn as the horizontal line on both plots. This resolved the step-0 ambiguity in
   §7 (scale's OOD edge is earned, not a starting-point artifact) and corrected the
   earlier "no OOD transfer" misread. *Still recommended for future runs:* add
   `eval_on_start: true` so step 0 is logged natively rather than reconstructed.
2. ~~Per-arm token counts and truncation rate at `cutoff_len` 32768.~~ **Done** — see
   the token-count table in §5. Truncation turned out negligible (≤0.03%), so the
   end-of-trajectory clipping concern doesn't materially apply.
3. ~~4×4 arm-by-family eval matrix (eval-only).~~ **Done** — see the matrix in §7. Each
   arm fits its own family best (diagonal minima); scale is a specialist (best own-fit,
   worst transfer), which the pooled scalar had hidden. Directly useful for the soup search.
4. **Second seed on the coderforge/swezero pair.** ~2 × 14 h × 8 A100s — cheap
   insurance that the (small) headline gaps are real rather than seed noise (§7).
5. **Same-pipeline verification toggle.** The only way to make the verification claim
   *causal*: one pipeline, verification switched on vs off, everything else held fixed
   (§1 confound).
