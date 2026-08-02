# ADP-v3 / Lever A1 — pre-registration (trajectory-only vs mixed)

_Written **before** the arm is trained and **before** its eval lands, per the v2
campaign's discipline. Numbers marked `[PHASE0]` are filled from the Phase-0 scoping
pass and are the only thing that may change after this document is committed; the
hypotheses, falsifiers and decision rules below are frozen as of this commit._

Status: **pre-registered, not yet run.**

---

## 1. Question

~34–41% of every ADP-v2 SWE subset consists of **context-condensation records** —
2-message pairs (`generation=openhands_sdk_condensation_prompt`) whose user turn is a
condenser prompt and whose assistant turn is a prose state summary. They train
*summarize-and-conclude*, not *act*. The v2 arms therefore spent ~40% of their gradient
on summarization.

The v2 root-cause finding is that SFT **traded depth for compliance**: arms became more
scaffold-compliant than base (swezero finishes 500/500 vs base 207/500, 0 empty patches
vs 138, ~half the actions) while producing valid-but-misdiagnosing patches on 82/82 of
the base-solved/arm-failed instances. Condensation records are the largest single
controllable slice of data that plausibly teaches exactly that behaviour.

**A1 asks: if we remove them and hold everything else fixed, does SBV pass@1 move?**

We do not currently know the sign. Condensation records could be net dilution (they
displace acting data), or they could carry genuine long-horizon context-management skill
that single-run pass@1 simply does not reward.

## 2. Design

| | treatment (A1) | control |
| --- | --- | --- |
| arm | `a1_traj_<source>` | existing v2 arm on the same source |
| data | 100% action-trajectory records | ~60% trajectory / ~40% condensation |
| source | `[PHASE0]` | same |
| records | `[PHASE0]` | 55,008 (1719 steps × 32) |
| recipe | frozen identical to v2 | — |

**Recipe (frozen, must not drift):** Qwen3.5-4B, `qwen3_5_nothink`, `cutoff_len` 32768,
LR 1e-5, cosine, warmup 0.03, 1 epoch, global batch 32, seed 42, ZeRO-2, liger + fa2,
8×A100. Same `pretokenize.py` and the same two patch scripts as v2.

**Record selection:** first-N in file order. The v2 sources are shuffled, so this is a
representative subsample *and* maximizes overlap with the control's trajectory records —
the two arms then differ as nearly as possible in composition alone. Records are copied
byte-for-byte (no JSON round-trip).

**Eval carve-out:** the matched v2 arm's `eval.llamafactory.jsonl`, shared by absolute
path, so in-training val-loss curves are directly comparable. Note this carve-out is
itself *mixed*, so A1's val loss on it is partly an out-of-distribution measurement —
that is intentional and it is a secondary metric only.

### The compute-matching caveat (stated up front)

Condensation records are short; trajectory records are long. A record-matched
trajectory-only arm therefore sees **more tokens** than the mixed arm at the same step
count. Exact ratio: `[PHASE0]`. This is unavoidable — you cannot delete 40% of a dataset
and hold both records and tokens fixed — so:

* The **primary** comparison is **record/step-matched** (same 1719-step schedule), which
  is the practically meaningful question: *at a fixed data budget and fixed compute
  schedule, is this composition better?*
* If the pure-trajectory pool is smaller than the control's 55,008 records (likely —
  see `[PHASE0]`), the arm is reported at its achievable budget and the shortfall is
  stated explicitly. **The builder warns and refuses to pad.**
* A token-matched variant is **not** pre-registered here. If A1 lands positive, a
  token-matched follow-up is the correct way to rule out "more tokens" as the driver,
  and it will be pre-registered separately. Until then, a positive A1 is reported as
  *composition-or-tokens*, not *composition alone*.

## 3. Hypotheses, falsifiers, decision rules

Primary metric: **SWE-bench Verified single-run temp-0 pass@1**, OpenHands scaffold,
reported on all three boards (full-500 / 456-ex-sphinx / clean-301). Paired stats:
**McNemar** vs the matched v2 arm and vs base θ₀=119, plus **TOST** with a ±10/500
equivalence margin → a three-state verdict (different / equivalent / inconclusive).

**H-A1:** trajectory-only > mixed on SBV pass@1.
→ *Supported if* McNemar vs the matched v2 arm is significant in the positive direction.

**Falsifier 1 (inert):** trajectory-only ≈ mixed, i.e. TOST concludes equivalence within
±10/500. ⇒ Condensation records are inert; the ~40% slice is **not** the lever, and the
campaign's weight shifts to Lever B (outcome verification).

**Falsifier 2 (reversed):** trajectory-only < mixed, significant by McNemar.
⇒ Condensation carries real value; **do not drop it**. Lever A2 (down-weight rather than
drop) becomes the interesting variant and the "summarize-over-act" mechanism story is
wrong as stated.

**Pre-committed non-claims:**
* A1 is **not** predicted to reach base (119). It removes one of two candidate defects;
  the campaign's headline arm is C1 (verified ∩ condensation-excluded).
* Any movement **below** base is still a real result. "Curated still loses to base" is
  informative, not a failure.
* We will **not** re-anchor the base. θ₀ = **119/500**, single-run,
  provenance-verified (`v2_init_singlerun_4b`). Not the retracted ~5%, not the
  non-instruct ~73.

## 4. Mechanism check (secondary, but pre-registered)

The score alone cannot distinguish "curation helped" from "curation helped for an
unrelated reason". Depth diagnostics, computed from the rollout JSONL for A1, the
matched v2 arm, and base:

* actions per instance (base 171, swezero 92)
* finish rate (base 207/500, swezero 500/500)
* empty-patch rate (base 138/500, swezero 0/500)
* turns-to-finish distribution

**Mechanism prediction if the depth-vs-compliance story is right:** A1 shifts *toward
base* — more actions, fewer premature finishes — even if the score moves little. A score
gain with **no** depth shift would mean the mechanism story is wrong even though the
lever worked, and should be reported as such rather than quietly folded into the
narrative.

## 5. What is NOT being tested here

Ruled out in v2, do not re-litigate: truncation (0% at the 32768 cutoff),
format/template/parser mismatch (byte-identical chat template, XML matches the
`qwen3_coder` parser), config sanity (LR/masking/packing), and contamination (audited;
re-rank identical on clean-301; dpf stood down).

Deferred to Lever B, gated on Phase-0 label recoverability: outcome/success filtering.
The records carry no resolved/reward field, so this needs externally recovered labels.

## 6. Provenance

* Kit: `openhands_sdk_training/v3_curation/` (this repo).
* Campaign design: `openhands_sdk_training/analysis/adp_v3_data_curation_campaign.md`.
* v2 result this builds on:
  `openhands_sdk_training/analysis/adp_v2_swebench_findings_report.md`.
* Phase-0 scoping output: `[PHASE0]`.
