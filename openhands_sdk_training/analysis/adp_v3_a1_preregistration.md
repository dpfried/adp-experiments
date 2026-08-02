# ADP-v3 / Lever A1 — pre-registration (trajectory-only vs mixed)

_Written **before** the arm is trained and **before** its eval lands, per the v2
campaign's discipline. Numbers marked `[PHASE0]` are filled from the Phase-0 scoping
pass and are the only thing that may change after this document is committed; the
hypotheses, falsifiers and decision rules below are frozen as of this commit._

Status: **pre-registered, not yet run.**
**See §7 — Amendment 1 (2026-08-02), filed before any arm finished training and before any
eval existed. It changes which arm is the headline. §1–§6 are the original text, unedited.**

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
* Phase-0 scoping output: `/checkpoint/dpf/adp-analysis/v3_scoping/PHASE0_SCOPING.md`.

---

## 7. Amendment 1 — two arms, headline is the token-matched one (2026-08-02)

_Filed after Phase-0 scoping, **before any arm finished training and before any eval
existed**. Nothing below was informed by a result. §1–§6 are left unedited so the change
is auditable — including their `[PHASE0]` placeholders, which are **not** back-filled in
place. Every Phase-0 number lives here in §7 instead, so the original design and the
revision can be read against each other._

### What Phase 0 changed

§2 assumed the shortfall from 55,008 records was the only matching problem, and treated a
record-matched arm as the natural primary. Phase 0 showed that framing was wrong on a
second, larger axis:

| | value | source |
| --- | --- | --- |
| trajectory record, mean tokens (swezero) | 21,357 | sample n=3,000, calibrated |
| condensation record, mean tokens (swezero) | 4,372 | same |
| **v2 swezero control, exact training tokens** | **852,523,454** | exact, read from its pre-tokenized Arrow cache (55,000 rows) |
| trajectory-only @ 52,473 records | ~1,120.7 M | estimated |

A max-pool trajectory-only arm is therefore **~1.32× the FLOPs** of the control while also
running a full 1-epoch schedule. **Records and tokens cannot both be matched** — dropping
the short records necessarily raises tokens-per-record.

This matters asymmetrically. A *win* for the 52,473-record arm would be uninterpretable
(better data vs ~32% more compute); only a *null or loss* would be clean. §2's original
mitigation — "report a positive result as composition-or-tokens" — is honest but concedes
the main question, since a positive result is the outcome we most want to interpret.

### Revised design: run both, headline the clean one

| arm | records | tokens | vs control | steps | role |
| --- | --- | --- | --- | --- | --- |
| **A1-tokmatch (HEADLINE)** | N ≈ 39.9k, set exactly | ≈ 852.5 M | **1.00×** | ≈ 1,247 | single-variable test of H-A1 |
| A1-maxpool (secondary) | 52,473 | ~1,120.7 M | ~1.32× | 1,640 | "all the trajectory data there is"; confounded upward |
| v2 control (already trained, 77/500) | 55,000 | 852,523,454 | 1.00× | 1,719 | — |

**N is not taken from the Phase-0 estimate (±2%).** It is computed *exactly*: the
A1-maxpool pretokenize pass produces an Arrow cache of all 52,473 trajectory records, and
N is the largest prefix whose cumulative `input_ids` length ≤ 852,523,454. Because
selection is first-N in file order, **A1-tokmatch's data is an exact prefix of
A1-maxpool's** — the two arms are nested, not independent samples.

### What A1-tokmatch does and does not control

Honest statement of the residual mismatch: holding tokens fixed **reduces optimizer steps
1,719 → ~1,247 (−27%)** at fixed batch size. There is no design that matches records,
tokens and steps simultaneously; the three are linked once record length changes. So:

* A1-tokmatch holds **FLOPs** fixed and gives up **steps**.
* A1-maxpool holds **neither** fixed and gives up interpretability on a win.
* Together they bracket the answer, which is why both are being run.

### Pre-committed joint reading (frozen now, before results)

* **Both beat control** ⇒ H-A1 supported, and robust to the compute axis. Strongest outcome.
* **maxpool beats control, tokmatch does not** ⇒ the gain is attributable to **extra
  compute/data, not composition**. Report as such. This is the outcome the original
  single-arm design could not have distinguished, and is the reason for the amendment.
* **tokmatch beats control, maxpool does not** ⇒ unexpected (more data hurting); would
  point at a length/schedule interaction and require a step-matched follow-up before any
  claim.
* **Neither beats control** ⇒ Falsifier 1 (§3). Condensation records are inert at this
  budget; Lever A is not the lever and the campaign's weight moves to Lever B.

Falsifiers, decision rules, metrics, the depth-diagnostic mechanism check (§4), and the
"do not re-anchor the base (θ₀ = 119)" commitment are all **unchanged**. The headline
comparison is A1-tokmatch vs the v2 swezero mixed arm; McNemar and TOST (±10/500) as in §3.

### Consequences for Lever B, recorded here so they are not lost

Phase 0 also settled two things that §5 listed as open:

* **`(repo, base_commit)` is recoverable from ~100% of scale/rebench/swezero trajectory
  records** — the base commit is written verbatim into the task prompt
  (`base commit ([0-9a-f]{40})`). Instance identity is therefore *not* the blocker for
  outcome verification; missing gold patches, FAIL_TO_PASS lists and network access are.
* **The cheap outcome proxy is dead.** Measured on n=3,000/subset, the "clean finish"
  clause passes 98–99% of finish-terminated records — it discriminates essentially
  nothing — and the "produced a diff" clause tracks scaffold behaviour rather than
  correctness (7.6% for coderforge vs 59.1% for scale). Worse, it selects for the
  confident-tidy finish that the v2 audit identified as the failure mode. **Do not use it
  as a filter**; doing so would amplify the disease. Lever B needs real labels or a
  carefully-designed judge, not this.

---

## 8. Amendment 2 — peer review (2026-08-02, still pre-result)

_Filed after @main-worker's review of §7 on the coordination channel, **before either full
arm was launched and before any eval existed**. Two of the three changes are corrections
to §3/§7, not elaborations. Both technical claims below were verified independently rather
than taken on assertion._

### 8.1 The TOST margin in §3 was wrong — ±10/500 → **±15/500**

§3 pre-registered a ±10/500 equivalence margin. **That is tighter than this instrument's
own documented resolution**, so it could essentially never return "equivalent".

Verified against this campaign's own findings report
(`adp_v2_swebench_findings_report.md`, and `SKEPTICAL_REVIEW_GUIDANCE.md`):

> "single temp-0 rollout × 500 cannot resolve effects <~15/500"; "for n=500, SE ≈ ±8".

Concretely: for a paired comparison the SE of the difference is driven by the discordant
pairs. The v2 swezero-vs-base pair had 82 + 40 = 122 discordant instances → SE ≈
√122/500 ≈ **11/500**. A TOST at ±10 requires the whole CI inside ±10 while the SE alone
is ~11 — unreachable by construction.

**Change:** the equivalence margin is **±15/500**, matching the documented noise floor.
This also gives "equivalent" the right meaning: *any true difference is below what this
measurement can resolve.*

**Pre-committed handling of the likely outcome:** "inconclusive" is an **expected**
verdict here, not a failure, and it will be **reported as inconclusive** — not rounded to
"no difference" or "equivalent". If a decision actually hinges on separating two arms
inside the floor, the escalation is more rollouts (multi-seed / pass@k), not a softer
margin. @soup-worker owns `paired_compare.py` and the final call on the statistic.

### 8.2 Falsifier 1 in §3/§7 overclaimed — scoped to the source actually tested

§3 and §7 read a two-arm null as "condensation records are inert; Lever A is not the
lever". **A swezero-only null does not license that.** swezero has the *lowest*
condensation share of the four subsets (34.3%), i.e. the least to remove — it is the
deliberately **conservative** source (chosen for the largest trajectory pool, the highest
terminal-record rate at 52.0%, and the strongest v2 baseline at 77/500). A null there
leaves rebench (41.3% condensation) untested.

**Revised Falsifier 1:** a null is recorded as **"Lever A is inert on the conservative
source"**, which *motivates a rebench replication* (max-contrast: 41.3% condensation,
weaker baseline 70/500, pool 46,914) **before** Lever A is declared dead. The
positive-case logic is unchanged and is why swezero stays the first source: an effect that
shows up on the least-condensation-heavy source is harder to dismiss than one that only
appears on the friendliest case.

### 8.3 Token-matching is a *good but imperfect* FLOP proxy — quantified

@main-worker noted Qwen3.5 is a **GDN hybrid**, so token-matching tracks FLOPs better here
than it would for a vanilla transformer. **Verified** from the model config
(`models--Qwen--Qwen3.5-4B/.../config.json`):

```
text_config.layer_types      = {linear_attention: 24, full_attention: 8}   # of 32
text_config.full_attention_interval = 4
text_config.hidden_size = 2560, intermediate_size = 9216
```

24/32 layers are linear-attention (O(L)), so most of the network is insensitive to how a
fixed token budget is distributed across sequence lengths. But the **8 full-attention
layers are not**: at a fixed token total, attention FLOPs scale ∝ average sequence length
(N=T/L sequences × O(L²) each = O(T·L)). A1-tokmatch averages ~21.4k tokens/record vs the
control's ~15.5k.

First-order estimate (dense projections `2(4d² + 3d·inter)` per token per layer; attention
`8·L·d` per token in full-attention layers only):

| arm | tokens vs control | **estimated FLOPs vs control** |
| --- | --- | --- |
| A1-tokmatch | 1.0000× | **~1.11×** |
| A1-maxpool | 1.3175× | ~1.46× |

So token-matching removes **most** of the compute confound (1.46× → 1.11×) but not all of
it, and the residual **favours A1**. Stated plainly so a positive A1-tokmatch result is
read as "better data, with a ~11% compute tailwind", not "matched compute". This is an
analytic estimate — it ignores GDN's own cost, embeddings/lm_head, gradient-checkpoint
recompute and kernel efficiency — so treat it as order-of-magnitude, direction-certain.

**This does not change the headline choice.** A ~1.11× residual is far more interpretable
than maxpool's ~1.46×, and running both still brackets the answer.
