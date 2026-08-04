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

---

## 9. Amendment 3 — devils-advocate + soup-worker review (2026-08-02)

_Filed at arm launch (jobs 330140 / 330141), **before either arm produced a checkpoint and
before any eval existed**. Records a confound that neither §7 nor §8 priced correctly, and
the statistical posture the campaign is actually committing to._

### 9.1 The action-data confound — priced, and the disambiguating arm defined

@devils-advocate's load-bearing objection: dropping condensation and refilling to a budget
**necessarily changes how much action data the model sees**, so a positive A1 has an
unpriced alternative explanation — *"the extra action trajectories helped"* — independent
of anything about condensation.

Exact numbers for swezero (Phase-0 prefix census). The control's 55,000 records decompose
into **36,040 trajectory + 18,968 condensation**. So, in action records seen:

| arm | action records | vs control | tokens vs control |
| --- | --- | --- | --- |
| control | 36,040 (+18,968 cond) | — | 1.00× |
| **A1-tokmatch (headline)** | 39,829 | **+3,789 (+10.5%)** | 1.00× |
| A1-maxpool | 52,473 | +16,433 (+45.6%) | 1.32× |

Amendment 1's token-matching **substantially shrinks but does not remove** this confound:
the headline arm sees +10.5% more action data, not the +45.6% a record-matched design
would have given. Worth stating that the confound is smaller than devils-advocate's
worst case, and still real.

**A useful structural property:** because selection is first-N in file order and the
control's action records are exactly the trajectory records inside the first 55,000, the
three arms are **exact nested prefixes**:

```
36,040  ⊂  39,829  ⊂  52,473
(control's own      (tokmatch)      (maxpool)
 action records)
```

**Pre-registered attribution and the gated third arm.** A positive A1 is attributable to
**{condensation-removal OR extra-action-data}** and will be reported that way, not as
condensation alone. To disambiguate, a third arm is **pre-defined now and gated on a
positive A1** (per devils-advocate's cost discipline — do not spend it on a null):

* **A1-subtract** = exactly the control's own **36,040** action records, condensation
  deleted, nothing added (~1,127 steps, ~0.90× control tokens). Action content is held
  *identical* to the control, so `control vs A1-subtract` isolates **condensation removal
  at fixed action content** — the cleanest form of the actual question. Its own caveat:
  at 0.90× tokens it is compute-*under*-matched, so a null there could be condensation-
  removal-helped cancelling against less-compute-hurt.
* `A1-subtract → tokmatch → maxpool` is then a **dose-response curve on action quantity at
  zero condensation**, which separates the two factors that A1 alone conflates.

No single two-arm comparison here is unconfounded. The nested series is what makes the
decomposition possible.

### 9.2 Statistical posture — corrected again, by soup-worker

@soup-worker (owner of `paired_compare.py`) corrected **both** §8.1 and the peer consensus.
main-worker, devils-advocate and I had all converged on "widen the margin toward ±15–20 to
match precedent". That is **backwards**:

> TOST declares equivalence iff the 90% CI ⊂ ±margin. Half-width = 1.645·SE. With 122
> discordant pairs, SE ≈ √122/500 ≈ 11/500, so even a **perfectly null** result gives a
> 90% CI of ±1.645·11 ≈ **±18/500**.

So the structural impossibility I "fixed" at ±15 in §8.1 was merely moved out a notch — the
instrument's floor is ~±18/500. **But inflating the margin to whatever the instrument can
resolve hides the underpowering rather than fixing it.**

**Committed posture (final):**
* **Margin stays ±15/500** — a defensible scientific line ("we care about effects ≥3%"),
  deliberately *not* inflated to match the instrument.
* A single 500-instance run reports **different / inconclusive only**. **"Equivalent" is
  out of reach at N=500 regardless of margin** and will not be claimed. n.s. ≠ equivalent.
* **McNemar@500 is powered for effects ≳4% (≳20/500).** Worked example at the expected
  split (122 discordant, ~73/49): z ≈ 2.17, p ≈ 0.03 — detectable. Below ~20/500, expect
  inconclusive.
* **Multi-seed is NOT the escalation.** At temp-0 the discordant pairs are *structural
  model disagreement, not sampling noise* — re-running reproduces the same labels, so SE
  does not shrink. The only lever is **more instances**. ⚠️ **Action: confirm the eval is
  actually temp-0 deterministic before relying on this**; if it is stochastic in practice
  (flaky infra flipping pass/fail), multi-seed *does* help (4 seeds ≈ SE/2 ≈ 5.5/500).

### 9.3 Two Lever-B constraints recorded now (pre-GPU, cheap to honour)

* **B2 (gold-patch distillation) is a train-on-test hazard and is hard-gated.** Phase 0
  established that `(repo, base_commit)` recovers from ~100% of records. Reconstructing
  trajectories to *end in the gold patch* on any instance overlapping SWE-bench Verified
  is literally SFT-on-the-eval-answer. v2 closed contamination at **repo** level, which is
  fine when not training on the label — B2 trains on the label. **B2 requires an EXACT
  (repo, base_commit) exclusion of all 500 SBV instances, not repo-level.** Same caution,
  lighter, for B-ii replay labels. Additionally, **rebench is the safer source for B1/B2**
  (decontaminated by design); swezero carries the SBV-overlapping repos
  (astropy/sympy/xarray/sphinx/seaborn).
* **B1 ("resolved-only") filters on OUTCOME, not DEPTH.** A resolved trajectory can still
  be shallow-tidy (a shallow fix is correct when the bug *was* shallow), and resolved
  trajectories over-represent **easier** instances the upstream agent could solve →
  distribution shift toward easy bugs. So H-B1 ("verification recovers depth") is **not
  mechanically entailed** by the filter; B1 could raise the score via easy-instance
  imitation with depth flat. Keep H-B1 hypothesis-voiced and let the **depth diagnostics**,
  not the /500, adjudicate the mechanism.

---

## 10. Amendment 4 — the "~11% compute tailwind" in §8.3 is wrong, measured (2026-08-02)

_Filed ~40 min into training, **before any checkpoint, eval or result existed**. This
corrects a claim I made in §8.3 and repeated in §9 — it was an analytic estimate, and the
measurement disagrees with it._

§8.3 estimated A1-tokmatch at **~1.11× the control's FLOPs** and told the reader to treat
a positive result as *"better data with a ~11% compute tailwind"*. Measured per-step wall
time says that framing is misleading.

**Measurement** (median seconds/optimizer-step, from each run's own `trainer_log.jsonl`,
same 8×A100 recipe, steps ≥15 to exclude warmup):

| run | mean tokens/record | median s/step | IQR | n |
| --- | --- | --- | --- | --- |
| v2 swezero mixed (control) | 15,500 | **25.20** | 25.00–25.40 | 339 |
| v3 A1-maxpool | 21,406 | **25.60** | 25.40–26.00 | 7 |
| v3 A1-tokmatch | 21,406 | **25.60** | 25.40–25.60 | 5 |

**Step time is flat (+1.6%) despite 1.38× more tokens per record.** At
`per_device_train_batch_size: 1` with gradient checkpointing, a single ~15–21k-token
sequence does not saturate an A100; the step is dominated by fixed costs (optimizer step,
the all-reduce over 4.2B params, recompute scheduling) rather than by token-proportional
matmul. So theoretical FLOPs are simply not the binding resource here, and a FLOP ratio is
the wrong currency for "did this arm get more compute".

**Corrected accounting for A1-tokmatch vs the control:**

| axis | A1-tokmatch vs control | direction |
| --- | --- | --- |
| tokens seen | 1.0000× (852.50 M vs 852.52 M) | matched |
| optimizer steps | 1,245 / 1,719 = **0.72×** | **fewer updates** |
| measured GPU-time | ~8.9 h / ~12.0 h ≈ **0.74×** | **less compute** |
| theoretical FLOPs | ~1.11× | slightly more |

**So the headline arm is not advantaged on any practical axis.** It is matched on tokens
and *disadvantaged* on optimizer steps and on actual GPU-seconds; the only axis on which
it leads is a theoretical FLOP count that the hardware demonstrably does not bill for.
**Replace "a positive A1-tokmatch comes with a ~11% compute tailwind" with: a positive
A1-tokmatch is obtained at matched tokens, ~28% fewer optimizer updates and ~26% less
GPU-time than its control** — which makes a win *harder* to explain away as compute, not
easier, and strengthens the headline arm rather than weakening it.

The symmetric consequence for the secondary arm: **A1-maxpool's 1.32× token advantage also
costs ~1.32× the wall-clock** (1,640 × 25.6 s ≈ 11.7 h), so its extra data is real extra
training. Its confound (§9.1) stands unchanged.

**Caveats, stated:** the v3 medians rest on n=5–7 early steps (the IQRs are tight, and the
control's n=339 median is stable at 25.20), and step time is not a perfect proxy for
learning-relevant compute either. **To re-verify with full n once both arms finish** — if
the completed-run medians diverge from this, this section gets corrected again rather than
quietly kept. What is already solid is the qualitative point: step time did **not** scale
with tokens/record, so §8.3's FLOP-ratio framing does not describe this system.

---

## 11. Amendment 5 — the eval is NOT reproducible; §9.2's escalation advice is reversed (2026-08-02)

_Filed while both arms were still training, **before any eval existed**. This reverses a
recommendation recorded in §9.2 and widens the campaign's stated noise floor._

### What §9.2 assumed

§9.2 recorded soup-worker's argument that **multi-seed is not the escalation** at temp-0:

> the discordant pairs in a temp-0 deterministic eval are structural disagreement between
> the two models, not sampling noise — re-running reproduces the same labels, so SE does
> not shrink. The only lever is more instances.

That reasoning is valid **conditional on the eval actually being deterministic**. It is not.

### Evidence that it is not

1. **dpf (2026-08-02):** *"I don't think it will be exactly reproducible given cuda
   non-determinism even with temp 0."*
2. **The serving path is nondeterministic by construction.** The eval runs vLLM with
   `--enable-prefix-caching` under continuous batching. Batch composition varies run to
   run, which changes float reduction order; at a near-tie that flips an argmax. In an
   agentic loop a single flipped token cascades into a different trajectory. Confirmed
   `"temperature": 0.0` in the OpenHands LLM config — greedy, but greedy ≠ reproducible.
3. **Observed label instability between the two base runs.** `v2_init_4b` (145) is a
   documented **best-of-3 union** (pass@3 artifact from a timeout + catch-up rerun);
   `v2_init_singlerun_4b` (119) is the verified single run. A union can only *add*
   resolutions, yet **23 instances are resolved in the single run and absent from the
   union** (96 concordant, 49 union-only). *Caveat, stated:* this is confounded — it could
   be genuine rollout variation, a different shard/scoring pass, or SWE-bench
   **test-execution** flakiness on identical patches. It was not decomposed (dpf accepted
   the premise, so no GPU was spent proving it). It is corroborating, not decisive.

### Consequences — three, and the third is the uncomfortable one

1. **Multi-seed IS a valid escalation.** If labels flip run to run, repeated rollouts
   average real noise and SE *does* shrink (~SE/√k). §9.2's "more instances is the only
   lever" is **withdrawn**; more instances and more seeds are both valid, and seeds are
   cheaper than sourcing new instances.
2. **Equivalence may be reachable after all** — with k seeds, SE ≈ 11/√k per 500, so
   k=4 gives ≈5.5/500 and a ±15 TOST becomes declarable. §9.2 said equivalence was
   structurally out of reach at N=500; that holds for a *single* run only.
3. **⚠️ Every paired comparison in this campaign is MORE underpowered than reported, not
   less.** McNemar and the SE ≈ √(discordant)/500 calculation both treat each instance's
   outcome as a *fixed property* of the model. If the outcome is itself stochastic, some
   of the observed discordance is run noise rather than model disagreement, and the true
   SE is **larger** than √(discordant)/500. So the campaign's ~15/500 noise floor is
   plausibly **optimistic**, and borderline calls are weaker than their p-values suggest.
   This applies retroactively to the v2 board and to the pooled220k panel, not just to A1.

### What this changes for A1 (committed now, pre-result)

* The margin stays **±15/500** and single-run verdicts stay **different / inconclusive**.
* **If A1 lands inside the noise band, the escalation is a second seed on both arms**
  (not more instances), and that is now the pre-registered response rather than an
  improvised one.
* Any A1 verdict is reported with **"single-run; eval is not bitwise reproducible"**
  attached. A near-threshold p-value will not be treated as decisive.
* **Unmeasured:** the *magnitude* of run-to-run flipping. The 23-instance figure is an
  upper-ish bound from a confounded comparison, not an estimate. A ~50-instance base
  re-run (~1 GPU-hour) would measure it directly and remains the cheapest way to size the
  effect if a borderline result makes it matter.

---

## 12. Amendment 6 — the GRADER is nondeterministic too (2026-08-02, measured)

_dpf asked whether the alternative explanations in §11 — (b) incomplete merge, (c) test
flakiness — were real. Both were checked. **(b) is refuted. (c) is confirmed and
quantified.** Filed while both arms were still training, before any eval existed._

### (b) Merge integrity — REFUTED, board is structurally sound

All 23 instances resolved in the single run but not in `init_4b` are **scored as
unresolved** in `init_4b`; **0 are absent**. The merge was complete, so the flips are
real label changes, not missing shards.

Campaign-wide check of every `score_*/merged.report.json`:

| check | result |
| --- | --- |
| board reports with union of resolved+unresolved = 500 | **13 / 13** |
| declared `total` = 500 | 13 / 13 |
| duplicate ids | **0** |
| missing ids | **0** |
| resolved ∩ unresolved overlap | 0 |

(The only partial report is `score_smoke10_inst4b`, a 10-instance smoke — not a board
number.) **No incomplete merges anywhere in the campaign.**

### (c) Scoring flakiness — CONFIRMED: the grader flips labels on byte-identical patches

Method: hash each instance's final `git_patch` in both base runs, keep instances whose
patch is identical, and compare the resolved/unresolved label.

| quantity | value |
| --- | --- |
| instances compared | 500 |
| identical patch in both runs | 207 (180 non-empty) |
| **identical non-empty patch, labels DISAGREE** | **11 / 180 = 6.1%** |
| different patch | 293 (61 label disagreements, 20.8%) |
| total label disagreement, same model | **72 / 500** — 61 generation, 11 scoring |

**Direction split (removes the union confound).** `init_4b` unions over attempts, so
"union=resolved, single=unresolved" could be a *different* attempt succeeding. Only the
opposite direction is clean:

* **CLEAN — 5 instances:** `init_4b` = **unresolved** (no attempt resolved) while the
  single run with a **byte-identical patch** = **resolved**. A union cannot explain this.
  `pytest-dev__pytest-6202`, `scikit-learn__scikit-learn-13496`, `sympy__sympy-20154`,
  `sympy__sympy-21379`, `sympy__sympy-21847`.
* AMBIGUOUS — 6 instances in the other direction.

**⇒ Grading nondeterminism is ≥5/180 (≈2.8%) of identical-patch instances, ≈1% of the
500-board; upper estimate 11 (6.1%).** Concentrated in sympy / pytest / scikit-learn,
consistent with slow or timeout-prone suites.

### Consequences

1. **~5–11 instances of the board can move with the model's output unchanged.** That is a
   third to two-thirds of the campaign's entire stated ~15/500 noise floor, and it was
   previously unaccounted for. It compounds §11: the floor is **not** a single-run
   artifact of generation sampling — part of it is the grader.
2. **This is irreducible by re-running the model.** Multi-seed averages generation noise
   (§11) *and* grading noise, so seeds still help — but a "deterministic decode" would
   not have removed it.
3. **Small deltas are less meaningful than the paired stats imply.** A ±5/500 difference
   between two arms is inside the grader's own flip range. Combined with §11's point that
   McNemar assumes fixed per-instance outcomes, **the ~15/500 floor should be read as a
   lower bound.**
4. **For A1:** the arms will be scored in a fresh pass while the v2 control's labels come
   from an older pass, so grader drift sits *between* treatment and control. Where
   feasible the control's stored rollouts should be **re-scored in the same pass** as A1
   so both sides see the same harness state. This does not remove per-run flakiness, but
   it removes systematic drift between passes.

*Caveat:* the 6.1% upper figure rests on a comparison against a union-scored run; the
defensible number is the clean **5**. Both are reported rather than the flattering one.

---

## 13. Correction — the val-loss curves are NOT comparable to the control (2026-08-02)

_Caught ~3h into training, before any eval existed. Affects a secondary metric only._

§2 states:

> **Eval carve-out:** the matched v2 arm's `eval.llamafactory.jsonl`, shared by absolute
> path, so in-training val-loss curves are directly comparable.

**The second clause is false.** Verified from the control's own
`all_results.json`, whose only eval metrics are:

```
eval_v2_id_mix_loss, eval_v2_swegym_ood_loss
```

and from its tokenized-cache name,
`tokenized_qwen35_4b_inst_seq32768_ev_v2_id_mix-v2_swegym_ood`. The v2 control was
evaluated on **`v2_id_mix` + `v2_swegym_ood`**, not on the per-arm `eval.llamafactory.jsonl`
carve-out the v3 arms use as `arm_eval`. Pointing both v3 arms at the same file makes the
**two v3 arms** comparable to each other — it does **not** make either comparable to the
control, which never measured that set.

**Impact: none on the primary result.** SBV pass@1 is the pre-registered primary metric and
is unaffected; val loss was always labelled secondary. The observed numbers stand on their
own terms — v3 train loss ~0.22–0.24 vs the control's ~0.34 at matched steps (expected:
condensation records are high-entropy prose, trajectory records are structured tool calls),
and `arm_eval` drifting up (0.4378 → 0.4451 tokmatch, 0.4368 → 0.4637 maxpool) is the
intentional OOD effect §2 predicted, since that carve-out is mixed while the arms train
pure. **What cannot be done is reading those against the control's 0.34/0.326.**

**Cheap remedy, if a val-loss comparison is wanted:** after training, run a loss-only pass
of {control, A1-tokmatch, A1-maxpool} over one common eval set (minutes on 1 GPU, no
retraining). Not required for the headline; recorded so the option is not forgotten.

**Root cause, for future kits:** `generate_v3_runs.py` defaults `--eval-set` to
`arm_eval=eval.llamafactory.jsonl`, inherited from the v2 generator — but the v2 *runs*
overrode that default with two named sets. The default looked like the control's
configuration and was not. Worth checking the comparison target's actual `all_results.json`
rather than the generator default next time.

---

## 14. Amendment 7 — variance decomposition and scope of the underpowering claim (2026-08-02)

_devils-advocate's review of §11–§12. Two of the three are corrections to me. Filed while
both arms were still training, before any eval existed._

### 14.1 §11 conflated two variance components

§11 concluded that because the eval is stochastic, "multi-seed IS a valid escalation" and
withdrew the "more instances is the only lever" line. **That framing was loose.** There are
two additive, non-interchangeable components:

| component | source | shrinks with |
| --- | --- | --- |
| **instance-sampling variance** | evaluating a finite 500-instance draw of the benchmark | **more instances** |
| **run-noise variance** | the stochastic decode established in §11 | **more seeds**, as ~1/√k |

soup-worker's SE ≈ 11/500 was measuring the *instance-sampling* term. Seeds do not touch
it. So §11's "k=4 ⇒ SE ≈ 5.5/500 and equivalence becomes declarable" holds **only under a
fixed-500 estimand** — and under that estimand the ±15 instance-sampling floor was never
the applicable floor in the first place.

**Corrected statement:** seeds and instances attack **different terms**; you now have *two
levers for two variance components*, not one lever that substitutes for the other. Which
applies depends on the estimand:

* **"pass-rate on these fixed 500"** → run-noise dominates; seeds are the lever.
* **"pass-rate on SWE-bench-Verified as a population"** → the instance-sampling term
  remains; **k=4 alone will not get a population comparison to 5.5/500.** Both levers needed.

Every comparison in this campaign should state which estimand it is making.

### 14.2 The scope of "more underpowered" — it does NOT touch the headline

§11 point 3 said "every paired comparison in this campaign is MORE underpowered than
reported". True, but **the scope was left too broad, and left that way it invites a reader
to discount the p<1e-10 spine.** The correction:

Symmetric run-noise inflates **both** McNemar discordant cells (b and c) roughly equally.
That grows the denominator of (b−c)²/(b+c) while leaving the numerator roughly unchanged,
so **the statistic shrinks and McNemar becomes MORE CONSERVATIVE** at detecting a true
difference.

⇒ **DIFFERENT verdicts are if anything SAFER under stochastic eval, not weaker.** A gap
that clears the bar *despite* inflated discordance is robust. So:

* **Unaffected (robust):** base 119 ≫ every arm; pooled220k −65/500 vs base; the whole
  p<1e-10 spine.
* **Genuinely at risk:** borderline/near-tie comparisons and any equivalence reading —
  i.e. exactly the inconclusive band this document already refuses to quote.

The underpowering bites the **inconclusive band, not the headline.**

### 14.3 §10's FLOP measurement is downgraded to provisional

§10's step-time medians for the v3 arms rest on **n=5–7 early steps**. Step time can drift
with later sequence composition and eval interleaving. Restated as **"early-step
measurement, +1.6% against 1.38× tokens/record, to be confirmed at full n"** rather than
settled. §10 already committed to re-verifying; this makes the provisional status explicit
in the claim itself. The qualitative conclusion (step time did not scale with tokens, so
the FLOP-ratio framing does not describe this system) is what the n=339 control median
supports.

### 14.4 Multi-seed stays gated

Accepted: do not let an unsized 23-instance figure become a board-wide k=4 program. The
pre-registered order is **size the flip rate first** (~50-instance base re-run, ~1 GPU-h),
*then* decide whether seeds are worth it. If the flip rate is small, this is a caveat, not
a re-run programme.

### 14.5 New asymmetric noise source (main-worker) — adopted into A1 practice

200/500 **base** rollouts stored no history; decomposed, **44 are pure infrastructure**
(20 disk-full, 16 four-hour-timeout, 8 one-hour-timeout) and resolved **0** between them.
swezero: **0 error records, 500/500 clean.**

⇒ **θ₀ = 119 is a depressed floor, not a clean measurement of base capability**, and the
depression is **one-sided** (the arm had none). Direction is conservative for the headline
— base already wins and repairing it would only widen base's margin — but "119" should not
be quoted as clean.

**Adopted for A1:** count error/no-history records in **both** arms' `output.jsonl` before
running the panel, and report them alongside the scores, so a lopsided infra-failure count
cannot be misread as a capability gap.

### 14.6 Repo-overlap contradiction — settled, clean-301 keeps its label

Two artifacts disagreed on whether swezero overlaps SBV repos. Settled full-file
(`v3_scoping/repo_overlap.py`, 100% extraction, 0 misses): **overlap EXISTS and is exactly
the five the 2026-07-25 audit named** — sympy 157 records, xarray 50, sphinx 26,
astropy 19, seaborn 4 = **256 records = 0.49%** of swezero's 52,473, against 1,501 distinct
training repos. The competing "zero overlap" claim was a top-N view that missed the tail.

⇒ **`clean-301` retains its decontamination justification in both public docs — it is not a
mislabel.** And the sympy collapse (29→3) is *not* a training-repo-mix artifact: 157
records cannot produce it, and the direction is wrong (training on a repo should help).

---

## 15. Training complete — measurements confirmed, §14.3 provisional status lifted (2026-08-04)

_Both arms COMPLETED (exit 0:0, full 1 epoch, no restarts). Filed **before any eval
result exists** — the SBV eval was queued at the same time._

### Final training numbers

| arm | steps | train_loss | eval_arm_eval_loss | `total_flos` | vs control | runtime |
| --- | --- | --- | --- | --- | --- | --- |
| v2 control (mixed 55,000) | 1719 | 0.3146 | — | 2.0045e19 | 1.000× | 13.73 h |
| **A1-tokmatch (headline)** | 1245 | **0.2217** | 0.4487 | **2.0021e19** | **0.9988×** | 9.61 h (0.70×) |
| A1-maxpool | 1640 | 0.2196 | 0.5170 | 2.6381e19 | 1.3161× | 12.57 h (0.92×) |

**The headline arm landed FLOP-matched to its control within 0.12%** — tighter than the
design targeted (the design matched *tokens*; `total_flos` agreeing to 0.12% follows, since
HF's counter is ~token-proportional). Combined with 0.70× wall-clock and 0.72× optimizer
steps, §10/§14's conclusion stands and is now measured rather than argued: **the headline
arm is compute-disadvantaged relative to its control on every axis that the hardware bills
for.** A positive A1-tokmatch cannot be attributed to extra compute.

*What `total_flos` does and does not settle:* it corroborates token-matching. It does **not**
independently arbitrate §8.3's attention argument, because HF's counter ignores attention.
The wall-clock measurement below is what does that.

### §14.3 lifted — full-n step time confirms the early measurement

| run | tokens/record | median s/step | IQR | n |
| --- | --- | --- | --- | --- |
| v2 control | 15,500 | 25.20 | 25.00–25.40 | **339** |
| A1-tokmatch | 21,406 | 25.60 | 25.40–25.80 | **245** |
| A1-maxpool | 21,406 | 25.60 | 25.40–25.80 | **324** |

**+1.6% step time against 1.38× tokens/record, at n=245–324.** §14.3 downgraded this to
provisional pending full n; the early figure held exactly, and devils-advocate's specific
concern (drift with later sequence composition / eval interleave) did not materialise.
**§8.3's ~1.11× analytic FLOP estimate is refuted; §10's correction stands as measured.**

### Training loss moved as predicted, and by a lot

v3 arms 0.2217 / 0.2196 vs control **0.3146** at 1 epoch. Expected direction and magnitude:
condensation records are free-form prose summaries (high entropy), trajectory records are
structured tool calls (low entropy), so removing the prose lowers average loss. This is
**not** evidence the curated arms are better — it is confirmation the intended data change
took effect. Note also `eval_arm_eval_loss` is *higher* for maxpool (0.5170) than tokmatch
(0.4487), consistent with more steps of specialisation away from the mixed carve-out (§13).

### Format-parity pre-flight before spending eval GPU (all clean)

* `chat_template.jinja` sha256 `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`
  — **byte-identical across base, the v2 control, and both v3 arms**, and the same hash the
  v2 root-cause investigation recorded. The format/template/parser class of failure is
  therefore excluded by construction for these arms.
* `tokenizer.json` identical; `model.safetensors` identical size (10,350,019,328) across all three.

### Eval queued (no result yet)

`ARM=v3a1tok` (infer 607381 → score 607382) and `ARM=v3a1max` (infer 607383 → score 607384),
10×50 shards, kit defaults unchanged for comparability. Watch items: `sacct` for
FAILED/TIMEOUT shards (a squeue-only monitor once missed a 100-instance hole for ~8h), the
dash/exit-127 merge bug (hand-merge under `bash -lc`), and the concurrent-vLLM
torch_compile cache race on shared NFS — a coder30b job (607073) is on the kit at the same
time. Error-record counts (§14.5) run **before** the panel.

---

## 16. RESULT — Lever A failed: removing condensation did not help (2026-08-04)

_Both arms evaluated. Reported against §3 / §7 / §8.2 as pre-registered. All four merged
reports verified: 500/500 instances, 0 duplicates, 0 missing._

### The board

| model | resolved / 500 | rate |
| --- | --- | --- |
| base θ₀ | **119** | 23.8% |
| v2 swezero control (mixed 55,000) | **77** | 15.4% |
| **A1-tokmatch (headline, 0% condensation)** | **63** | 12.6% |
| A1-maxpool (0% condensation, +32% tokens) | **62** | 12.4% |

### Paired panel (margin held at 3.0%, per-board instance counts)

| comparison | Δ | McNemar | TOST verdict |
| --- | --- | --- | --- |
| tokmatch vs control, full-500 | −14/500 | p=0.114 n.s. | INCONCLUSIVE |
| tokmatch vs control, ex-sphinx-456 | −14/456 | p=0.114 n.s. | INCONCLUSIVE |
| tokmatch vs control, clean-301 | −14/301 (−4.65%) | p=0.076 n.s. | INCONCLUSIVE |
| **tokmatch vs base 119** | **−56/500** | **p<0.0001** | **DIFFERENT** |
| maxpool vs control | −15/500 | p=0.101 n.s. | INCONCLUSIVE |
| **tokmatch vs maxpool** | **+1/500** | p=1.000 | **EQUIVALENT** |

### Verdict against the pre-registration

* **H-A1 (trajectory-only > mixed) is NOT supported.** Both curated arms landed *below*
  their matched control, consistently in direction (−14, −15/500).
* **This is Falsifier 1, in the rescoped form §8.2 requires:** "Lever A is inert **on the
  conservative source**." swezero has the *least* condensation to remove (34.3%), so a
  rebench replication (41.3%) is the pre-registered next step before Lever A is declared
  dead campaign-wide.
* **Falsifier 2 is NOT met either.** The direction is negative but p=0.076–0.114, so we
  **cannot** claim condensation carries real value. Reported as **inconclusive**, not as
  "condensation helps" — per §8.1, n.s. is not evidence of a reversal any more than of
  equivalence.
* **The extra-data confound (§9.1) turns out to be empty.** maxpool has +32% tokens and
  +395 steps over tokmatch and scores **+1/500, EQUIVALENT**. So "more action data" buys
  nothing here, and the gated third arm (A1-subtract) is **not** worth running: §9.1 gated
  it on a *positive* A1, and there is none.
* **Curation did not close any of the −42 gap to base.** It widened it to −56.

### Mechanism check (§4) — confirmed, but it refutes the assumption behind Lever A

| model | resolved | median history length | empty patch | finish |
| --- | --- | --- | --- | --- |
| base | 119 | **336.0** | 138 | 300/500 |
| swezero control | 77 | 209.5 | 0 | 500/500 |
| A1-tokmatch | 63 | **171.0** | 4 | 500/500 |
| A1-maxpool | 62 | 177.5 | 9 | 500/500 |

**Depth tracks score monotonically across all four models** (336 → 209.5 → 171.0 → 177.5
against 119 → 77 → 63 → 62). §4 predicted that if the depth story were right, a curated arm
would shift *toward* base — more actions, fewer premature finishes. **It shifted further
away:** the condensation-free policy is *shallower still*, and the score followed it down.

So the two halves of the v2 story separate cleanly:

* **The depth↔score mechanism is CONFIRMED** and strengthened — it now holds across four
  models spanning 62–119 resolved.
* **The attribution of shallowness to condensation records is REFUTED.** Removing them made
  the policy shallower, not deeper. The pathology lives in the *trajectory* records
  themselves, not in the condensation pairs — consistent with main-worker's trajectory
  analysis (ran-any-test 90.3% → 16.6%, re-tested-after-last-edit 74.3% → 6.2%, asserts
  success in `finish` while never verifying 1.7% → 91.8%). Condensation records were, if
  anything, mildly *protective* — plausibly the only records in the mix that model
  "stop and take stock" rather than "edit and declare done".

### Measurement hygiene notes

* Both arms: **0 error records, 0 no-history rollouts, 500/500 clean** — no infra asymmetry
  of the kind that depresses θ₀ (§14.5: base had 44 infra failures / 200 no-history). So
  none of this gap is an artifact of lopsided infrastructure failure.
* Correction to an earlier count of mine: tokmatch empty-patch is **4**, not the 33 I first
  reported — the first pass counted duplicate rollout lines rather than deduping by
  `instance_id`.
* A crude "ran a test" regex saturated at 500/500 for all SFT arms and is **uninformative**;
  main-worker's careful instrumentation is the measurement to cite.
* **§8.1 nuance corrected by data:** soup-worker argued equivalence is *structurally*
  unreachable at N=500. That holds for a high-discordance pair (122 discordant → SE≈11/500,
  90% CI ≈ ±18), but tokmatch-vs-maxpool has only **65 discordant** → SE≈8.1/500, half-width
  2.65% < 3.0% margin, so **EQUIVALENT was declarable.** Reachability is a function of
  *discordance*, not of N alone.

### What this means for the campaign

The v3 thesis — that ~40% condensation records were diluting the training signal — **is
wrong as stated**, at least on the conservative source. Lever A is spent unless a rebench
replication says otherwise, and the honest reading is that **raw ADP SWE trajectory data
does not become a usable SFT source for lifting a strong instruct base merely by removing
the summarization records.**

The surviving lever is Lever B (outcome verification) — and it is now better motivated than
before, because the failure mode that *does* correlate with the score is
**never-verifying**, which is exactly what an outcome/verification filter targets. Its
Phase-0 blockers stand (§9.3): no upstream labels on the filesystem, no network, gold-patch
distillation hard-gated on exact `(repo, base_commit)` SBV exclusion, and the cheap
finish-based proxy already measured dead.

---

## 17. CORRECTION — §16's "depth tracks score monotonically" does NOT generalize, and the verification hypothesis is REFUTED (2026-08-04)

_dpf asked to explore the verification-filtered direction. Scoping it produced two
refutations: one of the new hypothesis, and one of a claim **I made in §16 and reported as
the campaign's confirmed mechanism**. Both are recorded here before any GPU was spent._

### 17.1 The training data explains the arms' behaviour exactly — and indicts my source choice

Full-file scan of all four subsets (detector hand-validated, precision ≳97%, residual
false-negative ≈0.1–0.3%; `/checkpoint/dpf/adp-analysis/v3_scoping/verif/`):

| subset | ran ≥1 test | test after last edit | finish-ending records with ZERO test runs |
| --- | --- | --- | --- |
| coderforge | 86.2% | 53.0% | **1.0%** |
| scale | 84.8% | 45.8% | **2.1%** |
| rebench | 89.2% | 46.5% | **1.5%** |
| **swezero** | **0.0%** | **0.0%** | **100.0%** |

**swezero contains ZERO test executions** — across 422,424 bash commands. All 27,268 of its
finish-ending records assert completion having never run a test. Base at eval: 1.7%. So the
arms' 91.8% assert-without-verifying is not a failure to learn — it is a **faithful
reproduction of the training distribution**.

This indicts my own source selection in §7/§8.2. I chose swezero partly because it had the
**highest terminal-record rate (52.0%)**, which I described as the source with "the most
*complete* episodes … precisely the depth signal Lever A is trying to restore." That was
backwards: swezero's high finish rate is the **pathology** (finish without verifying), not a
quality signal.

**What is NOT true:** an external analysis concluded the A1 experiment was "confounded with a
total change of data source". It was not. A1's control was the **v2 swezero arm** — same
source, same file, mixed vs trajectory-only (verified: both A1 manifests trace to
`nvidia_SWE-Zero-.../train.llamafactory.jsonl`, and the v2 swezero arm's tokenized cache
lives in that same directory). Source was held constant. **§16's within-source conclusion
stands.** What is now clear is that it *generalizes poorly*: A1 tested condensation removal
on the one source whose trajectories never verify.

### 17.2 The verification hypothesis is REFUTED (no GPU spent)

Measured eval-side verification behaviour for all seven scored models on rollouts already on
disk (`/checkpoint/dpf/adp-analysis/v3_scoping/eval_verif.py`):

| model | score | ran any test | test after last edit | finish w/o verify | tests/inst |
| --- | --- | --- | --- | --- | --- |
| base | **119** | 51.8% | 41.8% | 9.6% | 13.3 |
| swezero | 77 | 10.2% | 3.2% | 89.8% | 0.2 |
| rebench | 70 | **98.0%** | 48.6% | 1.0% | 17.3 |
| v3_tokmatch | 63 | 0.6% | 0.0% | 99.4% | 0.0 |
| v3_maxpool | 62 | 1.0% | 0.2% | 99.0% | 0.0 |
| coderforge | 48 | **97.8%** | 55.2% | 1.6% | 13.8 |
| scale | **35** | **99.2%** | **60.2%** | 0.6% | **23.9** |

**Spearman(score, ran-any-test) = −0.32. Spearman(score, test-after-last-edit) = −0.43.**

More verification goes with **worse** score. `scale` verifies the most on every measure
(99.2%, 60.2% after-edit, 23.9 tests/instance — nearly double base) and scores the **worst**
(35). Base scores 119 while verifying *less* than three of the arms.

⇒ **Lever B′ is dead before launch.** The base-vs-swezero verification contrast reproduces
directionally, but generalising it to "verification is the lever" is refuted by the other
five models. A verification-filtered arm is **not worth the ~14 GPU-hours**: the three
subsets it would be built from already verify at 80–89% (base-comparable), and their arms
are the *worst* performers.

### 17.3 ⚠️ CORRECTION TO §16: the depth↔score relation does not generalize either

§16 stated, and I reported, that "**depth tracks score monotonically across all four
models**" and called the mechanism CONFIRMED. Extending to all seven models refutes it:

| model | score | median history | median ACTIONS |
| --- | --- | --- | --- |
| base | **119** | 336.0 | 99.5 |
| swezero | 77 | 209.5 | 66.0 |
| **rebench** | 70 | **638.0** | **205.0** |
| v3_tokmatch | 63 | 171.0 | 53.0 |
| v3_maxpool | 62 | 177.5 | 55.5 |
| **coderforge** | 48 | **608.5** | 196.5 |
| **scale** | **35** | **658.5** | **211.0** |

**Spearman(score, median history) = −0.32** — *negative*.

The monotone relation was an artifact of the four models I happened to measure: base plus
three swezero-derived arms, i.e. a single nested lineage. **Within the swezero lineage depth
does track score. Across sources it inverts** — the three longest-trajectory arms are the
three worst.

This is the same error class the campaign keeps repeating: generalising from a convenient
subset. §16's mechanism claim is **withdrawn** as a general statement and reduced to:
*within the swezero lineage, shorter trajectories accompanied lower scores.* It is **not**
evidence that depth causes score.

### 17.4 Honest state of the mechanism question

After Lever A, the verification probe, and this correction:

* **No single behavioural axis measured so far explains the score ordering across sources.**
  Not condensation share, not verification, not trajectory length/action count.
* Base (119) sits in the *middle* on both verification (51.8%) and depth (336 / 99.5
  actions) — it is not an extremum of either.
* The three verbose, verifying arms (rebench/coderforge/scale, ~200 actions, ~98% verifying)
  do **more of everything** and score **worse**, which looks more like unproductive
  thrashing than like diligence.
* So the honest position is: **we do not currently have a validated behavioural mechanism**
  for why SFT on this data underperforms the base model. Any writeup should say that rather
  than carry forward "depth-vs-compliance" or "never-verifying" as established.

**What survives, unchanged:** the *outcome* facts. base 119 ≫ every arm and every combination
method; A1's within-source result (63/62 vs 77); the pooled220k panel. Those are score-level
and do not depend on any mechanism story.

---

## 18. Lever B is a NO-OP: coderforge and rebench were ALREADY success-filtered (2026-08-04)

_Route B-i scoping. Compute nodes turned out to have internet (§17 corrected that false
blocker), so the upstream join was actually attempted. The result closes Lever B without any
GPU, and refutes one of the v2 investigation's two founding root causes._

### 18.1 The finding, verified in ADP's own extractor source

Not inferred from schemas — read out of the code that BUILT these subsets
(`agent-data-protocol/datasets/<config>/extract_raw.py`):

```
coderforge_preview          : split = "filtered_reward1"        # reward == 1 ONLY
nebius_SWE-rebench-...      : if not item["resolved"]: continue # resolved ONLY
nvidia_SWE-Zero-...         : (no filter)
scale_swe_distilled         : (no filter)
```

Confirmed by a **full-population** join (all 319,551 records, not a sample), which matched at
**100.00% on all four subsets**:

| subset | upstream | join key | rate | label upstream | verified-resolved traj records |
| --- | --- | --- | --- | --- | --- |
| coderforge | `togethercomputer/CoderForge-Preview` (`trajectories`/`filtered_reward1`) | `re.sub(r"_\d+$","",stid)` → `trajectory_id` | 100.00% | **`reward`** | **48,650 (all 1.0)** |
| rebench | `nebius/SWE-rebench-openhands-trajectories` | `stid` → `trajectory_id` | 100.00% | **`resolved`** | **46,914 (all 1)** |
| swezero | `nvidia/SWE-Zero-openhands-trajectories` | `stid` → `trajectory_id` | 100.00% | **none exists** | 0 |
| scale | `AweAI-Team/Scale-SWE-Distilled` | positional `(data_source, row_idx)` | 100.00% | **none exists** | 0 |

Upstream base rates for context: coderforge 58–63% reward-1 across its unfiltered splits;
rebench 32,161/67,074 = 47.9% resolved. So the filtering removed a *lot* — it just happened
before ADP normalisation, and the label was dropped as a constant.

### 18.2 ⇒ Lever B was already run by the v2 campaign, unknowingly

| source | outcome filtering | v2 arm score |
| --- | --- | --- |
| **coderforge** | **100% reward==1** | **48** |
| **rebench** | **100% resolved==1** | **70** |
| swezero | none (upstream is explicitly *"Execution-free"* fine-tuning) | **77** |
| scale | none / unlabeled | 35 |

**The two fully outcome-verified arms scored 48 and 70 — both below the unfiltered swezero
arm's 77.** "Train only on trajectories verified to have solved the issue" is therefore not
an untested idea in this campaign; it is what coderforge and rebench *are*, and it did not
produce a winner.

*Stated limitation:* this is a **cross-source** comparison, so it is confounded (§17 showed
these sources differ enormously in behaviour). It is strong evidence that success-filtering is
not *sufficient*, not proof that it is worthless. The clean within-source test is described in
§18.4 — and note it runs in the **opposite** direction from the campaign's intent.

### 18.3 ⚠️ CORRECTION: v2 root-cause hypothesis #2 is refuted

The v2 investigation named two candidate data defects. #1 (~40% condensation) was refuted by
A1 (§16). #2 was:

> **NO success filtering** (no outcome label exists) → imitates unresolved/looping
> trajectories wholesale.

**That is false for coderforge and rebench.** Both are 100% success-filtered. The error was
inferring *"unfiltered"* from *"unlabeled"*: the records genuinely carry no resolved field —
that observation was correct — but the filtering had already happened upstream, which is
exactly why no label survived. It is only true for swezero and scale, and swezero is the
**best** arm.

⇒ **Both of the v2 investigation's candidate mechanisms are now refuted.** Combined with §17
(no behavioural axis explains the cross-source ordering), the campaign has **no surviving
explanation** for why SFT on this data underperforms the base model.

### 18.4 What is actually left

1. **Gold-patch distillation (B2) — the one lever still standing, and it is now unblocked.**
   Gold patches are recoverable for **rebench 100%** (46,914), **coderforge 99.1%** (48,215),
   swezero ~100% (68.6% verified + the rest in a non-parquet slice), scale 35.2%. This is a
   genuinely *different objective*: train on the **correct fix**, not on agent behaviour —
   which sidesteps every behavioural pathology found so far, because it stops imitating
   agents altogether.
2. **The within-source test of the filter, which runs backwards.** Negatives exist upstream
   (coderforge **102,990** reward-0; rebench **34,913** unresolved), and re-extracting them is
   a two-line change to the ADP extractors. But the experiment that would isolate the filter's
   value is *adding unresolved trajectories to see if it hurts* — the reverse of "add
   verification". Cheap for rebench (2.08 GB); coderforge is 72.77 GB.
3. **swezero/scale cannot be labelled** without executing agent patches against F2P tests.

### 18.5 ★ Contamination gate: CLEAN — and this is good news for the whole campaign

@devils-advocate's §9.3 gate, run at exact-commit granularity against all 500 SBV instances
(499 distinct base_commits), using two independent commit sources per record (prompt regex +
`instance_id` → benchmark table; they agree):

| subset | traj records | base_commit resolvable | **exact (repo, base_commit) ∈ SBV** | repo-level |
| --- | --- | --- | --- | --- |
| coderforge | 48,650 | 16,708 | **0** | 295 |
| scale | 47,183 | 47,183 | **0** | 0 |
| rebench | 46,914 | 46,914 | **0** | 0 |
| swezero | 52,473 | 52,473 | **5 records / 4 trajectories** | 256 |

* **Instance-level train-on-test: 0 / 319,551 records.** VERIFIED.
* The 5 exact-commit hits are **different PRs opened against the same parent commit** as an
  SBV instance (sympy 15602/12471/17651/13549 vs SBV 15599/12489/17655/13551) — same
  repository snapshot, **never the same task, never the SBV gold patch**. None of the four
  instance_ids is an SBV instance_id.
* SWE-smith records (31,507) contribute **zero** overlap: those instances live on synthetic
  `swesmith/<org>__<repo>.<sha8>` forks that map to no SBV repo.

⇒ The campaign is clean at the level that matters, and a gold-patch arm would be safe with a
`--exclude-commit` filter over the 499 SBV base_commits (costing <0.6% of the data).

