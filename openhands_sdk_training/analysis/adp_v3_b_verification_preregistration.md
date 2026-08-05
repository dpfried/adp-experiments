# ADP-v3 / Lever B′ — pre-registration: filtering on VERIFICATION BEHAVIOUR

_Written **before** the scoping numbers were in and **before** any arm was trained, so the
hypothesis, the falsifiers, and — most importantly — the **confound design** are fixed in
advance. Numbers marked `[SCOPE]` are filled from the scoping pass by dated amendment;
nothing else moves. Companion to `adp_v3_a1_preregistration.md` (Lever A, refuted)._

Status: **pre-registered. No arm trained. No GPU spent.**

---

## 1. Why this lever, and why it is not the one the campaign doc proposed

The campaign doc's Lever B was **outcome verification**: train only on trajectories that are
*verified to have resolved the issue*. That is blocked on label recoverability.

Two results since then redirect it:

1. **Lever A is refuted** (`adp_v3_a1_preregistration.md` §16). Removing the ~40%
   condensation records did **not** help: 63 and 62 /500 vs the mixed control's 77 and
   base's 119. The condensation hypothesis is dead as an explanation.
2. **The failure mode that actually correlates with score is not-verifying.** From
   trajectory-level analysis of the eval rollouts (main-worker): ran-any-test
   **90.3% → 16.6%**, re-tested-after-last-edit **74.3% → 6.2%**, median test runs
   **28.5 → 0**, and *asserts success in the final `finish` message while never having
   verified* **1.7% → 91.8%**. SFT did not damage localization (median first-edit turn 37
   base vs 33 arm) — it removed the **verify loop** and substituted text claiming
   verification.

Meanwhile depth tracks score monotonically across four models (median trajectory length
336 / 209.5 / 171.0 / 177.5 against 119 / 77 / 63 / 62 resolved).

**So the lever becomes: filter the training data on verification behaviour that is visible
in the trajectory itself** — records where the agent *ran a test after its last edit*.

**Why this is strictly easier than the campaign doc's Lever B:** it requires **no outcome
labels, no upstream join, no network, no containers, no gold patches**. It is computable
from the action stream already on disk. It therefore sidesteps the entire label-recovery
blocker — and it targets the *measured* failure mode rather than a proxy for it.

---

## 2. THE CONFOUND, stated before any data is looked at

**Verifying costs actions.** A trajectory that runs tests after editing is, mechanically,
likely to be *longer* than one that does not. And we have just established that **depth
tracks score monotonically**. Therefore:

> A verify-positive filter may be, in substance, a **length filter** — and the A1 result
> already predicts that longer/deeper training data behaves differently. If we compare a
> verify-positive arm against the raw mixed control and it wins, **we will not know whether
> we selected for verification or merely for length.**

This is the same class of error as Lever A's, where "trajectory-only" silently also meant
"more action data per record". It is written down here so the design has to answer it.

Secondary confounds, also pre-declared:
* **Difficulty** — harder instances may induce more verification, so verify-positive may
  over-represent hard (or easy) instances.
* **Repo/environment** — some repos have a trivially runnable test command; verify-positive
  may over-represent those, which is a distribution shift, not a skill.
* **Detector validity** — a bad test-detector invalidates everything. A crude "ran a test"
  regex applied to the *eval* rollouts in this campaign **saturated at 500/500 and was
  worthless**. The detector must be hand-validated on both matches and near-misses, and its
  exact rule recorded, before any arm is built.

## 3. Design — the length-matched control is the whole point

| arm | data | role |
| --- | --- | --- |
| **B′1 — verify-positive** | N records that ran a test after their last edit | treatment |
| **B′2 — verify-negative, LENGTH-MATCHED** | N records that did **not** verify, resampled to match B′1's **token-length distribution** | the control that isolates verification from length |
| v2 swezero control (trained, 77/500) | mixed 55,000 | external reference |
| base θ₀ (119/500) | — | anchor |

**Recipe frozen identical to v2/A1** (Qwen3.5-4B, `qwen3_5_nothink`, cutoff 32768, LR 1e-5,
cosine, warmup 0.03, 1 epoch, global batch 32, seed 42, ZeRO-2, liger + fa2, 8×A100), and
**token-matched** to the reference where the yield allows — A1 showed token-matching also
lands FLOP-matched within ~0.1% on this setup, which is what made its result attributable.

**Sequencing (cost-disciplined, mirroring how the A1-subtract arm was gated):**
1. Run **B′1** alone first as a screen.
2. **Only if B′1 beats the v2 control** do we run **B′2**. If B′1 does not clear the control,
   there is nothing to attribute and B′2 is not worth the GPU.

This is deliberate: B′2 exists to answer *"verification or length?"*, a question that only
arises on a positive B′1.

## 4. Hypotheses, falsifiers, decision rules

Primary metric: **SWE-bench Verified single-run temp-0 pass@1**, OpenHands scaffold,
reported on all three boards (full-500 / 456-ex-sphinx / clean-301). Paired **McNemar** vs
each comparator plus **TOST**, margin held at **3.0% and converted per board** (15/500,
14/456, 9/301) — never a fixed instance count, which silently loosens the bar on smaller
boards.

**H-B′:** a verification-filtered arm beats its matched control on SBV pass@1, and shifts
the depth diagnostics toward base.

* **Supported if** B′1 > v2 control by McNemar **and** B′1 > B′2 (length-matched) — the
  second conjunct is what licenses attributing the gain to verification.
* **Falsifier 1 (no effect):** B′1 ≈ v2 control ⇒ verification-behaviour filtering does not
  lift pass@1 at this budget. Given Lever A also failed, this would mean **behavioural
  filtering of raw ADP trajectories is not sufficient**, and the remaining hope is genuine
  outcome labels (Route B-i) or a different objective entirely (RL / gold-patch
  distillation).
* **Falsifier 2 (it was length):** B′1 > control **but B′1 ≈ B′2** ⇒ the gain is from
  training on longer/deeper trajectories, **not** from verification behaviour. This would
  still be a real and useful finding — "train on deeper trajectories" is actionable — but it
  must **not** be reported as "verification is the lever".
* **Falsifier 3 (reversed):** B′1 < control ⇒ verify-positive records are actively worse,
  pointing at a difficulty/repo distribution shift rather than a skill.

**Pre-committed non-claims:**
* B′1 is **not** predicted to reach base (119). Partial recovery is still informative;
  Lever A moved the wrong way, so any move toward base is news.
* **Correlational until B′2 lands.** Until the length-matched control exists, a positive B′1
  is reported as *"verify-positive data trains better, attribution unresolved"*.
* Base stays anchored at **θ₀ = 119/500** single-run. Not re-anchored. Noted (§14.5 of the
  A1 prereg) that 119 is itself a *depressed* floor — 44 base rollouts failed on pure
  infrastructure and resolved 0 — so 119 understates base, which makes the SFT gap if
  anything larger, not smaller.
* Single-run verdicts are **different / inconclusive only**; equivalence is claimed only
  when the CI actually fits the margin (reachability depends on **discordance**, not N — 65
  discordant made it declarable in A1 where 122 would not).

## 5. Mechanism check (pre-registered, first-class)

The score alone cannot tell us whether the *behaviour* transferred. From the rollout JSONL,
for B′1 / B′2 / control / base:

* **ran-any-test rate** and **re-tested-after-last-edit rate** ← the direct analogues of the
  training filter; these are the ones that matter
* **asserts-success-while-never-verifying rate** (base 1.7% vs arm 91.8%)
* median trajectory length, finish rate, empty-patch rate

**Prediction if H-B′ is right:** B′1's *re-tested-after-last-edit* rate rises materially
above the control's 6.2% and toward base's 74.3%. **A score gain with no shift in
verification behaviour would falsify the mechanism even if the number went up**, and must be
reported that way — the same trap A1 §4 set and which A1 then walked into from the other
direction (depth moved, but away from base).

## 6. Yield risk (the thing most likely to kill this before it starts)

If the eval-side rate transfers to the training data, verify-positive records could be
**rare**. `[SCOPE]` will give the real number. Consequences, pre-declared:

* If yield ≥ ~40k from one source → run at a token-matched budget, clean.
* If yield is small (say < 10k) → the arm is in a **different compute regime** from the
  55k-record control, and a raw comparison against it is invalid. In that case the honest
  comparison is against a **same-size, same-source, unfiltered** arm, and the result is
  reported as compute-matched-not-data-matched.
* If yield is tiny (< ~2k) → **do not train.** Report the yield as the finding: raw ADP
  trajectories contain almost no verification behaviour to learn from, which itself explains
  why every SFT arm stopped verifying.

That last outcome would be a *result*, not a failure, and it is cheap to obtain.

## 7. Provenance
* Lever A result this builds on: `adp_v3_a1_preregistration.md` §16.
* Scoping (in flight, read-only): `/checkpoint/dpf/adp-analysis/v3_scoping/verif/VERIFICATION_FILTER.md`
  (detector + yield) and `.../leverB/UPSTREAM_LABELS.md` (Route B-i, now unblocked —
  compute nodes have internet, contrary to the earlier Phase-0 conclusion).
* Kit: `openhands_sdk_training/v3_curation/` (`build_curated_subset.py` already supports
  arbitrary record filters via `--drop-generation` / `--keep-record-type`; a
  verification-behaviour filter needs a new predicate, to be added with its own tests).
