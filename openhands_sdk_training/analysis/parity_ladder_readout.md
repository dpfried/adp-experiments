# Parity-ladder readout — arm ladder complete, base cells pending

Read in the order pre-registered in `parity_ladder_amendment.md`: manipulation
checks first, then the rungs, adjacent pairs only, assigning any effect to the
smallest rung that produces it.

**Status:** all four arm cells complete and scored at 100/100 (50 per shard, 0
error rows, 0 duplicate instances, merged totals exactly 50 each). Base cells
still inferring — F at 85 rows/50 unique on shard 00 and scoring on shard 01,
E at 21/50 and 19/50. **Everything about the base model below is provisional.**

---

## 0. Manipulation checks

**S0 — served prompt.** Verified on the live servers, not just from a flag: with
`chat_template_kwargs={"enable_thinking": false}` (what the harness always
sends), stock renders 13 tokens ending `[248068, 271, 248069, 271]` and nostub
renders 9, byte-identical prefix. The delta is exactly `<think>\n\n</think>\n\n`
and nothing else, on both models. Detail in amendment addendum 3.

**Native-`<think>` channel accounting.** The registered contingency fired, and
it fired asymmetrically — which is the whole reason it was registered:

| cell | model | template | records emitting native `<think>` |
|---|---|---|---|
| A | arm | stock | 0 / 100 |
| B | arm | nostub | **0 / 100** |
| C | arm | nostub | **0 / 100** |
| D | arm | nostub | **0 / 100** |
| F | base | nostub | **100 / 100** |

Blocks are `agent:thought` — model-generated, verified by role, not prompt text
echoed into history.

Two consequences. First, **S1 (B−A) stands exactly as registered for the arm**:
removing the stub does not open a native reasoning channel for the arm, so a
change in think-*tool* rate cannot be reasoning relocating between channels.
The arm emits zero native blocks in 300 rollouts across three nostub cells,
consistent with 0 `<think>` tokens in its 6.5M training tokens — it has been
trained *out of* the capability, not merely gated out of it.

Second, for the **base** model the stub is doing almost all the suppressing:
5/324 (1.5%) under stock → 100/100 under nostub. Base cells therefore require
the combined native+tool accounting registered in addendum 2, and F−B/F−E must
not be read on the tool channel alone.

---

## 1. The arm rungs

Paired on the same 100 instances. `b`/`c` are discordant counts (resolved in the
higher cell only / lower cell only); `SE = sqrt(b+c)` is the standard error of
the net difference under the null, i.e. a McNemar-style test. Registered gate:
`|net| < 2·SE` is uninformative.

| rung | what it adds | hi | lo | net | b | c | disc | SE | \|d\|/SE | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| **S1** B−A | stub removed | 8 | 7 | +1 | 6 | 5 | 11 | 3.32 | 0.30 | noise |
| C−B | wrapper & path matched | 7 | 8 | −1 | 2 | 3 | 5 | 2.24 | 0.45 | noise |
| **L2** D−C | prohibition & 5-phase list | 11 | 7 | +4 | 6 | 2 | 8 | 2.83 | 1.41 | noise |
| **L1** C−A | format rungs jointly | 7 | 7 | ±0 | 3 | 3 | 6 | 2.45 | 0.00 | noise |
| D−A | full training parity vs stock eval | 11 | 7 | +4 | 8 | 4 | 12 | 3.46 | 1.15 | noise |

**L1 PASSES, and cleanly.** Predicted `|C−A| ≤ 6/100` with the format rungs
jointly inert; observed exactly 0, with only 6 discordant instances. Removing
the stub and matching training's wrapper and absolute-path style changes the
arm's score not at all.

**S1: no score effect.** +1 at 0.30·SE. Note the churn underneath it — 11
instances flip to produce a net of 1 — which is the first sign that this design's
score channel is dominated by noise rather than treatment.

**L2 (D−C) is the largest single mover and still does not clear the gate.** +4 at
1.41·SE. No directional prediction was registered for D, so nothing is falsified.
The honest statement is *suggestive, underpowered* — restoring the semantic
content of the training prompt (the prohibition and the 5-phase list) is the only
rung that moves score at all, but not detectably at n=100.

Per the registered attribution rule, no effect is assigned to any format rung.

---

## 2. The result that reframes the ladder: a same-condition replicate

Cell A is, by construction, the board condition reproduced — same arm weights,
stock template, harness's own `default.j2`, same instances. So A versus the
board run `v2_swezero_4b` is a **same-condition replicate**, the thing I told
devils-advocate I did not have and would not fabricate. It arrived free.

| | resolved on the 100 |
|---|---|
| cell A (this ladder, 2026-08-05) | 7 |
| board `v2_swezero_4b` (2026-07-25) | 11 |
| resolved in **both** | **2** |
| resolved in exactly one | **14** |

Net −4 at 1.07·SE — consistent with noise, so the two runs do not disagree in
any tested sense. But the *composition* is the finding: **14 of 100 instances
flip between nominally identical runs, and the two runs' resolved sets overlap on
only 2 instances (Jaccard 0.12).** At a ~10% resolve rate this model+harness pair
barely reproduces *which* problems it solves.

**This churn is larger than every effect in the ladder.** Compare 14 discordant
against the rung table: 11, 5, 8, 6, 12. The ladder's score channel is measuring
a quantity whose run-to-run instability equals or exceeds all of its treatment
effects. That is a statement about the instrument, not about the treatments.

**One caveat, stated because it cuts against using this as a clean noise
estimate:** the board run is 2026-07-25 and cell A is 2026-08-05, and the harness
changed in between (this repo has a commit for a harness bug that cost rebench 6
instances). So 14 discordant **bundles decode nondeterminism with harness drift
and is an upper bound on pure decode churn.** For the ladder's *internal*
comparisons this bound is conservative in the right direction: all six cells ran
within the same hour on one harness build, so their mutual noise should be no
larger than this.

---

## 3. What this says about the motivating hypothesis

The ladder exists to test whether the arms' deficit against the base model is an
artifact of evaluating them out of distribution — trained one way, scored another.
The answer from the arm side is **no**.

Cell D is the arm evaluated as close to *as it was trained* as this harness
allows: no thinking stub, training's wrapper and absolute-path phrasing,
training's prohibition and phase list. It scores 11/100 against stock-eval's
7/100 — a +4 that does not clear noise. Full parity does not rescue the arm.

For scale, on these same 100 instances the base single-run θ₀ scored 20. Against
cell D that is net +9 at 2.18·SE, the only comparison in the entire table that
clears the gate, and it favours the **base** model. **That comparison is
cross-harness (2026-07-25 vs 2026-08-05) and therefore provisional** — cell E is
the within-harness base number and is still running. I am flagging the direction,
not banking the magnitude, until E lands.

Corroborates the existing finding that the adp-v2 SFT lift is null-to-negative;
adds that prompt/format mismatch is **not** the explanation for it.

---

## 4. What the score channel cannot do, and what to read instead

At n=100 and ~7–11% resolve rates, with same-condition churn of ~14 instances,
this design can only detect score effects of roughly ±7 instances or more. Every
arm rung is below that. The score channel has returned a real and useful negative
(L1 passes; format parity is inert) but it cannot adjudicate L2.

The behavioural channel is where the remaining information is: `n_actions`,
`ran_any_test`, `verified_after_edit`, `finish_claims_success`, and think-tool
rate are per-instance continuous or high-base-rate measures, so their paired SEs
are far tighter than a 7-vs-11 count comparison. That pass (`extract_traj_stats.py`
across all six cells, then `ladder_readout.py --stats`) is next, and is where S1's
actual registered outcome — the think-tool rate — gets read.

Deliberately not done: raising n to chase L2. That is new GPU scope and is dpf's
call, not mine. The number that would justify it is now measurable rather than
guessed — detecting a true +4 against this churn needs roughly 4× the instances.
