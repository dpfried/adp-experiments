# Parity-ladder readout — arm ladder complete, base cells pending

Read in the order pre-registered in `parity_ladder_amendment.md`: manipulation
checks first, then the rungs, adjacent pairs only, assigning any effect to the
smallest rung that produces it.

**Status:** all four arm cells complete and scored at 100/100 (50 per shard, 0
error rows, 0 duplicate instances, merged totals exactly 50 each). Base cells
still inferring — F complete on shard 01 and finishing shard 00, E at 22/50 and
24/50 with ~12h of walltime left at ~5.5 instances/h. **Everything about the base
model below is provisional.**

**Before reading any base number, see amendment Addendum 5.** The harness's
multi-attempt aggregator breaks a rank tie in favour of the *latest* attempt with
a rank function that never checks whether a patch exists, so a degenerate final
retry silently discards a good earlier patch. It is asymmetric by construction —
it can only bite a model that retries, which is base (0 of 400 arm records
affected; 9 of 50 in F shard 01, none rescued because F has no error rows). The
arm ladder below is untouched and the θ₀ = 119/500 anchor is immune (single-attempt
run, no attempt files). Base cells will be reported both as-harness and repaired.

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
arm's score not at all. (Score only — behaviourally the stub is *not* inert;
§3c-quater.)

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

## 3b. The behavioural channel — a clean format/semantic dissociation

Paired on all 100 instances (every arm cell has 100/100 transcripts, so the
intersection is the full set). `±SE` is the SE of the mean paired difference;
registered gate `|mean Δ| < 2·SE` is uninformative. 15 metrics; only those
clearing the gate are called out.

**S1 (B−A, stub removed) — the registered primary outcome is NULL.**

| metric | A | B | Δ ± SE | |
|---|---|---|---|---|
| `n_think` | 2.59 | 2.86 | +0.27 ± 0.225 | uninformative |
| `n_turns_with_thought` | 0.000 | 0.000 | ±0.000 | zero in every cell |
| `verified_after_edit` | 0.060 | 0.040 | −0.020 ± 0.028 | uninformative |
| `ran_any_test` | 0.120 | 0.140 | +0.020 ± 0.040 | uninformative |
| `n_edits` | 10.71 | 7.66 | −3.05 ± 1.268 | clears (2.4σ) |

Removing the thinking stub does **not** change the arm's think-tool usage
detectably. *(Superseded — see §3c-quater: it does, at 3.1σ, once the counts are
normalized by trajectory length. The raw comparison below is diluted by A's
longer trajectories. The rest of this subsection stands.)* `n_edits` is the only
metric of 15 that clears, and with 15 metrics ×
3 rungs at a 2σ gate roughly two false positives are expected — a single isolated
hit with no coherent companions is exactly what multiplicity noise looks like, so
**I am not claiming it.** Note also `n_turns_with_thought = 0.000` in every arm
cell, independently corroborating the 0/100 native-`<think>` count: the arm does
not use the native thought channel at all.

**C−B (wrapper & path matched) — nothing clears on any of 15 metrics**, raw or
length-normalized. So the wrapper and the absolute-path change really are inert
on both channels.

The **stub** rung is not: with `C−A = ±0` on score but think-per-action up 3.97σ
across A→C, L1 holds as *score* inertness and as behavioural inertness for
everything except the think channel. See §3c-quater. Stating it as "inert on both
channels" was too strong.

**D−C (prohibition & 5-phase list) — large, coherent, and all in one direction.**

| metric | C | D | Δ ± SE | σ |
|---|---|---|---|---|
| `ran_any_test` | 0.160 | 0.060 | −0.100 ± 0.036 | 2.8 |
| `verified_after_edit` | 0.080 | 0.010 | −0.070 ± 0.026 | 2.7 |
| `n_think` | 3.08 | 2.54 | −0.54 ± 0.211 | 2.6 |
| `n_actions` | 95.80 | 76.38 | −19.42 ± 7.810 | 2.5 |
| `n_terminal` | 43.75 | 33.94 | −9.81 ± 4.029 | 2.4 |
| `n_file_editor` | 47.95 | 38.87 | −9.08 ± 4.010 | 2.3 |

Six metrics clear, every one negative. Unlike the isolated `n_edits` hit, these
are mutually coherent and their direction was predictable from the prompt text
before looking (see below), so multiplicity does not explain them.

**Per the registered attribution rule, the effect is assigned to D−C — the
semantic rung — and to no format rung.** The ladder's central design question
("is the arm's deficit a format artifact?") gets a clean answer: format changes
nothing on either channel; the semantic content changes behaviour substantially.

## 3c. Why D behaves that way — the mechanism is explicit in the prompt

This is not a subtle distributional effect. Line 13 of the training prompt
(`prompts/train_swezero_noenv.j2`) says:

> The development environment is unavailable. This means you **CANNOT RUN PYTHON
> CODE for any purpose. Do not write or execute any tests. Do not use Python to
> check your work in any way.** Do not install any packages.

And the phase lists differ in exactly the way that implies. The eval prompt has 8
phases including **2. RUNNING** (install and run the tests), **4. TEST CREATION**,
and **7. VERIFICATION**. The training prompt has 5 and contains none of them:
READING → EXPLORATION → FIX ANALYSIS → FIX IMPLEMENTATION → FINAL REVIEW.

So `ran_any_test` −0.100, `verified_after_edit` −0.070 and `n_actions` −19.4 are
the model **complying with an instruction not to test or verify**. D is the arm
evaluated under the prompt it was trained on, and that prompt forbids the
verification behaviour SWE-bench rewards.

### 3c-bis. Decomposition — and a correction to the paragraph above

I audited the `ran_any_test` / `verified_after_edit` classifier rather than
trusting it, because both are low-base-rate flags carrying a 2.7–2.8σ claim. The
extractor's `test_turns` is a **composite**: a strict test-runner regex
(`pytest|tox|unittest|runtests|manage.py test|…`) **OR** a python-run command
mentioning `repro`/`test`. Decomposing D−C over the extractor's own categories:

| metric | C | D | Δ ± SE | σ | |
|---|---|---|---|---|---|
| `repro_created` | 0.300 | 0.020 | −0.280 ± 0.047 | **6.0** | clears |
| `cmd_pyrun` | 0.430 | 0.000 | −0.430 ± 0.118 | 3.6 | clears |
| `ran_any_test` | 0.160 | 0.060 | −0.100 ± 0.036 | 2.8 | clears |
| `verified_after_edit` | 0.080 | 0.010 | −0.070 ± 0.026 | 2.7 | clears |
| `cmd_pkg` | 0.070 | 0.000 | −0.070 ± 0.029 | 2.4 | clears |
| `cmd_test` (strict runner) | 3.140 | 1.040 | −2.100 ± 1.626 | 1.3 | **does not clear** |
| `ran_any_test_ok` | 0.050 | 0.050 | ±0.000 ± 0.014 | 0.0 | **no change at all** |

An independent recount straight off the raw terminal commands agrees: strict
test-runner *presence* is 0.04–0.07 in **every** cell with no D−C effect
(+0.020 ± 0.014), while any-python-execution presence drops 0.270 → 0.080
(−0.190 ± 0.044, 4.3σ).

**Correction to the paragraph above.** Saying the prohibition stops the arm
"running tests" was too broad. The arm almost never invoked a test runner in
*any* cell, and the rate of test runs that **actually executed successfully is
unchanged** (`ran_any_test_ok` 0.050 → 0.050). What the prohibition eliminates is
precisely the three things line 13 names: **writing reproduction scripts**
(0.300 → 0.020 — the ladder's single strongest effect at 6.0σ), **executing
Python** (0.430 → 0.000, literally zero), and **installing packages** (0.070 →
0.000). The `ran_any_test` and `verified_after_edit` drops are real but are
driven by the `repro`-script component of that composite, not by test-runner
invocation.

**This makes the picture more coherent, not less.** *Achieved* verification was
already at the floor (0.05) and did not move — and score did not move either
(D−A = +4, noise). Attempted verification changed a great deal; achieved
verification changed not at all. The two nulls agree.

Two independent lines now point the same way: aligning the eval prompt to
training does not improve score (D−A = +4, noise), and aligning it makes the arm
verify *less* (D−C, 2.7σ). The out-of-distribution explanation is refuted from
both directions.

### 3c-ter. A second correction: length normalization kills four of the six D−C effects

Prompted by devils-advocate, who noticed `n_think` fell in D with nothing in the
prompt to explain it. They were right, and the fix generalizes.

D contracts the **whole trajectory** by ~20% roughly uniformly: `n_actions`
×0.797, `n_terminal` ×0.776, `n_file_editor` ×0.811, `n_think` ×0.825. So a
per-trajectory *count* can fall simply because the trajectory got shorter.
Re-checking every D−C count metric as a rate per action:

| metric | kind | raw σ | σ per action | verdict |
|---|---|---|---|---|
| `repro_created` | binary | 5.9 | — | real |
| `cmd_pyrun` | count | 3.6 | **3.27** | real |
| `n_test_runs` | count | 1.4 | **2.26** | real (*stronger* normalized) |
| `cmd_pkg` | count | 2.4 | **2.20** | real |
| `n_terminal` | count | 2.4 | 1.21 | **length artifact** |
| `cmd_search` | count | 2.2 | 0.36 | **length artifact** |
| `n_file_editor` | count | 2.3 | 0.79 | **length artifact** |
| `n_think` | count | 2.6 | **0.56** | **length artifact** |

The think-per-action rate is flat — paired Δ = +0.00212 ± 0.00380 (0.56σ), pooled
0.0322 → 0.0333, i.e. if anything *up*. **There is no think effect in D to
explain**, which is why the prompt says nothing about thinking. The apparent
anomaly was an artifact of reporting a count for a shortened trajectory.

So §3b's "6 metrics clear in D−C" overstates the number of *independent* effects.
The honest decomposition of D−C is: **(i)** a ~20% uniform trajectory
contraction, and **(ii)** specific suppression of the three behaviours line 13
names, which survive normalization. The other count drops are (i) restated, not
separate findings. Binary presence flags (`repro_created`, `ran_any_test`,
`verified_after_edit`) are immune to the artifact by construction.

Reporting counts where rates were meant is the sixth silent-degradation defect of
the day. Like the others it produced plausible numbers and raised nothing.

### 3c-quater. The same correction run in reverse: L1 is **not** inert on 15/15

Normalization is not a filter that only removes findings. Applying it uniformly to
every rung — which I did only after wiring it into `ladder_readout.py` rather than
computing it by hand for D−C — turns up an effect on the **stub rung** that the
raw counts hid, in the opposite direction to the D−C corrections:

| rung | metric | raw Δ ± SE | σ | per-action Δ ± SE | σ | sign test |
|---|---|---|---|---|---|---|
| B−A | `n_think` | +0.27 ± 0.225 | 1.2 | **+0.0114 ± 0.0037** | **3.11** | 67↑/33↓, z=+3.40 |
| B−A | `think_arg_chars_total` | +472 ± 545 | 0.9 | **+21.5 ± 8.4** | **2.55** | 61↑/39↓, z=+2.20 |
| C−A | `n_think` | +0.49 ± 0.229 | 2.1 | **+0.0132 ± 0.0033** | **3.97** | 66↑/33↓, z=+3.32 |
| C−A | `think_arg_chars_total` | +829 ± 585 | 1.4 | **+17.7 ± 6.2** | **2.86** | 65↑/35↓, z=+3.00 |
| C−B | `n_think` | +0.22 ± 0.218 | 1.0 | +0.0018 ± 0.0041 | 0.45 | 56↑/43↓, z=+1.31 |

A ratio metric can be outlier-driven, so this is checked two further ways before
being claimed: the **medians** are positive and close to the means (B−A `n_think`
median +0.0098 vs mean +0.0114), and a distribution-free **sign test** on the
paired differences agrees at the same strength. It is not one instance carrying
the mean.

**What it means.** A's trajectories are longer (`n_actions` 109.1 vs B's 92.0,
itself only 1.7σ), so the raw think-count comparison was diluted. Per action, the
arm calls the `think` tool **~34% more often** with the thinking stub removed, and
writes proportionally more into it. By the attribution rule this belongs to the
**smallest rung that shows it**: B−A, the stub. C−B is flat, so the wrapper and
absolute-path changes remain genuinely inert; D−C is flat too, so this is not the
prohibition.

**The correction to make explicit:** §3b's "the format rungs are inert on 15/15
behavioural metrics" is **wrong as stated** and should read *inert on score, and
inert on 13 of 15 behavioural metrics — the two think-channel metrics move once
length-normalized*. The stub is not behaviourally inert; it is **score-inert**.

**What does not change.** The format-OOD explanation for the base-vs-arm score gap
is still refuted, and by the same evidence as before: `C−A = ±0` on score, and now
also a *named, measured* behavioural change that carries **no** score consequence.
An intervention that visibly moves the model's reasoning rate and moves score by
zero is stronger evidence against format-OOD than an intervention that moved
nothing at all, because it rules out "the manipulation never reached the model."

## 3d. The training-data claim, measured directly instead of inferred

In §3c-bis I wrote that the swezero trajectories "contain almost no
reproduction-script demonstrations," and offered cell D's 0.02 as the evidence.
**That inference was invalid** and devils-advocate was right to reject it: 0.02
is the rate under a prompt that *forbids* the behaviour, so it measures
suppression, not demonstration density. Eval-time behaviour cannot establish a
property of the training data.

The property is directly measurable, and the measurement already existed:
`analysis/traj_compare/extract_train_stats.py`, run 2026-08-02 over the arms' own
training files. Per source trajectory:

| training data | n traj | repro | ran_test | verified_after_edit | mean `cmd_pyrun` |
|---|---|---|---|---|---|
| **swezero** | 49,030 | **0.9%** | **0.7%** | **0.1%** | **0.00** |
| rebench | 30,096 | 74.0% | 88.0% | 59.5% | 11.11 |
| coderforge | 43,898 | 65.2% | 79.9% | 60.8% | 8.16 |
| scale | 34,599 | 55.0% | 80.1% | 53.6% | 9.23 |

**The claim survives, on proper evidence.** swezero's training data really is
essentially devoid of verification — repro at 0.9% over 49,030 trajectories,
`cmd_pyrun` at a flat 0.00 — and it is a two-orders-of-magnitude outlier against
the other three arms, whose data is *full* of verification (55–88%).

**But devils-advocate's counter-argument fails, and its failure is informative.**
Their step was "the arm cannot produce a behaviour at 0.30 that it never saw
demonstrated, therefore the data contains it." The scan refutes the conclusion, so
the premise is what's wrong: the arm produces repro at 0.30 under a neutral prompt
because that is **retained pretraining capability that SFT did not fully
remove**, not a learned demonstration. The three numbers line up as a gradient —
base ≈0.9 → arm under a neutral prompt 0.30 → training data 0.009 — so SFT moved
the arm most of the way toward its data's distribution, and the training prompt at
eval time closes the remaining gap (0.30 → 0.02 ≈ 0.009). That is a better story
than either of us had, and it is the one the data supports.

### And it retracts the lever I proposed

I previously wrote that the fix is to regenerate training data *with* an
environment and verification steps. **The cross-arm table does not support that
and mildly contradicts it.** The arm with essentially zero verification
demonstrations (swezero, 0.9%) is the *best* arm on the board at 77/500; the three
arms whose data is 55–88% verification score 70, 48 and 35. Whatever is driving
the board ordering, "more verification demonstrations" is not obviously it.

With n=4 arms differing in source corpus, size and task mix, this supports **no
causal claim in either direction** — I am not asserting that verification
demonstrations hurt. The point is narrower and it is about my own recommendation:
it was unsupported, and the one correlation available points away from it. It
should not be acted on without a controlled test.

This also converges with the v2 line's existing finding that verification is not
the board-level mechanism — *productive* verification is. Consistent with that,
`ran_any_test_ok` in this ladder sits at 0.05 in every arm cell regardless of what
the prompt says.

### Two claims that must stay in separate boxes

Per devils-advocate's scope caveat, which is correct:

* **(a) "swezero's data teaches a near-absent, prompt-suppressible verification
  habit."** Supported — directly measured at 0.9% over 49,030 trajectories, plus
  the D−C suppression at 5.9σ.
* **(b) "and that is why base ≫ arms on score."** **Not supported by anything
  here.** My own nulls decouple the two: D moves attempted verification enormously
  (repro 0.30 → 0.02) while achieved verification is flat (`ran_any_test_ok`
  0.050 → 0.050) and score is flat (D−A = +4, 1.15σ). Within this experiment,
  manipulating the arm's verification attempts changes neither achieved
  verification nor score.

The format-OOD refutation is unaffected by all of this: it rests on B and C being
**score-inert** (L1 passing on the score channel), not on D−A. §3c-quater sharpens
rather than weakens it — the stub demonstrably reaches the model (think rate per
action +3.97σ across A→C) and still moves score by zero.

## 4. What the score channel cannot do, and what to read instead

At n=100 and ~7–11% resolve rates, with same-condition churn of ~14 instances,
this design can only detect score effects of roughly ±7 instances or more. Every
arm rung is below that. The score channel has returned a real and useful negative
(L1 passes; format parity is inert) but it cannot adjudicate L2.

The behavioural channel carried the information, as expected: per-instance
measures have far tighter paired SEs than a 7-vs-11 count comparison, and they
resolved both L1 and L2 where score could not. See sections 3b/3c. Two
corrections apply to how the behavioural numbers are read, and they point in
opposite directions: §3c-ter removes four of D−C's six apparent effects as
trajectory-length artifacts, and §3c-quater **adds** an effect on the stub rung
that the raw counts hid (L1 is score-inert and inert on 13 of 15 behavioural
metrics, not 15 of 15). L2's effect is **a ~20% trajectory
contraction plus suppression of the three prohibited behaviours**, not the six
independent effects §3b's raw table implies — four of those six do not survive
normalization by trajectory length. The base cells' behavioural pass is still
pending their inference.

**A readout bug worth recording:** the metric list said `n_think_calls`, which
`extract_traj_stats.py` does not emit (the key is `n_think`). The metric loop
skips absent keys silently, so **S1's registered primary outcome was being
dropped with no warning** — the first behavioural run printed four metrics and
looked complete. `ladder_readout.py` now asserts every requested metric against
the data and prints a `!! METRICS not present` line, so a typo fails loudly.

Deliberately not done: raising n to chase L2. That is new GPU scope and is dpf's
call, not mine. The number that would justify it is now measurable rather than
guessed — detecting a true +4 against this churn needs roughly 4× the instances.
