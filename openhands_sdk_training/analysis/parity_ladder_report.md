# Prompt-parity / think-stub ladder — status report

*Written 2026-08-05, ladder completed 2026-08-06. Companion to `prompt_parity_prereg.md`, `think_stub_prereg.md` and
the running `parity_ladder_amendment.md` (10 addenda). Numbers here are the ones I am
willing to defend; everything I am not willing to defend is listed in
[What is not measured](#what-is-not-measured) rather than omitted.*

## Are the experiments done?

**No.** Rollouts (GPU work) are essentially complete; *scoring* is not, because
mid-campaign I found a defect in the scoring harness that depresses resolve counts
and had to re-score. State as of writing:

| piece | state |
|---|---|
| rollouts A–F | complete |
| rollout G (prefill-blocked base) | attempt 1 complete 100/100; **scored 2026-08-06 — G = 20/100** (see [Cell G](#cell-g--scored-and-still-not-a-causal-claim)) |
| clean re-score, B and F | **complete** — the primary rung is readable |
| clean re-score, A / C / D / E | **complete** (landed after this report was first written; see [The clean ladder](#the-clean-ladder-all-rungs)) |
| clean re-score, aggregates (all cells) | **not started** — larger compute, needs a decision |
| E aggregate | never scored at all (lost a shard to walltime) |

So: **the matched ladder is complete and readable at n=100 on every rung.** What remains
is the whole-campaign *aggregate* board, which needs re-scoring before any of its
numbers can be quoted. G has now been scored too, on the same attempt-1 footing (see
[Cell G](#cell-g--scored-and-still-not-a-causal-claim)). The findings below do not depend
on the aggregate re-score.

---

## Finding 1 — the scoring harness was silently losing resolves, and it is now fixed

This is the most consequential thing in this report, because it changes numbers that
were already circulating.

`run_score_shards.sbatch` created Apptainer sandboxes under one **shared** root and
pruned them **keyed by `instance_id` alone**. Every cell of a comparison scores the
*same* SWE-bench instance set, so two concurrent scoring jobs shared one sandbox
directory per instance and deleted it out from under each other mid-run. The failure
mode is silent and one-directional: a vanished sandbox scores as **unresolved**.

Measured, not theorised. Re-scoring byte-identical rollouts serially:

| cell | scored with 20 concurrent tasks | scored alone | flips |
|---|---|---|---|
| B (arm, nostub) | 6/100 | **14/100** | 9 gained, 1 lost |
| F (base, nostub) | 6/100 | **24/100** | 18 gained, **0 lost** |

The pre-registered reading in Addendum 10 was "≥5 disagreements ⇒ H-collision, no
matched rung may be quoted." Nine and eighteen. The fix (per-job sandbox subtree,
`d748808`) was validated twice on B independently (13 and 14, vs 6 contaminated).

Two things follow, and the second is the uncomfortable one:

1. **The collision is non-uniform across cells.** F lost 18 instances to it, B lost 8,
   and F lost *nothing* in the other direction. Addendum 8 had hoped instance-level
   scoring effects would subtract out of a within-batch difference. They do not. That
   hope is withdrawn.
2. **Every number scored before the fix is a floor, depressed by an unknown,
   cell-dependent amount.** That includes the aggregate ladder board, the arm totals,
   the `init`-checkpoint 145/500, the souping numbers (soup 50 / pooled55k 46 /
   pooled220k 54) and the aggregated F−B = +19. The identity gate confirms it from the
   inside: B's aggregate says 8 resolved while B's *clean* attempt-1 subset — a strict
   subset of the same rollouts — says 14, with only 3 instances ever retried. A subset
   cannot beat its superset by 11 unless the superset's scoring is broken.

Nothing here changes a *model*; it changes a *measurement*. But it changes it by more
than the size of most effects this campaign was built to detect.

## Finding 2 — the primary rung, at matched compute, on clean scoring

The rung is F−B: **base vs arm, both with the `<think>\n\n</think>` stub removed,
identical eval prompt**. Attempt-1-only, so one rollout per instance on both sides
(the harness's critic otherwise gave base 2.32 rollouts/instance against the arm's
1.04 — identical config, unequal compute, biasing *toward* base; Addendum 7).

| version of the rung | n | B (arm) | F (base) | net | b/c | SE | σ | verdict |
|---|---|---|---|---|---|---|---|---|
| contaminated attempt-1 | 100 | 6 | 6 | +0 | 6/6 | 3.46 | 0.00 | uninformative (both cells damaged) |
| **clean attempt-1** | 100 | **14** | **24** | **+10** | 17/7 | 4.90 | **2.04** | **significant** |
| aggregate (unequal compute, contaminated, cross-batch) | 100 | 8 | 27 | +19 | 22/3 | 5.00 | 3.80 | not quotable |

Paired McNemar; `b` = base-only resolves, `c` = arm-only, SE = √(b+c).

The clean value lands in the pre-registered band **`2·SE … +19`** whose registered
reading is: *the gap is real, and the aggregated form of it is partly inflated.* Both
halves hold. Base-nostub beats the SFT arm by 10 instances per 100 at matched compute
and a matched prompt — and the +19 that was on the board overstates that by about
half, from the retry-compute asymmetry plus scoring damage.

This is consistent with, and is now the cleanest available statement of, the finding
already in memory: **SFT lift on this line is null-to-negative against a properly
measured base.** It is not a new direction, it is the same direction measured properly.

## Finding 3 — reasoning census: where reasoning actually happens

dpf's question, prompted by DA's claim that blocking free text in G produces more tool
calls. Three channels, which are **not disjoint** — for base-nostub the `<think>` tag
sits at char 0 of the `thought` field, so channel (c) *is* channel (a), same text under
two names. Reported nested.

### Rollouts — paired on the 62 instances present in every cell

Denominator: one `ActionEvent` (an assistant turn that called a tool).

| cell | model | template | act/inst | free text | of which `<think>` | prose outside tag | `think()` | any |
|---|---|---|---|---|---|---|---|---|
| A | arm | stock | 96.3 | 0.0% | 0.0% | 0.0% | 2.5% | 2.5% |
| B | arm | nostub | 92.0 | 0.0% | 0.0% | 0.0% | 3.1% | 3.1% |
| C | arm | nostub+wrap | 92.6 | 0.0% | 0.0% | 0.0% | 3.3% | 3.3% |
| D | arm | nostub+train-prompt | 74.3 | 0.0% | 0.0% | 0.0% | 3.2% | 3.2% |
| E | base | stock | 288.4 | 48.3% | 0.0% | 48.3% | 8.2% | 52.2% |
| F | base | nostub | 211.2 | 99.9% | 99.9% | 16.4% | 9.2% | 99.9% |
| G | base | stock + prefill-block | 290.4 | 0.0% | 0.0% | 0.0% | **21.5%** | 21.5% |

Raw counts matter here because a rounded 0.0% hid a real event: cell A's free text is
**1 turn out of 5970**, not zero.

| cell | free-text turns | `think()` calls | `think()`/inst | free text/inst |
|---|---|---|---|---|
| A | 1 | 151 | 2.44 | 0.02 |
| B | 0 | 179 | 2.89 | 0.00 |
| C | 0 | 190 | 3.06 | 0.00 |
| D | 0 | 148 | 2.39 | 0.00 |
| E | 8645 | 1459 | 23.53 | 139.44 |
| F | 13083 | 1205 | 19.44 | 211.02 |
| G | 0 | 3874 | **62.48** | 0.00 |

**DA's claim is real, but it is channel substitution, not more work.** E → G, paired:

- `think()` calls per instance: 23.5 → 62.5 (**2.7×**)
- free-text turns per instance: 139.4 → 0 (the block works — 0/18002 turns)
- *total* actions per instance: 288.4 → 290.4 (**flat, +0.7%**)
- therefore **non-`think` actions per instance: 264.9 → 227.9 (−14%)**

So blocking prose does not make the model act more; it makes the model spend a seventh
of its turns narrating into a tool instead of acting. The reasoning volume is
conserved and re-routed. Whether that helps is a *score* question, answered null and
confounded (see [Cell G](#cell-g--scored-and-still-not-a-causal-claim)).

Cell F is the other half of the story: the stub does not suppress reasoning, it
**relocates** it. 99.9% of F's turns emit a `<think>` tag, **0.0% of them close it**,
82% are non-empty, mean 110 characters inside — and prose *outside* the tag drops from
E's 48.3% to 16.4%. The model pours its reasoning into an unclosed tag.

### Training data — 3000 trajectories per arm

Denominator: one assistant-**authored** turn. A structural asymmetry has to be stated
alongside, or the comparison misleads: the ADP → LLaMA-Factory conversion emits role
`function_call` as a bare JSON tool-call array with **no text field**, so a training
turn *cannot* carry free text next to its tool call. Prose exists only as a separate
`assistant` message.

| dataset | traj | prose turns | of those, terminal | `think()` % of calls | traj with `think()` | `<think>` |
|---|---|---|---|---|---|---|
| coderforge_preview | 3000 | 2.8% | **100.0%** | 6.6% | 50.9% | **0** |
| SWE-rebench (nebius) | 3000 | 3.0% | 99.8% | 4.3% | 40.8% | **0** |
| SWE-Zero (nvidia) | 3000 | 2.5% | **100.0%** | 7.1% | 56.6% | **0** |
| scale_swe_distilled | 3000 | 5.0% | 51.4% | **0.0%** | **0.0%** | **0** |
| swe-gym | 199 | 8.1% | 31.5% | **0.0%** | **0.0%** | **0** |

The three answers to dpf's question:

- **(a) free text before a tool call.** Structurally absent from the training data — and
  what prose exists is ~100% *terminal* (the closing summary), so mid-trajectory
  reasoning prose is ≈ 0. The arms reproduce this exactly: 1 free-text turn in ~6000.
  The behaviour is **inherited from the demonstrations**, not a template artifact.
- **(b) `think()`.** The one channel the data does teach — 4.3–7.1% of tool calls,
  ~0.6–1.0 calls/trajectory, in 41–57% of trajectories. The arms reproduce it at
  2.4–3.1% of calls, i.e. at roughly half the demonstrated rate. Note `scale_swe_distilled`
  and `swe-gym` contain **zero** `think()` calls.
- **(c) `<think>` tag.** **Zero occurrences in all five training datasets.** So F's
  99.9% tag rate is entirely a serving-template effect with no support in training —
  which is exactly why the B−A rung (stub on/off, arm side) is a **mechanistic null**:
  native `<think>` is 0/100 either way, because the arm never learned to use it.

## The clean ladder, all rungs

All six cells re-scored under the sandbox fix, attempt-1 only, paired McNemar on
n=100. **Nothing is pending; this is the final ladder.**

| rung | tests | n | low | high | net | b/c | SE | σ | verdict |
|---|---|---|---|---|---|---|---|---|---|
| **F−B** | base vs arm, both nostub | 100 | B 14 | F 24 | **+10** | 17/7 | 4.90 | **2.04** | **significant** |
| **E−A** | base vs arm, both stock | 100 | A 16 | E 28 | **+12** | 21/9 | 5.48 | **2.19** | **significant** |
| A−B | the stub, arm side | 100 | B 14 | A 16 | +2 | 9/7 | 4.00 | 0.50 | null |
| C−B | the wrapper and path | 100 | B 14 | C 16 | +2 | 8/6 | 3.74 | 0.53 | null |
| D−C | prohibition + phase list | 100 | C 16 | D 16 | **+0** | 5/5 | 3.16 | 0.00 | null |
| E−F | the stub, base side | 100 | F 24 | E 28 | +4 | 13/9 | 4.69 | 0.85 | null |
| **C−A** (registered **L1**) | all format rungs jointly | 100 | A 16 | C 16 | **+0** | 7/7 | 3.74 | 0.00 | **L1 PASSES** |
| G−E | prose blocked, base | 100 | G 20 | E 28 | −8 | 6/14 | 4.47 | −1.79 | null (and confounded — see below) |

Three things fall out, and one of them corrects an earlier claim of mine.

**L1 passes at the extreme.** The registered bound was `|C − A| ≤ 6/100`; the measured
value is **0**, with the resolved sets differing on only 7 instances each way. B and C
between them remove every non-semantic train/eval format difference for this arm, so
this is the cleanest available test of "the arms are being scored unfairly by serving
format" — and the answer is no. The registered falsifier (C − A ≥ +6, which would have
forced a re-measurement of every arm before further curation) does not fire. Curation
work does not need to wait on a format re-measurement.

**Every format rung is null and the model rung is not.** A−B, C−B, D−C and E−F all sit
under 1σ — and two of them (C−A, D−C) are *exactly* zero — while base-vs-arm is
significant on *both* template conditions (+10 nostub, +12 stock). The entire arm side
of the ladder spans 14–16 resolved out of 100 across four different prompt/template
conditions; the base side sits at 24–28. Per the attribution rule fixed in advance, the score difference is
attributed to the smallest rung that produces it — and no format rung produces one.
The gap is the model, not the prompt.

**Cross-check on the two significant rungs.** +10 and +12 per 100 sit alongside the
500-instance board's +8.4pp (base 119 vs best arm 77) — a separate measurement on 5×
the instances, a different scoring pass and a different rollout campaign. Two
independent instruments agreeing to within ~3pp is the reason to believe the matched
result is not a shard artifact. Before this ladder, base ≫ arm was only ever a
cross-harness claim carrying confounds; it is now within one harness, on the same
instances, with compute controlled, and it survives.

**I have to withdraw "removing the stub helped base."** That claim rested on F's
contaminated aggregate (27) against an E aggregate that was never scored. Clean and
matched, the stub rung on the base side is **E 28 vs F 24, +4, 0.85σ — null, with the
point estimate now pointing the other way.** The substantive part of the earlier
finding survives untouched (the stub collapses base's cap-hit rate, 31–35% → 0%, and
relocates its reasoning into an unclosed tag), but the *score* consequence does not.
`think_stub_prereg.md` §2 predicted removing the stub would hurt base; the honest
reading is now "no measurable score effect either way," not "it helped."

**And the mechanism is visible in the two cells, which kills a stronger claim than
mine.** E hits the 500-iteration cap on 35% of instances and F on 0%, so E resolves 28
of the ~65 instances it actually *completes* (**43%**) against F's 24 of 100 (**24%**).
The stub trades completion rate for per-completion quality: un-capping base adds
*finishers*, not *solves*. That refutes the separate claim — raised in review, and which
I did not carry into this report but which did reach memory — that **θ₀ = 119/500 is a
floor because the stub gags base into caps and costs it resolves**. The cap-hits the
stub causes do not cost base resolves; 119 is base's rate, not a floor beneath which a
truer value hides. (Distinct from, and not to be merged with, the sandbox-collision
floors in Finding 1, which stand. Also distinct from the unattributable-26/119
provenance question, which is about the board run, not the stub.) Credit where due: this
concession originated with the devil's-advocate agent, against their own earlier
position.

### Cell G — scored, and still not a causal claim

G was staged by the cell-G agent and I initially declined to score it, on the argument
that its score is uninterpretable in either direction. That was the wrong call, and not
because the argument is wrong: it costs two CPU shards, it completes a 7-cell ladder on
one footing, and an absent number gets filled in by guesswork. Scored on the identical
100-instance set with the sandbox fix: **G = 20/100, G−E = −8 at 1.79σ — null.**

It does not clear the registered 2·SE bar, so the score half of the prose-blocking
question is simply **null**, and the sign cannot be attributed either. The two confounds
that agent-b17cac1e pre-registered and measured point in opposite directions and are
unequal: the iteration cap **favours** G (22 of E's 35 cap-hitters run to completion
under G; 35.0% → 18.0%, exact McNemar p = 0.0115), while malformed tool calls
(1.94% → 4.66%, p = 0.0021) and `AgentErrorEvent`s (4 → 11, p = 0.0070) **disfavour** it.
A loss is what the malformed-call confound predicts, so it is not evidence about
reasoning content.

**What is worth noting is the direction, because it repeats.** Both manipulations that
reduce base's cap-hit rate score *lower* than the stock stub cell: E 28 → F 24 (nostub,
caps 35% → 0%) and E 28 → G 20 (prose blocked, caps 35% → 18%). Per-completion, all
three line up with the mechanism in the previous section — E resolves 43% of the
instances it completes, F 24%, G 24%. Two independent manipulations, each individually
null, both pointing the same way: **base's resolve count is not limited by running out
of iterations.** Un-capping it adds finishers, not solves. That is the same conclusion
the E−F rung reached, arrived at by a different route.

The clean deliverables from G remain the ones its own agent identified — the block fired
completely (0 of 17,288 action events carry prose, vs E's 5,929/12,315 = 48.14%) and the
relocation is decisive (narration → `think()`, +10.8pp, p = 1.3e-7) — not the score.

Rungs that scoring cannot rescue:

| rung | why not |
|---|---|
| F−E as a *compute-matched* claim | the registered L3 placebo fails (cap-hit E 31% vs F 0%; 35% vs 0% matched) and the aggregate form was never compute-matched — peer measured critic rejections 71/100 in F vs 15/100 in E. The a1x rung above sidesteps this by using attempt 1 only |
| D−C | **confounded by construction** — D drops 3 phases as well as changing the prompt, so its exact-zero bundles two manipulations that could in principle cancel. Unbundling needs new rollouts |
| G−E on score | **downgraded, do not wait for it** — see below |

## What is not measured

Stated explicitly so none of it reads as covered:

- **Every aggregate number in the campaign** — including the θ₀ = 119 and best-arm 77
  that anchor the headline — is a floor of unknown size until re-scored
  under the sandbox fix. That is the single largest outstanding item and it costs real
  CPU-hours across all cells — I have not started it because it is a scope call for dpf,
  not a detail.
- **E's aggregate does not exist** (lost shard). E also has 3 patches discarded by the
  harness's multi-attempt aggregator (Addendum 5) with no repaired scoring run, so any
  E aggregate that does appear will be biased low. Base cells should be reported both
  as-harness and repaired, labelled, never swapped silently.
- **G's score exists but carries no causal weight** (20/100, G−E null at 1.79σ). Its two
  confounds are of opposite sign and unequal size, so no direction is clean. Quote it as
  a descriptive cell, never as evidence about reasoning content. Details above.
- **The channel-blocking ladder should stop here**, on an argument from the
  devil's-advocate agent that I accept: blocking one surface reroutes reasoning to the
  next open one, and the remaining surfaces are undetectable by construction (free-text
  tool arguments, identifier choice, whitespace and ordering, which files get opened).
  So each drop-X rung can only ever establish "reasoning must live in some *other*
  channel," never "reasoning off," and the regress does not terminate in a channel that
  can be proven closed. A drop-`think()` rung would hit the same wall. The content
  question needs a different design — a model with no reasoning channel by
  training/RL, or an information-theoretic probe — not more ablation rungs.
- **F−E cannot be rescued by re-scoring** — its problem is unequal compute and a failed
  placebo in the *rollouts*, not the scoring.
- **D−C needs new rollouts** to unbundle. Not launched; GPU spend is dpf's call.
- No claim here rests on `<think>`-tag semantics being intended behaviour: the tag is
  unclosed 100% of the time in F, which is a serving-template artifact.

## Recommended next steps, in order

1. **Decide** whether to re-score the aggregate board under the fix. This is now the
   only thing standing between the campaign and a quotable set of numbers — until it
   happens the honest form of every aggregate is "≥ X". It is CPU-only but across all
   cells, so it is a scope call rather than a detail.

   Two things should inform that call. **The verdict does not depend on it.** Base ≫ arm
   is established by the matched ladder, which is already post-fix, compute-controlled
   and on identical instances; the board is now corroboration, not foundation, so the
   re-score is about *accurate absolute numbers* rather than about whether the
   conclusion holds. And **the expected direction is upside, not risk**: the collision
   only ever removed resolves, and on the ladder it was differentially anti-base (+18 to
   F against +8 to B). If that asymmetry repeats on the 500-board, a clean re-score
   *widens* the gap. That last part is a prior rather than a measurement — the proposed
   mechanism (base's longer, more numerous runs expose more sandboxes to the race) is
   plausible and untested, so it should not be quoted as a finding. Credit: framing from
   the devil's-advocate agent.

   If it is run, **report per-cell gained/lost, not just the net** — "18 gained, 0 lost"
   is what distinguishes a floor-correction from a re-roll, and a re-roll would show
   losses in both directions.
2. Teach `ladder_readout.py` to prefer the `_a1x` suffix and print which scoring pass it
   used — it still defaults to the contaminated `_a1`.
4. **Do not** spend GPU on further channel-blocking rungs; G's score is in and settles nothing about reasoning content, by construction.
5. Do not spend GPU on unbundling D−C unless the prompt question becomes decision-
   relevant: every format rung is null, so the prior that it matters is now low.
