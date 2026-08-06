# Prompt-parity / think-stub ladder — status report

*Written 2026-08-05. Companion to `prompt_parity_prereg.md`, `think_stub_prereg.md` and
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
| rollout G (prefill-blocked base) | running, 62/100 instances usable so far |
| clean re-score, B and F | **complete** — the primary rung is readable |
| clean re-score, A / C / D / E | running (jobs 876119–876126, submitted together) |
| clean re-score, aggregates (all cells) | **not started** — larger compute, needs a decision |
| E aggregate | never scored at all (lost a shard to walltime) |

So: one rung is final, four are pending a few hours of CPU scoring, and the
whole-campaign aggregate board needs re-scoring before any of its numbers can be
quoted. Three findings below are complete and do not depend on the pending work.

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
conserved and re-routed. Whether that helps is a *score* question, and G's score is
not in yet (job 876166 staged).

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

## Rungs that are readable, and rungs that are not

| rung | tests | status |
|---|---|---|
| **F−B** | base vs arm, matched prompt + matched compute | **+10, 2.04σ, readable** |
| B−A | the `<think>\n\n</think>` stub, arm side | mechanistic **null** — arm emits 0 native `<think>` either way (Finding 3) |
| F−E | the stub, base side | **not readable.** Registered L3 placebo fails (cap-hit E 31% vs F 0%; 35% vs 0% matched) and it is not compute-matched — peer measured critic rejections 71/100 in F vs 15/100 in E |
| D−C | train prompt vs eval prompt | **confounded** — D also drops 3 phases, so it bundles "prompt differs" with "instructions removed". Unbundling is new GPU scope |
| G−E | blocking prose in the base model | census done (Finding 3); **score pending** |
| A/C/D vs B | pending clean re-score | contaminated values (8/10/9) must not be quoted |

## What is not measured

Stated explicitly so none of it reads as covered:

- **Every aggregate number in the campaign** is a floor of unknown size until re-scored
  under the sandbox fix. That is the single largest outstanding item and it costs real
  CPU-hours across all cells — I have not started it because it is a scope call for dpf,
  not a detail.
- **E's aggregate does not exist** (lost shard). E also has 3 patches discarded by the
  harness's multi-attempt aggregator (Addendum 5) with no repaired scoring run, so any
  E aggregate that does appear will be biased low. Base cells should be reported both
  as-harness and repaired, labelled, never swapped silently.
- **G is incomplete** (62/100 instances). Its census rates are stable-looking but its
  score is unmeasured.
- **F−E cannot be rescued by re-scoring** — its problem is unequal compute and a failed
  placebo in the *rollouts*, not the scoring.
- **D−C needs new rollouts** to unbundle. Not launched; GPU spend is dpf's call.
- No claim here rests on `<think>`-tag semantics being intended behaviour: the tag is
  unclosed 100% of the time in F, which is a serving-template artifact.

## Recommended next steps, in order

1. Finish the four running clean re-scores; re-run `ladder_readout.py` (it still reads
   the `_a1` suffix and must be taught to prefer `_a1x` and print which it used).
2. Score G's attempt-1 subset (staged) and read G−E on score, not just on channel mix.
3. **Decide** whether to re-score the aggregate board under the fix. Until that happens
   the honest form of every aggregate number is "≥ X".
4. Leave D−C flagged as confounded rather than quoting it.
