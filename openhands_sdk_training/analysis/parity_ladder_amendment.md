# Parity ladder — amendment merging the stub and prompt probes

**Status: filed 2026-08-05 ~03:00 UTC. Both SLURM arrays were submitted at
02:58–03:00 UTC and were still `PENDING` when this was written — no rollout
output existed. This is an amendment to a design, not a reading of a result.**

Supersedes the *designs* in `think_stub_prereg.md` §2 and
`prompt_parity_prereg.md` §2. Both of those documents stay as filed; their
findings (§1) and predictions (§3) are unchanged and are not restated here.

## Why the two probes merged

They were written as separate GPU spends, with the prompt probe explicitly
deferred (`think_stub_prereg.md` §5). dpf asked to remove every format confound
at once rather than one at a time: *"remove all possible confounding vars and
just see if we can have the arm eval match the way the arms were trained."*

That request is well-posed for the format mismatches and needs a caveat for the
semantic one:

| mismatch | kind | removing it… |
|---|---|---|
| `<think>\n\n</think>` stub, 4 tokens per assistant turn | format | removes a confound |
| `<uploaded_files>` header; `django` vs `/workspace/django/` | format | removes a confound |
| prohibition on running python/tests; 5 phases vs 8 | **semantic** | **removes capability** |

The prohibition cannot be "controlled away." Restoring it tells the model not to
run tests in a container where tests run fine. So a single all-changes-at-once
"full parity" cell would confound three edits and could not be attributed. The
merged design is therefore a ladder in which each rung adds exactly one change.

## Design as launched

Runner `swe-bench-fair-evals/scripts/run_parity_ladder.sbatch`, 6 cells × 2
shards (`select/shard_{00,01}of10.txt`), n=100/cell, 12 jobs, one pass.

| cell | model | chat template | task statement | rung adds |
|---|---|---|---|---|
| **A** | arm | stock (stub) | harness `default.j2` | — control, as the board was measured |
| **B** | arm | `nostub.jinja` | harness `default.j2` | stub removed |
| **C** | arm | `nostub.jinja` | `train_wrapper.j2` | + header & path style matched |
| **D** | arm | `nostub.jinja` | `train_swezero_noenv.j2` | + prohibition & 5-phase list |
| **E** | base | stock (stub) | harness `default.j2` | — base control |
| **F** | base | `nostub.jinja` | harness `default.j2` | stub removed |

- arm = `v2_swezero_inst_4b_a100/output`; base = the cached `Qwen/Qwen3.5-4B`
  snapshot. Both paths taken from the existing shard-00/01 runs' `llm_config.json`
  so the comparators are the same weights the board used.
- Base gets the stub axis only. The swezero training statement is arm-specific
  and means nothing for a model that never saw it.
- Cells A/B/E/F are exactly the 8-job stub design in `think_stub_prereg.md` §2;
  C/D are exactly the T1/T2 idea from `prompt_parity_prereg.md` §2, moved onto a
  no-stub base so they test the task statement and not the stub again.
- **A and E are fresh controls, not reused board numbers.** vLLM prefix caching
  plus continuous batching make greedy decoding non-reproducible.
- Held constant: harness commit, temperature 0.0, condenser settings, tool
  preset, `NUM_WORKERS=16`, `max_input_tokens` 28000.

Submitted as two arrays only because the 12-job launch needed a second
authorization: `753617` (indices 0,1,4,5,6,7,10,11) and `753633` (2,3,8,9).
Same script, same environment, same pass — the split is administrative.

## Pre-launch verification

Done before submitting, recorded so a null is not later blamed on plumbing:

| check | result |
|---|---|
| all three task templates render under Jinja `StrictUndefined` | pass — no silently-empty variables |
| `workspace_dir_name` is in the harness render context | yes, `swebench/run_infer.py:get_instruction` |
| eval phase count | 7 `Phase N` + `8. FINAL REVIEW` = **8** |
| swezero phase count | 4 `Phase N` + `5. FINAL REVIEW` = **5** |
| the 3 phases eval adds | RUNNING, TEST CREATION, VERIFICATION |
| `train_wrapper.j2` carries no prohibition | confirmed by render probe |
| `train_swezero_noenv.j2` carries the prohibition | confirmed by render probe |
| 12-index cell mapping | dry-run enumerated, all 12 distinct and correct |

One template fix was made pre-launch: `train_wrapper.j2` line 14 used
`{{ instance.repo_path }}` (renders `/workspace/django/`) where the training
statement uses `/workspace/{{ workspace_dir_name }}` (`/workspace/django`).
Aligned, so cell C is exactly "training's wrapper and path style, eval's phase
content." Deployed sha256 `980ff5975d95698f…`.

## Added predictions for the new rungs

`think_stub_prereg.md` S1–S4 and `prompt_parity_prereg.md` P1–P4 stand as filed
and carry over: S1–S3 read on A/B/E/F, P1 on C, P2–P4 on D. Two predictions are
new because they only exist once the rungs are chained.

**L1 — the format rungs are jointly inert on score.** Registered:
|C − A| ≤ 6/100. *Rationale:* B and C between them remove every non-semantic
train/eval difference for this arm. If format mismatch were costing the arm real
points, this is where it would show, and it is the cleanest available test of
"the arms are being scored unfairly."
*Falsified if* C − A ≥ +6/100 → the board is confounded by serving format and
every arm must be re-measured before any further curation work.

**L2 — the ladder is monotone in nothing.** No directional prediction for
D relative to C. D adds capability-removing instructions on top of a
format-matched baseline, so it can move either way, and I decline to predict a
sign. Recorded so that whichever direction it lands is not narrated as expected.

**Attribution rule, fixed in advance.** With four arm cells, a score change is
attributed to the *smallest* rung that produces it, not to the largest change
present. B − A is the stub; C − B is the wrapper and path; D − C is the
prohibition and phase list. A difference that appears only at D − A and is
absent at every adjacent rung is noise, not a three-way interaction.

## Reading order, fixed in advance

1. **S1 first** (manipulation check: does removing the stub change `think`
   usage at all?). If nothing moves behaviourally in B vs A, the format half of
   the ladder is inert and B/C score deltas are uninterpretable noise.
2. **E vs F** before believing anything about the arm — it is the first
   empirical test of the never-measured assumption that base is stub-native.
3. Then the arm rungs, adjacent pairs only, per the attribution rule.
4. Behavioural extraction for all six cells in **one pass** with a pinned
   `traj_compare/extract_traj_stats.py`, matching the same-scoring-pass
   discipline used for the A1 readout.

Primary endpoint remains **behavioural** (`verified_after_edit`, `ran_any_test`,
`think`-call rate) at n=100 population proportions. **Score is secondary and
underpowered**: at a 10–16% base rate and n=100, sd ≈ 3.0–3.7, so only swings
≳6–7/100 clear 2sd. Stated again here so a null on score is not read as
evidence of absence.

## Addendum — the base iteration cap (filed 03:35 UTC, still pre-rollout)

Raised by agent-b17cac1e on the coordination channel after submission but
before any cell produced output. Independently re-verified here against the
existing board runs rather than taken on report:

| | instances with a transcript | only in `output_errors.jsonl` | of those, `MaxIterationsReached` |
|---|---|---|---|
| base `v2_init_4b` | **300 / 500** | **200 (40.0%)** | 155 |
| arm `v2_swezero_4b` | 500 / 500 | 0 | 0 |

The dropped rows carry `history_len: 0`, `git_patch: None`, and
`error: "…MaxIterationsReached: Agent reached maximum iterations limit (500)"`.
They are not blank *rows in* `output.jsonl` — they are **absent from it** and
present only in the error file. Scoring already reads both files
(`run_score_shards.sbatch` concatenates them), so **score is over 500 while
every behavioural statistic is over ~300**. That asymmetry applies to every
base number in the campaign, not just to this probe.

Consequences for cells E and F, registered now:

1. **Base behavioural n is ~60/cell, not 100.** At n=100 expect ~40 instances
   with no transcript. Any E/F behavioural proportion must state its surviving
   denominator; comparing a base proportion against an arm proportion without
   doing so compares ~60 against 100.
2. **The survivors are top-truncated.** The dropped instances are base's
   *longest* runs by construction — they were dropped for running too long. So
   base's tool-share and depth profiles are conditioned on "did not exhaust the
   budget," and are biased toward shorter trajectories. This is the sharpest
   available caveat on agent-b's item 5 (base 45.8% `terminal` / 16.2% message
   content vs arms' `file_editor` dominance): that contrast is measured on
   base's shorter 60%.
3. **E vs F must be compared on the intersection.** Only instances with a
   transcript in *both* E and F enter any behavioural diff, and the intersection
   size is reported alongside every number. Comparing E's survivors against F's
   survivors as two independent samples would let a shift in *which* instances
   survive masquerade as a shift in behaviour.
4. **Cap-hit rate becomes a registered secondary outcome per cell**, reported
   for all six cells, not only base.

**New prediction L3 (placebo / plumbing check).** The 500-iteration cap counts
**agent iterations, not tokens**, so trimming 4 tokens per assistant turn has no
mechanism by which it should move the cap-hit rate. Registered: |cap-hit(F) −
cap-hit(E)| ≤ 8pp.
*If falsified* — the cap-hit rate moves substantially between E and F — then
something larger than a 4-token trim changed, and the correct response is to
**inspect the rendered prompts and the vLLM logs before interpreting any E/F
result**. In that world an E−F resolve-rate delta is confounded by how much
budget each condition got, not by serving format, and must not be reported as a
format effect. This is the cheapest available guard against the ladder silently
measuring budget instead of parity.

Note this cuts against a reading of the existing board: base's 119/500 is
achieved while 40% of its instances never finish. Whether that makes base's
score an under- or over-statement is not something this probe measures, and I
am not claiming it does.

## Addendum 2 — reasoning-channel accounting and the F−B differential

Filed 04:00 UTC, still pre-rollout. Raised by devils-advocate reviewing the
three prereg docs.

### The objection

S1 reads the stub's effect on the arm's `think`-**tool** rate (B vs A). In cell
A the stub pre-closes an empty `<think></think>`, so the tool is the only
reasoning channel available. In cell B the assistant prefix is bare
`<|im_start|>assistant\n`, which in principle permits a **native** `<think>…</think>`
block. If nostub unlocks native blocks, a rise in tool rate at B could be
reasoning *relocating between channels* (a format shift) rather than
*un-suppressing* (a rate change) — identical S1 signature, different meaning.

### Resolved for the arm, live for base — measured, not assumed

Scanned assistant-generated content in the existing board runs, both served
with the **stock** (stub) template:

| | records with a native `<think>` block | total blocks |
|---|---|---|
| arm `v2_swezero_4b` | **0 / 500** | 0 |
| base `v2_init_4b` | **5 / 324 (1.5%)** | 5 |

The arm never emits a native block, consistent with 0 `<think>` occurrences in
6,508,136 training tokens. Base does, at a low rate, *despite* the stub already
being closed — so base retains the capability and the stub does not fully
suppress it.

Consequences:

1. **S1 (B−A) stands as written.** For the arm the relocation channel is
   empirically absent at n=500 under stock. It is still measured in B/C/D
   rather than assumed, because nostub changes the prefix and could in
   principle unlock what stock did not.
2. **Base cells require combined-channel accounting.** Any E/F reasoning
   comparison is reported as **native-block tokens + `think`-tool tokens**, and
   the two components are also reported separately so a pure relocation is
   visible as a compositional shift at constant total.
3. **Native-`<think>` rate becomes a registered secondary outcome for all six
   cells.** If it is non-zero in B/C/D, S1 is re-scored on the combined channel
   before being read.

### F vs B — the matched base-vs-arm differential

devils-advocate correctly notes that the base-vs-arm tool-profile question
(base 45.8% `terminal` / 16.2% message content vs arms' `file_editor`
dominance, previously n=1) is answered not by E/F but by **F vs B**: base-nostub
vs arm-nostub, same harness, same template, same shards, differing only in
model. This is already in the ladder and is now named as an explicit readout.

**Registered as a directional lower bound, not a point estimate.** The cap
asymmetry is ~40% (base) vs ~0% (arm), and base's dropped instances are its
longest by construction, so F and B cannot be intersected across models. Base's
terminal-heavy profile is therefore measured on its shorter-finishing runs, and
the true base-vs-arm terminal gap is if anything **larger** than F−B shows.
Reported with that direction stated.

### Noise band for E−F, and what L3 can and cannot do

No same-condition replicate exists, so an empirical run-noise band cannot be
pre-registered from data. Instead, registered procedure: because E and F are
compared **on the intersection**, each instance yields a paired difference, so
the standard error of the mean paired difference is computable directly from
this run. **Rule: treat |mean paired Δ| < 2·SE as uninformative** and say so,
rather than narrating a small tool-share move as a serving artifact.

Recorded explicitly, per devils-advocate: L3's 8pp threshold is ≈1.6·SE at
n=100, so it is deliberately insensitive — it will not false-alarm on decode
noise, and correspondingly **cannot detect a real sub-8pp cap-hit effect**. L3
is a plumbing guard, not a test of the stub's effect on budget, and must not be
reported as the latter.

---

## Addendum 3 — served-prompt verification (S0), measured on the live servers

The chat-template override was verified three ways before any rollout finished,
in increasing strength:

1. **Launcher echo** — each cell logs the template path it was handed, plus the
   `sha256` of `nostub.jinja` (`603813c5466bbf49`, identical across cells).
2. **vLLM arg dump** — the nostub cells list
   `'chat_template': '<...>/prompts_adp/nostub.jinja'` under `non-default args`;
   the stock cells do not. This proves vLLM *accepted* the flag.
3. **Rendered-token probe** — the decisive one. `POST /tokenize` on each live
   server with the same one-message conversation, `add_generation_prompt=true`
   **and `chat_template_kwargs={"enable_thinking": false}`** (the kwarg the
   harness always sends):

   | cell | model | template | tokens | tail |
   |---|---|---|---|---|
   | A | arm  | stock  | 13 | `… 248068, 271, 248069, 271` |
   | B | arm  | nostub |  9 | `… 74455, 198` |
   | E | base | stock  | 13 | `… 248068, 271, 248069, 271` |
   | F | base | nostub |  9 | `… 74455, 198` |

   The 4-token difference is exactly `<think>` `\n\n` `</think>` `\n\n`
   (`[248068, 271, 248069, 271]`) and nothing else — the prefix is
   byte-identical. The manipulation is confirmed **in the string the model
   actually conditions on**, on all four cells, not merely in a launcher flag.

**Recorded trap for anyone re-running this probe:** omitting
`chat_template_kwargs` makes stock and nostub render *identically* (both 11
tokens, ending `<think>` `\n`). That is correct behaviour, not a failed
override — `nostub.jinja` only edits the `enable_thinking=false` branch, and the
default branch opens a real `<think>` for the model to fill. A probe without the
kwarg tests the wrong branch and would wrongly read as "the override did
nothing."

### What the paired SE is not (limitation, per devils-advocate)

Added **after launch but before any results existed**; it states a limitation of
an already-registered instrument and changes no prediction, threshold, or
decision rule.

The paired-difference SE is the right gate for the mean-effect question asked
here, and it is self-correcting in the useful direction: decode run-noise
inflates `sd(dᵢ)`, which inflates the SE, which *widens* the `2·SE`
uninformative band — so nondeterminism makes the test more conservative, never
less. Run-noise cannot bias the mean paired Δ (symmetric flips cancel); it can
only widen the band.

But the SE **bundles run-noise with genuine per-instance treatment
heterogeneity**, and therefore is *not itself a measurement of run-noise*. Only
a same-condition replicate (E vs E′ on identical settings) could decompose the
two. No such replicate is being run, and none is needed for this decision — but
the SE must not be quoted as "the harness's run-noise."

---

## Addendum 4 — scoring-harness defect found and corrected before it touched a number

Recorded because it affects provenance of every score in this ladder, and
because it is the exact failure mode this document's discipline exists to catch.

My first `score_ladder.sh` had two defects, each of which would have corrupted
the readout **silently** rather than failing loudly:

1. **No completeness gate.** It submitted scoring as soon as `output.jsonl`
   existed, so cells were scored on 4/50, 5/50, 6/50 instances while inference
   was still appending rows. Paired with its `merged.report.json -> SKIP`
   branch, the first partial merge to land would have been treated as final and
   that cell would **never** have been re-scored on the full 50.
   `ladder_readout.py` takes its denominator from the merged report, so it would
   have printed a confident rate over ~6 instances and quietly shrunk the paired
   intersection — a silent-denominator error, the thing the "never assume 100"
   rule was written to prevent.
2. **Wrong idempotency key.** `run_score_shards.sbatch` writes per-shard
   `shard_NofM.report.json`; it does **not** write `merged.report.json` (that is
   a separate `merge_shard_reports.py` step). The "already merged" guard could
   therefore never become true, so every 10-minute tick resubmitted every
   eligible cell. 36 stray scoring jobs accumulated.

**Caught before any `merged.report.json` existed** — verified explicitly, all 12
cells showed `merged=no` — so no reported number was ever derived from partial
data. Remediation: cancelled all 36 stray jobs, deleted all 12 `score_par_*`
dirs (1.1G of derived artifacts; the 3.3G of `out_par_*` rollouts was not
touched), and rescored from scratch under a fixed gate. A cell is now eligible
only when `out+err >= expected` **and** its infer array element has left the
queue — the second condition matters, and held back `F__s01` while it sat at
50/50 with its element still queued.

One further trap the fix has to respect: `merge_shard_reports.py` **globs**
`shard_*.report.json`, so stale partial shard reports would be silently mixed
into a fresh merge. Rescoring a cell therefore removes its score dir rather than
overwriting it.

Scripts committed for provenance: `swe-bench-fair-evals/scripts/score_ladder.sh`
and `analysis/ladder_readout.py`, both requiring `SWEBENCH_ROOT` / `INFER_ARRAY`
rather than carrying cluster paths.

**Collection status at time of fix:** all four arm cells complete at 100/100
(A, B, C, D — 50/50 on each shard, 0 error rows), so the entire arm ladder
(S1 = B−A, C−B, D−C, and L1 = C−A) is fully collected. Base cells still running,
consistent with their cap-hitting long rollouts: F at 46/50 and 50/50, E at
7/50 and 4/50.

## Addendum 5 — a defect in the *harness's* multi-attempt aggregator, found while checking F's duplicate rows

Addendum 4's defects were mine. This one is in the benchmark harness, it is
still live, and it is **asymmetric between base and arm** in exactly the
direction that matters for the pre-registered base-vs-arm comparison. Found
2026-08-05 while chasing why `F__s00` held 87 rows for a 50-instance shard.

### What the rows meant

`out_par_F_base_nostub_evalp__s00` sat at 87 rows / 50 unique ids mid-flight,
while the *completed* `F__s01` had exactly 50 rows despite 35 attempt-2 retries.
So `output.jsonl` is not append-per-attempt: the harness rewrites it from all
attempt files at the end of the run (`aggregate_results`). The 87 rows were a
transient state. My dedup guard therefore never fires on a cleanly-finished
shard — it only matters if a job dies mid-consolidation.

Chasing which attempt survives consolidation is what surfaced the defect.

### The defect

`benchmarks/benchmarks/utils/iterative.py`:

* `aggregate_results()` iterates attempts **last → first**, replacing the
  incumbent only when `entry.beats(current)`.
* `_AggregatedEntry.beats()` is `return self.rank > other.rank` — strict.
* `_get_output_rank()` returns **0** = error, **1** = no-error/critic-failed,
  **2** = critic-passed.

An **empty patch** and a **substantive-but-critic-failing patch** are both
rank 1. Equal ranks therefore resolve in favour of whichever was seen first,
which — given the last→first iteration — is the **latest** attempt. The rank
function never looks at whether a patch exists. Retries also bump temperature
from 0.0 to 0.1, so a later attempt is a fresh non-greedy sample.

Net effect: **a degenerate final retry silently discards a good earlier patch.**

Note an empty final also implies no attempt ever reached rank 2 (a rank-2 entry
wins regardless of order), so every candidate in these cases is rank-1 and the
choice among them is precisely the tie the harness breaks arbitrarily.

Concretely, `django__django-12325` in `F__s01`: attempt 1 produced a 1868-byte
patch, attempt 2 produced 5806 bytes, and the record that got **scored** has a
patch of **0 bytes**.

### Why it is asymmetric — measured, not argued

| run | final records | empty patch | discarded a non-empty attempt |
|---|---|---|---|
| all 8 arm ladder cells (A/B/C/D × 2 shards) | 400 | **0** | **0** |
| base ladder cell F, shard 01 | 50 | 12 | **9** |
| θ₀ multi-attempt run `v2_init_4b`, 10 shards | 300 | 62 | 26 |
| arm board run `v2_swezero_4b`, 10 shards | 500 | **0** | **0** |

The exposure is behavioural, not configurational: all cells ran the same
protocol, but the base model fails the critic on ~70% of instances and so gets
retried constantly, while the arm retries 2–3 times per 50 and never lands on an
empty final. The defect can only bite a model that retries.

### Two things this does **not** change

1. **The arm ladder is untouched.** 0/400 affected, so A/B/C/D = 7/8/7/11 and
   every arm rung (S1, C−B, D−C, L1) stands exactly as reported in the readout.
2. **The θ₀ = 119/500 anchor is immune.** That number comes from
   `v2_init_singlerun_4b`, which has **zero** `critic_attempt` files — a
   single-attempt run cannot hit a multi-attempt tie-break. The affected
   `v2_init_4b` (145/500) is the separate retry-boosted protocol. I initially
   mis-stated this as a threat to the headline anchor; it is not.

### What it does change: cell F, and only because F has no error rows

In `v2_init_4b` the damage was largely self-cancelling by accident: the scoring
sbatch concatenates `output_errors.jsonl` with `output.jsonl`, and **24 of those
26** instances also appear in the error file carrying a non-empty patch, so they
were already scored on a real patch (at the cost of being scored twice).

`F__s01` has **zero error rows**, so **none** of its 9 are rescued. Nine of
fifty instances — 18% of the shard — will be scored on an empty patch despite
having produced patches of up to 12.8 KB. Every effect in this ladder is ≤4/100.
A bias of this size sits an order of magnitude above the signal it would
contaminate, and it points one way: it deflates base.

### Repair, and why it keeps parity

Minimal change to the tie-break only, leaving the rank ordering alone: *among an
instance's rank-1 attempts, prefer a non-empty patch, then the latest such
attempt.* Implemented in `analysis/repair_aggregate.py`, which emits only the
records whose selection actually changes, into an isolated directory (the
scoring sbatch pulls `output_errors.jsonl` from the input's own directory, so an
isolated dir scores exactly those rows).

Applied uniformly to all six cells. On the arm cells it is a **provable no-op**
(0/400 affected), so parity is preserved by construction rather than by
assumption — this is not a correction applied to base and withheld from arm.

**Both numbers get reported.** The original scoring of `F__s01` was left running
to completion rather than cancelled, so the readout will carry the as-harness
score (what the standard harness reports) alongside the repaired score (the
better estimate of what the model actually produced), clearly labelled. Swapping
one for the other silently is exactly the move this document exists to forbid.

Scoring of the repaired records was submitted with `--dependency=afterany` on
the original job: the scoring sbatch prunes Apptainer sandboxes keyed by
`instance_id` alone, so two concurrent jobs scoring the same instance would
delete each other's sandbox.

**This is the fifth silent-degradation defect in this pipeline in one day** —
partial-output scoring, a missing merge step, the `n_think_calls` typo, an
over-broad composite flag, and now a patch-blind tie-break in the harness
itself. None of the five raised an error. All five would have shipped a
confident wrong number.

---

## Addendum 6 — L3 is going to fail, and it fails *informatively*. Written before E completed.

**Committed in advance, on partial data, so the interpretation cannot be
retrofitted to the final number.** At the time of writing E has 34 of 100
transcripts and F has 100 of 100. The prediction below is what I will conclude
when E lands, stated now.

### What the partial data shows

| cell | model | template | seen | transcripts | cap-hit | native `<think>` |
|---|---|---|---|---|---|---|
| E | base | stock (stub) | 65 | 34 | 21 (32.3%) | **0 / 34** |
| F | base | nostub | 100 | 100 | **0 (0.0%)** | **100 / 100** |
| A–D | arm | either | 100 | 100 | 0 (0.0%) | 0 / 100 |

L3 registered `|cap-hit(F) − cap-hit(E)| ≤ 8pp`. The partial gap is **32pp** and
the direction of both components is already unambiguous. **L3 will fail.**

### Why the failure branch does not apply as written

L3's failure branch says: *inspect the rendered prompts and the vLLM logs before
interpreting any E/F result; an E−F delta is then confounded by budget rather
than serving format.* The first half is already done — **Addendum 3's rendered-
token probe (S0) settled it before any rollout finished**: E and F differ by
exactly `<think> \n\n </think> \n\n` (`[248068, 271, 248069, 271]`) with a
byte-identical prefix, verified on the live servers. There is no second
difference left to find. The mandated inspection has been performed and returns
"the only difference is the registered manipulation."

So the failure is not a plumbing failure. It is a **failure of the reasoning I
used to register L3**, and that reasoning is worth quoting against myself:

> The 500-iteration cap counts **agent iterations, not tokens**, so trimming 4
> tokens per assistant turn has no mechanism by which it should move the cap-hit
> rate.

The premise "4 tokens ⇒ no behavioural mechanism" is false for base. The four
tokens are not padding: a *pre-filled, already-closed* `<think>\n\n</think>`
occupies the reasoning slot and leaves nothing for the model to fill. Measured
consequence, not inference: base emits a native `<think>` block in **100/100**
nostub records and **0/34** stock records. The stub does not trim base's
reasoning, it **closes the channel**. A model denied its reasoning channel takes
more iterations to get nowhere, and 32% of the time it runs out.

### What this changes

1. **It is inert for the arm and drastic for base, for a reason that was
   predictable and that I did not predict.** SFT trained the native reasoning
   channel out of the arm entirely (0/100 native `<think>` in every arm cell,
   stock or nostub), so for the arm the stub occupies a slot the model was never
   going to use. Same four tokens, opposite consequence — the manipulation is
   only "small" relative to a model that still uses the channel.
2. **E is not a "base under parity" cell.** It measures what happens to base
   when its reasoning channel is closed. Any E−A number is a comparison between
   a gagged base and an arm that does not use the channel — interesting, but not
   the base-vs-arm contrast, and it must not be reported as one.
3. **F−B remains the right rung, and it was designated primary before any
   results existed.** No substitution after seeing results. **Corrected wording,
   per devils-advocate, who checked the timeline and was right:** an earlier
   draft said "before launch," which is false. Verified timestamps, all UTC —
   array *submitted* 02:58:54, array *started executing* 04:04:09, Addendum 2
   (naming the F−B differential) committed `b58b721` **03:53**, first arm
   `merged.report.json` written **07:28**. So F−B was named 55 min after
   submission, 11 min before the first rollout token was generated, and ~3.5 h
   before any result existed. Pre-*results* is the bar that matters and it is
   met with room to spare; pre-*launch* was a stronger claim I could not support
   and should not have made in a document whose entire subject is
   pre-registration integrity.
4. **The two known biases on F both still point the same way.** F is depressed by
   the aggregator tie-break (15/100 discarded good patches, Addendum 5) and E by
   cap-survivorship. Neither touches the arm cells. F−B stays a **directional
   lower bound**.
5. **L3 did its job.** It was registered as the cheapest available guard against
   the ladder silently measuring budget instead of parity. It caught a real
   budget asymmetry between E and F. That the cause turned out to be the
   treatment rather than the plumbing is what a placebo check is *supposed* to be
   able to tell you, and only because S0 had already excluded the alternative.

### What would falsify this reading

If E's native-`<think>` count rises materially above 0 as the remaining ~66
records land, the "closed channel" account is wrong and the cap-hit gap needs
another explanation. Recording the threshold now: **more than 5 of E's final
~100 records emitting a native `<think>` block** falsifies it. (5 is the stock
baseline already on record for base from the earlier 324-record board run.)

### 6b. The two findings this failure actually delivers — and the asterisk it costs

Added after devils-advocate's review argued I had under-sold the L3 failure.
Both points are theirs; the measurements below are mine, run on the θ₀ artifact
itself rather than inferred from the ladder.

**(i) It falsifies "base is stub-native," which `think_stub_prereg.md` §2
registered as a real open question.** That document was explicit that the whole
base-is-in-distribution premise rests on an untested step — "if base is
genuinely stub-native, removing the stub should **hurt** base." Removing it did
not hurt base; it un-gagged base. Cap-hit 32.3% → 0.0%, native `<think>`
0/34 → 100/100. The shipped template's `enable_thinking=False` stub is **not**
base's home format. This is a cleaner headline than "a placebo check failed,"
and it is a direct answer to a question that was registered before any of this
ran.

**(ii) It closes the base cap-hit mystery from the infra-asymmetry thread.**
Measured directly on `out_v2_init_singlerun_4b` — the θ₀ = 119/500 run itself,
not a proxy:

| | records | native `<think>` | MaxIterationsReached |
|---|---|---|---|
| θ₀ board run (base, **stub**) | 500 | **4 (0.8%)** | **131 (26.2%)** |
| ladder E (base, **stub**) | 34 so far | 0 | 32.3% |
| ladder F (base, **nostub**) | 100 | **100 (100%)** | **0 (0.0%)** |

The ~26% cap-hit rate that was flagged weeks ago and never mechanistically
closed is the same stub-gag: reasoning channel closed → iterations burned →
cap. Three independent runs line up on it. That is the strongest form an
informative failure takes — it resolves a previously-open question rather than
merely excusing itself.

**The asterisk, which is the reason to believe this is not a motivated
rescue.** θ₀ = 119/500 was measured with base's reasoning channel closed and
26.2% of its instances dying at the iteration cap. **119 is a base FLOOR, not a
point estimate**, and un-gagged base plausibly scores higher. Every future quote
of θ₀ — including in `adp-v2-swebench-arm-results` — needs "measured with the
reasoning channel gagged" attached.

Note what this does and does not do to the campaign verdict. It does **not**
rescue the arms: the stub is inert for them (0/100 native `<think>` in all four
arm cells, stock or nostub, because SFT removed the channel), so un-gagging base
only **widens** base ≫ arms. The SFT-lift-is-null-to-negative verdict gets
stronger, not weaker. Re-explaining a failed placebo in a way that softens my
own headline anchor while strengthening the conclusion I already hold is the
opposite of the shape motivated reasoning takes — but the asterisk is owed
regardless of which way it cuts.
