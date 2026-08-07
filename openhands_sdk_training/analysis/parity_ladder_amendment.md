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

### Outcome of the repair on F — final, both shards scored

| shard | discarded good patches | rescued by repair |
|---|---|---|
| `F__s00` | 6 | **0 / 6** |
| `F__s01` | 9 | **2 / 9** |
| F total | 15 / 100 | **2 / 15** |

**F = 27/100 as-harness, 29/100 repaired.** Both numbers are now final; an
earlier channel post said "29 can only go up" because s00's repair pass was
still running — it did not go up. Note what the 2/15 means: the tie-break was
throwing away real patches (15 of them, up to 12.8 KB), but 13 of those 15
patches would not have resolved their instance anyway. The defect is real and
the bias direction is real; on this cell its *magnitude* on score turned out to
be +2. That is a finding about the patches, not a reason to stop repairing —
15/100 discarded is the exposure, and which 15 get discarded is arbitrary with
respect to correctness.

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
   the aggregator tie-break (15/100 discarded good patches, Addendum 5 —
   measured cost on score now known: **+2**, 27 → 29) and E by
   cap-survivorship. Neither touches the arm cells. F−B stays a **directional
   lower bound**.
5. **L3 did its job.** It was registered as the cheapest available guard against
   the ladder silently measuring budget instead of parity. It caught a real
   budget asymmetry between E and F. That the cause turned out to be the
   treatment rather than the plumbing is what a placebo check is *supposed* to be
   able to tell you, and only because S0 had already excluded the alternative.

### 6a. E's failures are a *mixed* taxonomy — and where the cap-hits hide

Added after a peer scan of E reported "0 MaxIterationsReached across 146 base
rollouts." That scan read `output.jsonl`. **Capped instances never reach
`output.jsonl`** — the SDK harness writes them to `output_errors.jsonl` in the
same directory, and the two id sets are disjoint (verified: `overlap = 0` on
both E shards). Searching only the transcripts searches the one file from which
cap-hits are definitionally absent. Recording it because it is the same class of
error as the ones this document already catalogues, and because it would have
retracted a true finding.

Full untruncated `error` strings, both E shards, 53 rows over 99 instances seen:

| E failure mode | n | % of seen |
|---|---|---|
| **`MaxIterationsReached: … maximum iterations limit (500)`** | **35** | **35.4%** |
| instance timeout 14400s (wallclock) | 5 | 5.1% |
| remote conversation got stuck | 5 | 5.1% |
| run timeout 3600s (wallclock) | 4 | 4.0% |
| `OSError: [Errno 28] No space left on device` | 3 | 3.0% |
| remote conversation ended with error | 1 | 1.0% |
| — F, same harness/cap/shards/weights, nostub | **0** | **0.0%** |

Two things follow, and they must not be merged:

* The cap-hit claim **strengthens**: 35.4% (stub) vs 0.0% (nostub) on the same
  harness `sdk_43376f1`, same `maxiter_500`, same shard pairs, same weights.
  35.4% is if anything above the θ₀ board run's 26.2%, and it reproduces on new
  hardware and a new date, so the ~26% is a property of the **stub**, not of
  that one run.
* **But E's failure set is not pure cap.** 18 of 53 are wallclock or infra,
  including three ENOSPC. Quoting "E fails half the time" bundles a disk-full
  node into a behavioural claim. Every cap number in this document means
  `MaxIterationsReached` specifically and should be read that way.

## Addendum 7 — the primary comparison is not compute-matched, and this bias inflates base

**This is the most consequential defect found in the ladder so far, and it cuts
against the headline rather than for it.** Found while chasing an unrelated
discrepancy (E's `output.jsonl` rows were all tagged `attempt=1` while F's spanned
attempts 1–3).

The harness runs up to `n_critic_runs` attempts per instance and retries whichever
instances its critic judges failed (`_get_instances_for_attempt` →
`get_failed_instances` → `critic.evaluate(events, patch)`). Every cell was
launched with **identical** configuration — same critic, `n_critic_runs = 3`, no
per-cell flags. What the cells actually consumed:

| cell | attempt 1 | attempt 2 | attempt 3 | rollouts / instance |
|---|---|---|---|---|
| A arm stock | 100 | 5 | 0 | 1.05 |
| B arm nostub | 100 | 3 | 1 | **1.04** |
| C arm nostub wrapper | 100 | 4 | 0 | 1.04 |
| D arm nostub trainprompt | 100 | 2 | 0 | 1.02 |
| E base stock | 99 | 0* | 0* | (in flight) |
| F base nostub | 100 | 71 | 61 | **2.32** |

\* E is still running; s01 had just entered attempt 2 at the time of measurement.

The critic rejected base's first attempt **71/100** times and the arm's **2–5/100**
times. So F received ~2.2× the rollouts of B *and* a best-of-3 selection over
them, while the arm cells were effectively single-shot. Attempts 2–3 also sample
at **temperature 0.1** rather than 0 (`evaluation.py` raises it for `attempt > 1`),
so F's extra rollouts are additionally diversified.

**This is not a bug and not a misconfiguration.** It is the harness spending more
compute on whichever model its critic dislikes. It is nonetheless a confound for
any "base vs arm" claim, because F−B now bundles the model contrast with a
2.2× compute-and-selection advantage to base.

### The correction this forces to my own earlier claim

I wrote, in Addendum 6 point 4, that the two known biases on F "both still point
the same way" and therefore **"F−B stays a directional lower bound."** That is
now **wrong and withdrawn.** The tie-break (Addendum 5, worth +2) and E's
cap-survivorship both depress base; this one *inflates* base, and unlike those
two its magnitude is not yet measured. With biases pointing in both directions
and one of them unquantified, F−B = +19 (as-harness) / +21 (repaired) is a
**point estimate with two-sided uncertainty, not a lower bound.** I should not
have called it a lower bound on the strength of an enumeration of biases I had
not finished making — "both the biases I have found so far point one way" is not
the same claim as "the bias points one way," and I elided the difference.

### The compute-matched contrast, and why attempt 1 is the right one

Attempt 1 is matched on everything that differs above: exactly one rollout per
instance, temperature 0, all 100 instances in every cell, no critic selection,
and frozen in `output.critic_attempt_1.jsonl`, which `aggregate_results` never
rewrites. Scored via `analysis/attempt1_subset.py` +
`swe-bench-fair-evals/scripts/score_attempt1_ladder.sh` (CPU-only, scavenge,
same sandbox-collision and idempotency guards as the repair pass; 10 shards
submitted, E deferred until its inference finishes).

**Registered before those scores exist, so it cannot be retrofitted:** F−B at
matched compute should come out **smaller than +19**, because the advantage
being removed is base's. I expect it to remain clearly positive — F's 27
resolves cannot all be selection artifacts when 29 of its 100 instances
finalized at attempt 1 — but I am committing to reporting whatever it is,
including the case where the gap collapses. If matched-compute F−B falls below
~2·SE (SE computed from the actual discordant pairs, not assumed), then **the
campaign has no demonstrated within-harness base-vs-arm gap at matched compute**.
~~and the correct statement becomes that the arms are not distinguishable from
base rather than that base beats them.~~ **That last clause is too broad and is
struck** — see 7a, registered before the scores exist.

### 7a. How I will read the matched number, registered before it exists

Written after devils-advocate pointed out that my registration above had one
sentence that overreaches and one force I had not accounted for. Both corrections
accepted; both change what a small result would license me to say, so they have
to be on record *now*.

**Correction 1 — the collapse branch was scoped wrong.** "The arms are not
distinguishable from base" is a claim about the campaign; what this experiment
can speak to is 100 instances, one harness, one arm cell. The board number
(θ₀ = 119/500 vs swezero 77/500) is **single-rollout by construction on both
sides** — `out_v2_init_singlerun_4b` has zero critic-attempt files — so it never
had this confound to lose, and a small matched F−B cannot retract it. If the gap
collapses, the licensed statement is *"no demonstrated base-vs-arm gap at pass@1
on these 100 instances under this harness"*, and the board's 500-instance result
stays in its own box. Different n, different instance set, and (see below)
different base.

**Correction 2 — two forces pull on the magnitude in opposite directions, and I
had only counted one.** Removing attempts 2–3 removes base's compute advantage
(pushes F−B **down**, which is what I registered). But it also removes the
patch-blind tie-break defect, which is *measured* to cost F 2 points
(27 as-harness → 29 repaired, Addendum 5), and separately ladder-F is the
**un-gagged** base while the board's 119 is the gagged one, i.e. the ladder's
base should if anything be the stronger of the two. Both of those push **up**.
So "smaller than +19" is my expectation, not a boundary: **matched F−B landing at
or above +19 is a legitimate outcome of this design, not a sign something broke.**

**Input verification, run before any score existed** (so a broken input could not
be discovered after the fact and rationalised). For each of A/B/C/D/F: the subset
file at `runs/a1_<tag>__s{00,01}/output.jsonl` has **100 rows / 100 unique
`instance_id`s**, its id set is **exactly equal** to that cell's original
`output.critic_attempt_1.jsonl` id set, the scoring inputs
(`score_<tag>__s*_a1/shard_*of2.jsonl`) carry the same 100 unique ids, and **all
five cells cover an identical instance set** (symmetric difference 0 against A).
The pairing is therefore real, not assumed.

**A collision I permitted, and the validity check that catches it.** The a1 batch
runs 20 tasks that all score the **same 100 instances** concurrently. The scoring
sbatch prunes Apptainer sandboxes keyed by `instance_id` alone, so cross-cell
concurrency is exactly the hazard `reagg_ladder.sh`'s guard was written to prevent
— and that guard only covers *same-cell* jobs (`sc_$TAG`, `sc_${TAG}_reagg`), so
nothing blocked it. Mostly this costs wall-clock (sandbox rebuilds), but a sandbox
deleted mid-verify could produce a spurious *unresolved*, which would bias every
cell downward and look like a real effect.

Registered before the scores exist, as a cheap internal check: the arm cells
barely retried, so **each arm's a1 score must land within ~1–2 of its aggregate**.

| cell | aggregate | instances with an attempt 2 | a1 must be ≈ |
|---|---|---|---|
| A | 7/100 | 5 | 5–7 |
| B | 8/100 | 3 | 6–8 |
| C | 7/100 | 4 | 5–7 |
| D | 11/100 | 2 | 9–11 |

**If the arm a1 scores come in materially below those bands, the batch is
corrupted by sandbox collision and every a1 number including F's must be
discarded and re-run serially — not interpreted.** Only F's a1 is expected to
differ substantially from its aggregate, because only F retried heavily. This
check cannot be retrofitted: it is arithmetic on numbers that already exist.

Registered readings, by band (SE from the observed discordant pairs):

| matched F−B | what I will conclude |
|---|---|
| ≥ +19 | the aggregated gap was not compute-inflated on net; the tie-break deflation and the retry inflation roughly cancelled. Report matched as the headline, aggregated as the confounded companion. |
| between 2·SE and +19 | gap real, aggregated form partly compute-inflated. Headline the matched number and state the inflation size. |
| positive but < 2·SE | directionally consistent with the board's gagged-base gap, underpowered at n=100. **Not** "no gap" — "not independently significant at this n." |
| ≤ 0 | a genuine surprise *against the board*, not a confirmation of the arms. Owed a reconciliation, not a headline: shard-00/01 instance-set representativeness, n=100 power, and the fact that F is the un-gagged base and should have been the easier direction. I would report it as an unresolved conflict between two measurements, and say so. |

**A third finding, not about the ladder.** The critic rejects base's attempt 1
**71/100** and the arms' **2–5/100** under byte-identical config. That asymmetry
is the mechanism of this confound, but it is also an unmeasured *model-dependence
in the critic itself*, ~15–35×. Anything that reuses this critic — retry gating,
data filtering, an outcome-labeler — inherits a component that judges base and
SFT'd outputs by what is evidently not the same standard, for reasons nobody has
characterised. Flagging it here because it outlives this ladder; it is not
something I have scope to chase.

### 6b-pre. How the harness writes results, and the matched form of L3

Read out of `benchmarks/utils/evaluation.py` + `utils/evaluation_utils.py` after
noticing that E's `output.jsonl` had 99 rows all tagged `attempt=1` while F's had
exactly 100 spread over attempts 1/2/3. Three facts, none of them documented
anywhere in the kit:

1. Each attempt's rollouts are frozen in `output.critic_attempt_{N}.jsonl`.
2. `output.jsonl` is an **append log of every attempt while the job runs**, and is
   **rewritten** by `aggregate_results(final_output_file="output.jsonl")` only at
   the very end. Reading it mid-flight and reading it after completion are
   therefore *two different measurements*.
3. **Attempts 2–3 raise temperature 0.0 → 0.1.** Attempt 1 is deterministic;
   retries are not. Which instances get an attempt 2 at all is decided by the
   critic.

Consequence for L3: pooling all attempts compares *different mixtures* per cell,
and an incomplete cell is not comparable with a complete one at all. The matched
comparison is **attempt 1**: temp 0 in both cells, run on every instance in both
cells, frozen on disk, untouched by aggregation, and available for a cell that
never finishes.

| attempt-1 rollouts (temp 0, all instances) | n | `MaxIterationsReached` | cap% |
|---|---|---|---|
| E (base, **stub**) | 99 | **35** | **35.4%** |
| F (base, **nostub**) | 100 | **0** | **0.0%** |

35 / 99 vs 0 / 100 on the shared instance set. **L3 fails at −35.4pp against a
registered ≤8pp gate**, on the best-matched form of the test rather than the most
convenient one. This is a deviation from the letter of the registration, which
named the aggregated cell; it is reported alongside the registered form, not
substituted for it, and the direction was pre-committed in Addendum 6 at 32pp
before this number existed.

**Also a live hazard, and defect #6 in the silent-degradation catalogue:** if E is
killed by walltime, `aggregate_results` never runs and E's `output.jsonl` is left
as a raw append log — potentially with duplicate `instance_id`s across attempts.
Anything downstream that assumes one row per instance would then double-count
without erroring. `read_cell()` is set-based and safe; the scoring path is not
obviously so, and an E that times out must not be scored without checking this
first.

### What would falsify this reading

If E's native-`<think>` count rises materially above 0 as the remaining ~66
records land, the "closed channel" account is wrong and the cap-hit gap needs
another explanation. Recording the threshold now: **more than 5 of E's final
~100 records emitting a native `<think>` block** falsifies it. (5 is the stock
baseline already on record for base from the earlier 324-record board run.)

**Falsifier check at E = 52 records (2026-08-05): not falsified. 0 of 52.** Still
survivorship-limited; E is on attempt 1 at 9h40m of a 16h limit.

### 7b. What F's "native think" actually is — a correction to my own claim

Re-measured on all 21,763 of F's assistant messages while running the behavioural
pass, because a sampled record looked wrong. It was: the tag-level number holds,
the mechanism story I attached to it does not.

| F (base, nostub), all assistant messages | value |
|---|---|
| messages containing a literal `<think>` | **21763 / 21763** |
| `<think>` at character 0 of the message | **21763 / 21763** |
| messages ever emitting a closing `</think>` | **0** |
| chars between `<think>` and the first markdown header, p50 / p90 / p99 / max | **56 / 237 / 1112 / 6167** |
| messages with >200 such chars | **12.1%** |
| whole remainder after the tag, p50 | **68 chars** |

Per-cell tag counts (record level): A 0/100, B 0/100, C 0/100, D 0/100,
E **0/52**, F **100/100**.

So "F emits native `<think>` in 100/100 instances" is **true and the E-vs-F
contrast is real** — E emits the tag zero times, F every time, and that is
squarely attributable to the stub. But the reading I hung on it, that removing
the stub *restored base's reasoning channel*, overstates what is there. What
removing the stub restores is an **unclosed `<think>` opener with a median of one
sentence behind it** (56 chars; F's whole message is a median 68 chars), which
then runs straight into the same `## Phase N` prose E produces without any tag.
Only 12% carry more than 200 characters. Nothing ever closes the block.

**What this does and does not change.** The *score* comparison is untouched — F's
27/29 comes from resolved patches, not from this. The cap-hit collapse (E 35.4% →
F 0.0% at attempt 1) is untouched, and remains the substantive finding. What is
weakened is the causal narrative "the stub gags a reasoning channel base would
otherwise use productively": on this evidence the channel base opens is thin and
malformed, so the cap-hit collapse more likely comes from the *template edit
changing the whole generation distribution* than from reasoning content base was
being denied. Restating the falsified premise precisely: `think_stub_prereg.md`
§2's "base is stub-native" prediction still fails — removing the stub helped base
rather than hurting it — but "and it helped because base reasons natively" is my
addition and is not supported. Downgraded to: **removing the stub changes base's
output distribution substantially and in base's favour, by a mechanism not yet
identified.**

### 7c. Two analysis-code defects found by the same check

Both are silent-zero shapes, the same family as the harness defects above.

1. **`extract_traj_stats.py`'s `n_native_think` and `n_reason_content` are
   uniformly 0 for every cell, F included.** They read `thinking_blocks` /
   `reasoning_content`, which this harness never populates — the content lands in
   `thought` as a literal string. Any conclusion drawn from those two fields
   reads as "no cell reasons natively," which is wrong for F and indistinguishable
   from a true zero. `n_any_reason` is non-zero only because it also ORs in the
   `think` *tool*.
2. **The extractor's own comment is wrong about who calls the think tool.** It
   asserts base "never calls the think tool; the SFT arm is the exact mirror."
   Measured: the think tool is called in E 52/52, F 100/100, A 98/100, B 98/100,
   C 97/100, D 97/100. **Every cell calls it in ~all instances.** The comment
   justifies counting the channels separately, which is still right, but the
   stated base-vs-arm mirror does not exist and should not be cited.

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

## Addendum 8 — the scoring harness returns batch-dependent verdicts on identical patches

Found by the validity gate registered in 7a, which I had expected to pass
trivially. It failed, and the failure is not the one I was guarding against.

### What the gate found

The gate's premise was an identity: aggregate and attempt-1 scoring can only
differ on instances that were actually retried, because on a non-retried instance
both score the same rollout. Movement in the totals obeyed it. The *identity of
the resolved instances* did not:

| cell | agg | a1 | \|move\| | retried | instances disagreeing on a NEVER-retried rollout |
|---|---|---|---|---|---|
| A | 7/100 | 8/100 | 1 | 5 | **9** |
| B | 8/100 | 6/100 | 2 | 3 | **8** |
| C | 7/100 | 10/100 | 3 | 4 | **9** |
| D | 11/100 | 9/100 | 2 | 2 | **4** |

Registered tolerance was 2, on the grounds that a couple of genuinely flaky tests
are expected. 4–9 is not that.

### Ruling out the innocent explanations

**Not different patches.** For every non-retried instance in every cell, the
`git_patch` in `output.jsonl` and in `output.critic_attempt_1.jsonl` is
**byte-identical** — 0 differences out of 95/97/96/98/29 instances for A/B/C/D/F.
So the two runs scored the same bytes.

**Not random flakiness, and not the sandbox collision I was worried about.** Of
the 19 distinct instances that flip, **9 flip in more than one cell, and 8 of
those 9 flip in the same direction in every cell that flips them** — across
different models producing different patches:

| instance | cells | direction |
|---|---|---|
| `django__django-13363` | A, B, C | lost in all three |
| `django__django-13820` | A, B, C | gained in all three |
| `django__django-16527` | A, B | lost in both |
| `django__django-16255` | A, C | lost in both |
| `pytest-dev__pytest-7205` | A, C | gained in both |
| `django__django-16801` | B, D | lost in both |
| `scikit-learn__scikit-learn-10908` | B, D | gained in both |
| `django__django-12050` | C, D | lost in both |
| `scikit-learn__scikit-learn-13439` | C, D | **gained / lost** (the lone exception) |

Random contention does not pick the same instance in three independent cells and
push it the same way. This is an **instance-level property of the scoring
environment as built in a given batch**, and it is roughly balanced in sign
(gains ≈ losses), which also argues against collision-induced sandbox deletion —
that would produce excess *unresolved*.

### What this costs, and what it does not

**Cross-batch score comparisons carry it; within-batch cell differences largely
cancel it.** If `django-13363` fails to resolve for every cell in a batch, it
subtracts from both sides of any difference computed inside that batch.

That distinction is decisive for which number to quote, and it does not favour the
one I have been treating as the reference. Merge times: the arm aggregates landed
06:48–07:28, **F's aggregate landed 09:29 and 11:09** — a different, later batch.
The a1 scorings for A, B, C, D and F were submitted together and merged
14:41–14:51 — **one batch**.

- The aggregated **F−B = +19 is exposed to this effect**, because F and B were
  scored in different batches. On top of the compute confound in Addendum 7.
- The matched **F−B from the a1 batch is both compute-matched and batch-matched**,
  and is therefore the better instrument on two independent grounds. I did not
  anticipate the second one.

**Residual error that does not cancel.** 10 of the 19 flips occur in a single cell
only (~2.5 per cell). Those are the shape cross-cell sandbox collision would
produce — a collision hits whichever cell was verifying that instance at that
moment, not all of them. So the a1 numbers carry roughly **±2–3 per cell of
non-cancelling error**, and matched rungs smaller than that are not resolvable by
this batch.

**A statement I must therefore not make:** "F fell from 27 to X at matched
compute." That subtraction crosses two batches and mixes the compute effect with
this one. The defensible comparisons are cell-vs-cell *within* the a1 batch, and
cell-vs-cell within the aggregate batch, each quoted separately.

**Consequence for a memory claim.** The recorded run-noise calibration (14/100
instances flipping between nominally identical runs) was attributed to *rollout*
noise. Part of it is this instead: scoring noise on identical patches. The ±7/100
detection floor stands as a floor; its attribution was wrong.

---

## Addendum 9 — the readout was scoring an unmerged shard as fifty failures

Found while reading the first matched F−B number, and it moved that number, so it
goes on the record next to it rather than in a commit message.

`ladder_readout.py` paired each rung on the intersection of the cells' **rollout**
id sets — every instance present in the rollout file. That is the right universe
only when both cells are fully scored. Cell F had one of its two a1 shards merged,
so the primary matched rung printed

    F-B  base vs arm, nostub [PRIMARY, matched]   6 ->   2 on n=100  net= -4

pairing B's 100 scored instances against F's 50, and silently counting F's 50
**unscored** instances as unresolved. On the 50 instances actually scored in both
cells it is `2 -> 2, net 0`. The validity gate had the same defect from the other
side: it compared F's 100-instance aggregate (27 resolved) against its 50-instance
a1 (2 resolved) and reported a 25-point move; restricted to the shared scored set
it is 18 vs 2 against 36 retried.

Both blocks now intersect with the scored-id sets read out of the merged reports,
and partial coverage is labelled inline (`[PARTIAL 50/100]`) instead of being
absorbed into the denominator. This is the third instance this week of the same
failure family: **a missing input read as a zero rather than as missing** (the two
`extract_traj_stats.py` defects in §7c are the others). The lesson I will keep
applying: any metric whose denominator comes from a different file than its
numerator needs the two reconciled explicitly, because the failure is silent and
always in the direction of "the thing I was measuring didn't happen."

## Addendum 10 — how I will settle the failed gate, registered before the test

The registered validity gate FAILS in all five scored cells: 4–9 instances per
cell disagree between the aggregate and a1 scorings on rollouts that were never
retried (tolerance 2). The registered prescription was "re-run the attempt-1
scoring serially and re-read." I am not going to follow that literally, and here
is why, together with what I will do instead.

The gate compares the aggregate batch against the a1 batch, so it is itself a
**cross-batch** comparison — exactly the thing Addendum 8 shows is contaminated.
Its failure is therefore the *expected* consequence of Addendum 8, not new
evidence, and re-running the a1 scoring serially would produce a third batch with
its own offsets and would not distinguish the two live hypotheses:

- **H-collision:** the a1 batch is internally damaged, because 20 concurrent tasks
  scored one shared instance set and the sbatch prunes sandboxes keyed by
  `instance_id` alone. If so the a1 numbers are unusable as they stand.
- **H-batch:** verdicts are an instance-level property of a batch. Then the
  disagreement is real but **cancels inside** the a1 batch, and within-batch
  cell-vs-cell rungs remain the best available instrument.

The discriminating test is cheap and CPU-only: **re-score one cell's a1 subset
serially, alone in the queue, and compare it against that cell's first a1 result.**
Registered readings, before the numbers exist:

| serial a1 vs first a1, same cell | reading |
|---|---|
| 0–2 instances disagree | H-batch. The a1 batch is internally reproducible; the agg-vs-a1 gap is a batch offset that cancels within-batch. Matched rungs readable, ±2–3/cell. |
| ≥5 disagree | H-collision, or a third source. The a1 batch is not reproducible even alone, and **no** matched rung may be quoted; the whole matched ladder needs re-running serially. |
| 3–4 | ambiguous at this n; re-score a second cell before concluding. |

Cell B is the one to test: it is the arm side of the primary rung, fully merged,
and its a1 (6/100) sits close enough to its aggregate (8/100) that a collision
large enough to matter would be visible.

Until that test returns, the matched rungs in the readout stay flagged **do not
interpret**, and I will not quote the matched F−B as a result — only as a number
whose validity is pending one specific test.

## Addendum 11 — the Addendum 10 test returned H-collision. Root cause found, fixed, and it contaminates the aggregate board too

The registered test from Addendum 10 ran. Cell B's attempt-1 subset, re-scored alone
in the queue against the same bytes: **first run 6/100, serial re-run 13/100, 9
instances disagreeing** — and a second independent serial re-score returned 14/100.
The registered reading at ≥5 disagreements is **H-collision**: the a1 batch was
internally damaged, and no matched rung could be quoted as it stood. H-batch
(Addendum 8's "instance-level property of a batch, cancels within-batch") is
**withdrawn** as the explanation.

**Root cause, in the code rather than inferred.** `run_score_shards.sbatch` created
Apptainer sandboxes under one *shared* root and pruned them keyed by `instance_id`
alone. Every ladder cell scores the same instance set, so concurrent scoring jobs
shared a sandbox directory per instance and deleted it out from under each other
mid-run. A vanished sandbox scores as **unresolved** — silent, one-directional.
Fixed in `d748808` by giving each job a `job_${TAG}_${I}of${N}` subtree and pruning
the subtree rather than instance paths. (`env.sh`'s `ADP_WORKSPACE_LOCAL=1` already
gave a per-job root; correctness must not depend on an env var being switched on,
so the fix went in the sbatch.)

**The damage is not uniform across cells, which kills the cancellation argument.**

| cell | concurrent | serial | flips |
|---|---|---|---|
| B (arm, nostub) | 6/100 | 14/100 | 9 gained, 1 lost |
| F (base, nostub) | 6/100 | 24/100 | 18 gained, **0 lost** |

F lost more than twice what B lost, and lost nothing in the other direction. A
difference of differences cannot be assumed to subtract this out; Addendum 8's hope
that it would is retracted.

**Clean primary rung**, matched compute (attempt 1 only), matched prompt, paired
McNemar on n=100:

| version | B | F | net | b/c | SE | σ |
|---|---|---|---|---|---|---|
| contaminated a1 | 6 | 6 | +0 | 6/6 | 3.46 | 0.00 |
| **clean a1x** | **14** | **24** | **+10** | 17/7 | 4.90 | **2.04** |
| aggregate (unequal compute, contaminated) | 8 | 27 | +19 | 22/3 | 5.00 | 3.80 |

+10 at 2.04σ clears the registered `2·SE` gate and lands in the pre-registered
band `2·SE … +19`, whose reading is **"the gap is real and its aggregated form is
partly inflated."** Both halves stand.

**The aggregates are contaminated too, and this is the part that reaches beyond the
ladder.** The identity gate now fails on the *aggregate* side: B's aggregate reports
8 resolved while B's clean attempt-1 subset — a strict subset of the same rollouts —
reports 14, with only 3 instances ever retried. A subset cannot exceed its superset
by 11 unless the superset's scoring is damaged. Every number scored concurrently
before `d748808` is therefore a **floor** depressed by an unknown, cell-dependent
amount: the aggregate ladder board, the arm totals, `init` = 145/500, soup 50 /
pooled55k 46 / pooled220k 54, and the aggregated F−B = +19. Repairing them means
re-scoring under the fix — materially more CPU than the a1 subsets, so it is a scope
decision rather than something to quietly start.

What does *not* change: no model changed, and the direction of the campaign's
headline (SFT lift null-to-negative against a properly measured base) is unaffected —
the clean rung states it more defensibly than the contaminated one did.

## Addendum 12 — my own reasoning census was reading the wrong file, and the fix exposes a coverage asymmetry

Written 2026-08-07, correcting Finding 3 of `parity_ladder_report.md`. Found by taking
seriously a provenance warning agent-b17cac1e raised about a *different* artifact.

### The defect

`analysis/reasoning_census.py` globbed `output.jsonl`. That is the harness's **append
log**: it is rewritten only at the end of a run, and an instance whose attempt
terminated in an error is written to `output_errors.jsonl` instead, so its attempt-1
row is absent from the append log entirely. Two consequences, both measured:

| cell | instances in `output.jsonl` | instances in `output.critic_attempt_1.jsonl` |
|---|---|---|
| A, B, C, D, F | 100 | 100 |
| E | 78 rows / **68** distinct | **100** |
| G | **90** | **100** |

1. **Missing instances.** The census silently dropped 32 of E's and 10 of G's, and
   pairing across cells therefore collapsed to **62** instances — a number that appeared
   in the report as if it were a property of the experiment rather than of my glob.
2. **Wrong attempt.** Where a later attempt did land, the first surviving row is
   attempt 2, at a different temperature. On the 68 instances E has in both files the
   append log averages **291.6** ActionEvents against attempt 1's **181.1**, and the two
   files disagree on the action count for 22 of 68 (E), 44 of 90 (G) and 71 of 100 (F).
   Every per-instance rate in the old Finding 3 was computed over that mixture.

Fixed: read `output.critic_attempt_1.jsonl`, the single-attempt complete file that
**every rung of the ladder was already scored from**, with a fallback and a printed
provenance line. No ladder number is affected — the ladder never read the append log.

### The fix exposed a second, larger problem: row presence is not transcript presence

All seven cells have 100 attempt-1 *rows*, but where attempt 1 crashed the harness
writes a stub row — `instance_id`, an `error` string, **no history at all**:

| cell | rows | with a transcript | with a patch | resolved | resolved / patched | resolves with **no** transcript |
|---|---|---|---|---|---|---|
| A | 100 | 100 | 95 | 16 | 16.8% | 0 |
| B | 100 | 100 | 97 | 14 | 14.4% | 0 |
| C | 100 | 100 | 96 | 16 | 16.7% | 0 |
| D | 100 | 100 | 98 | 16 | 16.3% | 0 |
| E | 100 | **46** | 74 | 28 | **37.8%** | **11** |
| F | 100 | 100 | 82 | 24 | 29.3% | 0 |
| G | 100 | **65** | 64 | 20 | 31.2% | **5** |

Three readings, in order of how much they cost:

- **The ladder is unaffected and its direction is if anything conservative.** All 100
  instances were scored in every cell, so the paired McNemar rungs stand as computed. E
  reaches 28/100 having lost 54% of its attempt-1 conversations to harness crashes; had
  those run, E−A (+12) and E−F (+4) would plausibly be larger, not smaller. Likewise G
  had *more* usable coverage than E (65 vs 46) and still scored lower.
- **The loss is strongly condition-correlated** — E 54, G 35, everything else 0 — so an
  *unpaired* behavioural rate across cells compares instance mixes as much as
  conditions. `cell_instances()` now pairs on transcripts, which puts the honest
  cross-cell census at **n = 33**, not 62 and not 100.
- **A crashed conversation is not an empty one.** The harness recovers
  `test_result.git_patch`, and **11 of E's 28 resolves and 5 of G's 20 come from
  instances with no transcript**. So any behavioural account of E's score explains at
  most 17 of its 28 resolves. This also means a "completion rate" denominator has to be
  stated: resolved/patched is 37.8 / 29.3 / 31.2% for E / F / G, a much narrower spread
  than the cap-based 43 / 24 / 24% quoted earlier in the report, which rests on cap
  counts whose provenance I have not audited.

### Generalise

Same failure family as the three already recorded here: **a missing input read as a
zero rather than as missing.** An absent transcript read as "this cell did not reason";
an absent row read as "this instance does not exist". The lesson I actually take is
narrower and about me: agent-b17cac1e's warning named `output.jsonl` explicitly, I
verified it against the *scoring* path, found the scoring path clean, and stopped —
without checking my own analysis scripts against the same warning.
