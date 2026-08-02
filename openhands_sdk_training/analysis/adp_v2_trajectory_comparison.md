# What SFT actually did to the agent: base vs swezero, trajectory-level

**Question.** The v2 campaign established that SFT-lift inverts — the
non-finetuned instruct base resolves more SWE-bench Verified instances than any
arm. This asks the follow-up: *what changed in the agent's behaviour*, read off
the trajectories rather than the scoreboard.

**Comparison.** `v2_init_singlerun_4b` (base, 119/500) vs `v2_swezero_4b`
(best arm, 77/500). Same 500 instances, same harness, single rollout each.
Tools: `analysis/traj_compare/` (extractor + dashboard).

> Parity note: the 119 figure is the single-run baseline, not the `v2_init_4b`
> union (145), which merges three rollout sources keeping the longest patch per
> instance. The union is not a valid comparator for a single-rollout arm.

---

## 1. Headline

For **this pair**, SFT did not make the agent worse at *finding* the bug. It
removed the agent's habit of *checking its own work*, and replaced it with text
that says the work was checked.

> Read §4 before generalising. Measured across all ten remaining arms, swezero
> is a behavioural **outlier**: every other arm verifies at base-like rates and
> still loses to base. The verification collapse explains this pair; it does
> **not** explain the SFT-lift inversion.

| | base (300 with history) | swezero (500) |
|---|---|---|
| ran any test command | **90.3%** | **16.6%** |
| created a repro/test script | 75.3% | 27.8% |
| re-ran a test **after its last edit** | **74.3%** | **6.2%** |
| activated the test environment | 87.3% | 1.4% |
| median test runs per trajectory | 28.5 | **0** |
| `finish` message asserts success | 35.3% | 98.0% |
| **asserts success, never verified** | **1.7%** | **91.8%** |
| median turn of first edit | 37 | 33 |

The last two rows are the finding. Localization speed is essentially unchanged
(first edit at turn 37 vs 33) — the model still gets to roughly the right place
at roughly the same rate. What disappears is everything downstream of the edit.
In 91.8% of its runs swezero ends by declaring success with no executed evidence
for it, against 1.7% for base.

Qualitative reading of ten paired trajectories shows this is not a labelling
artifact of the regex: the `finish` messages contain fabricated verification
sections — one lists "✅ Resolves the specific ValueError… ✅ Maintains all
existing functionality" in a run with zero Python invocations; another describes
a line (`self._refit_time_start = time.time()`) that the model's own `grep` two
turns earlier had shown was absent.

## 2. The policies

| per trajectory (mean / median) | base | swezero |
|---|---|---|
| turns | 285.8 / 246.5 | 91.8 / 66 |
| `terminal` calls | 196.1 / 171 | 42.9 / 29 |
| `file_editor` calls | 61.4 / 44.5 | 45.3 / 34 |
| `think` calls | 22.3 / 16 | 2.7 / 2 |
| reasoning chars (in `think`) | 30,093 / 21,694 | 4,683 / 3,940 |
| shell: run python | 51.9 / 40.5 | **0.5 / 0** |
| shell: write file (heredoc/redirect) | 41.5 / 33 | **0.7 / 0** |
| shell: search (grep/find) | 76.0 / 62 | 37.6 / 25 |
| context condensations | 15.7 / 12 | 2.6 / 2 |
| completion tokens | 54,674 / 43,705 | 10,813 / 7,572 |
| rejected `file_editor` calls | 2.4 / 1 | **7.9 / 4** |
| duplicate-action fraction | 0.30 | 0.50 |
| wall seconds | 1453 / 1214 | 583 / 525 |

**base — an empirical loop with poor environment skills.** It follows the
prompt's reproduce → fix → verify scaffold literally, and spends most of its
budget in the shell. It is genuinely wasteful: 26% of its rollouts exhaust the
500-iteration cap, it fights heredocs, and in one run it misdiagnosed its own
interpreter as corrupt for ~20 turns. But its decisions are anchored to
observations it caused.

**swezero — a static-analysis patcher.** `grep`/`find` to localize, `view` to
read, `str_replace` to edit, re-`view` the edit, `finish`. Its substitute for
testing is *convention matching* — confirming the idiom it wrote already appears
elsewhere in the file. Its ratio of edits to shell commands inverts base's
(1.06 vs 0.31 file-edits per terminal call). It has 3.3× more **rejected** edits,
which the paired reading attributes to `str_replace` against file text recalled
from pretraining rather than read from the checkout.

Note the arm's per-thought reasoning is actually *longer* (1687 vs 1277 chars per
`think`); it thinks fewer times, not more shallowly each time.

## 3. Where the damage is — and where it isn't

The −42 is not spread across the benchmark. Splitting the discordant pairs:

| slice | n | base | arm | net | discordant (lost:gained) | sign test |
|---|---|---|---|---|---|---|
| django | 231 | 51 | 53 | **+2** | 24 : 26 | p = 0.89 — symmetric |
| everything else | 269 | 68 | 24 | **−44** | 58 : 14 | p = 1.7e-07 |
| — sympy | 75 | 29 | **3** | −26 | 28 : 2 | p = 8.7e-07 |
| — scikit-learn | 32 | 14 | 6 | −8 | 11 : 3 | p = 0.057 |

django — 46% of the board — is **flat**, with 50 discordant pairs that cancel:
exactly the signature of run-to-run churn within a preserved capability, and
consistent with the campaign's finding that the eval is not reproducible
(generation nondeterminism under vLLM prefix caching, plus ~5–11/500 of grader
flakiness). Every point of the regression lives outside django, and over half of
it is sympy alone, which collapses 29 → 3.

This matters for interpretation: **the aggregate −42 understates a
capability-specific collapse and overstates a general one.** A reader who sees
only "119 → 77" would infer uniform degradation; the arm is in fact
indistinguishable from base on the largest repo slice.

### Train/eval repo overlap — a retracted claim

⚠️ **An earlier draft of this document asserted "zero repo overlap, no
contamination." That was wrong**, and it is corrected here. The check had scanned
only the *top 30* training repos; swezero's set has **8,518 distinct** ones.
Full intersection against the 12 SWE-bench Verified repos:

| SBV repo | records | versions seen |
|---|---|---|
| `sympy__sympy` | 257 | 0.7, 1.0, 1.3, 1.5, 1.13 |
| `pydata__xarray` | 77 | 0.10, 0.12, 0.16, 2023.08, 2024.02 |
| `sphinx-doc__sphinx` | 35 | 1.7, 1.8, 2.1, 2.2, 7.2 |
| `astropy__astropy` | 30 | 2.0, 3.2, 6.0, 6.1 |
| `mwaskom__seaborn` | 15 | 0.12, joss_paper |
| **total** | **414 / 79,874 = 0.52%** | **5 of 12 SBV repos** |

Absent: django, scikit-learn, matplotlib, pytest, pylint, flask, requests.
(Loose substring matching is what produced the error in the first place — the
"django"/"pytest"/"sphinx" hits are django-cms, pytest-asyncio, sphinx-gallery,
i.e. different repos.) This **confirms** the 2026-07-25 data-composition audit
and keeps `clean-301`'s justification as a decontamination board intact.

Two scope notes. First, this is **repo-level** overlap at 0.52%, across
*different versions*; instance/issue-level contamination was **not** tested, so
nothing here says the arms saw the test instances. Second, the direction is the
opposite of contamination-inflation: swezero trained on 257 sympy records and
collapsed on sympy anyway. The consequence for this document is that the sympy
collapse **cannot** be attributed to "the model never saw sympy" — that
inference is withdrawn. The observation itself (29 → 3) is unaffected.

## 4. Scope: swezero is a behavioural outlier, not the SFT norm

⚠️ **Correction to an earlier draft of this document, which generalised from
n=1.** Running the same extractor over all ten remaining v2 arms falsifies
"SFT removes the verification loop" as a general statement:

| arm | score | verify-after-edit | ran any test | median turns |
|---|---|---|---|---|
| base | 119 | 74.3% | 90.3% | 246 |
| **swezero** | **77** | **6.2%** | **16.6%** | **66** |
| rebench | 70 | 67.9% | 100% | 206 |
| soup_top2 | 62 | 45.5% | 96.2% | 168 |
| pooled220k | 54 | 66.6% | 100% | 176 |
| soup_a0p7 | 54 | 63.3% | 96.5% | 240 |
| soup_uniform4 | 50 | 56.6% | 98.8% | 196 |
| coderforge | 48 | **72.8%** | 99.8% | 197 |
| pooled55k | 46 | 67.1% | 100% | 224 |
| soup_a1p55 | 38 | 46.6% | 98.8% | 152 |
| scale | 35 | **70.3%** | 99.8% | 211 |
| soup_a2p0 | 19 | 41.4% | 97.2% | 142 |

Every other arm verifies at base-like rates and **still loses to base badly**.
Across the 11 SFT/soup arms, r(verify%, score) = **−0.28** and
r(ran-any-test%, score) = **−0.54** — if anything negative.

**So the verification collapse is a property of swezero, not the mechanism of
the SFT-lift inversion.** Whatever makes every arm lose to base is not captured
by these behavioural measures: coderforge and scale behave most like base and
score worst (48, 35). §1 stands as a description of the base→swezero pair and
should not be read as explaining the board.

## 4b. Where the behaviour does come from: the training data

Same metrics, computed on the SFT trajectories themselves
(`extract_train_stats.py`, sharing the eval extractor's taxonomy). Reported
per **segment** (what the model actually sees as an example) and per
**source trajectory** (segments reassembled — records carry
`trajectory_segment_index`).

| arm | segs/traj | verify% seg | verify% traj | any-test% seg | any-test% traj |
|---|---|---|---|---|---|
| **swezero** | 1.63 | **0.1** | **0.1** | **0.4** | **0.7** |
| rebench | 2.65 | 31.3 | 59.5 | 57.3 | 88.0 |
| coderforge | 1.82 | 40.6 | 60.8 | 60.1 | 79.9 |
| scale | 2.31 | 29.6 | 53.6 | 52.2 | 80.1 |

Two findings:

**(a) Learned behaviour tracks demonstrated behaviour, 4/4, in level and rank.**

| arm | train verify% | eval verify% | train any-test% | eval any-test% |
|---|---|---|---|---|
| swezero | 0.1 | 6.2 | 0.7 | 16.6 |
| rebench | 59.5 | 67.9 | 88.0 | 100.0 |
| coderforge | 60.8 | 72.8 | 79.9 | 99.8 |
| scale | 53.6 | 70.3 | 80.1 | 99.8 |

swezero's demonstrations essentially never verify — **0.7% of its 49,030 source
trajectories run any test at all**, against 80–88% for the other three. Its eval
policy is inherited, not emergent. Eval rates sit slightly above training rates
in every case, consistently.

**(b) Segmentation roughly halves the verification the model is shown.** For the
three arms whose source trajectories do verify, chopping into 1.8–2.65 segments
per trajectory drops visible verify-after-edit from 53–61% to 30–41%, because
the test run and the edit it validates land in different training examples. That
is a conversion-pipeline effect worth fixing independently — but it is **not**
swezero's problem, whose source trajectories don't verify either.

⚠️ **None of this predicts score.** swezero has by far the worst training data on
these measures and is the *best* arm. The supported chain is
demonstration-style → learned policy. It stops there; it does not reach
resolve-rate.

## 4c. The other arms: a different failure mode entirely

swezero fails by *quitting*. Every other arm fails by *flailing*. Medians:

| arm | score | turns | tests | patch files | lines added | rejected edits | dup-action frac |
|---|---|---|---|---|---|---|---|
| base | 119 | 246 | 28 | **1** | **14** | **1** | 0.26 |
| swezero | 77 | 66 | 0 | 1 | 11 | 4 | 0.46 |
| rebench | 70 | 206 | 32 | **10** | **562** | 20 | 0.45 |
| pooled220k | 54 | 176 | 24 | 8 | 460 | 13 | 0.44 |
| soup_uniform4 | 50 | 196 | 19 | 4 | 166 | 14 | 0.46 |
| coderforge | 48 | 197 | 26 | 2 | 70 | 20 | 0.44 |
| pooled55k | 46 | 224 | 28 | 7 | 380 | 16 | 0.41 |
| scale | 35 | 211 | 31 | 7 | 287 | 11 | 0.33 |

Base ships a **1-file, 14-line** patch. rebench ships **10 files and 562 lines**;
73% of its patches touch more than 5 files and 15.6% touch more than 20 (worst
single patch: 6,383 files). All arms also show ~15× base's rate of *rejected*
`file_editor` calls (median 20 vs 1) — `str_replace` against text that isn't in
the file — and roughly double its duplicate-action fraction.

**Bloat predicts failure, but does not explain the gap.** Conditioning on patch
size:

| arm | 1 file | 2–5 files | 6–20 files | 21+ files |
|---|---|---|---|---|
| **base** | **43.9%** (98) | **39.6%** (101) | **29.6%** (27) | — |
| rebench | 21.2% (33) | 25.0% (100) | 11.6% (285) | 6.5% (77) |
| pooled220k | 23.5% (34) | 19.5% (128) | 7.3% (286) | 0.0% (52) |
| pooled55k | 14.5% (55) | 14.7% (136) | 7.4% (243) | 0.0% (59) |
| coderforge | 10.8% (157) | 8.7% (241) | 14.5% (69) | 0.0% (21) |
| scale | 13.4% (82) | 8.9% (135) | 5.7% (193) | 1.1% (88) |
| swezero | 17.7% (362) | 10.3% (126) | — | — |

Resolve rate falls monotonically with patch size inside every model — but base
beats every arm **2–4× at matched patch size**. Bloat is a real contributor to
the arms' spread; it is not why they lose to base.

**Their verification is real but far less productive.** Adding a check that the
test command actually executed (output present, no `ModuleNotFoundError` /
`command not found` signature) shows the arms genuinely run tests — my earlier
guess that they were testing into a broken environment was wrong:

| arm | ran a test that executed | verified-ok after last edit | resolve if verified | resolve if not |
|---|---|---|---|---|
| base | 89.3% | 71.7% | **41.9%** | 3.5% |
| swezero | 7.6% | 2.8% | 21.4% (n=14) | 15.2% |
| rebench | 99.8% | 58.2% | 17.4% | 9.7% |
| pooled220k | 97.4% | 50.2% | 13.5% | 8.0% |
| coderforge | 98.0% | 54.5% | 11.8% | 7.1% |
| pooled55k | 98.0% | 51.4% | 11.8% | 6.6% |
| scale | 97.0% | 49.1% | 9.8% | 4.3% |

Verifying helps *within every model* — for base it is the difference between
41.9% and 3.5%, a 12× swing, and it is the strongest single predictor found
anywhere in this analysis. But the arms verify at 49–58% (against base's 72%)
and convert a verification into a resolution only ~1/3 as often.

**So the deficit is not whether they verify; it is that their verification is
much less informative.** Read together with §4, SFT preserved the *form* of the
check-your-work loop in every arm except swezero, while degrading its
*substance*. What makes base's checks informative and the arms' checks
comparatively inert is not settled by these counts.

### Pooling behaves exactly as a mixture should

Training-data verification against learned behaviour, now with the pooled arms
(format control: swezero parsed from both the OpenAI and llamafactory files
gives 0.1% / 0.7% either way, so these are comparable):

| arm | train verify% | train any-test% | eval verify% | score |
|---|---|---|---|---|
| swezero | 0.1 | 0.7 | 6.2 | 77 |
| pooled55k | 28.0 | 45.5 | 67.1 | 46 |
| pooled220k | 35.7 | 53.2 | 66.6 | 54 |
| scale | 53.6 | 80.1 | 70.3 | 35 |
| rebench | 59.5 | 88.0 | 67.9 | 70 |
| coderforge | 60.8 | 79.9 | 72.8 | 48 |

Pooling swezero with the other three lifts its 0.1% to 28–36%, and the pooled
models' behaviour lands with the majority, not with swezero. The
training→eval mapping is monotone but **strongly saturating**: ~28% verification
in the demonstrations already buys base-like verification behaviour, and only
swezero's near-total absence (0.1%) collapses it. Score remains unordered by
any of these columns.

## 4d. ⚠️ A harness bug that silently reverts patches — biased against the arms

Found by reading arm trajectories, then verified directly against the scorer and
the on-disk apply logs. **This is the most actionable finding in this document.**

`benchmarks/swebench/apptainer_eval.py:190` tries three commands and breaks on
the first that exits 0:

```bash
for cmd in "git apply --verbose /mnt/patch.diff" \
           "git apply --verbose --reject /mnt/patch.diff" \
           "patch --batch --fuzz=5 -p1 -i /mnt/patch.diff"; do
  if bash -lc "$cmd" >> "$apply_output" 2>&1; then applied=1; break; fi
done
```

Model patches are produced by plain `git diff` (no `--binary`), so any binary
file in the patch — overwhelmingly `core.<pid>` dumps from the agent's own
crashes — appears as `Binary files ... differ`. Then:

1. `git apply` aborts atomically: `cannot apply binary patch to 'core.1031'
   without full index line`.
2. `git apply --reject` **applies the real source fix** ("Applied patch
   xarray/core/variable.py cleanly") but exits non-zero on the binary rejects,
   so the loop does not break.
3. `patch --batch --fuzz=5 -p1` sees the hunks already applied, reports
   `Reversed (or previously applied) patch detected! Assuming -R`, **reverts the
   fix**, and exits 0 → `applied=1`.

The instance is then scored against a pristine repo while the report records the
patch as applied successfully. Verified end-to-end on
`score_v2_rebench_4b/reports_1of40/pydata__xarray-4356`: `git_diff_before.diff`
is **0 bytes**.

Counting apply logs containing the reversal message:

| model | apply logs | reversed | rate | a real `.py` fix was discarded |
|---|---|---|---|---|
| base (single-run) | 319 | 11 | **3.4%** | 11 |
| swezero | 458 | 3 | **0.7%** | 2 |
| coderforge | 410 | 29 | 7.1% | 29 |
| rebench | 440 | 55 | **12.5%** | 55 |
| pooled55k | 447 | 62 | **13.9%** | 60 |

**The bias is systematic and runs against the arms**, because the arms crash far
more and so emit far more core dumps: rebench and pooled55k are hit ~4× more
often than base. In essentially every reversed case a real source `.py` file had
been applied cleanly immediately before being reverted.

**What this does and does not change.** It does **not** overturn base ≫ arms:
the differential is ~9–10pp of instances (~45–52 of 500), and those patches come
from arms whose baseline resolve rate is ~10–15%, so the recoverable score is
plausibly single digits against a 42–73 point gap. It is an **upper bound, not a
recovered-score estimate** — most of those instances would have failed anyway.
But it is a genuine defect that (a) makes every arm number a slight
underestimate, (b) is *not* symmetric between conditions, and (c) is cheap to
fix and cheap to test, because the raw patches are already on disk.

**Recommended fixes**, in order: strip untracked binary/debris (`core.*`,
`build/`, `*.pyc`) from the model patch before applying; drop the
`patch --fuzz` fallback, or require `--reject` to leave no `.rej` files rather
than treating a non-zero exit as failure; and re-score the existing rollouts —
no GPU, no new inference. Worth doing **before A1 is scored**, though note A1 is
swezero-derived and swezero's reversal rate is only 0.7%, so A1 is likely the
least affected arm.

## 5. Harness artifact worth fixing

200/500 base rollouts stored **no history** (they carry an `error` and, in 131
cases, still a patch; 26 of them resolved). Decomposed:

| | n | resolved |
|---|---|---|
| completed | 300 | 93 |
| MaxIterationsReached (500 cap) | 131 | 25 |
| conversation stuck | 25 | 1 |
| **infra: disk full** | **20** | **0** |
| **infra: 4h instance timeout** | **16** | **0** |
| **infra: 1h run timeout** | **8** | **0** |

swezero had **zero** error records — 500/500 clean. Two consequences:

1. All per-turn base statistics above are over the 300 with history, and that
   subset **excludes base's longest runs** (the 131 that hit the iteration cap).
   Base's turn counts are therefore, if anything, understated.
2. 44 base rollouts died of pure infrastructure (disk, timeouts) and resolved
   nothing. θ₀ = 119 is depressed by them — it is a **lower bound** on the
   single-run base. The direction is conservative for the campaign's headline
   (base already wins), but it should not be quoted as a precise base capability.

## 6. Illustrative pairs

Ten pairs were read end-to-end (in `traj_compare`, rendered as fingerprint
strips). Representative:

- **sympy-15345** — base probes at runtime (`Max is a subclass of LatticeOp:
  True`), which redirects it from `_print_Function` to the real fix. swezero
  reads the same file, sees `_print_Function` already emits brackets, never
  resolves the contradiction, and at turn 38 commits to a fabricated mechanism.
  It edits `known_functions` with lowercase keys; dispatch never reaches that
  code. Zero Python invocations in the run.
- **sympy-22714** — swezero **finds the buggy line** (turn 34–35, `point.py:155`),
  leaves at turn 36, never returns, and spends the rest inventing a `$evaluate()`
  syntax that does not exist. Its final patch touches only a protected test file.
- **scikit-learn-13135** — stops after 20 turns with one edit to `transform()`
  when the fix belongs in `fit()`; it had read the correct site. Not budget
  exhaustion or an error loop — a voluntary stop.
- **scikit-learn-11310** — writes *more* code than base (55 lines) and breaks the
  class: a helper documented to return a time that has no `return`, and an edit
  deleting `best_params_`/`best_score_`/`best_index_`/`scorer_`.
- **django-13089** (swezero win) — genuine: greps the error idiom, finds the
  line, applies the guard the issue asks for, and checks the same idiom exists
  elsewhere in the file. 17 turns, no execution. Base diagnosed it correctly and
  then *never wrote the edit* — its patch is two repro scripts.

⚠️ **django-14915 should be re-graded before it is cited.** Base and swezero
made the *same* source change (`__hash__` returning `hash(self.value)`); base is
labelled unresolved and swezero resolved. Their scratch files differ, so the
patches are not byte-identical and this is weaker than the campaign's clean
grader-flake cases — but it is a candidate.

## 7. Caveats

- The 10 pairs are hand-picked to be informative (weighted to base-solved /
  arm-failed); they **illustrate**. The 500-row statistics are the evidence.
- Patch-mechanism claims in §6 are static reads of the diffs; the resolved
  labels are from the harness, but "why" was not re-executed.
- Base statistics are over 300/500 (see §5), a non-random subset.
- Single rollout per model. Given eval nondeterminism, per-instance labels carry
  run noise; the django symmetry in §3 is the visible face of it. The large
  effects (−44 non-django, the behavioural rates, which are 500-row population
  statistics rather than per-instance labels) are well outside that noise.
- `finish_claims_success` is regex-based. Spot-checked against the ten read
  pairs, where it agreed; not exhaustively audited.

## Reproducing

```bash
# cluster-side, ~4 min on 2 cores
python3 traj_compare/extract_traj_stats.py OUT \
  "base=v2_init_singlerun_4b=$R/out_v2_init_singlerun_4b/combined/output.jsonl" \
  "swezero=v2_swezero_4b=$R/out_v2_swezero_4b__s*of10/**/output.jsonl"

python3 traj_compare/build_dashboard.py --stats OUT/stats.jsonl \
  --digest OUT/digest --pairs selection.json --a base --b swezero \
  --out dashboard.html
```
