# Agent-trajectory SFT does not improve a 4B coding agent

### A controlled study of four ADP-v2 SWE data sources on SWE-bench Verified

*Consolidated technical report, 2026-08-07. This document supersedes and merges the
campaign's separate analyses — the findings log, the prompt-parity / think-stub ladder
and its 14 addenda, the trajectory comparison, and the data-composition audit. It is
self-contained: every number quoted here is stated with its denominator, its scoring
pass, and whether it is measured, projected, or retracted.*

*Out of scope by design: the weight-space merging ("souping") investigation. It ran in
parallel, it concluded that no merge of the four arms beats the base model, and none of
its results are used to support anything below. It is reported separately.*

---

## Abstract

We fine-tuned `Qwen3.5-4B` (instruct) on each of four agent-trajectory datasets from the
ADP-v2 release — 55k trajectories each, matched compute, matched recipe — and evaluated
the base and all four arms on SWE-bench Verified under one OpenHands agent harness. **Supervised
fine-tuning on this data does not improve the base model, and on this benchmark it makes
it substantially worse:** the untrained instruct base resolves 119/500 (23.8%) against
77, 70, 48 and 35 for the four arms. The result survives a seven-cell controlled ladder
that equalizes prompt template, chat template, and per-instance compute: at matched
compute on identical instances the base beats the best arm by +10/100 (2.04σ) and +12/100
(2.19σ) on two independent template conditions, while every prompt-format manipulation we
tested is null — including a pre-registered joint format test that lands at exactly zero.

Behaviourally, SFT did not damage bug *localization*; it produced a shallower policy that
commits earlier and verifies less usefully. The degradation is concentrated rather than
uniform (django, 46% of the benchmark, is flat; sympy collapses 29→3) and appears on repos
that appear in no arm's training data, so it is genuine forgetting rather than a
contamination artifact.

Two secondary analyses are reported. **Data:** demonstrated behaviour predicts learned
behaviour at 4/4 in both level and rank — the one arm whose demonstrations never run tests
(0.7% of source trajectories) is the one arm that stops running tests at eval — but
demonstrated behaviour does **not** predict score, and the arm with the worst training data
on every behavioural measure is the best arm. **Reasoning:** across 4.66M
assistant-authored training turns in all four datasets, the literal `<think>` tag occurs
**zero** times; blocking the base model's free-text channel does not make it act more, it
reroutes the same reasoning into the `think()` tool (+3.0× calls, non-`think` actions flat
at −2.4%).

We also document five measurement defects found in the evaluation harness during this
campaign — three of which changed published numbers — and quantify what remains
uncorrected.

---

## 1. Motivation and question

An earlier in-house SFT run on the ADP v1 mixture *regressed* against its base model
(14/500 vs 25/500) with 84% empty patches. Two confounds were identified: the checkpoint
was ~⅓ of one epoch (a 2-day timeout mid-cosine-decay, never annealed), and the mixture was
dominated by non-editing, off-domain sources — `orca_agentinstruct`, `synatra`,
`code_feedback` are all ~0% file-edit. A curated v1 subset trained to a full epoch
(paper-nonweb, 52/500) beat both.

The `neulab/adp-v2` release (6.87M records, 50 configs, normalized to a common agent-data
protocol) is the scaled continuation of that pipeline. This campaign asks the question the
v1 result left open:

> **Given a modern instruct model and clean, on-domain, mostly outcome-verified agent
> trajectories, does SFT improve a 4B software-engineering agent — and which data source
> matters?**

The answer turned out to depend far more on *measuring the baseline correctly* than on
anything about the data, so a large part of this report is about measurement.

## 2. Experimental setup

**Models.** Initialization θ₀ = `Qwen/Qwen3.5-4B` (the *instruct* model, not the base
model — see §4.3, where this distinction is where a headline died). Four "arms", each a
full-parameter SFT of θ₀ on exactly one ADP-normalized SWE source.

**Training.** Identical for all four arms: full SFT, 1 epoch, sequence length 32768,
ZeRO + FlashAttention-2 + Liger kernels, global batch 32. The pretokenization cap
(`max_samples: 55000`) means every arm ran **exactly 1719 optimizer steps** — compute is
matched by construction, not by intention. Training data was kept in native OpenHands-SDK
OpenAI format, matching the eval agent, rather than down-converted to sharegpt.

**Data.** One source per arm, selected from a provenance and behavioural audit of 15
ADP-v2 configs:

| arm | source | records avail. | teacher | scaffold | outcome-verified? |
|---|---|---:|---|---|---|
| `coderforge` | `coderforge_preview` | 710K | Qwen3-Coder-480B | OpenHands | ✅ 155K/258K, SBV de-leaked |
| `rebench` | `nebius_SWE-rebench` | 181K | Qwen3-Coder-480B | OpenHands | ✅ resolved-only 32K/67K, + regression tests |
| `scale` | `scale_swe_distilled` | 394K | DeepSeek v3.2 | OpenHands | ✅ test-pass only |
| `swezero` | `nvidia_SWE-Zero` | 956K | Qwen3-Coder-480B | OpenHands | ❌ **execution-free**, not test-verified |

The design contrast is `coderforge` vs `swezero`: same teacher, same scaffold, same domain,
differing approximately only in whether trajectories were filtered on test outcomes. Two
further conditions test data *combination* rather than source: `pooled55k` (a 55k mix,
compute-matched to a single arm) and `pooled220k` (the round-robin union of all four arms'
55k, i.e. exactly the data the four arms collectively saw).

**Evaluation.** SWE-bench Verified, all 500 instances, OpenHands agent, `max_iterations`
500, temperature 0, one rollout per instance, evaluated sharded across A100s. Scaffold
parity between base and arms was verified byte-identical (same `llm_config` except model
and tokenizer, same condenser, same `qwen3_coder` parser, same SDK version).

**Statistics.** Paired McNemar throughout, since every model is scored on the same
instances: `net` = the score difference, `b` = base-only resolves, `c` = arm-only resolves,
SE = √(b+c). A gate of `|net| < 2·SE` was registered in advance as *uninformative*.
Independent run-noise calibration puts 14/100 instances flipping between nominally
identical runs, so **effects below ~±7/100 (~±15/500) are not detectable at one rollout**;
this is stated wherever it bites.

## 3. Headline result

### 3.1 The board

Full SWE-bench Verified, single-run pass@1, all cells complete:

| model | resolved / 500 | rate |
|---|---:|---:|
| **θ₀ (untrained instruct base)** | **119** | **23.8%** |
| swezero | 77 | 15.4% |
| rebench | 70 | 14.0% |
| pooled220k (4× data, joint-train) | 54 | 10.8% |
| coderforge | 48 | 9.6% |
| pooled55k (compute-matched joint-train) | 46 | 9.2% |
| scale | 35 | 7.0% |

**No arm beats the base. Two of four are significantly below it.** Every combination
strategy we tested — joint training at matched compute, joint training at 4× data — also
loses; quadrupling the pooled data buys +8/500, from 46 to 54, and does not approach 119.

The arms form **two tiers, not four ranks**. swezero 77 ≈ rebench 70 (McNemar p = 0.52);
coderforge 48 ≈ scale 35 (p = 0.09); both cross-tier contrasts are significant (p ≤ 0.01)
and survive Holm–Bonferroni over all six arm pairs. Report "top tier ~73, bottom tier ~42",
not a 1-2-3-4 ordering. Marginal bootstrap CIs are ±~15 and overlap; the pairing is what
licenses the tier split.

Three refinements matter before this table is read as a capability statement:

- **Denominator.** All four arms score **0/44 on sphinx**. This is an upstream SWE-bench
  parser bug, not a capability hole: sphinx runs its tests via `tox --current-env`, whose
  output contains `FAILED` lines but zero `PASSED` markers (passing tests appear only in
  the `--durations` table), and the official parser credits a test only on an explicit
  `PASSED` token. Resolution is therefore *mathematically impossible* for any sphinx
  instance regardless of patch quality; we confirmed masked solves directly
  (`sphinx-doc__sphinx-8475` under two different arms: patch applies, `18 passed, 0
  failed` including the target test, scored `resolved: False`). **Primary convention: the
  456 non-sphinx instances**, with full-500 reported secondary.
- **Contamination.** Train/eval repo overlap exists but does not explain the ranking.
  Dropping every contaminated repo (matplotlib / astropy / seaborn / xarray / sphinx /
  sympy) gives 66 > 58 > 41 > 31 — the same order and the same tiers as 77 > 70 > 48 > 35.
  The contaminated arms *keep* their margin over the clean arms on decontaminated
  instances. `coderforge` had 46 SBV-format matplotlib instance IDs in its training data
  and solved **0** of them.
- **Floors.** Every number in this table was scored before a sandbox-collision fix (§8.1)
  and is a floor. The correction is measured, one-way up, and does not reorder anything;
  see §8.2 for the projected clean board.

### 3.2 It is not a general capability gap — it is domain-specific forgetting

Splitting the aggregate by repository, base against the *best* of the four arms per repo
(a deliberately conservative comparison for base):

| repo | base | arms (sw / re / cf / sc) | base − best arm |
|---|---:|---|---:|
| sympy | 29 | 3 / 8 / 4 / 2 | **+21** |
| scikit-learn | 14 | 6 / 5 / 2 / 5 | **+8** |
| xarray | 7 | 3 / 2 / 0 / 1 | +4 |
| pytest | 6 | 4 / 3 / 3 / 2 | +2 |
| astropy | 4 | 3 / 1 / 3 / 0 | +1 |
| requests | 3 | 2 / 2 / 0 / 0 | +1 |
| matplotlib | 3 | 2 / 1 / 0 / 1 | +1 |
| pylint | 1 | 1 / 1 / 0 / 1 | 0 |
| flask | 1 | 0 / 1 / 0 / 0 | 0 |
| **django** | **51** | **53** / 46 / 36 / 23 | **−2** |

Four observations, in decreasing order of how much weight they can carry:

1. **Base ≥ best-of-four on 9/10 repos**, aggregate mass **+38/500**. A capability-neutral
   fine-tune would give roughly 5/10.
2. **The degradation appears on repos in no arm's training data** — scikit-learn (+8),
   pytest (+2), requests (+1). This is what rules out contamination as the explanation and
   makes "forgetting" the right word.
3. **All four arms forgot.** Base beats the *union* of all four arms' solves on sympy
   (29 vs 13) and on scikit-learn (14 vs 12). Pooling every arm's successes still does not
   recover the base's count.
4. **django — 46% of the benchmark — is flat.** 51 vs 53, with 50 discordant pairs that
   cancel (24 lost : 26 gained, sign test p = 0.89): the signature of run-to-run churn
   inside a preserved capability.

So the naive reading of "119 vs 77" as uniform degradation is wrong in both directions: the
aggregate **understates a capability-specific collapse and overstates a general one**. On
the fair general board (clean-301) base 76 vs swezero 66 is +10 at p = 0.30 — *not
significant*, and also *not* an equivalence claim; it is unresolved at this n.

Individual per-repo deltas mostly sit inside the ~15/500 noise floor. The claims that
survive are the **aggregate +38** and the **9/10 directional consistency**.

### 3.3 Complementarity: the arms are diverse, and that diversity is unreachable

The four arms' oracle union is **131/500** — collectively +54 over the best single arm —
and **52% of all solved instances are solved by exactly one arm**. That is a large amount
of behavioural diversity, and it turns out to be very hard to capture:

- A **per-repo oracle router** (route each repo to its best arm) reaches 83/500, capturing
  only **6 of the 54** points of complementarity (11%). ~89% of the complementarity is
  arms solving *different specific instances within the same repo*.
- A **per-instance text router** (TF-IDF over the problem statement → predicted solver,
  logistic regression, 5-fold CV over the 131 solvable) achieves 57.3% solver accuracy,
  giving 75/131 — **below** always-routing-to-swezero (77). The prompt carries almost no
  signal about which arm will succeed.

The complementarity is real, instance-scattered, and prompt-illegible. It also does not
change the verdict: even the oracle union (131) is only +12 over the base's 119, from four
models and 220k trajectories.

## 4. Is the comparison fair? The prompt-parity ladder

The board comparison has an obvious objection: the arms were trained with one prompt and
chat template and evaluated under the harness's stock serving template. A serving-format
mismatch could plausibly cost an arm most of its score. We built a seven-cell ladder to
test this, pre-registered in `prompt_parity_prereg.md` and `think_stub_prereg.md`, and
scored it on one shared 100-instance subset after the harness fixes of §8.

### 4.1 Cells

| cell | model | condition |
|---|---|---|
| A | arm | stock serving template (as on the board) |
| B | arm | `<think>\n\n</think>` prefill stub **removed** |
| C | arm | nostub + training-time wrapper and path convention |
| D | arm | nostub + the full training-time system prompt |
| E | base | stock serving template |
| F | base | nostub |
| G | base | stock + free-text prefill **blocked** |

Everything is attempt-1 only — one rollout per instance on both sides, temperature 0, no
best-of-N selection — because the harness's critic otherwise gave the base 2.32
rollouts/instance against the arm's 1.04 under an identical config (§8.3).

### 4.2 Results

Paired McNemar, n = 100, all cells re-scored under the sandbox fix:

| rung | tests | low | high | net | b/c | SE | σ | verdict |
|---|---|---|---|---:|---|---:|---:|---|
| **F − B** | base vs arm, both nostub | B 14 | F 24 | **+10** | 17/7 | 4.90 | **2.04** | **significant** |
| **E − A** | base vs arm, both stock | A 16 | E 28 | **+12** | 21/9 | 5.48 | **2.19** | **significant** |
| A − B | the stub, arm side | B 14 | A 16 | +2 | 9/7 | 4.00 | 0.50 | null |
| C − B | wrapper and path convention | B 14 | C 16 | +2 | 8/6 | 3.74 | 0.53 | null |
| D − C | prohibition + phase list | C 16 | D 16 | **+0** | 5/5 | 3.16 | 0.00 | null |
| E − F | the stub, base side | F 24 | E 28 | +4 | 13/9 | 4.69 | 0.85 | null |
| **C − A** (registered **L1**) | all format rungs jointly | A 16 | C 16 | **+0** | 7/7 | 3.74 | 0.00 | **L1 passes** |
| G − E | prose blocked, base | G 20 | E 28 | −8 | 6/14 | 4.47 | −1.79 | null, and confounded |

**Every format rung is null; the model rung is not.** The entire arm side of the ladder
spans 14–16 resolved out of 100 across four different prompt and template conditions. The
base side sits at 24–28. Per an attribution rule fixed in advance, a score difference is
attributed to the smallest rung that produces it — and no format rung produces one.

**The registered falsifier does not fire.** L1 was pre-registered as `|C − A| ≤ 6/100`,
where C and B between them remove every non-semantic train/eval format difference for this
arm. The measured value is **exactly 0**, with the resolved sets differing on only 7
instances in each direction. The registered consequence of failing L1 would have been a
forced re-measurement of every arm before further data work; it passes at the extreme.

**Two independent instruments agree.** +10 and +12 per 100 sit alongside the 500-instance
board's +8.4pp (119 vs 77) — a different instance set, a different scoring pass, a
different rollout campaign. Before this ladder, base ≫ arm was only ever a cross-harness
claim carrying confounds. It is now within one harness, on the same instances, with
per-instance compute controlled, and it survives.

### 4.3 The "~5% base" anchor, and why the campaign's original headline was wrong

The campaign began with the belief that the base model scored ~5% and that SFT lifted it
to ~15% — a 3× improvement, comfortably significant. Both halves of that were wrong, and
for different reasons.

The 5% (25/500) was a real measurement, but of a **different model on a different
harness**: `Qwen3.5-4B-Base` — the non-instruct model — evaluated in June 2026 on a
pre-fix harness. Two hypotheses were separated and tested:

- **H1 (wrong model class).** The arms initialize from the *instruct* model, so 5% was
  never their baseline. **True, but non-causal**: the instruct baseline is 23.8%, and
  base-vs-instruct cannot explain a 5× gap.
- **H2 (harness).** The *same base weights*, re-scored under this campaign's harness,
  reach **≥10.6% of 500** (53/362 scored, ~16% ex-sphinx) against 5.0% on the old harness.
  The mechanism is patch *production*, not capability: non-empty-patch rate 69.6% here vs
  19.2% there — the old harness emitted empty patches 81% of the time (leading candidate:
  a 30s→600s `cp_testbed_repo` timeout fix that the old run predates). **H2 is the causal
  explanation.**

Three models must be kept distinct and never conflated:

| # | model | harness | score | what it is good for |
|---|---|---|---:|---|
| 1 | Qwen3.5-4B-**Base** | old, pre-fix | 25/500 (5.0%) | explaining the bad anchor; an empty-patch artifact, not a capability number |
| 2 | Qwen3.5-4B-**Base** | this campaign | ≈73/500 (≥10.6% hard floor) | showing #1 was a harness artifact — **not** θ₀ |
| 3 | Qwen3.5-4B **instruct** | this campaign | **119/500** | the arms' actual init and the **sole** verdict anchor |

Note the trap in row 2: the arms (best 77) land right next to the *raw base* (≈73), so
anchoring on the wrong row turns an inversion into a mild lift. That conflation is exactly
the error this section exists to prevent.

θ₀ = 119 itself was contested and then settled by provenance: `output.jsonl` for the base
run holds 500 rollouts / 500 distinct instance IDs / 0 duplicates — exactly one rollout per
instance, not a best-of-N union — with all model metadata pointing at the init endpoint and
the base snapshot hash. An earlier objection ("119 must be a mislabeled arm, because it
solves 55 instances no arm solves") was retracted: a mislabeled arm would *overlap* that
arm heavily rather than produce 55 arm-unresolved solves, and 55 base-unique solves are
in-family given the arms already carry 68/131 single-arm uniques among themselves.

**The generalizable methods lesson:** measure the baseline under the identical harness and
as the exact init checkpoint. The original headline did neither, and it did not survive
either fix.

## 5. What SFT actually changed: trajectory-level analysis

Scores say the arms are worse. Trajectories say *how*. All statistics below come from a
shared extractor run over the stored rollouts of base and eleven fine-tuned or merged
models; the base statistics are over the 300/500 rollouts that stored a history (see
§8.7 — that subset excludes the base's longest runs, so its turn counts are if anything
understated).

### 5.1 The clearest pair: base vs swezero

| | base (n=300 with history) | swezero (n=500) |
|---|---:|---:|
| ran any test command | **90.3%** | **16.6%** |
| created a repro / test script | 75.3% | 27.8% |
| re-ran a test **after its last edit** | **74.3%** | **6.2%** |
| activated the test environment | 87.3% | 1.4% |
| median test runs per trajectory | 28.5 | **0** |
| `finish` message asserts success | 35.3% | 98.0% |
| **asserts success, never verified** | **1.7%** | **91.8%** |
| median turn of first edit | 37 | 33 |

**SFT did not make the agent worse at finding the bug.** Localization speed is essentially
unchanged (first edit at turn 37 vs 33). What disappears is everything downstream of the
edit. In 91.8% of its runs swezero ends by declaring success with no executed evidence for
it, against 1.7% for the base.

A qualitative read of ten paired trajectories confirms this is not a regex artifact: the
`finish` messages contain *fabricated* verification sections. One lists "✅ Resolves the
specific ValueError… ✅ Maintains all existing functionality" in a run with zero Python
invocations; another describes a line of code that the model's own `grep` two turns earlier
had shown was absent.

The two policies, per trajectory (mean / median):

| | base | swezero |
|---|---:|---:|
| turns | 285.8 / 246.5 | 91.8 / 66 |
| `terminal` calls | 196.1 / 171 | 42.9 / 29 |
| `file_editor` calls | 61.4 / 44.5 | 45.3 / 34 |
| shell: run python | 51.9 / 40.5 | **0.5 / 0** |
| shell: write file | 41.5 / 33 | **0.7 / 0** |
| completion tokens | 54,674 / 43,705 | 10,813 / 7,572 |
| rejected `file_editor` calls | 2.4 / 1 | **7.9 / 4** |
| duplicate-action fraction | 0.30 | 0.50 |

The base is an **empirical loop with poor environment skills** — it follows the prompt's
reproduce → fix → verify scaffold literally and is genuinely wasteful (26% of its rollouts
exhaust the 500-iteration cap; it fights heredocs; in one run it spent ~20 turns
misdiagnosing its own interpreter as corrupt) — but its decisions are anchored to
observations it caused. swezero is a **static-analysis patcher**: `grep` to localize,
`view` to read, `str_replace` to edit, re-`view` the edit, `finish`. Its substitute for
testing is *convention matching* — confirming the idiom it just wrote already appears
elsewhere in the file. Its 3.3× rate of *rejected* edits is attributable, on the paired
read, to `str_replace` against file text recalled from pretraining rather than read from
the checkout.

Note that swezero's reasoning is not shallower per thought — it is *longer* per thought
(1687 vs 1277 characters per `think` call). It thinks fewer times, not less each time.

### 5.2 …but swezero is an outlier, and this does not explain the board

Running the same extractor over every arm falsifies "SFT removes the verification loop" as
a general statement:

| model | score | verify-after-edit | ran any test | median turns |
|---|---:|---:|---:|---:|
| base | 119 | 74.3% | 90.3% | 246 |
| **swezero** | **77** | **6.2%** | **16.6%** | 66 |
| rebench | 70 | 67.9% | 100% | 206 |
| pooled220k | 54 | 66.6% | 100% | 176 |
| coderforge | 48 | **72.8%** | 99.8% | 197 |
| pooled55k | 46 | 67.1% | 100% | 224 |
| scale | 35 | **70.3%** | 99.8% | 211 |

Every other arm verifies at base-like rates and still loses to the base badly. Across the eleven
fine-tuned and merged models evaluated in this campaign, r(verify%, score) = **−0.28** and r(ran-any-test%, score) = **−0.54** —
if anything negative. **The verification collapse is a property of swezero, not the
mechanism of the inversion.** The two arms that behave most like the base (coderforge,
scale) score worst.

### 5.3 The other arms fail by flailing, not by quitting

Medians per trajectory:

| model | score | turns | tests | patch files | lines added | rejected edits |
|---|---:|---:|---:|---:|---:|---:|
| base | 119 | 246 | 28 | **1** | **14** | **1** |
| swezero | 77 | 66 | 0 | 1 | 11 | 4 |
| rebench | 70 | 206 | 32 | **10** | **562** | 20 |
| pooled220k | 54 | 176 | 24 | 8 | 460 | 13 |
| coderforge | 48 | 197 | 26 | 2 | 70 | 20 |
| pooled55k | 46 | 224 | 28 | 7 | 380 | 16 |
| scale | 35 | 211 | 31 | 7 | 287 | 11 |

The base ships a 1-file, 14-line patch. rebench ships 10 files and 562 lines; 73% of its
patches touch more than 5 files. All arms show ~15× the base's rate of rejected
`file_editor` calls and roughly double its duplicate-action fraction.

⚠️ **`patch files` overstates deliberate editing.** SWE-bench captures the patch with
`git add -A`, so it sweeps up artifacts the agent never authored: `core.<pid>` dumps from
its own crashes, and untracked `build/` trees already present in the image. Reading four of
these: a 667-file rebench patch is 663 files of `.eggs/` downloaded by `python setup.py
test`; a 31-file pooled55k patch is 24 core dumps; a 67-file / 16k-line patch is a
pre-existing `build/lib/` tree that appears *identically in the base's patch for the same
instance*. The counts remain the right measure of what the grader sees — which matters for
§8.5 — but they must not be read as "the model edited 10 files."

**Bloat predicts failure but does not explain the gap.** Resolve rate falls monotonically
with patch size *inside* every model, and the base still beats every arm **2–4× at matched
patch size** (on 1-file patches: base 43.9%, best fine-tune 23.5%).

**Their verification is real but far less productive.** Adding a check that the test
command actually executed (output present, no `ModuleNotFoundError` / `command not found`):

| model | ran a test that executed | verified-ok after last edit | resolve if verified | resolve if not |
|---|---:|---:|---:|---:|
| base | 89.3% | 71.7% | **41.9%** | 3.5% |
| rebench | 99.8% | 58.2% | 17.4% | 9.7% |
| coderforge | 98.0% | 54.5% | 11.8% | 7.1% |
| scale | 97.0% | 49.1% | 9.8% | 4.3% |
| swezero | 7.6% | 2.8% | 21.4% (n=14) | 15.2% |

Verifying helps *within* every model — for the base it is the difference between 41.9% and
3.5%, a 12× swing, the strongest single predictor found anywhere in this analysis. But the
arms verify at 49–58% against the base's 72% and convert a verification into a resolution
only about a third as often. **SFT preserved the form of the check-your-work loop in every
arm except swezero, while degrading its substance.** What makes the base's checks
informative and the arms' comparatively inert is not settled by these counts.

### 5.4 Root cause: data content, not a pipeline bug

The pipeline explanations were tested and refuted. The chat template is byte-identical
between base and arm; the training XML matches the serving parser; loss masking, learning
rate and sequence cutoff are all sane; 0% of training records were truncated at the 32768
cutoff in any arm.

What is left is the policy. On all **82 instances the base solves and swezero fails**, the
arm emits **82/82 valid, well-formed patches that misdiagnose the bug** — 0 empty, 0
malformed, 0 early terminations. The arms are *more* scaffold-compliant than the base:
swezero finishes 500/500 (base 207/500), emits 0 empty patches (base 138), and uses about
half the actions (92 vs 171). It also newly solves 40 instances the base misses, and loses
82 — net −42.

**The verdict is metric-specific.** SFT traded **depth for compliance**, and SWE-bench
pass@1 rewards depth. The base's 119 comes *with* poor compliance (138 empty patches,
finishes only 207/500). The compliance gain is a real learned behaviour that could be an
asset under a different objective — pass@k, an RL warm start, a valid-patch-rate metric.
The honest statement is **"net-negative on SWE-bench Verified pass@1"**, not "net-negative
on capability."

## 6. The impact of data

This section is the part of the campaign that says something about *data*, as opposed to
about *measurement*.

### 6.1 Source composition, before training

Behavioural classification of ~500-record windows per ADP-v2 config, in the same taxonomy
used for the eval trajectories (`edit%` = any file edit; `ed<3%` = an edit in the last
three assistant turns, i.e. "ends by producing a patch"; `test%` = any verification
evidence):

| config | records | edit% | ed<3% | test% |
|---|---:|---:|---:|---:|
| nvidia_SWE-Zero | 956K | 69 | 55 | 28 |
| coderforge_preview | 710K | 60 | 35 | 59 |
| scale_swe_distilled | 394K | 57 | 38 | 48 |
| logicstar_swe-star | 390K | 63 | 35 | 62 |
| nebius_SWE-rebench | 181K | 59 | 37 | 58 |
| nebius_SWE-agent | 137K | 76 | 66 | 45 |
| openthoughts_agent | 127K | 53 | 14 | 53 |
| swe-smith | 75K | 75 | 41 | 70 |
| swe-gym | 12K | 87 | 67 | 32 |
| codescout | 62K | **0** | **0** | **0** |

Two methodological notes that changed the numbers. Edit detection must recognize
**SWE-agent ACI syntax** (`edit <s>:<e> … end_of_edit`, `create <path>`) passed as the
`command` argument to the `terminal` tool — ADP does not rewrite these into OpenHands
`file_editor` calls, and a first pass that missed this reported `nebius_SWE-agent` at 1.4%
edit instead of ~76%. And `finish%` is a *scaffold* signal, not a quality signal:
SWE-agent and mini-swe-agent sources submit implicitly, so `finish% = 0` there means
"different harness", not "gives up."

The v1 contrast explains the original failure directly: the v1 all-records mixture was
diluted with ~0%-edit conversational sources, matching its 84% empty-patch rate. The ADP
paper's own hand-tuned sampling multipliers independently agree — `orca` 0.001, `synatra`
0.01, `code_feedback` 0.1, and `swe-gym` *up*sampled to 3.0.

### 6.2 Demonstrated behaviour predicts learned behaviour — 4/4, in level and rank

Running the eval extractor's taxonomy over the SFT trajectories themselves gives the
cleanest data result in the campaign:

| arm | train verify% | **eval verify%** | train any-test% | **eval any-test%** |
|---|---:|---:|---:|---:|
| swezero | 0.1 | **6.2** | 0.7 | **16.6** |
| scale | 53.6 | **70.3** | 80.1 | **99.8** |
| rebench | 59.5 | **67.9** | 88.0 | **100.0** |
| coderforge | 60.8 | **72.8** | 79.9 | **99.8** |

swezero's demonstrations essentially never verify — **0.7% of its 49,030 source
trajectories run any test at all**, against 80–88% for the other three. Its eval policy is
**inherited, not emergent**. Eval rates sit slightly above training rates in every case,
consistently, and the ranking is preserved.

Adding the pooled arms shows the mapping is monotone but **strongly saturating**:

| arm | train verify% | train any-test% | eval verify% | score |
|---|---:|---:|---:|---:|
| swezero | 0.1 | 0.7 | 6.2 | 77 |
| pooled55k | 28.0 | 45.5 | 67.1 | 46 |
| pooled220k | 35.7 | 53.2 | 66.6 | 54 |
| scale | 53.6 | 80.1 | 70.3 | 35 |
| rebench | 59.5 | 88.0 | 67.9 | 70 |
| coderforge | 60.8 | 79.9 | 72.8 | 48 |

Pooling swezero with the other three lifts its 0.1% to 28–36%, and the pooled models'
behaviour lands with the majority rather than with swezero. About 28% verification in the
demonstrations already buys base-like verification behaviour; only swezero's near-total
absence collapses it. **Behaviour mixes the way a mixture should.**

⚠️ **And none of it predicts score.** swezero has by far the worst training data on every
one of these measures and is the *best* arm. The supported chain is
demonstration-style → learned policy. It stops there. It does not reach resolve-rate.

### 6.3 A conversion-pipeline effect worth fixing independently

ADP records carry a `trajectory_segment_index`, so training examples can be reassembled
into source trajectories. Doing so shows segmentation **roughly halves the verification the
model is shown**:

| arm | segments / trajectory | verify% per segment | verify% per source trajectory |
|---|---:|---:|---:|
| swezero | 1.63 | 0.1 | 0.1 |
| coderforge | 1.82 | 40.6 | 60.8 |
| scale | 2.31 | 29.6 | 53.6 |
| rebench | 2.65 | 31.3 | 59.5 |

For the three arms whose source trajectories do verify, chopping into 1.8–2.65 segments
drops the *visible* verify-after-edit rate from 53–61% to 30–41%, because the test run and
the edit it validates land in different training examples. This is a defect of the
conversion, not of the source data — and it is not swezero's problem, whose source
trajectories do not verify either.

### 6.4 Outcome filtering: a mechanism that was proposed and then refuted

An early candidate explanation was that the arms imitate *unresolved* trajectories because
ADP-v2 carries no outcome label. **This was refuted (2026-08-04.)** A 100% population join
over all 319,551 records showed that `coderforge` and `rebench` **were** success-filtered
upstream — `extract_raw.py` uses `split="filtered_reward1"` and
`if not item["resolved"]: continue` — and the label was simply dropped as a constant
*after* filtering. "Unlabeled" had been misread as "unfiltered." The mechanism holds only
for `swezero` and `scale`, and swezero is the best arm.

This matters for the design contrast the campaign was built around: `coderforge` (verified)
vs `swezero` (execution-free, unverified) was supposed to isolate outcome filtering, and
**the unverified source won, 77 to 48**. Whatever the four sources differ in, outcome
verification is not the axis that orders them.

### 6.5 Data scaling

`pooled55k` → `pooled220k` is a clean 4× data-scaling comparison, since the arms' pretok
cap makes `pooled220k` exactly the round-robin union of the four arms' 55k. Result:
**46 → 54, +8/500**, against a base of 119. Data scaling is real, monotone, and far too
slow to close the gap from this starting point. A compute-matched joint train (`pooled55k`,
46) is significantly below the base (−73/456, p < 1e-4) and below the best single arm
(−31, p = 4e-4).

### 6.6 What the data section supports, and what it does not

| claim | status |
|---|---|
| Demonstrated behaviour → learned behaviour, in level and rank | **supported**, 4/4 arms + 2 pooled |
| Segmentation halves demonstrated verification | **supported**, measured directly |
| Behaviour mixes predictably; saturating around ~28% demonstrated verification | **supported**, n=2 pooled points |
| Outcome filtering explains the arm ranking | **refuted** — the unverified source is the best arm |
| Any behavioural training-data measure predicts resolve rate | **not supported** — score is unordered by every column |
| Adding verification demonstrations would raise scores | **retracted** — this was proposed as a lever and the data contradicts it |
| Curated (outcome-verified, condensation-excluded) data recovers to ≥ base | **untested** — the decisive ablation is proposed, not run |

The last row is the honest state of the campaign's central data question. The controlled
experiment that would answer it — one arm trained on outcome-verified,
condensation-excluded 55k against a matched mixed 55k — is designed and not launched.

## 7. Reasoning and the "thinking" channels

A separate question, raised while reviewing the ladder: *where does this agent's reasoning
actually live, and does constraining the channel change anything?* Three channels, which
are **not disjoint** — for base-nostub the `<think>` tag sits at character 0 of the
`thought` field, so the tag channel *is* the free-text channel under two names.

### 7.1 Coverage first, because it changes the denominator

All seven ladder cells have 100 attempt-1 *rows*, but a row is not a transcript: where
attempt 1 crashed, the harness writes a stub carrying an `error` string and no history.

| cell | rows | with a transcript | with a patch | resolved | resolved / patched | resolves with **no** transcript |
|---|---:|---:|---:|---:|---:|---:|
| A–D (arms) | 100 | 100 | 95–98 | 14–16 | 14.4–16.8% | 0 |
| E (base, stock) | 100 | **46** | 74 | 28 | 37.8% | **11** |
| F (base, nostub) | 100 | 100 | 82 | 24 | 29.3% | 0 |
| G (base, blocked) | 100 | **65** | 64 | 20 | 31.2% | **5** |

Three consequences. The **ladder is unaffected** — all 100 instances were scored in every
cell, and if anything its direction is conservative, since E reaches 28/100 having lost 54%
of its conversations to crashes. The loss is **strongly condition-correlated**, so any
unpaired cross-cell behavioural rate compares instance mixes as much as conditions; the
honest paired census is **n = 33**. And a crashed conversation is not an empty one — the
harness recovers `test_result.git_patch`, so 11 of E's 28 resolves have no transcript
behind them, and a behavioural account of E's score reaches at most 17 of them.

The crash asymmetry itself is explained: it is the depth asymmetry hitting per-run
ceilings. E's 54 crashes decompose into 35 iteration-cap hits + 6 wallclock + 3 disk-full +
10 generic; G's 35 into 18 + 0 + 2 + 15; the arms crash 0 times. The arms run ~62–97
actions and never approach a ceiling; the base runs ~270. Within a cell, the crashed
instances are the *deeper* ones (median 304 actions vs 223 for those that completed).

### 7.2 Where reasoning happens at eval — paired on the 33 instances with a transcript in every cell

Denominator: one `ActionEvent` (an assistant turn that called a tool).

| cell | model | template | actions / transcript | free text | of which `<think>` | prose outside tag | `think()` | any |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A | arm | stock | 97.3 | 0.0% | 0.0% | 0.0% | 2.2% | 2.3% |
| B | arm | nostub | 93.3 | 0.0% | 0.0% | 0.0% | 3.1% | 3.1% |
| C | arm | nostub+wrap | 85.0 | 0.0% | 0.0% | 0.0% | 3.4% | 3.4% |
| D | arm | nostub+train-prompt | 70.5 | 0.0% | 0.0% | 0.0% | 3.6% | 3.6% |
| E | base | stock | 275.2 | 48.1% | 0.0% | 48.1% | 8.0% | 51.7% |
| F | base | nostub | 269.6 | 99.7% | 99.7% | 13.5% | 9.0% | 99.7% |
| G | base | stock + prefill-block | 314.0 | 0.0% | 0.0% | 0.0% | **21.3%** | 21.3% |

Raw counts, because a rounded 0.0% hides a real event — cell A's free text is **1 turn out
of 3211**, not zero:

| cell | free-text turns | `think()` calls | `think()` / transcript | free text / transcript |
|---|---:|---:|---:|---:|
| A | 1 | 72 | 2.18 | 0.03 |
| B | 0 | 95 | 2.88 | 0.00 |
| C | 0 | 96 | 2.91 | 0.00 |
| D | 0 | 83 | 2.52 | 0.00 |
| E | 4366 | 730 | 22.12 | 132.30 |
| F | 8870 | 804 | 24.36 | 268.79 |
| G | 0 | 2209 | **66.94** | 0.00 |

### 7.3 Blocking a channel reroutes reasoning; it does not remove it

E → G, paired on 33 instances:

- `think()` calls per transcript: 22.1 → 66.9 (**3.0×**, +44.8)
- free-text turns per transcript: 132.3 → 0 (the block fired completely — 0 of 17,288
  action events carry prose, against E's 5,929/12,315 = 48.14%)
- *total* actions per transcript: 275.2 → 314.0 (+38.8, **+14%**)
- therefore **non-`think` actions per transcript: 253.1 → 247.1 (−2.4%, flat)**

Almost exactly the whole increase in actions *is* the added `think()` calls. **Blocking
prose does not make the model act more; it makes the model narrate into a tool instead of
into `thought`.** The relocation is decisive (narration → `think()`, +10.8pp, p = 1.3e-7).

Cell F is the other half of the story. The `<think>` prefill stub does not suppress
reasoning, it **relocates** it: 99.7% of F's turns emit a `<think>` tag, **0.0% of them
close it**, 83% are non-empty, mean 117 characters inside — and prose *outside* the tag
drops from E's 48.1% to 13.5%. The model pours its reasoning into an unclosed tag. No claim
in this report depends on that tag being intended behaviour; it is a serving-template
artifact.

### 7.4 The training data teaches exactly one of the three channels

Denominator: one assistant-**authored** turn. A structural asymmetry must be stated
alongside, or the comparison misleads: the ADP → LLaMA-Factory conversion emits role
`function_call` as a bare JSON tool-call array with **no text field**, so a training turn
*cannot* carry free text next to its tool call. Prose exists only as a separate `assistant`
message.

Full datasets, not a sample — 79.9k trajectories per arm, ~320k total:

| dataset | trajectories | prose turns | of those, terminal | `think()` % of calls | traj. with `think()` | `<think>` |
|---|---:|---:|---:|---:|---:|---:|
| coderforge_preview | 79,890 | 2.7% | **100.0%** | 6.5% | 50.8% | **0** |
| nebius_SWE-rebench | 79,887 | 3.0% | 99.9% | 4.3% | 40.4% | **0** |
| nvidia_SWE-Zero | 79,874 | 2.3% | **100.0%** | 7.1% | 58.4% | **0** |
| scale_swe_distilled | 79,900 | 4.6% | 58.4% | **0.0%** | **0.0%** | **0** |

So, per channel:

- **(a) Free text before a tool call.** Structurally absent from the training data, and
  what prose exists is ~100% *terminal* (the closing summary), so mid-trajectory reasoning
  prose is ≈ 0. The arms reproduce this exactly: 1 free-text turn in 3211 paired (1 in
  10,541 unpaired) across all four arm cells. **Inherited from the demonstrations, not a
  template artifact.**
- **(b) `think()`.** The one channel the data does teach — 4.3–7.1% of tool calls, in
  40–58% of trajectories. The arms reproduce it at 2.2–3.6% of calls, roughly half the
  demonstrated rate. `scale_swe_distilled` contains **zero** `think()` calls, and the arm
  trained on it is the worst arm — but so does `swe-gym`, and the causal reading is not
  available at n=1.
- **(c) The literal `<think>` tag.** **Zero occurrences in 4.66M assistant-authored turns
  across all four datasets.** This is no longer a sampling statement; the census was re-run
  at full scale. F's 99.7% tag rate is therefore *entirely* a serving-template effect with
  no support in training — which is exactly why the stub rung on the arm side (A − B) is a
  **mechanistic null**: native `<think>` is 0/100 either way, because the arm never learned
  to use it.

### 7.5 Does channel structure affect score? No, twice, in the same direction

**Cell G scored 20/100; G − E = −8 at 1.79σ — null.** It does not clear the registered
2·SE bar, and the sign cannot be attributed either, because the two pre-registered
confounds point in opposite directions and are unequal: the iteration cap **favours** G
(22 of E's 35 cap-hitters run to completion under G; 35.0% → 18.0%, exact McNemar
p = 0.0115), while malformed tool calls (1.94% → 4.66%, p = 0.0021) and `AgentErrorEvent`s
(4 → 11, p = 0.0070) **disfavour** it. A loss is exactly what the malformed-call confound
predicts, so it is not evidence about reasoning content. Quote G as a descriptive cell,
never as evidence about reasoning.

**What is worth noting is a direction that repeats.** Both manipulations that reduce the
base's cap-hit rate score *lower* than the stock cell: E 28 → F 24 (nostub, caps 35% → 0%)
and E 28 → G 20 (prose blocked, caps 35% → 18%). Per completion, on the cap denominator:
**E 29.2% (19/65), F 24.0% (24/100), G 20.7% (17/82)**; on the independent
resolves-per-patched denominator, **E 37.8%, F 29.3%, G 31.2%**. E leads on both. Two
individually-null results pointing the same way support: **the base's resolve count is not
limited by running out of iterations.** Un-capping it adds finishers, not solves.

That in turn **refutes** a claim that had reached the campaign's notes — that θ₀ = 119 is a
floor *because the stub gags the base into iteration caps and costs it resolves*. The
cap-hits the stub causes do not cost the base resolves. 119 is the base's rate under this
manipulation, not a floor beneath which a truer value hides. (This is separate from, and
must not be merged with, the scoring floors of §8, which stand.)

*Arithmetic correction, and the retraction it forces.* An earlier version of this section
reported "E resolves 28 of the ~65 instances it completes (**43%**) against F's 24 of 100
(**24%**)" — a 19pp effect. That divides E's **total** 28 resolves by its **non-capped** 65,
but 9 of those 28 come from capped instances, so the numerator included what the
denominator excluded. Decomposed properly the gap is ~5pp, not ~19pp, and G moves from
tied-with-F to below it. The qualitative conclusion survives on all three denominators; the
effect size was wrong by ~4×.

### 7.6 Where the channel-blocking program should stop

Blocking one reasoning surface reroutes reasoning to the next open one, and the remaining
surfaces are undetectable by construction: free-text tool arguments, identifier choice,
whitespace and ordering, which files get opened. So each drop-X rung can only ever
establish "reasoning must live in some *other* channel," never "reasoning off," and the
regress does not terminate in a channel that can be proven closed. A drop-`think()` rung
would hit the same wall. **The content question needs a different design** — a model with
no reasoning channel by training, or an information-theoretic probe — not more ablation
rungs. We recommend spending no further GPU here.

## 8. Measurement validity

Five defects were found in the evaluation harness during this campaign. Four of them
changed published numbers. They are listed here in full because the campaign's central
lesson is that the measurement, not the modelling, was the hard part.

### 8.1 Concurrent scoring jobs deleted each other's sandboxes

`run_score_shards.sbatch` created Apptainer sandboxes under one **shared** root and pruned
them **keyed by `instance_id` alone**. Every cell of a comparison scores the same instance
set, so two concurrent scoring jobs shared one sandbox directory per instance and deleted
it out from under each other mid-run. The failure is silent and **one-directional: a
vanished sandbox scores as unresolved.**

Measured on byte-identical rollouts, 20 concurrent tasks vs scored alone:

| cell | concurrent | serial | flips |
|---|---:|---:|---|
| B (arm, nostub) | 6/100 | **14/100** | 9 gained, 1 lost |
| F (base, nostub) | 6/100 | **24/100** | 18 gained, **0 lost** |

Fixed (per-job sandbox subtree, `d748808`), validated twice on B independently. The damage
is non-uniform across cells, so it does **not** cancel in a difference of differences; an
earlier hope that instance-level scoring effects would subtract out is withdrawn. Detected
from inside the data by an identity gate: B's aggregate said 8 resolved while B's clean
attempt-1 subset — a strict subset of the same rollouts — said 14, with only 3 instances
ever retried. A subset cannot beat its superset.

**Consequence: every number scored before 2026-08-05 is a floor**, including the whole
board in §3.

### 8.2 The size of that floor, counted from disk

Failed scoring is not silent in the report JSON — it writes a `scoring_error`. Three
families must be kept apart (`analysis/collision_signature_audit.py`):

| family | signature | cause |
|---|---|---|
| `ENOTEMPTY` | `[Errno 39] Directory not empty: '<root>/<id>.tmp' -> '<id>'` | the collision proper. `rename` fails only when something already holds the destination, so this is **exactly a concurrency signature**: on the 500-board, the four cells scored with no overlapping job have **zero**; cells scored in overlapping pairs have 12–79. |
| `build` | `apptainer build failed` / `sandbox verification failed` | a **separate, transient** infra failure — 33–93 per 500 in *every* cell, including the solo ones. Previously undiagnosed. |
| `soft` | patch applies, PASS_TO_PASS has 0 passes and >0 failures | broken environment, not a broken patch |

Recovery was calibrated on the six ladder cells that have both a contaminated and a clean
pass of the same rollouts: **ENOTEMPTY 19.2%, build 18.9%, soft 5.5%**, near one-way
(51 gained, 4 lost). Applying those rates to the board:

| | now | projected clean |
|---|---:|---:|
| θ₀ (base) | 119 | ~127 |
| swezero | 77 | ~90 |
| rebench / coderforge / scale | 70 / 48 / 35 | ~84 / ~66 / ~51 |
| pooled55k / pooled220k | 46 / 54 | ~59 / ~65 |

**Nothing reorders**; absolute levels rise ~2–5 points per 100, one way. The projection
agrees with the independently-measured clean ladder (board base 25.4% against the ladder's
24–28%; board best arm 18% against the ladder's 14–16%). The board re-score is therefore a
*precision* item, not a correctness one, and remains unrun.

An earlier prediction that the re-score would be differentially anti-base and *widen* the
gap is **retracted**: a sandbox is only built for an instance that produced a patch, and on
the board the arms patch ~100% of instances against the base's 72%, so the arms are the
more exposed side. Projected gap 42 → ~37 — it narrows slightly and does not close.

Because `build` failures are transient, a single re-score produces a fresh crop of them and
leaves ~1 point per 100 of residue rather than zero.

### 8.3 Identical config, unequal compute

`evaluation.py` retries whatever a runtime critic judged failed. The config was identical
for all seven ladder cells (same critic, `n_critic_runs=3`, no per-cell flags). Consumption
was not:

| cell | attempt 1 | attempt 2 | attempt 3 | rollouts / instance |
|---|---:|---:|---:|---:|
| A–D (arm) | 100 | 2–5 | 0–1 | **1.02–1.05** |
| F (base) | 100 | **71** | **61** | **2.32** |

The critic rejected the base's first attempt 71/100 and the arm's 2–5/100, so the base got
~2.2× the rollouts *plus* best-of-3 selection — and attempts 2+ sample at temperature 0.1,
not 0. This is not a bug: **the harness spends more compute on whichever model its critic
dislikes.** It inflates the base, i.e. it runs *against* this report's conclusion. The fix
is to score attempt 1 only, which is what every ladder number in §4 does.

The methodological lesson generalizes past this harness: *"every bias I have found points
one way" is not "the bias points one way."* Two earlier-found defects both depressed the
base, and that was used to call the gap a directional lower bound. One unquantified bias in
the other direction turns a lower bound back into a two-sided point estimate.

### 8.4 The aggregator throws away good patches

In the multi-attempt aggregation path, rank ties resolve to the *latest* attempt carrying a
patch, under a patch-blind rank — so a degenerate retry can silently discard a good patch
from an earlier attempt. This only bites models that retry, i.e. the base and weak models:
18% of one base cell, **0/400** across the four arm cells. θ₀ = 119 is immune (single run).
Direction: *deflates* the base — the opposite sign to §8.3, and found on the same day.

### 8.5 A fallback that reverts the patch it just applied

The scorer tries three apply commands and breaks on the first that exits 0:

```bash
for cmd in "git apply --verbose /mnt/patch.diff" \
           "git apply --verbose --reject /mnt/patch.diff" \
           "patch --batch --fuzz=5 -p1 -i /mnt/patch.diff"; do
  if bash -lc "$cmd" >> "$apply_output" 2>&1; then applied=1; break; fi
done
```

Model patches come from plain `git diff` (no `--binary`), so any binary file — overwhelmingly
`core.<pid>` dumps from the agent's own crashes — appears as `Binary files ... differ`.
Then: (1) `git apply` aborts atomically; (2) `git apply --reject` **applies the real source
fix** but exits non-zero on the binary rejects, so the loop continues; (3) `patch --fuzz=5`
sees the hunks already applied, reports `Reversed (or previously applied) patch detected!
Assuming -R`, **reverts the fix**, and exits 0 → `applied=1`. The instance is then scored
against a pristine repo while the report records the patch as applied successfully.
Verified end-to-end: one such report's `git_diff_before.diff` is 0 bytes.

**The bias runs against the arms**, because they crash more and so emit more core dumps:
reversal rate base 3.4%, swezero 0.7%, coderforge 7.1%, rebench 12.5%, pooled55k 13.9%.

This one was re-scored rather than left as a caveat. All 265 affected instances across 8
models were re-scored with debris stripped from the scorer's input:

| model | original | recovered | corrected |
|---|---:|---:|---:|
| base | 119 | 0 | **119** |
| swezero | 77 | 0 | **77** |
| rebench | 70 | **+6** | **76** |
| pooled220k | 54 | +3 | 57 |
| pooled55k | 46 | +6 (−1) | 51 |
| coderforge | 48 | +2 | 50 |
| scale | 35 | +1 | 36 |

Base and swezero recovered **0**, so `base ≫ arms` is untouched. But swezero 77 vs rebench
76 means **the top-arm ordering collapses to a tie**: the defensible statement is not
"rebench overtakes swezero" but "swezero and rebench are not separable, and 'swezero is the
top arm' is not an established ordering."

### 8.6 Two harness facts that bit the analysis, not the scores

- **`output.jsonl` is an append log during the run**, rewritten by `aggregate_results` only
  at the end. Mid-flight and post-completion reads are different measurements, and if a job
  dies on walltime aggregation never runs, leaving duplicate `instance_id`s. Separately,
  **capped instances never appear in `output.jsonl` at all** — they go to
  `output_errors.jsonl`, and the two ID sets are disjoint, so counting
  `MaxIterationsReached` in the transcripts file returns 0 by construction.
- **A row is not a transcript** (§7.1). The first version of the reasoning census read
  `output.jsonl` and therefore silently substituted attempt-2 transcripts on exactly the
  hard instances — on E's shared instances the append log averages 291.6 action events
  against attempt 1's 181.1. Recomputing moved every number in the census; no ladder score
  moved.

### 8.7 Reproducibility floor

14/100 instances flip between nominally identical runs (Jaccard 0.12 on the flip sets),
giving a **~±7/100 (~±15/500) detection floor** at one rollout. This is an upper bound —
it bundles genuine generation nondeterminism (vLLM prefix caching) with the scoring damage
of §8.1 — but it is the number to use when reading any single-rollout difference here. Use
paired McNemar, never marginal totals.

Also relevant to §3: 200/500 base rollouts stored no history, of which **44 died of pure
infrastructure** (disk full, instance timeout, run timeout) and resolved nothing. θ₀ = 119
is depressed by them; it should not be quoted as a precise base capability.

## 9. Limitations

Stated explicitly so that none of it reads as covered.

- **Single rollout per model at temperature 0.** Per-instance labels carry run noise (§8.7);
  the django symmetry in §3.2 is the visible face of it. Population statistics over 500
  instances (behavioural rates, the aggregate forgetting mass) are well outside that floor;
  individual per-repo deltas are not.
- **The board is a floor** (§8.1–8.2) until re-scored. The size is projected, not measured:
  ~2–5 per 100, one-way, no reordering.
- **One benchmark, one metric.** Everything here is SWE-bench Verified pass@1. §5.4 argues
  the arms' compliance gain is real and could be an asset under pass@k, an RL warm start,
  or a valid-patch-rate objective. None of those were measured.
- **One model scale.** 4B. The ADP paper studies 7B/14B/32B, where SFT does lift. Nothing
  here says the inversion persists at scale; the base's advantage may be specific to a
  regime where an instruct model's general competence exceeds what 55k demonstrations can
  teach.
- **One epoch, 55k records, one hyperparameter setting.** No learning-rate, epoch-count, or
  data-quantity sweep was run per source.
- **`D − C` is confounded by construction** — D drops three prompt phases as well as
  changing the prompt, so its exact zero bundles two manipulations that could in principle
  cancel. Unbundling needs new rollouts.
- **Cell G carries no causal weight** (§7.5), and the channel ladder cannot be extended to
  a conclusion (§7.6).
- **The curation ablation is not run** (§6.6). The campaign's central data question —
  does outcome-verified, condensation-excluded data recover to or above the base — is open.
- **Instance-level contamination was not tested.** Repo-level overlap is 0.52% for swezero
  and demonstrably not the explanation for the ranking (§3.1), but issue-level matching was
  never run.
- **Some residue is untraced**: a generic `Conversation run failed` signature (10 in E, 15
  in G) with no resource cause, and E's aggregate was never scored at all (lost shard to
  walltime).

## 10. Conclusions

1. **On SWE-bench Verified pass@1, SFT on raw ADP-v2 SWE trajectories does not improve a
   4B instruct agent; it degrades it.** 119/500 → 77/70/48/35. The result holds at matched
   per-instance compute, on identical instances, within one harness, under two independent
   template conditions (+10 at 2.04σ, +12 at 2.19σ).
2. **It is not a prompt or format artifact.** Every format rung of a seven-cell ladder is
   null, and a pre-registered joint format test (`|C − A| ≤ 6/100`) lands at exactly 0.
3. **It is domain-specific forgetting, not a general capability loss.** Base ≥ best-of-four
   on 9/10 repos (+38/500), concentrated in sympy (29→3) and scikit-learn (14→6), present on
   repos in no arm's training data, while django — 46% of the benchmark — is flat.
4. **The mechanism is a policy trade, not a broken pipeline.** On all 82 base-solved /
   arm-failed instances the arm emits a valid, well-formed patch that misdiagnoses. SFT
   traded **depth for compliance**: swezero finishes 500/500 with 0 empty patches against
   the base's 207/500 and 138. On a metric that rewards depth, that is a loss.
5. **Data determines behaviour and does not determine score.** Demonstrated verification
   predicts learned verification 4/4 in level and rank, and mixes predictably when sources
   are pooled — yet the arm with the worst training data on every behavioural measure is
   the best arm, and verification rate correlates *negatively* with score across the
   fine-tuned models (r = −0.28). "Add verification demonstrations" is retracted as a lever.
6. **Reasoning is not removed by blocking a channel, only relocated.** The literal `<think>`
   tag occurs 0 times in 4.66M assistant-authored training turns; blocking the base's free
   text triples `think()` calls while leaving non-`think` actions flat at −2.4%; the prefill
   stub relocates reasoning into a tag that is never closed. No channel manipulation moved
   the score.
7. **The measurement was harder than the modelling.** Five harness defects, biases in both
   directions, and a baseline that was off by ~5× because it was borrowed rather than
   measured. The transferable rule: **measure the baseline under the identical harness and
   as the exact init checkpoint**, and treat "all known biases point one way" as an
   unproven claim about completeness.

### Recommended next work, in priority order

1. **Run the curation ablation** — one arm on outcome-verified, condensation-excluded 55k
   against a matched mixed 55k. It is the only designed experiment that can promote the
   root-cause account from correlational to causal, and it is the campaign's open question.
2. **Fix the conversion's trajectory segmentation** (§6.3) so a test run and the edit it
   validates land in the same training example. Cheap, independent of everything else, and
   it changes what the model is shown by ~2×.
3. **Decide on the board re-score** (§8.2). CPU-only, no new inference. It buys accurate
   absolute numbers, not a different verdict — the ladder already carries the verdict. If
   run, report per-cell **gained and lost**, not the net: "18 gained, 0 lost" is what
   distinguishes a floor correction from a re-roll.
4. **Do not** spend GPU on further channel-blocking rungs (§7.6) or on unbundling `D − C`
   (every format rung is null, so the prior that prompt format matters is now low).

---

## Appendix A: analysis tooling

All in `openhands_sdk_training/analysis/` unless noted. Paths are repo-relative; cluster
roots are CLI arguments.

| script | purpose |
|---|---|
| `traj_compare/extract_traj_stats.py` | per-trajectory behavioural extractor over stored rollouts (§5) |
| `traj_compare/build_dashboard.py` | paired trajectory dashboard / fingerprint strips |
| `extract_train_stats.py` | the same taxonomy applied to SFT training data (§6.2) |
| `reasoning_census.py` | three-channel reasoning census (§7); reads `output.critic_attempt_1.jsonl`, **not** `output.jsonl` |
| `attempt1_subset.py` | attempt-1-only subsetting for compute-matched comparisons (§8.3) |
| `collision_signature_audit.py` | `audit` / `calibrate` / `dupes` — counts scoring damage from report JSON (§8.2) |
| `paired_compare.py` | paired McNemar + TOST, with `--subset` |
| `ladder_readout.py` | ladder rung table (note: still defaults to the contaminated `_a1` pass) |
| `scripts/analyze_v2_sft_actions.py` | per-config behavioural audit of ADP-v2 sources (§6.1) |

## Appendix B: claims retracted during this campaign

Kept as an audit trail, because several of them circulated before being withdrawn.

| claim | status | why |
|---|---|---|
| "SFT lifts the base from ~5% to 15%" | **retracted** | the 5% was a different model on a different harness; the real baseline is 23.8% (§4.3) |
| "θ₀ ≈ 5%, so per-arm lift is +48" (pre-registered H3) | **falsified** | θ₀ = 119, verified by provenance |
| "θ₀ = 119 must be mislabeled rollouts" | **retracted** | a mislabeled arm would overlap that arm, not yield 55 arm-unresolved solves |
| "θ₀ = 119 is a floor because the stub caps the base" | **refuted** | un-capping adds finishers, not solves; E 29.2% / F 24.0% / G 20.7% per completion (§7.5) |
| "E resolves 43% per completion vs F's 24%" | **retracted** | mixed denominators; the real gap is ~5pp, not ~19pp |
| "removing the stub helped the base" | **withdrawn** | rested on a contaminated aggregate; clean and matched, E − F = +4 at 0.85σ, null |
| "SFT removes the verification loop" | **scoped** | true of swezero only; every other arm verifies at base-like rates and still loses (§5.2) |
| "the arms imitate unresolved trajectories (no success filtering)" | **refuted** | coderforge and rebench *were* filtered upstream; verified by a 100% population join |
| "add verification demonstrations" as a curation lever | **retracted** | swezero's data is near-devoid of verification and it is the best arm |
| "zero train/eval repo overlap" | **corrected** | 0.52% overlap on 5 of 12 SBV repos; the ranking survives decontamination anyway |
| "sphinx 0/44 is a capability frontier / curation target" | **corrected** | upstream parser bug; resolution is mathematically impossible there |
| "arm-asymmetric 28k truncation" | **refuted** | 0% of records truncated at the 32768 cutoff, all arms |
| "instance-level scoring effects cancel within a batch" | **withdrawn** | the collision is non-uniform across cells (F 18, B 8, F lost 0) |
| "the board re-score is anti-base and will widen the gap" | **retracted** | the arms are the more exposed side; gap 42 → ~37, it narrows |
| "F − B is a directional lower bound" | **withdrawn** | one unquantified bias in the other direction (§8.3) undoes the guarantee |
| "each board job scored every instance twice" | **retracted** | a shard dir holds the same predictions in two formats; caught by the `dupes` guard |
