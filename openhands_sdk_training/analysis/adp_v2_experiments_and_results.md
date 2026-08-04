# Fine-tuning Qwen3.5-4B on agentic SWE trajectories: experiments and results

**Summary.** We fine-tune a 4B instruction-tuned language model on four different agentic
software-engineering (SWE) trajectory datasets and evaluate on SWE-bench Verified. None of the four
resulting models, nor any weight-space combination of them, nor a jointly-trained model on their
pooled data, beats the *untrained* instruct model's own resolve rate. The apparent large "lift" from
fine-tuning reported in earlier analyses was an artifact of comparing against an incorrectly-measured
baseline (wrong checkpoint, wrong evaluation harness); once the baseline is measured correctly, under
the same harness and from the exact checkpoint the fine-tunes started from, fine-tuning shows no
significant general improvement and a systematic, measurable *loss* of capability on domains the
instruct model already handled well. A patch-level audit suggests fine-tuning shifts the model's behavior from
deep, exploratory problem-solving toward a faster, more scaffold-compliant but shallower policy — a
trade that this particular benchmark's single-attempt scoring penalizes. We also find that whether a
data source's trajectories were execution-verified (i.e., only kept if the fix actually passed tests)
was not, by itself, predictive of downstream benchmark performance: the *unverified* source produced
one of the two best fine-tunes, on par with a verified source and significantly ahead of the other two
verified sources (though this comparison is across only four, otherwise-uncontrolled sources — see
§1.1/§2.2 for the caveat).

---

## 1. Setup

### 1.1 Instruct model and training data

All fine-tuning below starts from **Qwen3.5-4B**, the instruction-tuned checkpoint — we call this the
**instruct model** throughout. This is a distinct checkpoint from **Qwen3.5-4B-Base**, the non-instruct
pretrained variant of the same model family, which shows up separately below only as a comparison
point (§2.1, §2.8) and never as a fine-tuning starting point in this study.

We fine-tune four separate models — one per training-data source — on trajectories drawn from four
distinct agentic-SWE datasets that a shared normalization pipeline had converted into a common
tool-calling trajectory format. We refer to each resulting model as an **arm**. The four sources,
their distillation teacher model, and whether their trajectories were execution-verified (kept only
if the agent's patch actually made the target tests pass) are:

| arm | data source | teacher model | execution-verified? | non-action/condensation turns |
|---|---|---|:--:|:--:|
| `coderforge` | `coderforge_preview` | Qwen3-Coder-480B | yes | 39% |
| `scale` | `scale_swe_distilled` | DeepSeek v3.2 | yes | — |
| `rebench` | a SWE-bench-style rebuild dataset | Qwen3-Coder-480B | yes (+ regression tests) | — |
| `swezero` | an execution-free distillation set | Qwen3-Coder-480B | **no** | 34% |

`coderforge` and `swezero` share the same teacher model and scaffold and are the source pair that
differs most directly in execution-verification status, but they are otherwise disjoint datasets
with different underlying problem sets and pipelines, and (as the table shows) somewhat different
proportions of non-action, condensation/summarization turns — so this pair is *not* a clean,
single-variable controlled ablation of verification status alone; treat the comparison in §2.2 as
suggestive of a real effect (n=4 sources, uncontrolled), not as an isolated causal test.

Each arm's training set is a fixed-size, deterministically sampled 55,000-record subset of its
source (reservoir-sampled from a larger pool with a fixed seed), converted to an OpenAI-style
tool-calling chat format matching the evaluation harness described in §1.3.

Two additional models are trained on **pooled** data — the union of the four arms' training
records — rather than on a single source, to compare weight-space combination (§2.5) against joint
training on combined data (§2.6):

| model | training data | records |
|---|---|---:|
| `pooled55k` | round-robin interleave of the four sources | 55,000 (compute-matched to one arm) |
| `pooled220k` | full round-robin union of what all four arms trained on | 220,000 (data-matched: the exact union) |

### 1.2 Training configuration

All six trained models (four single-source arms + two pooled models) use an identical recipe, varying
only in training data: full-parameter supervised fine-tuning, context length 32,768 tokens, one epoch,
global batch size 32, peak learning rate 1e-5 with cosine decay to zero and 3% linear warmup, bf16
precision, gradient checkpointing, fixed random seed. Training ran on 8×A100 (80GB) nodes per model.
A single arm's 55,000-record epoch takes 1,719 optimizer steps; `pooled220k`'s 220,000-record epoch
takes 6,875 steps. Checkpoints retain full optimizer and scheduler state so that any preemption/resume
continues the same learning-rate schedule rather than resetting it — an earlier, unreported run of
this recipe found that resets from partial checkpointing state can by themselves account for a large
fraction of the score differences between models, so this is treated as a load-bearing methodological
detail, not an implementation nicety.

### 1.3 Evaluation protocol

All resolve rates below are on **SWE-bench Verified** (500 human-vetted issue/patch pairs across 12
open-source Python repositories), using an OpenHands-based coding-agent scaffold: a default system
prompt, maximum 500 agent iterations, greedy decoding (temperature 0, i.e. a single deterministic
rollout per instance — not averaged over repeated samples), a maximum input context of 28,000 tokens
with a context-condenser triggered above that threshold, maximum 2,047 output tokens per turn, and
native tool-calling parsing matched to the model's chat template. Scoring uses the official SWE-bench
test harness (containerized, per-instance execution of the target test suite against the model's
patch).

**Two evaluation-methodology points are important for interpreting every result below:**

1. **A benchmark resolve rate is a property of the model *and* the evaluation harness together, not
   of the model alone.** In a controlled check (§2.1), we re-scored the identical (untrained) Qwen3.5-4B-Base checkpoint
   under two different harness configurations and observed resolve rates differing by roughly
   2–3× (see §2.1) — driven almost entirely by how often each configuration allowed the agent to
   produce *any* patch at all, not by any difference in patch quality. Absolute numbers should
   therefore only be compared across models evaluated under the *same* harness configuration;
   cross-configuration comparisons (§2.8) are flagged as such throughout.
2. **44 of the 500 instances (all from one repository) cannot be resolved under the standard scoring
   harness for a benchmark-specific reason**: that repository's test suite does not emit the
   pass/fail marker the harness's parser looks for, so every instance from it scores as failed
   regardless of patch quality, for every model. This affects all models identically, so it does not
   bias any *relative* comparison, but it deflates every absolute resolve rate reported on the
   full 500-instance board by a model-dependent (but bounded, small) amount. Since this repository
   contributes zero resolved instances for every model in this study, raw resolved counts are
   identical whether expressed "out of 500" or "out of 456" (only the percentage denominator differs);
   we report raw counts out of 500 throughout for readability, and compute paired significance tests
   (McNemar/TOST) on the 456-instance board that excludes this repository unless otherwise noted.

**Statistical treatment.** With single-sample (non-repeated) rollouts at n≈500 or n≈300, the smallest
difference between two models that is reliably distinguishable from sampling noise is roughly ±15
resolved instances. We report paired McNemar tests (which use the per-instance agreement/disagreement
pattern between two models, and are substantially more powerful than comparing marginal totals) for
head-to-head comparisons, and treat a non-significant paired test as "not resolved at this sample
size," not as evidence of equality.

**A note on training/evaluation overlap.** Some training sources contain a modest number of records
whose formatting coincidentally matches SWE-bench-Verified-style instance identifiers, creating
partial train/eval overlap for some (not all) of the four data sources, concentrated in six of the
benchmark's repositories. A decontaminated re-ranking that excludes every instance from any repository
with any such overlap for any arm preserves both the ranking and the approximate size of the gaps
between arms reported below (§2.2), so the comparisons in this report do not appear to be contamination
artifacts. This check is at repository granularity, not exact instance-level matching.

---

## 2. Results

### 2.1 The importance of a correctly-measured baseline

Before fine-tuning, we measure the untrained instruct model's own resolve rate, under the *identical*
evaluation harness used for every fine-tuned model, with a single deterministic rollout per instance
(matching the fine-tuned models' evaluation protocol exactly, so the comparison is apples-to-apples).

**The untrained instruct model resolves 119/500 (23.8%; 26.1% on the 456-instance board).** This is the correct comparison point for every result below, since it is the exact
checkpoint every fine-tuned arm started from, evaluated the same way.

This number is substantially higher than an earlier, informal reference point of "≈5%" that had been
used as a sanity anchor for the instruct model. Investigating the discrepancy traced it to two compounding
issues, both instructive beyond this particular study:

- **Harness sensitivity.** The ≈5% figure came from evaluating Qwen3.5-4B-Base (the non-instruct
  variant of the same model family) under an earlier, different harness configuration. Re-evaluating
  those exact same model weights under the harness configuration used in this report (identical model,
  different harness) recovered a resolve rate of *at least* 10.6% of the full board — more than double
  the earlier figure, and a hard floor rather than a point estimate: that re-evaluation run did not
  successfully score every instance (infrastructure errors left some unscored), and treating every
  unscored instance as unresolved is what gives 10.6%; the resolve rate among the instances that did
  score cleanly was somewhat higher, though that subset may not be representative, since harder
  instances plausibly fail to score more often. The gap was concentrated almost entirely in how often
  the agent produced *any* patch at all (roughly 70% of episodes ended with a non-empty patch under the
  harness used here, versus roughly 19% under the earlier configuration, on the identical model
  weights), not in the quality of patches once produced.
- **Wrong checkpoint.** Separately, that ≈5% figure was for Qwen3.5-4B-Base, whereas all four
  fine-tuned arms in this study start from the instruct model. The instruct model's own untrained
  resolve rate (119/500, measured here) is itself more than double Qwen3.5-4B-Base's resolve rate
  under the same (corrected) harness.

Both effects compounded: an informal "≈5%" reference for the wrong checkpoint under an inferior
harness configuration made a ~75-point fine-tuning arm look like a large improvement, when the correct
baseline for that arm's actual starting checkpoint, under the same harness, is 119. Every result in the
remainder of this report uses the correctly-measured 119/500 baseline. **General methodological
takeaway:** an untrained-baseline comparison point for an agentic-coding benchmark must be (a)
evaluated under the identical scaffold configuration as the models being compared — resolve rate on
this benchmark can move several-fold on identical model weights purely from harness details — and
(b) the *exact* checkpoint that the fine-tuning starts from, not a related-but-different model variant.

### 2.2 Single-source fine-tuning: scores and a verification-quality surprise

Full-board resolve rates for the four single-source arms, alongside the instruct model:

| model | resolved / 500 | resolve rate |
|---|---:|---:|
| instruct model (no fine-tuning) | **119** | **23.8%** |
| `swezero` | 77 | 15.4% |
| `rebench` | 70 | 14.0% |
| `coderforge` | 48 | 9.6% |
| `scale` | 35 | 7.0% |

**No arm beats the instruct model**, and the gap is large: the strongest arm resolves 42 fewer instances
than the untrained instruct checkpoint it was fine-tuned from (§2.3 examines where this gap comes from).

Among the four arms, paired testing reveals a **two-tier structure rather than a strict four-way
ranking**: `swezero` and `rebench` are statistically indistinguishable from each other (McNemar
p = 0.52), as are `coderforge` and `scale` (p = 0.09), while every top-tier-vs-bottom-tier pair differs
significantly (p ≤ 0.01, and as low as p < 10⁻⁴). This two-tier split survives a Holm–Bonferroni
correction across all six pairwise comparisons. We therefore describe the arms as forming a strong tier
(`swezero` ≈ `rebench`, roughly 70–77) and a weak tier (`coderforge` ≈ `scale`, roughly 35–48), rather
than ranking them 1st through 4th.

**A verification-quality surprise.** `coderforge`, `scale`, and `rebench` are all built from
execution-verified trajectories (kept only when the agent's fix actually passed tests); `swezero`'s
source data is execution-free — trajectories were never checked for a correct outcome. If verification
quality predicted downstream agent performance, `swezero` should be the *weakest* arm. Instead it is
tied for the *strongest*, matching the verified `rebench` and significantly ahead of both other
verified sources (`coderforge`, `scale`) by more than 20 resolved instances — a gap that persists (66
vs. 41, decontaminated) after removing every instance with any training/evaluation overlap. `swezero`
and `coderforge` are the pair sharing a teacher model and scaffold whose most salient documented
difference is verification status — yet the unverified source produced the substantially stronger
arm. As noted in §1.1, this is not an isolated, single-variable ablation (the two sources are
otherwise disjoint and differ somewhat in composition too), so we treat this as suggestive rather
than a controlled causal test: across these four sources, execution-verification of training
trajectories was not, by itself, predictive of downstream benchmark performance; whatever separates
the strong tier from the weak tier here is at least partly a different property of the
data than pass/fail filtering.

### 2.3 Where does fine-tuning gain and lose capability?

The instruct model (119) outscores the strongest single arm, `swezero` (77), by 42 points in aggregate.
That 42-point gap is not spread evenly across the benchmark. To localize it, we compare the instruct
model against a **per-repository oracle** — the best of the four arms *on each repository individually*,
which may be a different arm in each row (this is the same routing oracle discussed in §2.4, whose total
across repositories is 83/500):

| repository | instances | instruct model | best of 4 arms (per repo) | instruct model − best |
|---|---:|---:|---:|---:|
| sympy | 75 | 29 | 8 | **+21** |
| scikit-learn | 32 | 14 | 6 | **+8** |
| xarray | 22 | 7 | 3 | +4 |
| pytest | 19 | 6 | 4 | +2 |
| astropy | 22 | 4 | 3 | +1 |
| requests | 8 | 3 | 2 | +1 |
| matplotlib | 34 | 3 | 2 | +1 |
| pylint | 10 | 1 | 1 | 0 |
| flask | 1 | 1 | 1 | 0 |
| django | 231 | 51 | 53 | **−2** (the one repository where an arm beats the instruct model) |
| seaborn | 2 | 0 | 0 | 0 |

(The twelfth SWE-bench Verified repository, `sphinx`, is excluded here as in §1.3; `seaborn` has only 2
instances and both the instruct model and every arm score 0 there, so it is a zero-contribution row.)
Because the "best of 4 arms" column is a per-repository oracle rather than a single fixed arm, its total
(83) is higher than `swezero`'s own aggregate score (77), and this table's differences sum to **36**, not
the 42-point single-arm gap quoted above — the two numbers answer related but distinct questions (one
arm's aggregate deficit vs. the *best-case*, per-repository deficit against an oracle that can pick a
different arm for every repository), and both are used elsewhere in this report (§2.4 uses the 83/36 view).

The instruct model matches or exceeds the best arm on **10 of the 11** scoreable repositories; only on
`django` (the largest repository, 46% of the benchmark) does any arm edge ahead of the instruct model,
and only by 2. The aggregate gap concentrates heavily in `sympy` (+21) and, to a lesser extent,
`scikit-learn` (+8); the remaining repositories contribute only a few points each, individually within
the noise floor for a benchmark of this size.

Two checks rule out the more mundane explanations for this pattern:

- **Not a training-data leak.** `scikit-learn` and `pytest` are not present in *any* arm's training
  data, yet the instruct model beats every arm there too — so the pattern is not an artifact of contaminated
  overlap between training and evaluation data.
- **Not just the weakest arm underperforming.** On both `sympy` and `scikit-learn`, the instruct model
  beats even the *union* of what all four arms can solve combined (29 vs. 13 on sympy; 14 vs. 12 on
  scikit-learn) — pooling every arm's solved instances together still does not recover the instruct
  model's count. All four fine-tunes independently lost capability the instruct model had on these
  domains, including the arm that was directly trained on `sympy`-repository data.

We characterize this as **systematic degradation on domains the instruct model already handled well**,
concentrated in a couple of specific repositories, rather than either a uniform capability loss or an
isolated one-repository anomaly. This degradation, not a general capability gap, is the dominant
contributor to the raw 42-point instruct-model-vs-best-arm difference: restricting to a "clean" 301-instance
subset (excluding all repositories touched by training-data overlap for *any* arm, including
`sympy`), the instruct model's advantage over the best arm shrinks to a statistically non-significant +10
(76 vs. 66, McNemar p = 0.30). **We do not find a significant general improvement in single-attempt
resolve rate from fine-tuning on this benchmark, and we find a significant, repository-concentrated
loss of pre-existing solve-rate that is not attributable to contamination** (§2.7 discusses why this
should be read as specific to this benchmark's single-attempt scoring, not as a protocol-independent
claim about capability).

### 2.4 Complementarity across sources: how much could be gained by combining them?

The four arms solve substantially different subsets of the benchmark. The union of all four arms'
solved instances — an oracle that always picks whichever arm (if any) solves a given instance — reaches
**131/500 (26.2%)**, 54 instances more than the best single arm. Of those 131 jointly-solvable
instances, only 8 are solved by all four arms; 68 (52%) are solved by exactly one arm. This behavioral
diversity motivated asking how much of that 54-instance gap could realistically be captured by a
practical combination method, rather than an oracle.

Two cheap combination strategies both fall well short:

- **Routing by repository** (always sending a given repository's instances to whichever arm is
  historically strongest on that repository) reaches 83/500 — capturing only 6 of the 54 available
  points (11%). Most of the complementarity is *within* a repository, between specific instances, not
  *between* repositories.
- **Routing by instance text** (a simple text classifier trained to predict, from the issue
  description, which arm is most likely to solve a given instance, cross-validated on the 131
  jointly-solvable instances) achieves only 57% held-out accuracy at picking the correct arm, and the
  resulting routed resolve rate (75/131 on the solvable subset) is *below* simply always using the best
  single arm. The issue text carries essentially no exploitable signal about which arm will succeed.

We take this as evidence that the complementarity between these four data sources is **scattered at
the level of individual instances and not predictable from surface features of the problem** — capturing
it would require either an oracle/ensemble (impractical at inference time) or combining the sources at
training time rather than routing between already-trained models. This motivates the training-time
combination experiments in §2.5–2.6.

### 2.5 Combining models by weight averaging

Since all four arms are fine-tuned from the same initialization, each defines a **task vector**
τᵢ = θᵢ − θ₀ (the displacement from the shared starting checkpoint θ₀, i.e. the instruct model). We
decompose each task vector into a shared component and an arm-specific residual: τᵢ = s + rᵢ, where s
is the mean of the four task vectors and rᵢ = τᵢ − s is what remains for arm i (by construction, the
four residuals sum to zero).

Measuring this decomposition: all four task vectors have nearly identical norms (14.9–15.3, in the
model's parameter space), and each is only weakly aligned with the others (pairwise cosine similarity
0.20–0.26 — well above what unrelated random directions would produce, but far from parallel). The
shared direction s captures only **65% of a typical arm's norm** (‖s‖ / mean‖τᵢ‖ = 0.646), meaning
each arm's residual rᵢ carries roughly 75–78% of that arm's total displacement from the instruct model
(these two percentages are norms of different, close-to-orthogonal components, not fractions of a
whole, so they need not sum to 100% — by the Pythagorean relation for near-orthogonal vectors,
0.646² + 0.76² ≈ 1.0) — a large amount of arm-specific structure that a *simple average* of the four
task vectors necessarily discards, since the residuals cancel out of an unweighted mean by construction. Breaking the
comparison down by model component shows this shared/residual split is not uniform across the network:
embedding and output layers align strongly across arms (cosine ≈ 0.67 — consistent with a shared
format/tool-use adaptation), while the attention and MLP layers — where task-specific skill is
presumably learned — are far less aligned (cosine ≈ 0.10–0.11), most divergent in early layers.

This geometry entails a mathematical fact and raises a separate empirical question. The fact, which
follows directly from the definitions above and needs no experiment to confirm: an equal-weight average
of the four fine-tuned models' weights keeps *exactly* the shared component s (≈65% of a typical arm's
displacement from the instruct model) and cancels the four residuals exactly, by construction. The open
empirical question is whether the *performance* associated with each arm's residual survives being
discarded this way, or whether it was concentrated in the shared component instead — that is not
determined by the geometry and has to be measured:

| combination | resolved / 500 | vs. instruct model | vs. best single arm |
|---|---:|---|---|
| instruct model | **119** | — | — |
| best single arm (`swezero`) | 77 | — | — |
| best soup found (average of the two strong-tier arms only) | 62 | significantly below (p < 10⁻⁴) | significantly below on the decontaminated 301-instance board (p = 0.014); not significant on the full 456-instance board (p = 0.10) — see note below |
| equal-weight average of all four arms | 50 | significantly below (p < 10⁻⁴) | significantly below (p = 0.0016) |

The results show the discarded residuals, not the retained shared component, are where each arm's
performance lived: every soup variant lands well below the arm(s) it was averaged from.

The equal-weight average of all four arms' weights lands in the **weak tier** — statistically
indistinguishable from the two weakest single arms, and significantly below both strong-tier arms
(swezero: p = 0.0016; rebench: p = 0.025). Restricting the average to just the two strong-tier arms
recovers somewhat but is still well below the best individual arm and far below the instruct model —
though this specific comparison against `swezero` is a near-tie on the full 456-instance board (p =
0.10) and only becomes clearly significant on the decontaminated 301-instance board (p = 0.014). The
discrepancy traces to `sympy`: the two-arm soup includes `rebench` (the sympy specialist, §2.2) and
scores 11/75 there against `swezero`'s 3/75, a repository-specific edge that narrows the overall
456-board gap; the 301-instance board excludes `sympy` (along with the other contaminated repositories,
§1.3) and reveals the larger, more representative deficit underneath it.

We further tested scaling the shared direction alone, θ(α) = θ₀ + α·s, for α ranging from 0.7 to 2.0
(α = 1 is the equal-weight average above; α ≈ 1.55 restores the *norm* of a typical single arm along
this one direction, though not its actual residual direction). Resolve rate **decreases monotonically**
as α increases past roughly 0.7 (54 → 50 → 38 → 19 for α = 0.7, 1.0, 1.55, 2.0). Since α = 0 recovers
the instruct model by definition (119), the entire measured curve sits below the instruct model, and pushing
further along the shared direction only makes things worse. This indicates the shortfall is not a
matter of *magnitude* — simply rescaling the merged model does not recover lost performance — but a
genuine loss of the residual, arm-specific directions that a weight average necessarily cancels out.
**No weight-space combination we tried recovers the performance of the best single arm, let alone the
instruct model.**

### 2.6 Combining models by joint training

An alternative to combining already-trained models is to train a single model on the pooled data from
all four sources directly. We trained two such models: one on a data-matched pool (the full union of
what all four arms trained on, 4× a single arm's data budget) and one on a compute-matched pool (the
same total training budget as one arm, spread across all four sources).

| model | training data | resolved / 500 | vs. instruct model | vs. best single arm |
|---|---|---:|---|---|
| instruct model | — | 119 | — | — |
| best single arm | 55K, one source | 77 | — | — |
| joint-trained, compute-matched | 55K, all four sources | 46 | significantly below (p < 10⁻⁴) | significantly below (p = 4×10⁻⁴) |
| joint-trained, data-matched | 220K, all four sources | 54 | significantly below (p < 10⁻⁴) | — |

Neither pooled model beats the instruct model or the best single-source arm; the data-matched model is
significantly below the instruct model on every board tested (−65/500 and −65/456, p < 10⁻⁴; −37/301 on
the decontaminated board, p < 10⁻⁴).

Two comparisons that are *not* resolvable at this sample size, and which we therefore decline to state
as equivalences: 4× the training data (55K → 220K) buys **+8 resolved instances (p = 0.34)**, and the
data-matched joint-trained model sits **+4 above the equal-weight weight-averaged soup** from §2.5
(50 → 54, p = 0.69). Both are well inside the noise floor of §1.3, so the honest reading is
"indistinguishable at n = 500," *not* "data scaling is flat" and *not* "joint training and weight
averaging land in the same place" — those are equivalence claims, and as noted in §1.3 a
non-significant result does not establish equivalence. (In fact equivalence is structurally
unreachable on a single 500-instance run here: with ~120 discordant pairs the standard error is
≈11/500, so even a *perfectly* null result yields a 90% confidence interval of roughly ±18/500 —
wider than any margin one would want to call "equivalent." Only more instances, not more seeds, would
tighten this; at temperature 0 the discordant pairs are structural disagreement between two
deterministic models, not sampling noise that re-running would average away.)

**What the board does support: combining the four data sources — whether by weight averaging or by
joint training, at either data budget we tested — does not recover the performance lost relative to
the instruct model, nor does it beat the strongest single-source arm.**

### 2.7 Why does fine-tuning underperform? A mechanistic case study

To understand *why* fine-tuning underperforms the instruct model, we audited every instance the
instruct model solves but the strongest arm (`swezero`) does not (82 instances). In every one of these
82 cases, the fine-tuned model's submitted patch is syntactically valid and applies cleanly to the
repository — none are empty, malformed, or the product of an early, incomplete termination. The
failures are not scaffold breakdowns; they are patches that are well-formed but do not fix the
underlying bug (misdiagnosis).

Looking at overall episode behavior confirms a systematic behavioral shift, not a broken model:

| | instruct model | fine-tuned arm (`swezero`) |
|---|---:|---:|
| episodes reaching a final answer | 207 / 500 | 500 / 500 |
| episodes ending with an empty patch | 138 / 500 | 0 / 500 |
| mean agent actions per episode | ~171 | ~92 |

These two instruct-model counts do not sum to 500: 362 episodes (500 − 138) ended with a non-empty
patch, but only 207 explicitly reached a final answer, so 155 episodes exhausted the iteration budget
without an explicit final action yet still left a non-empty diff in the workspace, which the harness
scores directly (we have not established what fraction of the instruct model's 119 resolved instances
fall in this category, only that it is a substantial share of its episodes overall).

The fine-tuned model is *more* compliant with the agent scaffold in every respect measured here — it
always finishes, it always produces a patch, and it does so in about half as many steps as the
instruct model. But this comes at a cost: relative to the instruct model, the fine-tuned arm gains 40
newly-solved instances the instruct model missed, while losing 82 the instruct model had — a net
loss of 42. We interpret
this as **fine-tuning shifting the model's policy from a slower, more exploratory, "messier"
problem-solving style toward a faster, tidier, more scaffold-compliant one that is quicker to settle on
an answer and less likely to keep investigating a wrong initial diagnosis.**

This reframes the headline result: it is not that fine-tuning made the model unconditionally worse,
but that it moved the model along a depth-vs-compliance trade-off, and the specific evaluation
protocol used throughout this report — a single deterministic attempt per instance, scored only on
whether the final patch resolves the issue — rewards the instruct model's deeper (if less efficient and
less reliably-terminating) exploration. A different protocol — for example, allowing multiple attempts
per instance, or an evaluation that penalizes non-termination more heavily — could plausibly favor the
fine-tuned behavior instead. The result reported throughout this study should be read as **"fine-tuning
on this data does not improve single-attempt resolve rate on this benchmark,"** not as a
protocol-independent claim about capability in general.

**A candidate, correlational (not yet causally tested) explanation.** Auditing `swezero`'s training data
(the arm audited above) finds two properties that plausibly contribute to this shift: 34% of its
records are condensation/summarization turns (the trajectory narrates and concludes progress rather
than taking a new action; a similar proportion, 39%, holds for `coderforge`, though the other two
sources are not measured on this axis, so we do not know whether this generalizes across all four),
and none of the training data carries a task-outcome label.

⚠️ **This second property was misread, and the correction is material (2026-08-04).**
We inferred "unfiltered" from "unlabeled." Two of the four sources were in fact
**success-filtered upstream, before ADP normalisation**, with the label then
dropped as a constant: `coderforge` keeps only `reward == 1` and `rebench` only
`resolved == 1` (verified in ADP's own `extract_raw.py`, and by a join over all
319,551 records matching at 100%). Upstream base rates were 58–63% and 47.9%, so
the filter removed a great deal. The claim holds only for `swezero` and `scale`
— and `swezero`, the *unfiltered* source, produced the **best** arm. So
"unresolved trajectories are imitated on equal footing with resolved ones" is
false for half the corpus, and outcome-verified training data is not an untested
idea in this campaign: `coderforge` and `rebench` **are** that experiment. On the
harness-corrected board they score 50 and 76 against unfiltered `swezero`'s 77 —
i.e. one clearly below and one statistically indistinguishable, which is evidence
that success-filtering is not *sufficient*, not that it is worthless. The
comparison is also cross-source and therefore confounded.

The first property (conclusion-oriented records) remains as described. A direct test — fine-tuning one
arm on an outcome-filtered, condensation-excluded version of the same data source, holding everything
else fixed — would establish whether curating along these two axes recovers some or all of the lost
depth; this experiment has not yet been run and is the most direct open follow-up to this study.
(`adp_v2_commentary.md`, a companion discussion note, goes further into candidate mechanisms — including
whether this is better characterized as a policy shift, a partial erosion of general capability from
full-parameter fine-tuning, or an imitation-learning/partial-observability effect specific to agentic
tasks — with additional falsifiable predictions beyond the single curation ablation proposed here.)

### 2.8 Comparison to a companion campaign

A related campaign trained similar agentic-SWE data recipes starting from **Qwen3.5-4B-Base** (the
same non-instruct checkpoint introduced in §2.1), using different training and evaluation
infrastructure than the one described in this report. Its headline single-run pass@1 results on the
same benchmark: a curated recipe reached 52/500 (10.4%); a different data recipe reached 74/500
(14.8%) starting from Qwen3.5-4B-Base, and — in an isolated ablation that changed only the
initialization, holding the data fixed — 82/500 (16.4%) starting from the instruct model instead.

Because that campaign used different infrastructure, and §2.1 established that infrastructure details
alone can move absolute resolve rates several-fold on identical model weights, any cross-campaign
absolute comparison must be read as suggestive, not confirmed. With that caveat: the four single-source
arms in this report (35–77) and that campaign's results (52–82) occupy a broadly similar overall range,
but not all fine-tunes land in the same place — our two weaker arms (48, 35) are well below that
campaign's results. What is notable is specifically at the *top* end: the strongest recipe on each side
lands in a similar 74–82 band regardless of whether training started from Qwen3.5-4B-Base or from the
instruct model — that campaign's strongest Qwen3.5-4B-Base-initialized recipes rise to 74 and 82, and
our strongest arm (77, instruct-initialized) sits inside that same band too, well below the instruct
model's own untrained resolve rate of 119 that it started from. This is consistent with (though not, on
its own, proof of) the same phenomenon identified mechanistically in §2.7: this style of agentic-SWE fine-tuning may pull
models of different starting strength toward a common point on this benchmark, rather than uniformly
adding capability on top of wherever a model started. Confirming this would require a matched-harness
instruction-tuned baseline evaluated on that campaign's own infrastructure, which does not currently
exist.

---

## 3. Discussion and related work

**The central finding** is a well-supported negative result: for these four data sources, at this data
scale (55K–220K records), fine-tuning a capable instruction-tuned 4B model does not improve — and on
several specific domains, measurably degrades — its single-attempt SWE-bench Verified resolve rate,
and no combination of the resulting models (by weight averaging or joint training) closes that gap.
The mechanistic audit (§2.7) suggests this is not simply "the data is low quality" but a specific,
identifiable behavioral trade-off (depth vs. compliance) that this benchmark's scoring protocol
penalizes. Whether **curated** training data — filtered to outcome-verified, non-condensation
trajectories — can retain the compliance gains without the depth loss is, at time of writing, the most
concrete and highest-value open question raised by this work, and the natural next experiment.

**Relationship to prior work on model merging.** Averaging the weights of independently fine-tuned
models sharing an initialization ("model soups") and additively composing task vectors ("task
arithmetic") are established techniques for combining specialized capabilities without joint
retraining, resting on the empirical observation that same-initialization fine-tunes tend to remain
linearly mode-connected. Follow-on methods (e.g. TIES, DARE) address destructive sign-interference
between task vectors, primarily for merges across less-related tasks than the same-domain fine-tunes
studied here. Related work uses the pairwise angle between task vectors as a cheap signal for whether
weight averaging is likely to preserve capability — consistent with that framework, we find that
averaging preserves only the shared (mean) component of the four task vectors, and that a
substantial fraction of each arm's useful, complementary behavior lives specifically in the
residual, arm-specific component that an unweighted average cancels by construction. Other work
uses merge coefficients (or task-vector geometry more generally) as a cheap proxy for what a full
data-mixing ratio would produce for a subsequent joint-training run — an idea our joint-training
comparison (§2.6) speaks to directly: at the two data budgets we tested, joint training on pooled data
did not outperform weight averaging, so — at least in this setting — neither combination strategy
recovers what separate single-source training achieves, let alone the instruct model. We stress that
this is a "did not outperform," not a demonstration that the two are interchangeable: the merge-vs-joint
gap here (+4/500) is simply too small to resolve at this sample size (§2.6), so our data neither
validates nor refutes the merge-as-proxy-for-data-mixing idea — it only shows both landed well short of
the starting checkpoint.

---

## 4. Limitations

- **Statistical power.** At n≈500 (or n≈300 on the decontaminated subset) with a single deterministic
  rollout per instance, differences smaller than roughly ±15 resolved instances are not reliably
  distinguishable from sampling noise. Several comparisons among the weaker arms and combination
  methods fall near or within this range and should be read as directional.
- **Benchmark scoring artifact.** 44 of 500 instances (one repository) cannot be resolved under the
  standard scoring harness regardless of model quality, for a benchmark-specific test-output-parsing
  reason. This affects all models identically and does not bias relative comparisons, but deflates
  every model's absolute resolve rate by a small amount; we report raw counts out of 500 throughout
  (see §1.3) and compute significance tests on the 456-instance board excluding these instances.
- **Contamination check is repository-level, not instance-exact.** The decontamination check in
  §1.3/§2.3 excludes entire repositories with any known training/evaluation overlap for any arm; a
  fully instance-exact audit for every source has not been completed.
- **The mechanistic explanation in §2.7 is a case study, not yet a controlled experiment.** The
  patch-level audit compares one arm against the instruct model; it has not been replicated across all
  four arms, and the proposed curation ablation to test the hypothesized cause causally has not been
  run.
- **Cross-campaign comparisons (§2.8) cross different infrastructure** and are explicitly flagged as
  suggestive given the harness-sensitivity finding in §2.1.

## 5. Open questions and future work

- Run the proposed curation ablation (outcome-filtered, condensation-excluded training data vs. an
  equally-sized unfiltered control) to test whether it recovers some or all of the lost depth
  identified in §2.7.
- Evaluate the fine-tuned arms under a protocol less sensitive to the depth-vs-compliance trade-off
  (e.g. multiple attempts per instance, or a metric that separately credits reaching a well-formed
  final answer) to test whether the "compliant" policy fine-tuning produces is a worse or better
  starting point for further training (e.g. reinforcement learning) even though it underperforms on
  single-attempt resolve rate.
- A matched-harness instruction-tuned baseline for the companion campaign in §2.8 would let the
  apparent common "attractor" pattern be tested directly rather than only suggestively.
- Investigate whether a stronger, non-text-based router (e.g. using model confidence signals rather
  than surface problem-text features) can approach the 131-instance oracle union identified in §2.4,
  since the simple routing strategies tested there captured only a small fraction of it.
- `adp_v2_commentary.md` proposes several cheap, discriminating experiments aimed at separating a pure
  policy-shift explanation of §2.7 from a partial capability-erosion one (e.g. evaluating the fine-tuned
  arm on a non-agentic reasoning benchmark; forcing longer exploration at inference time via a prompt
  intervention; checking whether the fine-tuned arm's decision to stop correlates with actually having
  read the relevant files, or just with turn count) — these are lower-cost than the training-data
  curation ablation above and could be run first to sharpen which mechanism the ablation should target.
