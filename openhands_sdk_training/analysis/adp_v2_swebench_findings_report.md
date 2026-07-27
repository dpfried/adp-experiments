# ADP-v2 SWE-bench Verified — Findings Report (living doc)

> _Public snapshot of the internal living findings doc. File paths to analysis scripts/results have been reduced to basenames (internal cluster paths removed)._


_Maintained by the main-worker session. All numbers file-backed under
`merged.report.json`.
Framing: this is about **what the experiment teaches**, not leaderboard rank._

## TL;DR
_STATUS 2026-07-26 (VERIFIED, **PROVISIONAL pending pooled220k joint-train**). θ₀ resolved = 119/500 provenance-clean base; all 4 arms + full soup/α-sweep scored. See the "θ₀ RESOLVED" + "SFT-LIFT VERDICT" sections at the end for file-backed detail._

**Bottom line (one sentence):** Raw ADP first-55k SFT does **not** improve a **~24% base** (Qwen3.5-4B instruct, single-run pass@1) and **significantly degrades it across the base's domains** (base ≥ best arm on 9/10 repos; sympy 29/75 → 3/75); **no weight-merge of the arms recovers it** — every soup is significantly below base and the best arm (joint-train pooled220k **PENDING**, honest prior ≤ base). The apparent "5%→15% SFT lift" was mostly a **harness artifact (H2)**: the *same* Qwen3.5-4B-*Base* weights score **≥10.6% (~16% ex-sphinx) under this campaign's harness** vs 5.0% on Graham's pre-fix June harness (non-empty-patch rate ~70% vs 19%), so ~5% reflects patch-production, not model capability. A base-vs-instruct mix-up (arms are *instruct*-init, θ₀=119≈24%; H1) is a real but *non-causal* secondary defect.

**Three findings, kept separate (per-board, so they don't blur):**
1. **No significant general lift** — clean-301 (fair general board): base 76 vs best arm (swezero) 66 = **+10, McNemar p=0.30, n.s.**; django (clean, 231) 51≈53 tie. ("No significant lift," NOT equivalence, NOT "base wins.")
2. **Systematic degradation across the base's domains** — base ≥ best-of-4-arms on **9/10 repos** (only django favors an arm, +2); aggregate **+38/500**, concentrated in sympy (+21) & sklearn (+8) but present even on clean never-trained repos (sklearn/pytest) ⇒ genuine forgetting, not contamination. Resolvable claims = the aggregate +38 and the 9/10 directional consistency (per-repo deltas mostly within-noise). This — not a general-capability gap — is the true source of the naive full-500 "+42 base≫arms". (See MULTI-REPO FORGETTING section.)
3. **Weight-merging strictly loses** — combine board base 119 ≫ swezero 77 > … > top2-soup 62 > uniform4 50 > α1.55 38 > α2.0 19; the α-line maximum is α=0 (=base) by construction; every measured soup < base & < best arm (residual-drop, not a norm deficit; (0,0.7) unmeasured → no strict-cliff claim). Cemented claim = **no WEIGHT-MERGE beats base**; joint-train is the one untested method that could.

**Data defects = candidate LEVERS, not a proven cause (devils-advocate):** agent-b's data findings (~40% non-solve/condensation records, no quality filter, arm-asymmetric 28k truncation) are *candidate* curation levers with **untested** effect — the 40%-condensation sign is two-sided (dilution to cut, OR a real context-mgmt skill SBV doesn't score). The real remaining research question = does **quality-filtered** data lift over the measured ~24% base? (Data curation, not post-hoc merging.)

**Methods lesson (transferable):** measure the baseline **under the identical harness** and as the **exact init checkpoint**. The "~5%" conflated two problems, the bigger being the harness: the *same* base weights score ≥10.6% (~16% ex-sphinx) under this campaign's harness vs 5.0% on Graham's pre-fix harness (~70% vs 19% non-empty patches) ⇒ mostly a *patch-production/harness* artifact (H2), not capability. The wrong-checkpoint-class baseline (instruct-init arms vs a base anchor; H1) is real but *non-causal*. (Provenance RESOLVED 2026-07-26 by the Babel-side agent — see "ANCHOR PROVENANCE — RESOLVED" below.)

- **HEADLINE RETRACTED (2026-07-25, see H3-FALSIFIED below): the "SFT lift" was computed against the WRONG baseline (wrong model class + wrong harness).** "θ₀ ≈5%" was a borrowed sanity-anchor (25/500) — a REAL measurement, but of Graham’s *base* model on a *broken (pre-timeout-fix) harness* (H2, see provenance below), never the instruct θ₀ the arms init from; "+50/500 hugely significant" was an arithmetic gap to that placeholder, NOT a paired McNemar (no measured θ₀ resolved-set existed). MEASURED single-run θ₀ = 119/500 (~24%, provenance-verified 2026-07-26 — see "θ₀ RESOLVED" section below) => SFT-lift is NULL-to-NEGATIVE: no arm significantly beats base (clean-301 +10, p=0.30 n.s.); coderforge/scale significantly BELOW base — the one
  effect comfortably resolvable at n=500 ("the paper sentence"; θ₀ denominator now fixed = 119/500 verified).
  Everything finer (arm ranking, soup-vs-pool, data-scaling) is **≤ a 2-tier split, mostly "unresolved at
  n=500"** — single temp-0 rollout can't resolve <~15/500.
- 4 ADP SWE data sources, each SFT'd on **Qwen3.5-4B** (θ₀ = instruct), produce
  **complementary, specialized** capabilities — not redundant ones.
- Full-500 SWE-bench Verified: **swezero 77 (15.4%), rebench 70 (14.0%),
  coderforge 48 (9.6%), scale 35 (7.0%)**  (anchor CORRECTED: provenance-verified single-run instruct θ₀ = 119/500 (~24%), NOT ~5% — see retraction above).
- **Oracle union = 131/500**: collectively they solve +54 over the best single arm; **52%
  of solved instances are solved by exactly ONE arm** → high behavioural diversity.
- **Specialization:** swezero broad (best on 8/10 repos); **rebench→sympy**; **scale→scikit-learn**
  (a real niche w/ unique solves *despite* being weakest overall); **ALL FOUR = 0/44 on sphinx**
  (a shared coverage hole — top data-curation target).
- **Failure mode:** scale's deficit is patch *quality*, not giving up (all arms emit a patch
  ~99.8% of the time, error <1%).
- **Merge-vs-joint-train is a CLEAN test:** arms trained 55k each (pretok cap; 1719 steps); pooled220k =
  the 4×55k union ⇒ soup(all-4) vs pooled220k is data-matched. (Earlier 'confound' retracted — see below.)

## Experiment design
Base θ₀ = `Qwen/Qwen3.5-4B` (instruct). Four "arms" = SFT on one ADP-normalized SWE source each
(~80k ex): coderforge_preview, nebius_SWE-rebench, scale_swe_distilled, nvidia_SWE-Zero.
Two "pooled" = joint-train on a mix (pooled55k=55k, pooled220k=220k). "Soup" = weight-average of the
4 arms. Eval = SWE-bench Verified (500), OpenHands agent, maxiter-500, temp 0, sharded 10×A100.

## Methods & rigor (bugs found + fixed — they gate validity)
1. **Apptainer `/private` scoring bug**: `--writable` sandbox aborted on host $HOME/cwd binds →
   every instance falsely 0-resolved. Fixed `--no-mount hostfs`→`hostfs,home,cwd`. (Caught because a
   uniform 0-across-arms is a scoring artifact, not model perf.)
2. **Duplicate rollouts** (e.g. 515 lines/495 distinct) inflated denominators → added instance_id dedup.
3. **Per-job vLLM port collision** (2 shards/node) → per-job port.
4. **Comparability:** arms compared on the same full 500. A 96-instance proxy was shown to **mis-rank**
   rebench vs coderforge vs full-500 → NOT usable to rank close models.

## Results
### Arm scores (full-500, file-backed)
| arm | resolved/500 | rate |
|---|---|---|
| swezero | 77 | 15.4% |
| rebench | 70 | 14.0% |
| coderforge | 48 | 9.6% |
| scale | 35 | 7.0% |


**Statistical rigor (paired McNemar + bootstrap 95% CI, seed 1234):** the arms form **TWO TIERS, not
four ranks**:
- swezero 77 [62,93] ≈ rebench 70 [55,87] — **n.s.** (McNemar p=0.52). Top tier, statistically tied.
- coderforge 48 [36,61] ≈ scale 35 [24,47] — **n.s.** (p=0.09). Bottom tier, tied.
- {swezero,rebench} > {coderforge,scale} across the tier boundary is significant (p≤0.01, up to p<1e-4).
=> report as "top tier ~73 / bottom tier ~42", NOT a strict 1-2-3-4 ranking. Marginal CIs are wide
(±~15) and overlap; the paired test is what licenses the tier split.

**Multiple-comparison + equivalence (team-converged, file-backed):** the 2-tier split **survives
Holm-Bonferroni** over the 6 arm-pairs (all 4 cross-tier reject; both within-tier retain; rebench↔coderforge
p=0.009 is the fragile one under a larger family — re-run Holm over arms+soups+pooled once those exist).
**CRITICAL — n.s. ≠ equivalence:** non-significant pairs are **"unresolved at n=500," NOT "tied/equal."**
A positive equivalence claim (e.g. "merge ≈ joint-train", "4× data ≈ no gain") needs a **TOST** test vs a
pre-set margin (pre-registered **±10/500 = 2%**, dpf-adjustable); at n=500 the close pairs come out
**INCONCLUSIVE** (90% CI wider than the margin). So all fine contrasts report three-state:
**different / equivalent / inconclusive** — most will read *inconclusive*. Tool: soup-worker's
`paired_compare.py` (McNemar + TOST). This makes "n=500 is underpowered for the interesting soup/pool
contrasts" a *quantified* claim, and is the concrete case for more instances/seeds (the top 9am item).

### Complementarity (from resolved_ids)
Oracle union (any-arm) = **131/500 (26.2%)**. Solved by exactly 1 arm: 68; by 2: 35; by 3: 20; by all 4: 8.
Unique-to-arm: swezero 29, rebench 25, coderforge 7, scale 7. Backbone = swezero+rebench (54 unique combined).
NOTE: a weight-space soup is NOT an ensemble → it will not reach 131; realistic soup target = "beat 77".

### Specialization map (per-repo)
- **sphinx-doc__sphinx = 0/44 by ALL arms** (coverage hole).
- swezero best on 8/10 repos (broad). rebench niche = sympy (8/75 vs 3). scale niche = scikit-learn
  (ties swezero 5–5, 3 unique sklearn solves + unique matplotlib/xarray).

### Failure mode
All arms produce a git_patch ~99.8% of instances; error/stuck <1%. So score differences = solution
*quality*, not give-up/timeout rate.

### Task-vector geometry (mechanistic) — tau_i = theta_arm_i - theta0
**Norms** (distance moved from theta0): swezero 14.86, rebench 15.07, coderforge 15.04, **scale 15.26**.
=> ALL arms moved ~the same distance; the WORST arm (scale) moved the FARTHEST, the BEST (swezero) the
LEAST. So performance is NOT explained by "how much it learned" — scale learned a *different, less useful
direction*, not a smaller amount. (‖τ‖ is ~uncorrelated / slightly anti-correlated with SWE score.)

**Pairwise cosine(tau_i, tau_j):** all in 0.20–0.26 (near-orthogonal, mildly positive). swezero is the
*least* aligned with the others (0.199–0.218) — its useful direction is the most distinct; rebench↔coderforge
most aligned (0.262). => the 4 ADP sources push the model in largely **different weight-space directions**,
consistent with the complementary capability niches.

**Uniform-soup direction:** ‖mean(τ)‖ / mean(‖τ‖) = 9.73/15.06 = **0.646** (orthogonal baseline for n=4
= 0.500; aligned = 1.0). => mostly-orthogonal with mild positive alignment ⇒ averaging should be **mildly
constructive, not destructive** (no cancellation), BUT the soup moves only ~65% as far as any single arm —
a smaller step in a compromise direction. Whether that compromise beats the best arm is the empirical
question the soup eval answers.

**Per-component / per-depth (where specialization lives):**
- **embed / lm_head: mean cos = 0.669 (HIGH alignment)** — all 4 sources push the embedding/output layer
  the *same* way. This is shared **format/token adaptation** for agentic SWE, not source-specific skill.
- **attn: 0.103, mlp: 0.114 (near-orthogonal)** — the source-specific learning lives in the transformer
  **body**; sources diverge there. (norm layers move ~0 — ignore.)
- **by depth: early 0.066 (most divergent) > mid 0.134 ≈ late 0.133.** Divergence concentrates in **early
  layers**.
- ‖τ‖ is near-identical across arms within every component (e.g. mlp 10.4–10.7) — reinforcing: arms move
  the same *distance*, differ in *direction*.
=> **Mechanistic story: ADP sources AGREE on the format (embeddings, cos .67) but DIVERGE on the skill
(attn/MLP body, cos ~.10, strongest early).** Souping averages the aligned embedding safely; the
dilution/interference risk is concentrated in the near-orthogonal body

**Shared-component decomposition (CORRECTS the "near-orthogonal/complementary" framing):**
Decompose τᵢ = s + rᵢ, s = mean(τ). Measured: **‖s‖/mean‖τ‖ = 0.646; cos(τᵢ,s) ≈ 0.64 for every arm**;
residual rᵢ ≈ 75–78% of each arm's norm. **Correction (per devils-advocate's null-floor point):** pairwise
cos ~0.22 is ~10⁴× the high-d random null (1.6e-5) — so the arms are NOT "orthogonal/diverse"; they share a
**DOMINANT common direction s** (the shared SWE-agentic-SFT adaptation, 64% of each arm) plus a *modest*
arm-specific orthogonal residual (~76% of norm) that carries the niches + the 77/70/48/35 spread. My earlier
"near-orthogonal → complementary" wording is retracted.
**Souping consequence (task-arithmetic, corrected):** uniform4's task-vector = s *exactly* (residuals cancel);
norm = 0.646·τ̄ ⇒ it keeps the shared SFT lift at ~full strength but **drops every arm's residual skill**.
All equal-weight combos ∝ s; convex reweights (C2/C3/greedy/LOO) move toward a single-arm vertex ⇒ ≤ best arm.
The only lever with upside is **global scale α** (θ₀+α·s); norm-matching a single arm needs **α ≈ 1/0.646 ≈ 1.55**
(NOT √N=2, which overshoots ~29%). The α≈1.55 soup — not α=1 — is the fair merge-vs-joint comparator to
pooled220k (joint SGD reaches full norm in-combined-direction, escaping the mean-soup √N shrinkage).
**Sharpened prediction:** uniform4 = **tie-to-below the arms, NOT a win** (shared lift kept, residual skill dropped).

**Residual geometry (arm-specific directions rᵢ = τᵢ − s):** ‖rᵢ‖ ≈ 11.3–11.7 (75–78% of each arm's ‖τ‖ — the
residuals are LARGE). cos(rᵢ,rⱼ) ≈ **−0.33 for all pairs = −1/(N−1)**, exactly the zero-sum null (Σrᵢ=0 by
construction) → the residuals form a **symmetric simplex**: maximally-spread, no extra pairwise structure, the
niches are genuinely independent directions. Substrate for task-arithmetic: subtracting s isolates each arm's
large, independent residual — proposed experiments (θ₀+rᵢ niche-retention; θ₀−α·s negation; 3-way ablation)
tabled for soup-worker/dpf. — exactly where averaging 4
distinct directions shrinks the step (the global 0.646 ratio).

## Design caveat: what soup-vs-pool can/can't show
Arms ≈80k single-source; pooled = 55k/220k mixed; no pool on the ~320k union. So soup(arms) ≈ pooled220k
would be confounded by data amount/mix (can't isolate merge-vs-joint-train). pooled55k→220k = clean
scaling. To cleanly test merge-vs-joint-train would need a union-matched pool. (Pooled mix inferred from
sizes+dataset_info; exact per-source recipe unconfirmed.)

> ✅ **RESOLVED — CONFOUND RETRACTED (2026-07-25, verified).** The arms trained on
> **55k each** (arm `pretok.yaml` `max_samples: 55000`; each ran exactly **1719 steps = max_steps**
> = 55008/batch-32 — the ~80k jsonl was truncated to first-55k). `pooled220k` (`max_samples: 220000`)
> is **the round-robin union of the 4 arms' 55k = 220k** (per generate_arm_runs.py provenance). So the
> arms' trained union == pooled220k EXACTLY ⇒ **soup(all-4) vs pooled220k IS a clean, data-matched
> merge-vs-joint-train test.** My original 'no union-matched pool / ~320k' claim was WRONG (I read
> jsonl line-count as trained-count, missing the pretok cap). Confound withdrawn.

## PENDING (this doc updates as they land)
- **θ₀ baseline** `v2_init_4b` (agent-b17cac1e) → per-arm **SFT delta** over the starting model.
- **uniform soup** `v2_soup_uniform4_4b` (soup-worker) → **compositionality test** (harness ready:
  `compositionality.py` — union recovery, niche retention, interference).
- **pooled55k/220k** evals → data-scaling; soup-vs-pool (w/ caveat above).
- ~~task-vector geometry~~ **DONE** (see Results): scale weak due to *direction* not magnitude; sources near-orthogonal (mild + alignment); soup predicted mildly constructive.


## Pre-registered hypotheses (LOCKED before soup/θ₀/pooled results exist)
_Recorded now so the pending results are genuine tests, not post-hoc stories. Each has a falsifier._

- **H1 (uniform soup /500):** near-orthogonal τ with mild + alignment (ratio 0.646 > orthogonal 0.5,
  no cancellation) ⇒ souping is **mildly constructive, not destructive**. Predict soup ≈ the **top tier**:
  **point 72, range [66, 82]** (≈ tied with swezero/rebench). **Falsifier:** soup < mean-of-arms (58) ⇒
  destructive interference (would contradict the geometry); soup > 90 ⇒ strong super-additivity.
- **H2 (compositionality):** soup recovers **55–70% of the 131 oracle union (~72–92)**; **emergent**
  (none-of-arm) solves ≈ **0–3** (weight-averaging rarely invents capability); retains top-tier niches,
  partially loses coderforge/scale-unique. **Falsifier:** emergent >5, or union recovery >85%.
- **H3 (θ₀ baseline):** θ₀ ≈ base anchor 5% ⇒ **[20, 32]/500** (point 25). Implies per-arm SFT delta
  ~ **+48 top tier / +13 bottom tier**. **Falsifier:** θ₀ > 40 (base already strong).
- **H4 (pooled scaling):** monotonic in data ⇒ **pooled220k > pooled55k**; predict pooled55k [45,65],
  pooled220k [60,80]. **Falsifier:** pooled220k ≤ pooled55k.
- **H5 (soup vs pooled220k — CLEAN, data-matched: both = the 4×55k union):** joint-train ≥ merge ⇒
  **pooled220k ≥ soup** by a few. The *interesting* outcome = soup ≈/> pooled (merge competitive with
  joint-train, at a fraction of the cost). **Falsifier for "≥":** soup beats pooled220k by >5.

_locked 2026-07-25 04:22 UTC_

## Adversarial review (devils-advocate session, 2026-07-25)
A 4th dpf-invoked session stress-tested the souping science. Verdicts:
- **#1 (confound amplification):** RETRACTED along with mine — pooled220k IS the data-matched union (above).
- **#2 (null-zone):** VALID + accepted. swezero(77)≈rebench(70) is n.s. (p=0.52); soups will differ ~1–5/500
  = **sub-noise on single-rollout×500**. ⇒ uniform4 is the pivotal cheap probe; **hold C2/C3/C4/LOO** unless
  uniform4 clears 77 *significantly*. Don't grow a cheap probe into a search that optimizes noise.
- **#3 (prior mismatch):** VALID. Classic Model Soups = same-data/diff-hyperparam; here = **diff-data-source
  = task arithmetic**, where destructive interference is common. High 1-arm-only diversity (68/131) is
  *good for ensembles, bad for weight-averaging*. ⇒ tempers my geometry-based "uniform4≈top-tier" prior.
- **COMPETING PRE-REGISTRATION for H1 (devils-advocate):** uniform4 will be **statistically
  indistinguishable from swezero** (Δ<~15/500 unresolvable at 1 rollout×500); direction uncertain, interference
  could put it **≤ swezero**. Note this AGREES with my H1 *number* (~72≈77) but reframes the *meaning*: a "hit"
  on H1 does NOT show souping helps — only that it doesn't hurt. Data adjudicates.
- **POWER FLOOR (whole board):** single temp-0 rollout × 500 cannot resolve effects <~15/500. Most head-to-heads
  here are underpowered. Fixes: more seeds / temp>0 on key contrasts, or report paired-McNemar + "indistinguishable"
  honestly. Flagged as a top question for dpf before spending more A100-hours on sub-noise comparisons.
- **Pairing rigor (soup-worker):** uniform4/perfw4 (all-4 = 220k) ARE data-matched to pooled220k; top2 (swezero+
  rebench = 110k) is NOT ⇒ compare top2 vs arms/soups only, never vs pooled220k.

## Interpretation (so far)
The headline is **not** a score — it's that ADP-normalized sources induce **localized, complementary**
capabilities, and that the benchmark has a **shared blind spot (sphinx)**. The scientifically decisive
comparisons (compositionality of merge vs joint-train; whether combination recovers niches or interferes)
are pending the soup/pool/θ₀ results and are wired to compute automatically.

_generated 2026-07-25 04:12 UTC_

## Pre-registered hypotheses — TASK-ARITHMETIC / SUBTRACTION (H6–H8, LOCKED before any subtraction eval)
Family: theta = theta0 + alpha*s + sum_i beta_i*r_i, with s=mean(tau) (shared agentic-SFT axis, 64% of each
arm's norm; aligned in embeddings=format + body-skill), r_i = tau_i - s (arm residual, ~76% of norm; the 4 form
a symmetric simplex, cos(r_i,r_j) = -1/(N-1) = -0.33 -> independent niche directions). Known points: base
(a=0,b=0), arm_i (a=1, b_i=1), uniform soup (a=1, b=0). These lock the UNMEASURED points. (author: main-worker)

- **H6 (s = the general agentic-coding axis) — theta0 + alpha*s sweep.** Predict monotone rise alpha:0->~1.55
  (norm-matched to a single arm; 1/0.646), plateau/slight decline by alpha=2 (overshoot). FREE anchors: alpha=0
  = theta0 (init eval running), alpha=1 = uniform4 (running). NEW striking point: **alpha=-1 (negation) -> BELOW
  base** (subtracting s un-does the agentic SFT). Falsifier: alpha=-1 >= theta0, OR alpha=2 beats alpha=1.55 by
  >5/500 (no overshoot). [overlaps soup-worker's positive-alpha sweep -> coordinate, don't duplicate]

- **H7 (niches live in the residual r_i) — theta0 + r_i (niche isolation).** Predict OVERALL << arm_i (shared
  lift removed) -> global resolved in [theta0, theta0+10]. BUT on arm i's niche repos (rebench->sympy,
  scale->sklearn) retains a DISPROPORTIONATE share vs the wrong-residual control theta0 + r_j. Falsifier:
  theta0+r_i ~= arm_i overall (shared lift wasn't needed), OR niche retention no better than theta0+r_j (niches
  not in residual). CAVEAT: theta0+r_i may emit degenerate rollouts (missing format-s in embeddings) ->
  variant H7' keeps embedding-component of s, zeros only body-s, as the cleaner "niche-skill-in-body" test.

- **H8 (compositional niche union) — theta0 + s + r_i + r_j (2-niche targeted merge).** A point UNREACHABLE by
  any equal-weight soup (soup over all 4 cancels residuals: sum r = 0). Predict: keeps shared lift AND unions
  arm i+j niches -> BEATS uniform soup on {niche_i U niche_j} repos, ~= or > max(arm_i,arm_j) there. The
  weight-space compositionality UPSIDE test. Falsifier: <= uniform soup on the union repos (niches don't add in
  weight space). Gate H8 on H7 first confirming residuals carry niches.

_locked 2026-07-25 (subtraction program; pending dpf scope sign-off + soup-worker coordination on merge tooling)_

### EVAL-METHOD REFINEMENT for H6-H8 (power-floor fix, per soup-worker 2026-07-25 19:30) — pre-data
The original H6-H8 wording implied *aggregate /500* readouts. That is UNDERPOWERED: the soup/scale
spread already sits near the ~15/500 single-rollout noise floor, so a residual-isolation effect measured
on all 500 would be swamped. Corrected measurement design (predictions/directions UNCHANGED — this
sharpens the readout, not the hypothesis):
- **H7 (theta0+r_i niche isolation):** primary readout = resolved-rate on **arm i's OWN unique-solve set**
  (its resolved_ids uniques: swezero 29, rebench 25, scale 7, coderforge 7), NOT /500. Control = theta0+r_j
  (wrong residual) on the SAME instance set. Effect is large there, negligible in aggregate.
- **H8 (theta0+s+r_i+r_j union):** readout on {unique_i U unique_j}; compare vs uniform soup on the same set.
- **Gating:** all H6-H8 builds gated on the uniform4 (=theta0+s) result — if uniform4 ~= arms (shared s
  carries the lift), residual-isolation upside is low-prior. EXCEPTION: **H6 alpha=-1 negation runs regardless**
  (it tests whether s is the general agentic axis, independent of niche structure).
- **Tooling:** builds via soup-worker's build_taskvec.py (canonical);
  taskvec_merge.py (main-worker, independent impl) kept ONLY as an arithmetic
  cross-check. Targeted-subset eval via paired_compare.py --subset <id-list> (soup-worker adding).
_refined 2026-07-25 19:xx UTC_

### REVISION of H6-H8 (adversarial review by devils-advocate, 2026-07-25 19:33) — SUPERSEDES the above
devils-advocate raised three valid objections; accepted. The subtraction program is cut from ~4 evals to
~1 eval + free analysis:
- **H6 alpha=-1 negation: DROPPED.** Dominated — a monotone alpha>=0 sweep entails the sign (no new bit),
  and a collapse can't be attributed to s's *semantics* without a norm-matched random-direction control
  (theta0 - random_||s||) that wasn't budgeted; without it 'below base' only means 'large perturbation breaks
  the model'. (Retracts the earlier "run alpha=-1 regardless" note.)
- **H7 theta0+r_i: REFORMULATED to a FREE marginal test, raw build DROPPED.** theta0+r_i is confounded — it
  deletes the shared competence s (base SWE skill + format), so it may solve ~nothing incl. its own niche,
  conflating 'niche not in residual' with 'removed the base'. Correct question = the marginal contribution of
  r_i OVER the shared model. KEY IDENTITY: theta0+s+r_i = arm_i exactly (s+r_i=tau_i), so the marginal test
  (theta0+s+r_i)-(theta0+s) = **arm_i vs uniform4**, analyzed PER-REPO on arm i's niche/unique-solve set. This
  needs ZERO new builds/evals (arm evals done; gated only on uniform4). Do it the moment uniform4 lands.
- **H8 theta0+s+r_i+r_j: KEPT as ONE eval, REFRAMED 'upside' -> 'interference measurement'.** Residuals are
  anti-correlated by construction (sum r=0 -> cos(r_i,r_j) = (0.22-0.646^2)/0.583 = -0.34 = -1/(N-1) floor), so
  ||r_i+r_j|| = 0.879*tau_bar = only 1.15x a single residual; each niche keeps ~66% of its solo projection.
  Predict theta0+s+r_i+r_j <= best single-niche arm on {unique_i U unique_j}. Value = a clean citable NEGATIVE
  ('weight-averaging anti-correlated niche vectors != MoE routing'); the 66% retention leaves it empirically
  open whether that suffices, so worth one eval to demonstrate rather than assume.
- **Gate:** all new builds gated on soup-worker's alpha-sweep {1.0,1.55,2.0} showing capability responds to the
  s-direction norm at all; if flat, residuals carry no resolvable capability and H8 is moot before a GPU-hour.
_revised 2026-07-25 19:xx UTC (main-worker, accepting devils-advocate)_

### Contamination robustness of the arm ranking (output-side check, 2026-07-25) — RANKING SURVIVES
agent-b flagged train/eval overlap: coderforge trained on 46 matplotlib SBV-format instance-ids; swezero on
5 SBV repos as workspace envs (astropy/seaborn/xarray/sphinx/sympy); scale+rebench clean. Output-side check
(per-arm resolved_ids stratified by repo; contaminated repo union = matplotlib/astropy/seaborn/xarray/sphinx/sympy):

| arm | full/500 | clean-only | on OWN-contaminated repos |
|---|---|---|---|
| swezero | 77 | 69 | 6 (astropy 3, sympy 3; 0 seaborn/xarray/sphinx) |
| rebench | 70 | 60 | 0 (uncontaminated) |
| coderforge | 48 | 41 | 0 matplotlib (46 training ids -> 0 solves) |
| scale | 35 | 32 | 0 (uncontaminated) |

**Conclusion: contamination does NOT explain the ranking.** Clean-only order+gaps preserved (69>60>41>32 = same
as 77>70>48>35). coderforge solved 0/its-46 leaked matplotlib instances; swezero solved 0 on 3 of 5 contaminated
repos incl. 0/44 sphinx despite sphinx training. Own-contamination benefit <= ~6/77 for swezero, ~0 for
coderforge. If anything the contaminated repos are where arms are WEAK -> opposite of an inflation confound. The
swezero lead rests on CLEAN repos (django 53/231, sklearn, pytest, pylint, requests — none in any subset). Niche
claims rebench->sympy and scale->sklearn are on uncontaminated arms -> unaffected. CAVEAT: repo-level; agent-b's
instance-level exact-match pending, but coderforge 0-matplotlib implies even exact leak didn't help. (main-worker)

### CORRECTION to the contamination table above (parser bug, 2026-07-25) — use THESE numbers
My first pass parsed repo as the ORG (before "__"), so it missed pydata__xarray / sphinx-doc__sphinx /
mwaskom__seaborn (counted them clean). Fixed to package-level (part after "__"); independently reproduced by
devils-advocate — numbers agree exactly.

| arm | full/500 | decontaminated (drop matplotlib/astropy/seaborn/xarray/sphinx/sympy) | on-contam breakdown |
|---|---|---|---|
| swezero | 77 | 66 | astropy 3, matplotlib 2, xarray 3, sympy 3 |
| rebench | 70 | 58 | astropy 1, matplotlib 1, xarray 2, sympy 8 |
| coderforge | 48 | 41 | astropy 3, sympy 4 (matplotlib 0!) |
| scale | 35 | 31 | matplotlib 1, xarray 1, sympy 2 |

Decontaminated re-rank = swezero 66 > rebench 58 > coderforge 41 > scale 31 — SAME order + tiers as full.
**Arm-differential confound test (the correct framing — contamination is arm-specific, only swezero+coderforge):**
swezero(contam) − rebench(clean): full +7 → decontam +8; coderforge(contam) − scale(clean): full +13 → +10.
=> contaminated arms KEEP their margin over clean arms on decontaminated instances => ranking is NOT a
contamination artifact. coderforge solved 0 of its 46 leaked matplotlib instances (repo-familiarity, not
memorization). rebench's sympy 8 is a genuine clean niche. Supersedes the 69/60/41/32 numbers above. (main-worker)

### BREAKTHROUGH-RELEVANT: the complementarity is INSTANCE-level, not repo-level (free, resolved_ids, 2026-07-25)
The +54 gap (oracle union 131 vs best-single swezero 77) is the prize. WHERE does it live?
- **per-repo ORACLE router** (route each repo to its best arm) = **83/500** -> captures only **6/54 = 11% of the gap.**
- **intra-repo diversity lost to repo-routing = union - router = 48** -> ~89% of the complementarity is arms
  solving DIFFERENT SPECIFIC INSTANCES WITHIN THE SAME REPO, not different repos.
=> The "obvious" MoE (route by repo/language) is nearly worthless here (77->83). The specialization map
  (rebench->sympy etc.) is real but explains almost NONE of the complementarity. Capturing the 54 needs
  per-INSTANCE routing or ensemble DISTILLATION, not repo-routing and not souping (which cancels residuals).
- **Per-source marginal value (Shapley over the 4 resolved-sets, sums to 131):** swezero 48.2 / rebench 42.8 /
  coderforge 22.8 / scale 17.2 (LOO-unique 29/25/7/7). Order matches raw score but COMPRESSED; even worst arm
  (scale) contributes 17.2 irreducible -> no source is redundant. Complementarity doesn't flip the ranking but
  confirms every source carries unique capability that any single-model or soup approach loses. (main-worker)

### Complementarity STRUCTURE + ROUTING PREDICTABILITY (free analyses, 2026-07-25) — routing NOT viable
Script adp_analysis_taskAB.py (SWE-bench_Verified local cache; sklearn to
pylibs). Solve-dist: 0->369, 1->68, 2->35, 3->20, 4->8; union 131 (sanity OK).

**A. What the complementarity is ABOUT = gold-patch DIFFICULTY, not patch type.**
- unsolved (369): patch mean 17 lines, 2.8 hunks, 17% multi-file (the frontier = LARGE/multi-file patches).
- all-four-easy (8): smallest/simplest/shortest-spec (mean 6 lines, 1.1 hunks, 0% multi-file).
- single-arm (68): in between — small patches but LONGER problem statements ("solvable but finicky").
- Per-arm unique profiles differ: rebench = terse-spec small multi-file bugfixes; swezero+scale = long-spec;
  scale distinctively longest problem statements (median 2399 chars) + largest patches -> weak overall but a
  few unique high-verbosity/large-diff wins. (coderforge/scale n=7 -> noisy.)

**B. Per-instance ROUTING from problem text is NOT viable.** TF-IDF(problem_statement)->solver-arm,
LogReg, 5-fold CV over the 131 solvable: correct-solver acc = 0.573 => routed-resolved = **75/131**, which is
BELOW always-route-swezero (77) and == best-single. Classifier collapses to the majority arm; the prompt
carries almost no signal about WHICH arm will succeed.
=> Combined with (soup cancels residuals) + (per-repo router captures only 11%, =83), **NO cheap inference-time
or weight-merge method captures the +54.** The deep complementarity is instance-scattered AND prompt-illegible
=> it can only be combined at TRAINING time (joint-train/pooling, or off-SBV RFT). This elevates pooled220k as
THE test of whether training-time combination captures what merge/route cannot. (main-worker + subagent)

### M1 sharpened -> a STRUCTURAL merge-vs-(train/route) decision rule (soup-worker + main-worker, 2026-07-25)
Prior art: "low task-vector interference => merge works" = task-arithmetic (Ilharco); TIES/DARE/RegMean/
Model-Stock reduce/measure interference for merging. **Novel, writeup-worthy piece:** the residual
anti-correlation is FORCED, not empirical. For N fine-tunes of a shared base, r_i = tau_i - mean(tau) sum to
zero by construction => mean pairwise cos(r_i,r_j) = -1/(N-1) (exact when ||r_i|| equal, as here ~11.3-11.7).
=> equal-weight merging of same-base fine-tunes PROVABLY captures only the shared component s; residual
complementarity is unmergeable by averaging and must be ROUTED or TRAINED-IN. Turns task-vector geometry into
a merge-vs-(train/route) DECISION RULE. And routing is empirically dead here (text-router 75<77, repo-router
83), so for THIS campaign the rule collapses to merge-vs-TRAIN-IN => pooled220k is the pivotal test.
CAVEAT (soup-worker): the STRENGTH (how much capability lives in residuals vs s) is the empirical, N=4-single-
datapoint part (here large: residuals 76% of norm, +54 complementarity). Don't over-generalize the magnitude
without more arm-counts / base-models. (This is the candidate paper headline.)

### M1 CORRECTION (devils-advocate + soup-worker, 2026-07-25) — retract "headline"; it's a CASE STUDY
The "forced-direction decision rule" I headlined above is a TAUTOLOGY: cos(r_i,r_j) = -1/(N-1) holds for ANY
mean+residual decomposition of same-base vectors (never varies with data) => it cannot predict anything.
Retracted as a novel rule. The genuinely data-dependent quantity is the pairwise cosine c of the TASK VECTORS
(||s||/tau_bar = sqrt((1+(N-1)c)/N)): high c => small residuals => merge~=arms; low c => residuals dominate =>
merge loses them. But that ~= prior art (Model-Stock: weight-angle for merge decisions; task-arithmetic/TIES/
DARE: measure/reduce interference). And even c predicts NORM survival, not CAPABILITY survival (hole-1) — only
the eval (does theta0+s ~= arms?) decides, not geometry. HONEST FRAMING = a case study: "for these 4 same-base
SWE arms, low pairwise cos (0.22) + instance-scattered complementarity => equal-weight merge recovered only the
shared skill; routing/joint-train needed for the rest — consistent with Model-Stock / task-arithmetic
interference theory, N=4." NOT a universal law. Would BECOME a rule only with geometry->outcome replicated
across multiple arm-counts / task-sets (future work).
ROUTING-DEAD calibration: the B result shows "no CHEAP inference-time TEXT router helps" (TF-IDF collapses to
majority; 75<77). It does NOT prove complementarity is unrouteable — a stronger router (embedding features,
model logprob/confidence, cross-encoder) could beat 77; that's off the critical path. Claim scoped accordingly.

### SPHINX 0/44 IS A HARNESS BUG, not a capability gap (M2 diagnostic, 2026-07-25) — EVAL-VALIDITY
Read-only subagent over runs/score_v2_<arm>_4b/reports_*/sphinx-doc__sphinx-*/. Verdict = (b) harness artifact
(devils-advocate predicted this from 'swezero trained on sphinx yet 0/44').
- 145/147 arm x instance emit a git_patch; 141/147 APPLY cleanly; 0/147 resolved.
- ROOT CAUSE: sphinx runs tests via `tox --current-env -epy39`; that output has FAILED lines but ZERO 'PASSED'
  markers (passing tests appear only in the --durations table). SWE-bench's pytest parser credits a test only on
  an explicit PASSED token => every sphinx PASS_TO_PASS + FAIL_TO_PASS is scored FAIL => resolved is
  MATHEMATICALLY IMPOSSIBLE for any sphinx instance regardless of patch quality.
- CONFIRMED masked solves: sphinx-8475 under BOTH rebench and scale (patch applies, real pytest '18 passed / 0
  failed' incl. the target test, scored resolved=False); coderforge sphinx-8269 (6 passed/0 failed, unresolved).
  >=2 airtight; ~36/141 applied instances show real 0-failures yet 0 resolved.
IMPLICATIONS: sphinx is NOT a capability frontier / data-curation target (my earlier M2 framing was WRONG).
Treat the 44 sphinx as UNSCORED => honest denominator ~456. Arm scores are UNDERSTATED by each arm's true sphinx
solves (rebench + scale >=1-2 each confirmed; swezero's 8475 was a real miss). AFFECTS ALL PENDING EVALS
(uniform4/pooled/soup) identically unless the parser is fixed. Proposed fix: parser credits tests from the tox
--durations table (or force explicit PASSED output), then re-score the 44 sphinx instances for all arms.
Blast-radius on shared kit -> coordinate before patching. Diag: sphinx_diag.py (main-worker+subagent)

### SPHINX finding — VERIFIED + SCOPE-CORRECTED (soup-worker challenge, 2026-07-25)
soup-worker asked: is the sphinx mis-score in my triage script or the OFFICIAL swebench-eval parser? VERIFIED
against ground truth (sphinx-doc__sphinx-8475 / rebench): official report.json = resolved:False,
patch_successfully_applied:True, FAIL_TO_PASS 0/1 fail, PASS_TO_PASS 0/17 fail (all 18 tests marked failure);
test_output.txt = "18 passed, 673 warnings in 3.02s". So the OFFICIAL per-repo parser scored a genuinely-passing
cleanly-applied patch as 0/18 resolved=False => real UPSTREAM sphinx-parser bug (sphinx tox output lacks the
per-test PASSED markers the parser credits), NOT the kit's scripts, NOT "sphinx genuinely hard".
SCOPE CORRECTION (retract my earlier "affects the board"): the mis-scoring is bounded to sphinx (44 inst, one
repo) and is UNIFORM across all arms/soup/pooled => it does NOT bias any COMPARISON (arm ranking, soup-vs-arms,
pooled-vs-soup all live in django/clean repos). It only understates each model's ABSOLUTE total by its true
sphinx solves (effective denom ~456; rebench+scale >=1-2 masked solves confirmed). => panel resolved_ids are
trustworthy for comparisons; honest fix = a <=44-instance absolute-score caveat + optional upstream-parser
sphinx re-score, NOT a board rescore. (main-worker, verified)

### SPHINX — final scope precision + CONVENTION (devils-advocate, 2026-07-25)
Precision fix to my "uniform => no comparison bias": not strictly true. The parser zeroes ALL sphinx, but each
model's MASKED true-sphinx solves differ (rebench/scale >=1-2; swezero possibly more, it trained on sphinx) =>
a small comparison bias FAVORING models worse at sphinx. Bounded by the true-sphinx-competence differential
(~a couple instances) => SUB-NOISE (< +-8/500), practically negligible but not exactly zero.
CONVENTION (adopted): **primary board = the 456 non-sphinx instances** (drop the mis-scored repo uniformly for
all models) — exact, removes BOTH the absolute deflation AND the residual bias. full-500 reported secondary with
a "sphinx (44) unscored: upstream swebench parser under-credits sphinx tox passes" footnote. Optional: re-score
the 44 from test_output.txt directly, and a one-line upstream SWE-bench bug report (outward-facing => dpf's call).
Sphinx thread CLOSED. (main-worker + devils-advocate + soup-worker)

### ★ uniform4 RESULT — pre-registered H1/H2 SCORED (2026-07-25) — H1 FALSIFIED (soup significantly WORSE than arms)
uniform4 (equal-weight soup = theta0+s, alpha=1) = 50/500 · 50/456 non-sphinx (PRIMARY) · 36/301 clean. Panel
(soup-worker, McNemar+TOST, 456): vs swezero -27 p=0.0016 SIG(worse); vs rebench -20 p=0.025 SIG(worse); vs
coderforge +2 p=0.89 n.s.; vs scale +15 p=0.067 n.s. => BOTTOM tier, significantly below both top arms.
- **H1 [66,82] pt72: FALSIFIED** — actual 50 << 66; soup did NOT tie/slightly-trail the arms, it dropped to the
  bottom tier significantly below swezero(77)/rebench(70). My locked range was too optimistic.
- **Geometry DIRECTION confirmed, MAGNITUDE underestimated:** decomposition predicted merge keeps only s (0.646x
  norm), drops ~76% residual => 'tie-to-below'. Direction right (soup<arms, significant), but the drop is bigger
  than the norm-ratio implied: s alone => ~50, so residuals carry MORE than half each arm's capability. (The <66
  outcome does NOT contradict the geometry as H1's falsifier-label implied — it CONFIRMS residual-drop, more strongly.)
- **H2 (union recovery 55-70%): FALSIFIED low** — 50/131 = 38%; soup doesn't even retain best-single 77.
  (emergent none-of-arm solves pending resolved_ids overlap; expect ~0.)
- **devils-advocate competing pred ('indistinguishable from swezero'): direction right (<=swezero), 'tie' WRONG**
  — significant loss (p=0.0016), resolvable not within-noise.
NET: clean, significant, citable NEGATIVE — naive equal-weight souping of diverse same-base SWE arms SIGNIFICANTLY
underperforms the best single arm; residual capability is lost. M1 qualitative story confirmed w/ significance; my
H1/H2 quantitative ranges too optimistic. NEXT: gated alpha-sweep (1.55, 2.0) adjudicates norm-deficit (recoverable)
vs residual-drop (unfixable) — prior leans residual-drop. (main-worker)

### ★★ SFT-LIFT reframe — H3 FALSIFIED, verdict INVARIANT (2026-07-25, agent-b θ₀ rescore)
Scaffold parity PASS (init vs 4 arms byte-identical llm_config except model/tokenizer: temp0, max_input28000,
max_output2047, max_iter500, condenser, qwen3_coder parser, SDK1.27). Arms confirmed single-run pass@1; only
θ₀ had the spurious best-of-3 union (=145, a pass@3 artifact from θ₀'s lone infer TIMEOUT + catch-up). HARD
BOUND from patch comparison: single-run θ₀ in [70,145]/500, LB **70 CERTAIN** (byte-identical-patch resolves).
Clean v2_init_singlerun_4b merging ~1-2h; soup-worker runs θ₀-vs-arms-vs-soups McNemar+TOST (456/clean-301) then.
- **H3 (θ₀≈5% => [20,32], SFT delta +48top/+13bottom): FALSIFIED, invariantly.** θ₀ >= 70 >> 32. The "~5% base"
  was NEVER file-backed — an assumption. Measured single-run base is ~3x+ that.
- **VERDICT (invariant over θ₀∈[70,145]): SFT-lift is NULL-to-NEGATIVE.** No arm significantly beats base;
  2/4 (coderforge 48, scale 35) significantly BELOW base. Even at LB θ₀=70: swezero +7 (n.s., cf swezero-rebench
  +7 p=0.52), rebench <= base, coderforge/scale inverted.
- RETRACTS the campaign's assumed headline ("arms lift base 5%->77, 3/4 clear the paper bar"). The unmeasured
  baseline was the load-bearing assumption; measuring it collapses the lift. Arm scores (77/70/48/35 single-run)
  stand — only the baseline they're compared against changed. This is now the campaign's most important (and
  most rigorous) finding: on SWE-bench Verified, these ADP SFT arms do NOT beat the base instruct model.
  (main-worker scoring H3; agent-b rescore + scaffold check)

### θ₀ STILL UNRESOLVED — downgrading the "invariant null-to-negative" verdict (2026-07-26) — ⚠️ SUPERSEDED
_⚠️ SUPERSEDED by "θ₀ RESOLVED" below: θ₀=119 is provenance-verified clean single-run base (500 distinct / one-per-instance / base snapshot 851bf6e8). My wrong-model/mislabel hypothesis is **WITHDRAWN** — a mislabeled arm would OVERLAP that arm heavily, not yield 55 no-arm solves, and base-unique solves are in-family (arms have 68/131 single-arm-uniques among themselves). Kept below for the audit trail. — main-worker_
Second θ₀ attempt v2_init_singlerun_4b merged = 119/500, but it FAILS agent-b's wrong-model diagnostic: θ₀
solves **55 instances NO arm solves** (46% of its solves base-unique; dist by #arms-also-solving {0:55,1:26,
2:19,3:13,4:6}) — implausible for a true base vs its own 4 SFT descendants under byte-identical scaffold. Likely
still mislabeled/mixed rollouts (same class as the 145 pass@3 union). So **BOTH θ₀ numbers (145, 119) are
artifacts.**
CONSEQUENCE: the earlier "θ₀∈[70,145], LB 70 certain ⇒ SFT-lift null-to-negative, INVARIANT" is **DOWNGRADED**.
The LB 70 (byte-identical-patch resolves) is itself provenance-contingent — if init rollouts are source-mixed, a
"byte-identical" patch may literally BE an arm's patch mislabeled, so it doesn't prove base solves it. Net:
**no trustworthy θ₀ baseline exists yet; the SFT-lift DIRECTION (positive/null/negative) is UNRESOLVED** pending
a provenance-verified single base rollout (agent-b's lane).
What IS solid regardless: "~5% base" was never measured AND no paired arm-vs-base McNemar was ever run, so the
"SFT-lift significant" headline was never established in EITHER direction. (main-worker — catching my own prior
over-claim via the wrong-model cross-check; calibration #5.)

### θ₀ RESOLVED — v2_init_singlerun_4b PROVENANCE-VERIFIED = true single-run base 119 (2026-07-26, soup-worker) — SUPERSEDES the 02:03 "θ₀ UNRESOLVED" downgrade
The 02:03 downgrade (θ₀=119 "fails wrong-model diagnostic" via 55 base-unique solves) is **retracted** — the provenance check settles it:
- `out_v2_init_singlerun_4b/combined/output.jsonl` = **500 rollouts / 500 distinct / 0 dup ids → exactly ONE rollout per instance** (NOT a best-of-N union). Model metadata = all `adp-eval-v2_init_4b__sKKof10` init endpoints; agent-b's identity check = base snapshot 851bf6e8 (no arm/pooled ckpts). ⇒ NOT a union, NOT mislabeled-arm rollouts.
- **The "55 base-unique ⇒ wrong-model" argument is refuted** (devils-advocate, concurred): a mislabeled ARM would OVERLAP that arm's resolved set heavily, not produce 55 solves NO arm has. And base-unique solves are EXPECTED here, not anomalous — the arms already have 68/131 single-arm-unique among themselves (high temp-0 cross-model diversity), and base genuinely out-solves the SFT'd arms on repos they forgot (sympy). 55 base-unique is in-family.
- **83↔119 reconciled:** 119 = init rollouts scored STANDALONE; 83 = the subset where init's patch won the union's *longest-patch* tiebreak; the "instruct 36" were instances init ALSO solved but instruct's patch was longer (⊂ 119); union 145 = init's 119 ∪ 26 initcatch-only. So 83+36=119 is arithmetic, not an init+instruct union.
⇒ **θ₀ = 119/500 is the verified single-run pass@1 base. The SFT-lift direction is now RESOLVED (below), not unresolved.** (soup-worker ran the decider; devils-advocate closed it 02:14.)

### ★★ SFT-LIFT VERDICT (verified base) + COMPLETE SOUP INVESTIGATION (2026-07-26, soup-worker) — the campaign's bottom line
**Base θ₀ (single-run, verified) = 119/500 · 119/456 (26.1%) · 76/301 (25.2% clean).** Keep the two SFT findings SEPARATE (per-board; devils-advocate's framing) — the naive full-500 "+42 base≫arms" conflates them:
1. **No significant general SFT lift.** clean-301 (fair general-capability board): θ₀ 76 vs swezero 66 = **+10, McNemar p=0.30, n.s.** django (231 inst, 46% of SBV, clean): θ₀ 51 ≈ swezero 53 (tie). State as "no significant lift," not equivalence (underpowered), not "base wins."
2. **Repo-specific CATASTROPHIC FORGETTING** on repos the base already knew: **sympy θ₀ 29/75 → swezero 3/75** (swezero TRAINED on sympy yet collapsed), sklearn 14→6. Provenance-confirmed base ⇒ genuine forgetting, not labeling. This is where the full-456 "θ₀ vs swezero +42 (p=2e-4)" actually comes from — mechanism (2) leaking into the aggregate, NOT a general-capability gap.
Calibration: base ≈ 24-26%, ~5× the never-measured "~5%" anchor — the whole "SFT lifts 5%→15%" premise was an artifact of an unmeasured baseline.

**COMPLETE combine-strategy board (raw /500, all single-run pass@1, official swebench, 500-complete):**
```
base θ₀ (single-run)   119    ← ceiling
swezero (best arm)      77
rebench                 70
─ soups (soup-worker) ─
top-2 (drop scale+cf)   62     best soup
α=0.7                   54
uniform4 (α=1.0)        50
α=1.55                  38
α=2.0                   19     collapse
   arms: coderforge 48, scale 35 interleave the weak soups
```
- **Every soup is SIGNIFICANTLY below base (all p<1e-4) AND below the best arm** (top2 vs swezero −19 clean-301 p=0.014; a0p7 vs swezero −23/456 p=0.0095). uniform4 vs swezero −27/456 p=0.0016.
- **α-sweep is MONOTONE DECREASING** (α0.7=54 → 1.0=50 → 1.55=38 → 2.0=19): capability *decreases* with ‖soup-τ‖. ⇒ the merge deficit is **RESIDUAL-DROP, not a norm deficit** — averaging keeps only the shared direction s (0.646× arm norm) and discards the ~76% residual where arm-specific capability lives; rescaling s can't recover it and overshoot (α>1) adds incoherence. Hole-1 resolved.
- Dropping the weak/wrong-direction arms (top-2, and scale has the largest ‖τ‖ but worst score) helps most among soups (+12 vs uniform4) but does NOT rescue merging.

**NET CAMPAIGN VERDICT (soup + base sides):** on SWE-bench Verified, ADP-v2 SFT on this raw data gives **no general lift over the base instruct model** and **systematically degrades the base across its domains (base ≥ best arm on 9/10 repos; concentrated in sympy/sklearn)**; **weight-merging the arms is strictly a loss** (every soup significantly below base and best arm, via residual-drop). The only combine method not yet scored is **pooled220k (joint-train)** — the last candidate that could beat base. Real remaining research question (per devils-advocate): does **quality-filtered** data lift over the measured ~24% base? (Data curation, not post-hoc merging.)

### ★★ θ₀ "~5%" ANCHOR PROVENANCE — RESOLVED (2026-07-26, Babel-side agent) — H2 (HARNESS) CAUSAL, H1 NON-CAUSAL
The Babel-side investigation (findings on branch `babel-provenance-findings`) resolves the "~5%" anchor:
**it is a HARNESS artifact (hypothesis H2), not the base-vs-instruct mismatch (H1) earlier suspected.**
- The 25/500 (5.0%) was Graham's June-2026 eval of **Qwen3.5-4B-*Base***. But the **same base weights**,
  re-scored in-house under **this campaign's harness** (`base4b` / `score_base4b254`, provenance-verified from
  shard metadata `custom_tokenizer=Qwen/Qwen3.5-4B-Base`), score **53/362 scored ⇒ ≥10.6% of 500 (hard floor),
  ~16% ex-sphinx** — ~2–3× the 5%. The gap is **patch production, not capability**: non-empty-patch rate 69.6%
  (this harness) vs 19.2% (Graham's) on identical weights; Graham emitted empty patches 81% of the time. Leading
  mechanism (hypothesis): the `cp_testbed_repo` 30s→600s timeout fix that Graham's run predates.
- **H1 is *nominally* true but *non-causal*:** the anchor really is the base (non-instruct) model and the arms
  are instruct-init (a real but secondary methodological defect), yet base-vs-instruct cannot explain 5%→~24%
  because the base itself clears ≥10.6% under the right harness.
- **H3/H4 false.** "paper-nonweb ≈ 52/500 (10.4%)" is a real in-house **Base-init** SFT-checkpoint eval
  (`papernonweb1154`), not the ADP paper's number.
- **No valid Babel instruct anchor:** `rawinstruct4b` died at 13/47 scored (survivorship-biased) ⇒ the FAIR
  rerun's θ₀ = 119/500 (~24% instruct) is the only trustworthy untrained-instruct baseline. (Bonus:
  `swesmithinstruct540` has a *final* 82/500 = 16.4% on disk vs the stale RESULTS.md partial "20/96".)
- **Verdict UNCHANGED (anchor = instruct θ₀ = 119):** the SFT-lift still **inverts** against the arms' *actual* init — **instruct θ₀ = 119/500** — with all four arms (77/70/48/35) far below (best-arm −42). ⚠️ Keep THREE models distinct; do NOT conflate them: **(1)** Graham's Qwen3.5-4B-*Base* + broken harness = 5% (empty-patch artifact — explains the bad anchor, not a capability number); **(2)** Qwen3.5-4B-*Base* (non-instruct) under THIS harness ≈ 73/500 — a real RAW-BASE number, relevant ONLY to explaining Graham's 5% (same base weights, better harness), **NOT θ₀**; **(3)** instruct θ₀ = 119 (851bf6e8) = the arms' init and the SOLE verdict anchor. The ≈73 base figure must not re-anchor the SFT-lift (arms 77 vs base ≈73 would be a mild *lift*, not an inversion — wrong model class; that conflation is exactly the base-vs-instruct slip this section warns against). *Aside (soup-worker):* the arms (best ≈77) land near the raw-base ≈73 — ADP SFT dragged the instruct init (119) DOWN to roughly raw-base level. (main-worker, folding in the Babel-side
  agent's findings; supersedes the earlier "H1 confirmed" reading.)

### ★ MULTI-REPO FORGETTING — systematic, not sympy-only (2026-07-27, devils-advocate; base = init_singlerun vs arms resolved_ids, per repo)
Resolves DA's "n=1 repo" concern. base = verified single-run θ₀ (119); "best arm" = per-repo max over the 4 arms.

| repo | base | arms (sw/re/cf/sc) | base − best-arm |
|---|---:|---|---:|
| sympy | 29 | 3/8/4/2 | +21 |
| scikit-learn | 14 | 6/5/2/5 | +8 |
| xarray | 7 | 3/2/0/1 | +4 |
| pytest | 6 | 4/3/3/2 | +2 |
| astropy | 4 | 3/1/3/0 | +1 |
| requests | 3 | 2/2/0/0 | +1 |
| matplotlib | 3 | 2/1/0/1 | +1 |
| pylint | 1 | 1/1/0/1 | 0 (tie) |
| flask | 1 | 0/1/0/0 | 0 (tie) |
| django | 51 | 53/46/36/23 | −2 (only repo an arm > base) |

- **base ≥ best-of-4-arms on 9/10 repos** (only django favors an arm, by +2) — conservative for base (vs the *best* arm per repo). Capability-neutral SFT would give ~50/50, not 9/10.
- **Aggregate forgetting mass = +38/500**, spread sympy(+21) → sklearn(+8) → xarray(+4) → pytest(+2) → … — NOT sympy-only.
- **Present on clean, never-trained repos** (sklearn +8, pytest +2, requests +1 — in no arm's training data) ⇒ genuine capability degradation, not a contamination/training-mix artifact.
- **Calibration:** most per-repo deltas sit within the ~15/500 single-rollout noise floor individually; the *resolvable* claims are the AGGREGATE (+38 full-500) and the DIRECTIONAL consistency (9/10). Frame as "systematic degradation across the base's domains", NOT "significant forgetting on each of N repos". This aggregate — not a general-capability gap — is the true source of the naive full-500 "+42 base≫arms". (devils-advocate analysis; folded in by main-worker)

### Forgetting is MULTI-REPO, not sympy-only (DA strengthening #1, soup-worker 2026-07-26)
Checked base-vs-arms per repo (free, resolved_ids) to test whether the "catastrophic forgetting" claim rests on sympy alone. It does NOT — it's a pattern across ≥3 repos (base beats the BEST of all 4 arms by ≥3):
```
repo          tot  base  best-arm  base−best  arm-union
sympy          75   29      8        +21         13
scikit-learn   32   14      6         +8         12
pydata(xarray) 22    7      3         +4          5
(django 231: base 51 ≈ best 53, tie — the bulk is unaffected)
```
- **base beats the arm-UNION on sympy (29 vs 13) and sklearn (14 vs 12)** — i.e. ALL FOUR arms lost capability the base had on these domains, not just the weakest; pooling every arm's solves still doesn't recover the base's count. Strong catastrophic-forgetting signature.
- **sklearn is a CLEAN (non-contaminated) repo** (no arm trained on it) → the forgetting is genuine domain capability loss, NOT a contamination/leakage artifact.
- Scope: forgetting is concentrated on specialized-library repos (sympy/sklearn/xarray); the bulk (django, 46% of SBV) is a base≈arm tie. So the honest wording is "SFT erodes pre-existing capability on multiple specialized domains the base already knew, while leaving the bulk unchanged" — a pattern (≥3 repos), not an n=1 sympy anecdote.
- Base-rate note: base = 119/500 = 23.8% ≈ **~24%** (standardized throughout).
