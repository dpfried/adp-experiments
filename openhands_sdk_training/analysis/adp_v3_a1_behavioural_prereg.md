# v3 A1 — behavioural readout, pre-registered

**Status: filed 2026-08-02 ~18:30 UTC, while arms 330140 (A1-tokmatch) and
330141 (A1-traj-only) are still training (~3h in of ~10–13h) and no A1 eval
exists.** Nothing below has been computed on A1 data.

Companion to the score-level prereg (`adp_v3_a1_preregistration.md`). That one
registers what A1's *resolve rate* must do. This registers what A1's *behaviour*
must do, measured with `traj_compare/extract_traj_stats.py` — the same tooling
and identical metric definitions used for the v2 board.

## Why bother

A1 is one number away from being uninterpretable. If it lands inside the noise
band (~15/500, itself a single-run lower bound once grader flake ~5–11/500 is
counted), the score alone will not say whether the curation did anything.

Behavioural rates are a **different and better-powered instrument** for that
question: they are 500-row population proportions, not per-instance graded
labels, so they are immune to grader flakiness and only mildly exposed to
generation nondeterminism. A curation change can therefore be shown to have
moved the policy even when it fails to move the score — and that is a real,
reportable result rather than a null.

## What v2 established (the baseline these predictions sit against)

Verified, all four v2 SFT arms, training data vs learned policy:

| arm | train verify% | eval verify% | train any-test% | eval any-test% | score |
|---|---|---|---|---|---|
| swezero | 0.1 | 6.2 | 0.7 | 16.6 | 77 |
| rebench | 59.5 | 67.9 | 88.0 | 100.0 | 70 |
| coderforge | 60.8 | 72.8 | 79.9 | 99.8 | 48 |
| scale | 53.6 | 70.3 | 80.1 | 99.8 | 35 |
| *(base, no SFT)* | — | 74.3 | — | 90.3 | 119 |

Two established facts: **demonstration style predicts learned policy (4/4)**,
and **learned policy does not predict score** (r(verify%, score) = −0.28 across
11 arms). Both A1 arms are built from swezero data, so both inherit the 0.1%/0.7%
demonstration baseline.

## Predictions

Registered before any A1 rollout exists. P1–P3 are the falsifiable ones.

**P1 — A1 arms stay in swezero's behavioural basin, not base's.**
Both A1-tokmatch and A1-traj-only will show `verified_after_edit` **< 25%** and
`ran_any_test` **< 45%** — far closer to swezero (6.2% / 16.6%) than to base
(74.3% / 90.3%). *Rationale:* both curate swezero data, which contains almost no
verification to preserve. Curation can drop examples; it cannot add behaviour
that is not demonstrated.
*Falsified if* either arm exceeds those thresholds — which would mean curation
can induce behaviour absent from the demonstrations, a genuinely surprising and
more interesting result than the score.

**P2 — the two A1 arms will differ from each other by less than they differ from
swezero.** |tokmatch − traj-only| on `verified_after_edit` will be **< 15pp**,
while both sit > 25pp from base. *Rationale:* they differ in token budget and
condensation handling, not in whether the demonstrations verify.

**P3 — `claims-success-never-verified` stays high (> 60%) in both arms.**
swezero is at 91.8%, base at 1.7%. This is the sharpest v2 discriminator and
should track P1.

**P4 (directional, weaker) — trajectory-only curation should move
`n_condensation` and median turns more than it moves any verification metric.**
That is the axis A1 actually manipulates. No threshold registered; recorded so
that a post-hoc "it changed what we expected" cannot be claimed retroactively.

## Decision rules, fixed in advance

1. **If P1 holds and A1's score is inside the noise band** → report as
   "curation changed neither policy nor score," and do not spend a second seed
   chasing the score. The behavioural null is informative and cheap.
2. **If P1 holds but A1's score moves outside the band** → the gain is *not*
   mediated by verification behaviour. Do not attribute it to "better
   trajectories" without a mechanism; look at localization and patch shape
   (`first_edit_turn`, `patch_files`, `n_files_edited`) instead.
3. **If P1 is falsified** (curation restored verification) → this is the
   headline regardless of score, and it makes "curate for verifying
   trajectories" a directly testable A2 arm. Note in advance that per v2 this
   would still **not** predict a score gain — the two are decoupled.
4. **Any A1-vs-v2-control behavioural comparison must be extracted in the same
   pass**, matching @agent-b17cac1e's same-scoring-pass discipline. Metric
   definitions are pinned by `extract_traj_stats.py` at the commit recorded
   below; if that file changes, re-extract both sides.

## Pinned

- Tooling: `analysis/traj_compare/extract_traj_stats.py` (+ `extract_train_stats.py`)
  as of branch `traj-compare-viz`.
- Metrics: `verified_after_edit`, `ran_any_test`, `n_test_runs`,
  `finish_claims_success`, `env_activated`, `first_edit_turn`, `n_condensation`.
- Comparators: swezero (77) and base (119, single-run) v2 rows, already extracted.
- Denominator caveat carried over: base's per-turn stats are over the 300/500
  rollouts that stored history; A1 arms must be checked for error records before
  comparing (`grep -c '"error"'`), since v2 base had 200 and swezero had 0.

---

## RESULT — scored 2026-08-04, after A1 landed

A1 result: base 119, swezero control 77, **A1-tokmatch 63, A1-maxpool 62**.
Lever A (condensation removal) failed; both curated arms are below their control.

| model | /500 | verify-after-edit | ran-any-test | med turns | claims-success-never-verified |
|---|---|---|---|---|---|
| base | 119 | 74.3% | 90.3% | 246 | 1.7% |
| swezero control | 77 | 6.2% | 16.6% | 66 | 91.8% |
| A1-tokmatch | 63 | **3.0%** | **5.6%** | 53 | 94.8% |
| A1-maxpool | 62 | **3.4%** | **6.4%** | 56 | 94.8% |

**All three predictions PASS:**

- **P1 PASS** — both arms `verified_after_edit` < 25% (3.0 / 3.4) and
  `ran_any_test` < 45% (5.6 / 6.4). Passed with wide margin.
- **P2 PASS** — |tokmatch − maxpool| verify = **0.4pp** < 15pp. Corroborates the
  campaign's formal EQUIVALENT on score (+1/500) on an independent axis.
- **P3 PASS** — claims-success-never-verified 94.8% (both) > 60%.

**Decision rule #2 applied** (P1 holds, no score gain to mediate). The finding:
verification runs **74.3% → 6.2% → 3.0%/3.4%** against scores **119 → 77 →
63/62**. Curation did not restore verification, it **halved what remained**, and
the score followed.

⚠️ **Correction (2026-08-04, same day).** An earlier version of this paragraph
added "— the same monotone direction as the campaign's median history-length
chain (336 → 209.5 → 171 → 177.5), on an independent instrument." **That
corroboration is withdrawn.** The campaign's depth↔score chain was measured on a
single nested lineage (base + three swezero-derived arms) and **inverts across
sources**: extended to all seven scored models, the three deepest arms are the
three worst (rebench 206 actions / 70, coderforge 197 / 48, scale 211 / 35).
The four-model monotone run was a subset artifact, so it cannot corroborate
anything. The A1-lineage observation above stands on its own; it is a
**within-lineage** statement only, exactly like the depth claim it can no longer
lean on.

This also gives a sign to the "condensation may have been mildly protective"
conjecture: stripping condensation records took verification 6.2% → 3.0%, i.e.
on this source those records were the last ones modelling stop-and-check.
Lever A's premise is inverted, not merely unsupported.

**Limits.** Population rates over 500 rollouts (immune to grader flake, mildly
exposed to generation nondeterminism), but *correlational* with score and n=4 on
the model axis. Nothing here shows that raising verification would raise the
score — that remains Lever B's question.
