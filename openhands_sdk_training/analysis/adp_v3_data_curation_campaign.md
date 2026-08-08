# ADP-v3 — Data-Curation Campaign (condenser handling + outcome verification)

_New campaign proposal, opened 2026-07-28 (agent-b). Motivated by the ADP-v2 souping campaign's
verified bottom line: **raw first-55k ADP SFT is net-negative vs a ~24% base, and no combination
method recovers it.** This campaign asks the one question v2 left open: **does QUALITY-FILTERED data
lift over the measured ~24% base?** Companion to the v2 findings report
(`adp_v2_swebench_findings_report.md`) and the data-composition audit
(memory `adp-v2-data-composition`)._

---

## 1. Why this campaign (the v2 result in one screen)

All numbers single-run temp-0 pass@1 on SWE-bench Verified (500), OpenHands scaffold, file-backed
under `merged.report.json`. Board (raw /500):

**base θ₀ 119 (~24%) ≫ swezero 77 > rebench 70 > top2-soup 62 > α0.7 54 ≈ pooled220k 54 > uniform4 50 > coderforge 48 ≈ pooled55k 46 > α1.55 38 > scale 35 > α2.0 19.**

- **No SFT arm beats base**; the best (swezero 77) is −42/500 below θ₀=119. **No combine method beats
  base** either — every weight-soup and both joint-trains (pooled55k 46, pooled220k 54) land in the
  weak cluster (CONFIRMED, all budgets).
- **Root cause is data content/objective, NOT a pipeline bug** (file-backed; format / template /
  parser / masking / LR all refuted). On the 82 base-solved/arm-failed instances, **82/82 are valid,
  well-formed patches that MISDIAGNOSE the bug** — 0 empty, 0 malformed. The arms are *more*
  scaffold-compliant than base (swezero finishes 500/500 vs base 207/500; 0 empty patches vs base's
  138; ~half the actions, 92 vs 171). **SFT traded DEPTH for COMPLIANCE:** it learned to make a
  minimal symptom-level edit, write a confident "fixed it" summary, and finish early, in place of
  base's messier deep root-causing. This is metric-specific — SBV pass@1 rewards depth — not a flat
  "SFT degrades capability."
- **Two candidate data defects** (correlational until tested here) plausibly drive the depth→compliance
  shift:
  1. **~40% of every subset is context-condenser records** (a 2-message user=condenser-prompt →
     assistant=prose-state-summary pair, `generation=openhands_sdk_condensation_prompt`), which train
     *summarize-and-conclude* rather than *act*. Two-sided: could be a genuine context-management skill
     SBV pass@1 doesn't score, or net dilution.
  2. **NO success/outcome filtering.** ⚠️ **REFUTED 2026-08-04 (prereg §18)** — coderforge and rebench were success-filtered upstream; see note below. The records carry no resolved/reward/pass field, so unresolved
     and looping trajectories are imitated wholesale. Only ~39% of trajectory records even end at a
     real finish (the rest are mid-task segments).
- Ruled OUT as causes (do not re-litigate): truncation (0% at the 32768 cutoff), format/template
  mismatch (byte-identical), config (LR 1e-5 / masking / packing all sane), contamination (dpf stood
  down; re-rank identical on clean-301). Each single arm and pooled55k trained on **55k = equal
  compute**.

**Thesis for v3:** the SFT delta is net-harmful because the *objective* teaches a shallow-tidy policy
from unfiltered, summary-heavy data. Curating the data — removing the summarize-over-act signal and
keeping only *verified-good* trajectories — should shift the learned policy back toward depth. Whether
that reaches or exceeds the ~24% base is the empirical question; partial recovery is still informative.

---

## 2. The two levers

### Lever A — Condenser handling (~40% of the data)
The condensation records are the single largest controllable slice. We do not yet know their sign.

- **A1 — trajectory-only (drop all condensation), matched compute.** Train one arm on **55k of
  pure action-trajectory records** (0% condensation) vs the raw mixed-55k arm (~60% action / ~40%
  condensation) from the *same source*. Same recipe, 1719 steps.
  - *H-A1:* trajectory-only > mixed on SBV pass@1 (condensation is net dilution/harm).
  - *Falsifier:* trajectory-only ≈ mixed (n.s. by McNemar/TOST) ⇒ condensation is inert; the lever is
    elsewhere (→ Lever B). If trajectory-only < mixed ⇒ condensation carries real context-mgmt value.
- **A2 — down-weight instead of drop (optional, gated on A1).** Keep condensation at loss-weight ~0.25
  (or as a separate auxiliary objective) to test whether they contribute *some* long-horizon
  context-compression skill without dominating the policy. Only run if A1 shows condensation is not
  purely harmful.
- **Depth diagnostics (all arms):** actions/instance, turns-to-finish, finish-rate, empty-patch rate.
  Prediction if depth is recovered: the curated arm looks *more like base* (more actions, fewer
  premature finishes) — a mechanism check independent of the /500 score.

### Lever B — Outcome verification (success filtering)
The higher-value but higher-risk lever: train only on trajectories we can *verify* ended in a correct
fix. **The critical unknown is label recoverability** — the ADP records were normalized and carry no
outcome label and (apparently) uuid ids, so this needs a scoping pass FIRST (Phase 0 below). Three
routes, cheapest first:

- **Route B-i — recover upstream labels.** The records retain `metadata.source_dataset` +
  `source_trajectory_id`. The upstream sources (nvidia SWE-Zero, nebius SWE-rebench, coderforge, scale)
  plausibly ship a resolved/pass flag and/or a **gold patch** per trajectory that ADP normalization
  dropped. If we can join back on `source_trajectory_id`, we get real labels for free. **This is the
  preferred path — Phase 0 tests it.**
- **Route B-ii — test-replay.** Re-score each trajectory's final `git_patch` against the instance's
  test suite (SWE-bench-style). Definitive but requires mapping each record to repo + base commit +
  tests — uncertain given the anonymized ids. Feasible only where instance identity is recoverable.
- **Route B-iii — LLM-judge proxy.** An LLM verifier scores (issue, final patch) for plausible
  correctness. Cheap, no harness, but noisy and correlated with the very shallow-tidy failure we're
  trying to remove — use only as a fallback / coarse filter, and validate its labels against a
  test-replayed sample.

Experiments (whichever labeling route Phase 0 makes viable):
- **B1 — resolved-only SFT, matched compute.** Train on 55k of *verified-resolved* trajectories vs the
  raw mixed-55k arm.
  - *H-B1:* resolved-only > mixed, and ideally ≥ base (verification is the lever).
  - *Falsifier:* resolved-only ≈ mixed ⇒ success-filtering alone doesn't fix it.
  - *Risk:* if the upstream resolved-rate is low, one source may not yield 55k resolved records ⇒ pool
    sources or run at a smaller matched budget (report compute-matched, not data-matched).
- **B2 — gold-patch distillation (if gold patches recoverable).** Reconstruct trajectories to end in the
  *gold* patch rather than the agent's guess — teaches the correct fix directly, the strongest positive
  signal. Compare vs B1 and base.
- **B3 — preference/outcome-weighting (stretch, post-SFT).** With resolved/unresolved pairs, a DPO or
  outcome-weighted pass on top of the best SFT arm. Only if B1/B2 show curation moves the needle.

### Lever C — the decisive combined arm
- **C1 — verified + condensation-excluded 55k** (Lever A ∩ B) vs the matched raw arm AND vs base. This
  is the campaign headline: the cleanest data we can construct at matched compute.
  - *H-C1 (hypothesis, NOT a promise):* C1 recovers depth and closes some/all of the −42 gap to base.
    Recovery to/above 119 is the hypothesis; **partial recovery (<119) is still a positive result**
    (curation helps but doesn't fully close it). Pre-committing the magnitude is out.

---

## 3. Rigor & metrics (reuse the v2 kit)

- Eval via `swebench-eval` (10×50 A100 shards → aggregate → score → merge). **Watch the
  exit-127 dash/merge bug** (hand-merge with `bash -lc "source env.sh && …merge_shard_reports.py <TAG>"`).
- **Per-board reporting:** full-500 / 456 (ex-sphinx, the shared 0/44 hole) / clean-301 (ex the 6
  contaminated repos). Report all three so domain-forgetting and any residual contamination don't blur.
- **Paired stats:** McNemar vs **base(119)** and vs the **matched raw-SFT arm**, + TOST (±10/500 margin)
  → three-state different/equivalent/inconclusive. Use soup-worker's `paired_compare.py`.
- **Depth diagnostics** as a first-class outcome (see Lever A) — they test the *mechanism*, not just the
  score, and are cheap (from the rollout jsonl).
- **Pre-register** each arm's hypothesis + falsifier before its eval lands (v2 discipline).
- Anchor: **base θ₀ = 119/500** single-run provenance-verified (`v2_init_singlerun_4b`). Do NOT re-anchor
  on the retracted ~5% or the non-instruct ~73.

---

## 4. Phase 0 — data-provenance scoping (MUST run first; gates Lever B)

Before training anything, a read-only data pass (SSH to FAIR, `v2_swe_subsets/ (the ADP SWE subsets)`):
1. **Label recoverability:** can `metadata.source_dataset` + `source_trajectory_id` join back to the
   upstream datasets, and do those carry a resolved/pass flag and/or gold patch? (Determines B-i vs
   B-ii vs B-iii.)
2. **Instance identity:** is repo + base-commit + test recoverable per record (enables B-ii test-replay)?
3. **Yield:** if labels are recoverable, what is the resolved-rate per source, and can we assemble 55k
   resolved (single-source) or must we pool? Same for trajectory-only (post condensation-drop counts —
   already ~60% × source size, ample).
4. **Deliverable:** a one-page viability note picking the labeling route + the source(s) + the achievable
   matched-compute budget. **The whole Lever-B arm of the campaign is gated on this.** (Lever A / A1 is
   unblocked and can start immediately — it needs no labels.)

---

## 5. Sequencing & compute

Each arm ≈ one v2-arm budget (55k records, 1719 steps, 8×A100; then a 10×A100 eval). Suggested order:
1. **A1 (trajectory-only)** — unblocked, cheap, resolves the ~40% condensation sign. Start now.
2. **Phase 0 scoping** — in parallel with A1; determines whether/how B is possible.
3. **B1 (resolved-only)** — gated on Phase 0.
4. **C1 (verified + condensation-excluded)** — the headline, once A1 + B1 inform the recipe.
5. A2 / B2 / B3 — optional, gated on the above.

Start with **one source** (candidate: **rebench** — decontaminated-by-design, clean; or **swezero** —
strongest raw signal) to keep contrasts clean; generalize to a second source only if a lever pays off.

---

## 6. Risks & open questions (calibrated)

- **Label recovery may fail** (uuid ids, upstream labels stripped). This is the campaign's central
  uncertainty → Phase 0 exists to answer it before spend. If all three routes fail, Lever B degrades to
  a noisy LLM-judge filter and the campaign leans on Lever A + condensation work.
- **Enough clean data at matched compute?** Resolved-only may undershoot 55k from one source ⇒ pool or
  drop to a smaller matched budget (report as compute-matched).
- **Mechanism is still a hypothesis.** A1/B1/C1 + depth diagnostics are what promote "condensation +
  no-filter → shallow-tidy" from correlational to causal. Keep it hypothesis-voiced until they land.
- **Curation might not reach base.** A null result (curated ≈ raw, both < base) would itself be a strong
  finding: raw ADP SWE trajectories are not a usable SFT source for lifting a strong instruct base on
  agentic SWE — pointing to RL/off-policy or gold-patch distillation instead of trajectory SFT.
- **SBV is a proxy.** Depth-vs-compliance is metric-specific; a curated arm that helps pass@k or as an
  RL init but not pass@1 is still a real result — consider a pass@k or valid-fix-rate secondary metric
  if pass@1 stays flat.

## 7. Success criteria
- **Primary:** does any curated arm (A1 / B1 / C1) significantly beat its matched raw arm on SBV pass@1
  (McNemar), and does C1 close a meaningful fraction of the −42 gap to base 119?
- **Mechanism:** do curated arms shift the depth diagnostics toward base (more actions, fewer premature
  finishes, lower confident-wrong rate)?
- **Either way:** a file-backed answer to "can quality-filtered ADP data lift over a measured ~24%
  base," which is the question the v2 campaign could not answer.
