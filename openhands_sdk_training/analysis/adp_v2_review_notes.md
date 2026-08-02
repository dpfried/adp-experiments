# Review notes on `adp_v2_experiments_and_results.md`

Editorial review of the writeup itself — expected to go stale as items get fixed.
For discussion of *why* fine-tuning degrades performance, see `adp_v2_commentary.md`.
(Claude review, 2026-08-02, from a session with dpf.)

**Status (2026-08-02, main-worker session): all items below addressed in the writeup, each
independently re-verified against the raw eval data (not just re-read) before fixing — see
per-item notes. Kept below as a historical record per this file's own stated purpose.**

Overall the writeup is strong: the argument structure is sound, hedging is calibrated
(the verification-status finding and the §2.8 cross-campaign comparison are appropriately
flagged as suggestive), and the internal arithmetic checks out — the task-vector geometry
is self-consistent (pairwise cosines of 0.20–0.26 imply ‖s‖/‖τ‖ ≈ 0.65 for four
equal-norm vectors, and 0.646 implies ~76% residual norm), the per-repo table sums to 119,
and the union/routing arithmetic in §2.4 is consistent. The §2.7 reframe
(depth-vs-compliance trade-off under single-attempt scoring, rather than "fine-tuning is
bad") is the best part of the report.

## Issues, in rough order of importance

1. **The primary-board promise isn't kept.** §1.3 says "unless noted, 'resolve rate'
   below refers to the 456-instance board" — but every table in §2 reports /500. Only
   §2.1 gives the 456 number at all. Either change the note to say the 500-board is
   primary (with 456 as a robustness check), or actually report the 456 board.
   **[ADDRESSED]** §1.3 now states the actual practice precisely: raw counts /500
   throughout (justified — the unscoreable repository contributes 0 for every model, so
   the numerator is identical either way), significance tests on /456. Also fixed the
   parallel claim in §4/Limitations, which had the same stale wording.

2. **§2.3's headline number doesn't match its own table.** The section opens with "the
   42-point gap" (119 vs. swezero's 77), but the table compares against the
   *per-repository best* of the four arms, whose column sums to 83 — so the table's
   diffs sum to 36, not 42. One sentence noting that the per-repo-best comparison is
   deliberately more conservative (36 vs. 42) would fix it.
   **[ADDRESSED]** Re-derived the whole table directly from the raw resolved_ids (not
   just re-read) to confirm 83/36 exactly, then added an explicit paragraph distinguishing
   the single-arm 42-point gap from the per-repository-oracle 36-point one and tying the
   83 to §2.4's routing-oracle number.

3. **§2.7's episode accounting has an unexplained 155-episode gap.** The instruct model
   reaches a final answer in 207/500 episodes but only 138 end with an empty patch — so
   155 episodes apparently produced a non-empty patch *without* reaching a final answer
   (presumably the harness extracts the workspace diff at the iteration limit). That's
   plausible but unstated, and it matters: 119 resolves come from a pool that's mostly
   not "clean finishes," which is itself an interesting fact about the instruct model's
   behavior. Also, the section's first sentence says "every instance the model solves" —
   should be "the instruct model."
   **[ADDRESSED]** Both fixed: "the model" → "the instruct model"; added a sentence
   deriving the 155-episode gap and its likely mechanism (iteration-limit + workspace-diff
   extraction), explicitly flagged as NOT established what fraction of the 119 resolves
   come from this category (checked whether this was cheaply verifiable from the raw
   rollout logs — the harness's event schema didn't make it a quick check, so left as an
   honest open point rather than a guess, per this file's own norm of not overclaiming).

4. **§2.5's "testable prediction... confirmed" framing overreaches.** That the
   equal-weight average retains exactly s and cancels the residuals is a mathematical
   identity, not a prediction the score table can confirm. What the results actually show
   is that the *discarded residuals carried the performance*. Reword to: the geometry
   says the average keeps only s; the scores show that what it discards is where the
   performance lived.
   **[ADDRESSED]** Reworded exactly along the suggested lines: the mathematical fact
   (average keeps s, cancels residuals — true by construction) is now separated from the
   empirical question (whether performance survives) and the scores are described as
   answering the latter, not "confirming" the former.

## Smaller items

- §2.7 says "roughly 40% of training records are condensation turns," but the §1.1 table
  says 34% for `swezero` (the arm being audited) and 39% for `coderforge`, with the other
  two unmeasured. "Roughly 40%" is loose, and "of training records" is broader than the
  evidence.
  **[ADDRESSED]** Rewritten to name `swezero` specifically (34%) since that's the arm
  §2.7 actually audits, with `coderforge`'s 39% noted as the only other measured source
  and an explicit caveat that generalization to all four is unknown.
- §2.3's table has 10 rows, but 12 repos minus the one unscoreable repo leaves 11.
  Presumably the missing one had zero solves on both sides — worth a footnote.
  **[ADDRESSED]** Computed the real numbers directly from the raw dataset rather than
  guessing: `seaborn` has exactly 2 instances in the benchmark, and the instruct model and
  every arm score 0 there — added as an explicit row plus a footnote, and fixed the
  "9 of 10" claim just below to "10 of 11" to stay consistent with the added row.
- §2.1: "at least 10.6%" — the "at least" is unexplained. If some instances were
  unevaluated in that re-run, say so; otherwise drop it.
  **[ADDRESSED]** Added the explanation: it's a hard floor because that re-evaluation run
  didn't score every instance and unscored ones count as unresolved, plus a caveat that
  the scored-subset rate may not be representative.
- §2.8: `swezero` at 77 sits in the *middle* of the 74–82 band, not "at the bottom"
  (74 is the bottom).
  **[ADDRESSED]** Changed "sits at the bottom of" to "sits inside" the band.
- §2.5: the 65%-shared / 75–78%-residual pair will trip readers who add them to >100%.
  A parenthetical noting these are norms of near-orthogonal components (Pythagorean, not
  additive) would preempt that.
  **[ADDRESSED]** Added the Pythagorean parenthetical as suggested (0.646² + 0.76² ≈ 1.0).
- §2.5's soup table uses a clean-subset p-value (p = 0.014) for exactly one comparison
  with no explanation of why that comparison needed the clean subset.
  **[ADDRESSED]** Recomputed the McNemar test directly on both boards to confirm (456:
  p=0.10 n.s.; clean-301: p=0.014 sig) and identified the mechanism (the two-arm soup
  includes `rebench`, the sympy specialist, which gives it an 11-vs-3 sympy edge over
  `swezero` that narrows the full-board gap; the clean board removes that repository) —
  added both the numbers and the mechanism as a footnote under the table.

None of these threaten the central findings; items 1–3 are the ones to address before
sharing more widely.
