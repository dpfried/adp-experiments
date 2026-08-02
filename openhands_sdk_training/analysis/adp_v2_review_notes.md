# Review notes on `adp_v2_experiments_and_results.md`

Editorial review of the writeup itself — expected to go stale as items get fixed.
For discussion of *why* fine-tuning degrades performance, see `adp_v2_commentary.md`.
(Claude review, 2026-08-02, from a session with dpf.)

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

2. **§2.3's headline number doesn't match its own table.** The section opens with "the
   42-point gap" (119 vs. swezero's 77), but the table compares against the
   *per-repository best* of the four arms, whose column sums to 83 — so the table's
   diffs sum to 36, not 42. One sentence noting that the per-repo-best comparison is
   deliberately more conservative (36 vs. 42) would fix it.

3. **§2.7's episode accounting has an unexplained 155-episode gap.** The instruct model
   reaches a final answer in 207/500 episodes but only 138 end with an empty patch — so
   155 episodes apparently produced a non-empty patch *without* reaching a final answer
   (presumably the harness extracts the workspace diff at the iteration limit). That's
   plausible but unstated, and it matters: 119 resolves come from a pool that's mostly
   not "clean finishes," which is itself an interesting fact about the instruct model's
   behavior. Also, the section's first sentence says "every instance the model solves" —
   should be "the instruct model."

4. **§2.5's "testable prediction... confirmed" framing overreaches.** That the
   equal-weight average retains exactly s and cancels the residuals is a mathematical
   identity, not a prediction the score table can confirm. What the results actually show
   is that the *discarded residuals carried the performance*. Reword to: the geometry
   says the average keeps only s; the scores show that what it discards is where the
   performance lived.

## Smaller items

- §2.7 says "roughly 40% of training records are condensation turns," but the §1.1 table
  says 34% for `swezero` (the arm being audited) and 39% for `coderforge`, with the other
  two unmeasured. "Roughly 40%" is loose, and "of training records" is broader than the
  evidence.
- §2.3's table has 10 rows, but 12 repos minus the one unscoreable repo leaves 11.
  Presumably the missing one had zero solves on both sides — worth a footnote.
- §2.1: "at least 10.6%" — the "at least" is unexplained. If some instances were
  unevaluated in that re-run, say so; otherwise drop it.
- §2.8: `swezero` at 77 sits in the *middle* of the 74–82 band, not "at the bottom"
  (74 is the bottom).
- §2.5: the 65%-shared / 75–78%-residual pair will trip readers who add them to >100%.
  A parenthetical noting these are norms of near-orthogonal components (Pythagorean, not
  additive) would preempt that.
- §2.5's soup table uses a clean-subset p-value (p = 0.014) for exactly one comparison
  with no explanation of why that comparison needed the clean subset.

None of these threaten the central findings; items 1–3 are the ones to address before
sharing more widely.
