# Analysis house rules — ADP SWE-bench eval campaign

**Status:** binding on anything written into a report, a preregistration, a memory file, or
the coordination channel. Added 2026-08-05 after the campaign's sixth published-then-reverted
conclusion.

**Read this alongside [`SKEPTICAL_REVIEW_GUIDANCE.md`](SKEPTICAL_REVIEW_GUIDANCE.md).** That
document is dispositional — how to *be* when reviewing. This one is operational — what you
must mechanically *do* before a number leaves your session. The distinction matters, because
`SKEPTICAL_REVIEW_GUIDANCE.md` checklist item 5 already says *"Whole denominators? Any silent
shard/merge failures?"* — and rule 1 below still got violated twice after it was written.
**Prose guidance did not prevent these errors. That is the evidence for making the rules
executable.**

---

## Why this file exists: six reverts, three causes

| # | Reverted conclusion | Cause |
|---|---|---|
| 1 | θ₀ = 145 → **119** | best-of-3 union sold as pass@1 |
| 2 | "differential attrition between cells" | read `output.jsonl` only |
| 3 | base "has 44 infra failures" → **200** | stale number restated, not re-measured |
| 4 | "F's repaired score can only go up" (29) → **27 as-harness / 29 repaired** | projected from a partial pass |
| 5 | "the 40% cap-hit rate is a property of that run, not of base" | read `output.jsonl` only |
| 6 | "F−B is a directional lower bound" | enumerated the confounds found, called it the direction |

Three causes, not six mistakes:

- **#2 and #5** are the *same* harness defect (rule 1). It fails silently and it has now bitten
  the two people who know this pipeline best.
- **#1 and #6** are *unequal compute between the things being compared* (rule 2), twice, in
  different costumes.
- **#3, #4, #6** are inferences stated at the confidence of measurements (rules 3–4).

**The diagnostic that makes this tractable: no *measurement* in this campaign has ever been
reverted. Every revert has been of an *inference*.** §0b's numbers (300/500 transcripts, 26
unattributable resolves, 54.7k vs 10.8k completion tok/instance) survived all of it; what died
was the "therefore" attached to them. Rules 3 and 4 exist to keep those two things physically
separate so a revert is a one-line edit to a labelled block, not a headline retraction.

---

## Rule 1 — Never count a rollout directory by hand. Use the loader.

```python
from load_rollouts import load_cell
cell = load_cell(out_dir, select=".../select/shard_00of10.txt", attempt=1)
```

It raises `RolloutIntegrityError` rather than returning a plausible wrong number. Run it as a
CLI to eyeball a cell: `python load_rollouts.py <out_dir> --select <shard.txt>`.

The three traps it closes, each measured on real cells 2026-08-05:

1. **`output.jsonl` omits every instance that ended in a terminal error** — those are in
   `output_errors.jsonl`. So a `MaxIterationsReached` search over `output.jsonl` returns zero
   *by construction*. This produced reverts #2 and #5.
2. **The two files are NOT disjoint.** This was the folklore correction proposed after revert
   #5 ("just union them"), and it is *also wrong*: cell `E_base_stock_evalp` had **3**
   overlapping `instance_id`s in shard 00 and **6** in shard 01. Naive concatenation
   double-counts. ⇒ **Rule 7.**
3. **`output.jsonl` is an append log during the run**, rewritten by `aggregate_results` only at
   the end. Mid-flight it carries duplicate `instance_id`s (measured: 11 duplicate rows across
   E/s01's two files). A cell whose job died on walltime stays a raw append log forever, and a
   non-set-based reader silently double-counts it.

**Prefer `output.critic_attempt_N.jsonl`** — frozen at write time, exactly one row per
instance, attempt-matched. The loader picks it automatically. Measured proof that the choice
changes answers: on E/s01, the frozen attempt-1 file gives `cap500=16, ok=27`; the append-log
fallback on the same cell gives `cap500=14, ok=31`.

**Always pass `select=`.** Coverage against the intended instance list is the invariant that
catches the whole silent-truncation family. Without it you cannot tell a complete cell from a
truncated one — both parse fine.

## Rule 2 — Compute-match before you compare. Score attempt 1.

The harness retries whichever instances **its critic** judges failed, and attempts ≥2 sample at
**temperature 0.1**, not 0. Under *identical* config the critic rejected base's first attempt
**71/100** and the arms' **2–5/100** ⇒ base consumed 2.32 rollouts/instance plus a best-of-3
selection, the arms 1.04. Not a misconfig; the harness spends more compute on whichever model
the critic dislikes.

- Any cross-model number is **attempt-1 only** (one rollout, temp 0, no critic selection) unless
  you are explicitly reporting a pass@k, labelled as such.
- The critic's 71-vs-5 base-vs-arm rejection asymmetry is itself an **unexplained
  model-dependence** in a shared component. Anything built on this critic (selection, filtering,
  an outcome-labeller) inherits it.
- Retry rate is not even constant within one model: base's cells differ enormously
  (F 71/100 vs E 15/100 at attempt 2), because errored instances are retried on a different
  path from critic-rejected ones. **You cannot assume two cells of the same model are
  compute-matched.** Check.

## Rule 3 — Separate **Measured** from **Inferred**, physically.

Every analysis section carries two labelled blocks. Measured = a number with a file path and a
loader invocation behind it. Inferred = anything with a "therefore." Never let a "therefore"
inherit the confidence of the number above it, and never put one in a TL;DR or a headline
without the label. When an inference dies, the edit is confined to its block.

## Rule 4 — Keep a confound **budget**, not a confound list.

Banned sentence shape: *"both biases I found point the same way, therefore the bias points that
way."* That is an unfinished search reported as a complete one, and it caused revert #6 — after
being authored by the reviewer whose job was to catch it.

Required instead: **"I checked k mechanisms of an unknown total; here they are; the residual is
unbounded."** A direction claim needs either an exhaustive mechanism argument or a matched
control — never an enumeration.

## Rule 5 — State **both** denominators, every time.

These runs have two, and they differ by up to 2×:

- **N scored** — every selected instance, because grading reads `test_result.git_patch`, so a
  transcript-less run is still graded. Cell E: 100.
- **M transcript-bearing** — the denominator for *every behavioural* statistic (tool mix, depth,
  narration rate, token counts). Cell E: **46**.

The missing M−N are **systematically the longest runs** (they are the ones that hit the cap), so
behavioural stats over M are **top-truncated**, not randomly sampled. Say which denominator a
number uses, in the table, next to the number.

## Rule 6 — Do not ship a correction in the same hour you measure it.

Revert #5 was measured, committed, written to three memory files, and posted to the channel
inside one hour — with no interval in which anyone could check it, and it was retracting a
finding others had already built on. Each step was individually defensible; the sequence was
noise. **A correction that retracts someone else's load-bearing result gets posted for review
first and committed second.**

## Rule 7 — Verify the folklore too, including the correction to your own error.

The fix proposed after revert #5 — "the two files are disjoint, so union them" — was asserted by
two independent reviewers, was consistent with the evidence they had, and was **false** (rule 1,
trap 2). Being corrected is not the end of the check. A consensus among agents is not a
measurement; it is a hypothesis with social proof. Run the probe.

---

## Pre-publication gate

Before a number enters a report, prereg, memory file, or channel post:

1. Did `load_cell(..., select=..., attempt=1)` produce it, strict, without raising?
2. Is it attempt-1, or explicitly labelled pass@k?
3. Which denominator — N scored or M transcript-bearing? Is that stated *at* the number?
4. Is it in the **Measured** block or the **Inferred** block?
5. If it claims a direction: how many bias mechanisms did I check, and did I say the residual is
   unbounded?
6. If it corrects someone else: has it been posted for review, or am I committing it in the same
   hour I measured it?
7. Is any premise here folklore I inherited rather than a probe I ran?
