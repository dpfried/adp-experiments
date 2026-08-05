#!/usr/bin/env python3
"""Emit a cell's FIRST-critic-attempt rollouts as a standalone output.jsonl.

Why this exists
---------------
The eval harness is configured identically for all six ladder cells (same critic,
`n_critic_runs=3`), but the number of attempts a cell actually *consumes* is
decided by the critic at runtime, and it came out wildly unequal:

    cell                       attempt1  attempt2  attempt3   rollouts/instance
    A  arm  stock                   100         5         0        1.05
    B  arm  nostub                  100         3         1        1.04
    C  arm  nostub wrapper          100         4         0        1.04
    D  arm  nostub trainprompt      100         2         0        1.02
    F  base nostub                  100        71        61        2.32

The critic rejects base's first attempt 71/100 times and the arm's 2-5/100
times, so base is handed ~2.2x the rollouts and a best-of-3 selection while the
arm is effectively single-shot. That is not a configuration asymmetry and not a
bug -- it is the harness spending more compute on whichever model the critic
dislikes. It is still a confound for any "base vs arm" claim, and it inflates
base, i.e. it points the OPPOSITE way from the two biases already catalogued
(the patch-blind tie-break and cap-survivorship, which both depress base).

Attempt 1 is the compute-matched contrast: exactly one rollout per instance,
temperature 0 in every cell (`evaluation.py` raises it to 0.1 only for
attempt > 1), all 100 instances, no critic selection, and frozen on disk in
`output.critic_attempt_1.jsonl` where `aggregate_results` never rewrites it.

Usage
-----
    attempt1_subset.py <out_TAG dir> <dest output.jsonl>

Writes every attempt-1 row (error rows included, so the denominator stays the
full instance set) and prints a one-line summary. Deduplicates by instance_id,
keeping the first occurrence -- a resumed job can legitimately re-run an
instance into the same attempt file.
"""
import glob
import json
import os
import sys


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dest = sys.argv[1], sys.argv[2]

    rows, seen, dups = [], set(), 0
    pat = os.path.join(src, "**", "output.critic_attempt_1.jsonl")
    files = sorted(glob.glob(pat, recursive=True))
    for f in files:
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            iid = r.get("instance_id")
            if not iid:
                continue
            if iid in seen:
                dups += 1
                continue
            seen.add(iid)
            rows.append(line)

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with open(dest, "w") as fh:
        for line in rows:
            fh.write(line + "\n")

    # No output_errors.jsonl is written alongside: the scoring sbatch picks up
    # output_errors.jsonl from the INPUT's own directory, so leaving it absent
    # makes the job score exactly these rows and nothing else.
    nerr = sum(1 for line in rows if (json.loads(line).get("error") or ""))
    print(f"{len(rows)} attempt-1 records ({nerr} error rows, {dups} dup skipped) "
          f"from {len(files)} file(s) -> {dest}")


if __name__ == "__main__":
    main()
