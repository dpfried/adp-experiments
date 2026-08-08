#!/usr/bin/env python3
"""Repair the patch-blind tie-break in the harness's multi-attempt aggregator.

DEFECT (benchmarks/benchmarks/utils/iterative.py):
  aggregate_results() iterates attempts LAST -> FIRST and replaces the incumbent
  only when `entry.beats(current)`, which is `self.rank > other.rank`.
  _get_output_rank() returns 0=error, 1=no-error/critic-failed, 2=critic-passed.
  An EMPTY patch and a substantive-but-critic-failing patch are BOTH rank 1, so
  equal ranks resolve in favour of the LATEST attempt and the rank never looks
  at whether a patch exists. Retries additionally run at temperature 0.1 rather
  than 0.0, so a later attempt is a fresh non-greedy sample. Net effect: a
  degenerate final retry silently discards a good earlier patch.

  An empty final implies NO attempt reached rank 2 (a rank-2 entry would have
  won regardless of order), so every candidate here is rank-1 and the choice
  among them is exactly the tie the harness breaks arbitrarily.

REPAIR (minimal; changes only the tie-break, not the rank ordering):
  among an instance's rank-1 attempts, prefer a NON-EMPTY patch, then the LATEST
  such attempt.

Emits ONLY the instances whose selection actually changes, as a standalone jsonl
that can be scored on its own. Deliberately written to a fresh directory: the
scoring sbatch concatenates `output_errors.jsonl` from the input file's own
directory, so an isolated dir scores exactly these rows and nothing else.

Applied uniformly to every cell of a comparison. On this ladder's arm cells it
is a provable no-op (0 of 400 records affected), so parity is preserved by
construction rather than by assumption.

Usage: repair_aggregate.py <out_dir> <dest.jsonl>
"""
import json, glob, os, sys


def plen(r):
    tr = r.get('test_result') if isinstance(r.get('test_result'), dict) else {}
    return len((tr.get('git_patch') or r.get('git_patch') or ''))


def main():
    out_dir, dest = sys.argv[1], sys.argv[2]
    g = glob.glob(os.path.join(out_dir, '**', 'output.jsonl'), recursive=True)
    if not g:
        print(f"no output.jsonl under {out_dir}")
        return 1
    dd = os.path.dirname(g[0])

    empty = set()
    for line in open(os.path.join(dd, 'output.jsonl')):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if plen(r) == 0:
            empty.add(r['instance_id'])
        else:
            empty.discard(r['instance_id'])   # a non-empty row for it exists

    best = {}   # iid -> (attempt, raw_line)
    for a in (1, 2, 3, 4, 5):
        f = os.path.join(dd, f'output.critic_attempt_{a}.jsonl')
        if not os.path.exists(f):
            continue
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            iid = r['instance_id']
            if iid not in empty or r.get('error') or plen(r) == 0:
                continue
            if iid not in best or a >= best[iid][0]:
                best[iid] = (a, line)

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, 'w') as fh:
        for iid, (a, line) in sorted(best.items()):
            fh.write(line + '\n')
    print(f"{os.path.basename(out_dir)}: empty_final={len(empty)} "
          f"repaired={len(best)} -> {dest}")
    for iid, (a, _) in sorted(best.items()):
        print(f"    {iid}  <- attempt {a}")
    return 0


sys.exit(main())
