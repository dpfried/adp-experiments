#!/usr/bin/env python3
"""Count instances whose FINAL aggregated record has an empty patch while an
earlier critic attempt produced a non-empty one.

Root cause (benchmarks/benchmarks/utils/iterative.py):
  aggregate_results() walks attempts LAST -> FIRST and replaces the incumbent
  only when `entry.beats(current)` i.e. `rank > other.rank`, with
  _get_output_rank() in {0=error, 1=no-error/critic-failed, 2=critic-passed}.
  An empty patch and a substantive-but-critic-failing patch are BOTH rank 1, so
  a tie is resolved in favour of the LATEST attempt and the rank is blind to
  whether a patch exists. Retries also run at temperature 0.1 (not 0.0).
  => a degenerate final retry silently discards a good earlier patch.

The exposure is behavioural, not configurational: only a model that retries a lot
can be bitten. Measured on the parity ladder, all 8 arm cells were 0/400 affected
while base cell F lost 9 of 50 on one shard.

Note the scoring sbatch concatenates `output_errors.jsonl` alongside
`output.jsonl`, which can accidentally rescue an affected instance if an error
row happens to carry a patch. That rescue is incidental, not a fix -- cells with
no error rows get nothing. Always check both.

Usage: scan_discarded_patches.py <runs_root> <run_glob> [<run_glob> ...]
Set RESCUABLE_OUT=<path> to dump the affected instance ids.
"""
import json, glob, os, sys, collections


def patchlen(r):
    tr = r.get('test_result')
    p = (tr or {}).get('git_patch') if isinstance(tr, dict) else None
    if p is None:
        p = r.get('git_patch') or ''
    return len(p or '')


def scan_dir(dd):
    per = collections.defaultdict(dict)   # iid -> attempt -> patchlen
    fin = {}
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
            per[r['instance_id']][a] = patchlen(r)
    f = os.path.join(dd, 'output.jsonl')
    if os.path.exists(f):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            fin[r['instance_id']] = (r.get('attempt'), patchlen(r))
    return fin, per


def main():
    root = sys.argv[1]
    tot_fin = tot_empty = tot_rescuable = 0
    rescuable_ids = []
    for pat in sys.argv[2:]:
        for d in sorted(glob.glob(os.path.join(root, pat))):
            g = glob.glob(os.path.join(d, '**', 'output.jsonl'), recursive=True)
            if not g:
                continue
            dd = os.path.dirname(g[0])
            fin, per = scan_dir(dd)
            empty = [i for i, (a, L) in fin.items() if L == 0]
            resc = [i for i in empty if max(per.get(i, {}).values() or [0]) > 0]
            tot_fin += len(fin)
            tot_empty += len(empty)
            tot_rescuable += len(resc)
            rescuable_ids += resc
            print(f"{os.path.basename(d):46} final={len(fin):4} "
                  f"empty={len(empty):4} discarded-good={len(resc):4}")
    print(f"\nTOTAL final={tot_fin} empty={tot_empty} "
          f"discarded-good={tot_rescuable} "
          f"({100.0*tot_rescuable/tot_fin if tot_fin else 0:.1f}% of scored)")
    out = os.environ.get('RESCUABLE_OUT')
    if out:
        open(out, 'w').write('\n'.join(sorted(set(rescuable_ids))) + '\n')
        print(f"wrote {len(set(rescuable_ids))} ids -> {out}")


main()
