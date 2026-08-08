#!/usr/bin/env python3
"""Count sandbox-collision damage in already-written score reports, and project
what a clean re-score would recover -- without re-running any scoring.

Scoring failures leave a signature in the per-instance report JSON rather than
failing silently. There are THREE distinct families and they do not share a
cause -- keep them apart:

  ENOTEMPTY  "scoring_error": "[Errno 39] Directory not empty:
                 '.../swebench-sandboxes/<id>.tmp' -> '.../<id>'"
             The sandbox collision proper (fixed 2026-08-05, per-job subtree).
             Only fires when something else already holds the destination path,
             so it tracks CONCURRENCY: on the 500-instance board every cell
             scored with an overlapping job has 12-79 of these, and all four
             cells scored alone have exactly ZERO.
  build      "apptainer build failed" / "sandbox verification failed".
             A SEPARATE, transient infrastructure failure -- present in every
             cell including the solo ones. Do not attribute these to the
             collision; that conflation makes the collision look universal.
  soft       patch applied cleanly, but PASS_TO_PASS has zero successes and >0
             failures -- a broken environment, not a broken patch.

`calibrate` diffs contaminated (_a1) against clean (_a1x) passes of the same
rollouts to measure what fraction of each family actually comes back:
ENOTEMPTY 19.2%, build 16.7%, soft 5.5%, near one-way (51 gained / 4 lost on the
ladder). `audit` applies those rates to any other set of score dirs.

Note on exposure: a sandbox is only built for an instance that produced a patch,
so exposure tracks patch rate. On the board the arms patch ~100% of instances
and base 72%, which makes the ARMS the more exposed side -- the opposite of the
ladder's asymmetry. Do not extrapolate a direction from one to the other.

Usage:
  collision_signature_audit.py audit     <runs_root> <dir_glob> [<dir_glob> ...]
  collision_signature_audit.py calibrate <runs_root> <cell> [<cell> ...]
      where <cell> resolves to score_<cell>__s??_a1 vs ..._a1x
  collision_signature_audit.py dupes     <runs_root> <dir_glob> [<dir_glob> ...]
      hygiene guard: asserts predictions are one row per instance. (They are --
      a shard dir holds shard_N.jsonl AND shard_N.swebench.jsonl, the same
      predictions in two formats; counting both looks like double-scoring.)
"""
import json, glob, os, sys, collections

# Recovery rates measured on the six ladder cells that have both a contaminated
# and a clean pass; see parity_ladder_amendment.md Addendum 14.
RATES = {"ENOTEMPTY": 0.192, "build": 0.167, "other": 0.167, "soft": 0.055}
FAMS = ["ENOTEMPTY", "build", "other", "soft"]


def load(d):
    """Merge every shard report in a score dir -> (reports, resolved_ids, total)."""
    rep, res, tot = {}, set(), 0
    for p in glob.glob(os.path.join(d, "shard_*of*.report.json")):
        j = json.load(open(p))
        rep.update(j.get("reports") or {})
        res |= set(j.get("resolved_ids") or [])
        tot += j.get("total") or 0
    return rep, res, tot


def sig(r):
    """One of FAMS, or None, for one instance report."""
    e = str(r.get("scoring_error") or "")
    if e:
        if "Directory not empty" in e:
            return "ENOTEMPTY"
        if "apptainer build failed" in e or "sandbox verification failed" in e:
            return "build"
        return "other"
    p2p = (r.get("tests_status") or {}).get("PASS_TO_PASS") or {}
    if r.get("patch_successfully_applied") and not p2p.get("success") and p2p.get("failure"):
        return "soft"
    return None


def patched(r):
    return bool(r.get("patch_exists") or r.get("scoring_error"))


def audit(root, globs):
    print(f"{'cell':30s} {'tot':>4} {'res':>4} {'patch':>6} "
          + ' '.join(f"{f:>9s}" for f in FAMS) + f" {'proj':>6}")
    for g in globs:
        for d in sorted(glob.glob(os.path.join(root, g))):
            rep, res, tot = load(d)
            if not rep:
                continue
            c = collections.Counter(sig(r) for r in rep.values())
            proj = len(res) + sum(RATES[f] * c[f] for f in FAMS)
            print(f"{os.path.basename(d):30s} {tot:4d} {len(res):4d} "
                  f"{sum(patched(r) for r in rep.values()):6d} "
                  + ' '.join(f"{c[f]:9d}" for f in FAMS) + f" {proj:6.0f}")


def calibrate(root, cells):
    print(f"{'cell':30s} {'a1':>4} {'a1x':>4} {'gain':>5} {'lost':>5} | "
          + ' '.join(f"{f:>5.5s}" for f in FAMS))
    tot = collections.Counter()
    for c in cells:
        ra, Ra, rb, Rb = {}, set(), {}, set()
        for s in sorted(glob.glob(os.path.join(root, f"score_{c}__s*_a1"))):
            x = load(s); ra.update(x[0]); Ra |= x[1]
        for s in sorted(glob.glob(os.path.join(root, f"score_{c}__s*_a1x"))):
            y = load(s); rb.update(y[0]); Rb |= y[1]
        if not rb:
            print(f"{c:30s} (no clean _a1x pass)")
            continue
        for f in FAMS:
            S = {i for i, r in ra.items() if sig(r) == f}
            tot[f] += len(S); tot[f + "_R"] += len(S & Rb)
        tot.update(gain=len(Rb - Ra), lost=len(Ra - Rb))
        print(f"{c:30s} {len(Ra):4d} {len(Rb):4d} {len(Rb-Ra):5d} {len(Ra-Rb):5d} | "
              + ' '.join(f"{sum(1 for r in ra.values() if sig(r)==f):5d}" for f in FAMS))
    if tot:
        print(f"\ntotals: gained {tot['gain']} / lost {tot['lost']}")
        for f in FAMS:
            print(f"  recovery {f:10s} n={tot[f]:4d} recovered={tot[f+'_R']:3d} "
                  f"{tot[f+'_R']/max(tot[f],1):6.1%}")


def dupes(root, globs):
    """How many times is each instance scored within a single job?"""
    print(f"{'cell':34s} {'shards':>7} {'rows':>6} {'distinct':>9} {'rows/inst':>10}")
    for g in globs:
        for d in sorted(glob.glob(os.path.join(root, g))):
            files = [p for p in glob.glob(os.path.join(d, "shard_*of*.jsonl"))
                     if not p.endswith(".swebench.jsonl")]
            if not files:
                continue
            c = collections.Counter()
            for p in files:
                for line in open(p):
                    line = line.strip()
                    if line:
                        c[json.loads(line).get("instance_id")] += 1
            hist = dict(collections.Counter(c.values()))
            print(f"{os.path.basename(d):34s} {len(files):7d} {sum(c.values()):6d} "
                  f"{len(c):9d} {str(hist):>10}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    mode, root, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    {"audit": audit, "calibrate": calibrate, "dupes": dupes}[mode](root, rest)
