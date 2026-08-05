#!/usr/bin/env python3
"""Parity-ladder readout: coverage, cap-hit rate, score, and intersection-safe
behavioural diffs for the 6 cells.

Pure stdlib. Runs FAIR-side.

  python ladder_readout.py <SWEBENCH_ROOT> [--stats <stats.jsonl>]

Coverage/cap-hit works as soon as infer lands (no scoring needed).
Score columns fill in once score_<tag>/merged.report.json exists.
Behavioural columns fill in once extract_traj_stats.py has been run.

Registered discipline this script enforces (parity_ladder_amendment.md):
  * base behavioural n is ~60/cell, not 100 -- every proportion prints its own
    surviving denominator, never an assumed 100.
  * E vs F is compared on the INTERSECTION of instances with a transcript in
    both cells, so a shift in *which* instances survive cannot masquerade as a
    shift in behaviour. Same rule applied to every adjacent arm rung.
  * cap-hit rate is a secondary outcome for all six cells.
"""
import json, os, sys, glob, collections

CELLS = [
    ("A", "par_A_arm_stock_evalp",   "arm",  "stock",  "eval default.j2"),
    ("B", "par_B_arm_nostub_evalp",  "arm",  "nostub", "eval default.j2"),
    ("C", "par_C_arm_nostub_wrap",   "arm",  "nostub", "train_wrapper.j2"),
    ("D", "par_D_arm_nostub_trainp", "arm",  "nostub", "train_swezero_noenv"),
    ("E", "par_E_base_stock_evalp",  "base", "stock",  "eval default.j2"),
    ("F", "par_F_base_nostub_evalp", "base", "nostub", "eval default.j2"),
]
SHARDS = ["00", "01"]


def _native_think(rec):
    """Count native <think> blocks in assistant-generated content.

    Registered secondary (addendum 2): if nostub unlocks a native reasoning
    block, a rise in think-TOOL rate could be reasoning relocating between
    channels rather than un-suppressing. Measured for every cell, not assumed.
    Baseline under stock: arm 0/500, base 5/324.
    """
    n = 0
    for ev in (rec.get("history") or rec.get("events") or []):
        for key in ("content", "thought", "message", "text"):
            v = ev.get(key)
            if isinstance(v, list):
                v = " ".join(str(p.get("text", "")) if isinstance(p, dict) else str(p)
                             for p in v)
            if isinstance(v, str) and v:
                n += v.count("<think>")
    return n


def read_cell(root, tagbase):
    """Return per-instance coverage for a cell, pooled over its two shards."""
    have, capped, errored, all_ids = set(), set(), set(), set()
    native = {}
    for sh in SHARDS:
        tag = f"{tagbase}__s{sh}"
        d = os.path.join(root, "runs", f"out_{tag}")
        for f in glob.glob(os.path.join(d, "**", "output.jsonl"), recursive=True):
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
                all_ids.add(iid)
                hist = r.get("history") or r.get("events") or []
                if hist:
                    have.add(iid)
                    native[iid] = _native_think(r)
        for f in glob.glob(os.path.join(d, "**", "output_errors.jsonl"), recursive=True):
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
                all_ids.add(iid)
                errored.add(iid)
                if "MaxIterationsReached" in str(r.get("error") or ""):
                    capped.add(iid)
    # an instance only counts as capped/no-transcript if it never produced one
    capped -= have
    blanks = all_ids - have
    return {"all": all_ids, "have": have, "blank": blanks, "capped": capped,
            "native": native}


def read_score(root, tagbase):
    resolved, scored = set(), 0
    for sh in SHARDS:
        rep = os.path.join(root, "runs", f"score_{tagbase}__s{sh}", "merged.report.json")
        if not os.path.exists(rep):
            continue
        try:
            j = json.load(open(rep))
        except Exception:
            continue
        resolved |= set(j.get("resolved_ids") or [])
        scored += len(j.get("resolved_ids") or []) + len(j.get("unresolved_ids") or [])
    return resolved, scored


def load_stats(path):
    by_model = collections.defaultdict(dict)
    if not path or not os.path.exists(path):
        return by_model
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        by_model[r.get("model")][r.get("instance_id")] = r
    return by_model


def pct(k, n):
    return f"{100.0*k/n:5.1f}%" if n else "    - "


def main():
    root = sys.argv[1]
    statsp = None
    if "--stats" in sys.argv:
        statsp = sys.argv[sys.argv.index("--stats") + 1]
    stats = load_stats(statsp)

    cov, sc = {}, {}
    print("=" * 100)
    print("COVERAGE + CAP-HIT (secondary outcome, all six cells)")
    print("=" * 100)
    print(f"{'cell':4} {'model':5} {'tmpl':7} {'task stmt':21} "
          f"{'seen':>5} {'transcript':>11} {'no-tx':>6} {'capped':>7} {'cap%':>7} "
          f"{'nativeThink':>12}")
    for c, tagbase, model, tmpl, stmt in CELLS:
        v = read_cell(root, tagbase)
        cov[c] = v
        n = len(v["all"])
        nat = sum(1 for x in v["native"].values() if x)
        print(f"{c:4} {model:5} {tmpl:7} {stmt:21} "
              f"{n:5} {len(v['have']):11} {len(v['blank']):6} {len(v['capped']):7} "
              f"{pct(len(v['capped']), n):>7} "
              f"{nat:5}/{len(v['have']):<6}")
    print("  nativeThink = records emitting a native <think> block / records with a "
          "transcript.\n  Registered secondary: baseline under stock is arm 0/500, "
          "base 5/324. If non-zero\n  in B/C/D, S1 must be re-scored on native+tool "
          "combined before being read.")

    print()
    print("=" * 100)
    print("SCORE (over ALL instances incl. no-transcript rows; scoring reads both files)")
    print("=" * 100)
    print(f"{'cell':4} {'resolved':>9} {'scored':>7} {'rate':>7}")
    for c, tagbase, *_ in CELLS:
        res, n = read_score(root, tagbase)
        sc[c] = (res, n)
        print(f"{c:4} {len(res):9} {n:7} {pct(len(res), n):>7}")

    print()
    print("=" * 100)
    print("ADJACENT-RUNG DELTAS (attribution rule: assign an effect to the SMALLEST rung)")
    print("=" * 100)
    rungs = [("B", "A", "stub removed"),
             ("C", "B", "wrapper & path matched"),
             ("D", "C", "prohibition & 5-phase list"),
             ("F", "E", "stub removed (base)"),
             ("C", "A", "L1: format rungs jointly"),
             ("E", "A", "base vs arm, both stock"),
             ("F", "B", "base vs arm, both nostub [LB]")]
    for hi, lo, what in rungs:
        rh, nh = sc.get(hi, (set(), 0))
        rl, nl = sc.get(lo, (set(), 0))
        if not nh or not nl:
            print(f"  {hi}-{lo:2} {what:32} score: (awaiting scoring)")
            continue
        # paired on the intersection of scored instances
        inter = (cov[hi]["all"] & cov[lo]["all"])
        ph = len(rh & inter); pl = len(rl & inter)
        d = ph - pl
        flip_up = len((rh - rl) & inter); flip_dn = len((rl - rh) & inter)
        print(f"  {hi}-{lo:2} {what:32} score: {ph:3} vs {pl:3} on n={len(inter):3} "
              f"delta={d:+3}   (gained {flip_up}, lost {flip_dn})")

    if not stats:
        print("\n(no stats.jsonl supplied -- behavioural section skipped)")
        return

    print()
    print("=" * 100)
    print("BEHAVIOURAL, INTERSECTION-PAIRED (denominator printed; never assumed 100)")
    print("=" * 100)
    METRICS = ["verified_after_edit", "ran_any_test", "n_think_calls",
               "finish_claims_success", "n_actions"]
    for hi, lo, what in rungs[:4] + [("F", "B", "base vs arm, both nostub [LB]")]:
        sh_, sl_ = stats.get(hi, {}), stats.get(lo, {})
        inter = sorted(set(sh_) & set(sl_) & cov[hi]["have"] & cov[lo]["have"])
        if not inter:
            print(f"  {hi}-{lo}: no paired transcripts yet")
            continue
        cross_model = (hi, lo) == ("F", "B")
        print(f"\n  {hi} vs {lo} -- {what}   paired n={len(inter)} "
              f"(of {len(cov[hi]['have'])} / {len(cov[lo]['have'])} transcripts)")
        if cross_model:
            print("      NOTE: cross-model. base drops ~40% to the iteration cap "
                  "(its LONGEST runs) vs ~0% for\n            the arm, so this is a "
                  "DIRECTIONAL LOWER BOUND, not a point estimate -- the true\n"
                  "            base-vs-arm gap is if anything larger.")
        for m in METRICS:
            pairs = [(sh_[i].get(m), sl_[i].get(m)) for i in inter]
            pairs = [(a, b) for a, b in pairs
                     if isinstance(a, (int, float, bool)) and isinstance(b, (int, float, bool))]
            if not pairs:
                continue
            d = [float(a) - float(b) for a, b in pairs]
            n = len(d)
            mh = sum(float(a) for a, _ in pairs)/n
            ml = sum(float(b) for _, b in pairs)/n
            md = sum(d)/n
            # SE of the mean paired difference, computed from this run --
            # registered in place of an unavailable same-condition replicate.
            var = sum((x-md)**2 for x in d)/(n-1) if n > 1 else 0.0
            se = (var/n) ** 0.5
            flag = "" if (se and abs(md) >= 2*se) else "   (|d| < 2*SE: UNINFORMATIVE)"
            print(f"      {m:24} {ml:8.3f} -> {mh:8.3f}   delta {md:+8.3f} "
                  f"+/- {se:.3f}{flag}")


if __name__ == "__main__":
    main()
