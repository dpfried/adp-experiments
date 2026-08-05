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


def read_attempt1(root, tagbase):
    """Cap-hit over the FIRST critic attempt only, pooled over both shards.

    Why attempt 1 rather than the aggregated `output.jsonl`:

      * `output.jsonl` is an append log of every attempt while the job runs, and
        is REWRITTEN by `aggregate_results(final_output_file="output.jsonl")`
        only at the very end. Reading it mid-flight and reading it after
        completion are two different measurements, so an incomplete cell is not
        comparable with a complete one.
      * `benchmarks/utils/evaluation.py` raises temperature 0.0 -> 0.1 for
        attempt > 1. Attempts 2-3 are therefore a different sampling regime, and
        which instances even get an attempt 2 depends on the critic. Pooling all
        attempts compares different mixtures per cell.
      * Attempt 1 is deterministic (temp 0), runs on ALL instances in both
        cells, and its rollouts are frozen in `output.critic_attempt_1.jsonl`,
        which aggregation never touches.

    So attempt 1 is the matched comparison, and it is available for a cell that
    never finishes. Returns None if the attempt-1 files are absent.
    """
    ids, capped, tx = set(), set(), set()
    found = False
    for sh in SHARDS:
        d = os.path.join(root, "runs", f"out_{tagbase}__s{sh}")
        pat = os.path.join(d, "**", "output.critic_attempt_1.jsonl")
        for f in glob.glob(pat, recursive=True):
            found = True
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
                ids.add(iid)
                if r.get("history") or r.get("events"):
                    tx.add(iid)
                if "MaxIterationsReached" in str(r.get("error") or ""):
                    capped.add(iid)
    if not found:
        return None
    return {"all": ids, "capped": capped, "have": tx}


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
            "errored": errored, "native": native}


def read_score(root, tagbase, suffix=""):
    resolved, scored = set(), 0
    for sh in SHARDS:
        rep = os.path.join(root, "runs", f"score_{tagbase}__s{sh}{suffix}",
                           "merged.report.json")
        if not os.path.exists(rep):
            continue
        try:
            j = json.load(open(rep))
        except Exception:
            continue
        resolved |= set(j.get("resolved_ids") or [])
        scored += len(j.get("resolved_ids") or []) + len(j.get("unresolved_ids") or [])
    return resolved, scored


def count_retried(root, tagbase):
    """Instances that got a 2nd or later critic attempt, i.e. the ONLY instances on
    which the aggregated score and the attempt-1-only score can possibly differ.

    Used as an exact validity check on the attempt-1 batch, not as a heuristic:
    |aggregate_resolved - attempt1_resolved| <= count_retried() is an identity, so a
    violation means the attempt-1 scoring is corrupt rather than informative. The
    live risk is cross-cell Apptainer sandbox collision -- every ladder cell scores
    the SAME instance set and the scoring sbatch prunes sandboxes keyed by
    instance_id alone, so a sandbox deleted mid-verify yields a spurious
    `unresolved` that would depress every cell and mimic a real effect.
    """
    ids = set()
    for sh in SHARDS:
        d = os.path.join(root, "runs", f"out_{tagbase}__s{sh}")
        for f in glob.glob(os.path.join(d, "**", "output.critic_attempt_*.jsonl"),
                           recursive=True):
            n = os.path.basename(f).rsplit("_", 1)[-1].split(".")[0]
            if not n.isdigit() or int(n) < 2:
                continue
            for line in open(f):
                line = line.strip()
                if not line:
                    continue
                try: ids.add(json.loads(line)["instance_id"])
                except Exception: pass
    return ids


def _patchlen(r):
    tr = r.get("test_result") if isinstance(r.get("test_result"), dict) else {}
    return len((tr.get("git_patch") or r.get("git_patch") or ""))


def count_discarded(root, tagbase):
    """Instances whose scored record has an empty patch while an earlier critic
    attempt produced a non-empty one -- the harness aggregator defect documented
    in parity_ladder_amendment.md Addendum 5.

    `aggregate_results()` breaks a rank tie in favour of the LATEST attempt with a
    rank function that never checks whether a patch exists, so a degenerate final
    retry silently discards a good earlier patch. Only a model that retries a lot
    is exposed, which makes it asymmetric between base and arm -- exactly the
    comparison this script exists to make. It must therefore be COUNTED, not
    assumed absent: measured 0/400 on the arm cells and 9/50 on base cell F s01.
    """
    empty_tot = disc_tot = 0
    for sh in SHARDS:
        d = os.path.join(root, "runs", f"out_{tagbase}__s{sh}")
        g = glob.glob(os.path.join(d, "**", "output.jsonl"), recursive=True)
        if not g:
            continue
        dd = os.path.dirname(g[0])
        fin, att = {}, collections.defaultdict(int)
        for line in open(os.path.join(dd, "output.jsonl")):
            line = line.strip()
            if not line:
                continue
            try: r = json.loads(line)
            except Exception: continue
            fin[r["instance_id"]] = max(fin.get(r["instance_id"], 0), _patchlen(r))
        for a in (1, 2, 3, 4, 5):
            f = os.path.join(dd, f"output.critic_attempt_{a}.jsonl")
            if not os.path.exists(f):
                continue
            for line in open(f):
                line = line.strip()
                if not line:
                    continue
                try: r = json.loads(line)
                except Exception: continue
                att[r["instance_id"]] = max(att[r["instance_id"]], _patchlen(r))
        empty = [i for i, L in fin.items() if L == 0]
        empty_tot += len(empty)
        disc_tot += sum(1 for i in empty if att.get(i, 0) > 0)
    return empty_tot, disc_tot


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

    # --- L3, the pre-registered placebo/plumbing check (amendment Addendum 1).
    # The 500-iteration cap counts AGENT ITERATIONS, not tokens, so trimming ~4
    # tokens per assistant turn has no mechanism by which it should move the
    # cap-hit rate. Registered: |cap-hit(F) - cap-hit(E)| <= 8pp. Deliberately
    # insensitive (~1.6*SE at n=100): it is a guard against the ladder silently
    # measuring budget instead of parity, NOT a test of the stub's effect on
    # budget, and must not be reported as the latter.
    # Gate on BOTH base cells being complete. A capped instance only lands in
    # output_errors.jsonl when it finishes hitting the cap, so a mid-flight cell
    # has a meaningless cap-hit denominator -- evaluating L3 early prints a loud
    # FAIL that is purely a partial-data artifact.
    ce, cf = cov.get("E"), cov.get("F")
    NEXP = 100                       # 2 shards x 50 instances per cell

    # Attempt-1-matched form, printed first because it is the better-matched
    # measurement and does not depend on either cell finishing. See
    # read_attempt1() for why the aggregated file is not comparable mid-flight.
    _tag = {c[0]: c[1] for c in CELLS}
    a1e, a1f = read_attempt1(root, _tag["E"]), read_attempt1(root, _tag["F"])
    if a1e and a1f and a1e["all"] and a1f["all"]:
        inter = a1e["all"] & a1f["all"]
        pe = 100.0 * len(a1e["capped"] & inter) / max(1, len(inter))
        pf = 100.0 * len(a1f["capped"] & inter) / max(1, len(inter))
        d1 = pf - pe
        v1 = "PASS" if abs(d1) <= 8.0 else "FAIL"
        print(f"\n  L3 placebo (cap-hit), ATTEMPT-1 MATCHED (temp 0 in both, "
              f"n={len(inter)} shared instances):")
        print(f"       E={pe:.1f}% ({len(a1e['capped'] & inter)})  "
              f"F={pf:.1f}% ({len(a1f['capped'] & inter)})  "
              f"delta={d1:+.1f}pp  -> {v1} the registered <=8pp gate")
        print(f"       E attempt-1 coverage {len(a1e['all'])}/{NEXP}, "
              f"F {len(a1f['all'])}/{NEXP}. This is a deviation from the letter of "
              f"the\n       registration (which named the aggregated cell) and a "
              f"strictly better match on\n       attempt index, temperature and "
              f"instance set. Both forms are reported; neither\n       is quietly "
              f"substituted for the other.")
    if ce and cf and (len(ce["all"]) < NEXP or len(cf["all"]) < NEXP):
        print(f"\n  L3 placebo (cap-hit), REGISTERED FORM: awaiting completion "
              f"(E {len(ce['all'])}/{NEXP}, F {len(cf['all'])}/{NEXP}) -- "
              f"cap-hit is only interpretable on a complete cell")
        # E runs 3 attempts per shard and a gagged base burns the full 500
        # iterations, so E may exhaust its walltime before 100/100. If it does,
        # the registered form is never evaluable and would silently print
        # "awaiting" forever. Report the intersection-paired version too, but
        # only as an explicitly labelled DEVIATION: every instance that reached
        # a terminal state has a real classification, so the intersection is a
        # legitimate paired comparison -- it is just not the test that was
        # registered, and swapping it in silently is the move this whole
        # analysis exists to forbid.
        inter = (ce["have"] | ce["errored"]) & (cf["have"] | cf["errored"])
        if inter:
            re_ = 100.0 * len(ce["capped"] & inter) / len(inter)
            rf_ = 100.0 * len(cf["capped"] & inter) / len(inter)
            dd = rf_ - re_
            would = "would FAIL" if abs(dd) > 8.0 else "would PASS"
            print(f"  L3 (DEVIATION, intersection-paired, n={len(inter)} "
                  f"terminal in both cells): E={re_:.1f}%  F={rf_:.1f}%  "
                  f"delta={dd:+.1f}pp  -> {would} the registered <=8pp gate")
            print("       Not a substitute for the registered test. The reading of an L3 "
                  "failure was\n       pre-committed in amendment Addendum 6 at 32pp, on "
                  "partial data, before this\n       number existed -- so it cannot have "
                  "been retrofitted to it.")
    elif ce and cf and ce["all"] and cf["all"]:
        re_ = 100.0 * len(ce["capped"]) / len(ce["all"])
        rf_ = 100.0 * len(cf["capped"]) / len(cf["all"])
        dd = rf_ - re_
        verdict = ("PASS (plumbing guard clear)" if abs(dd) <= 8.0 else
                   "FAIL -- inspect the rendered prompts and the vLLM logs BEFORE "
                   "interpreting any E/F\n           result; an E-F delta would be "
                   "confounded by budget, not serving format")
        print(f"\n  L3 placebo (cap-hit, registered |F-E| <= 8pp): "
              f"E={re_:.1f}%  F={rf_:.1f}%  delta={dd:+.1f}pp  -> {verdict}")
    else:
        print("\n  L3 placebo (cap-hit): awaiting both base cells")

    print()
    print("=" * 100)
    print("SCORE (over ALL instances incl. no-transcript rows; scoring reads both files)")
    print("=" * 100)
    print(f"{'cell':4} {'resolved':>9} {'scored':>7} {'rate':>7} "
          f"{'emptyPatch':>11} {'discarded':>10} {'repaired':>18} {'attempt1':>12}")
    warn = []
    for c, tagbase, model, *_ in CELLS:
        res, n = read_score(root, tagbase)
        empty, disc = count_discarded(root, tagbase)
        rres, rn = read_score(root, tagbase, "_reagg")
        # The repaired resolved set is the original PLUS anything the rescued
        # records resolve; the rescued instances were scored on an empty patch in
        # the original, so they are necessarily unresolved there.
        if disc and rn:
            pair = f"{len(res | rres):3} / {n:3}  ({len(rres)} new)"
        elif disc:
            pair = "MISSING"
            warn.append((c, disc))
        else:
            pair = "n/a (no-op)"
        sc[c] = (res, n)
        if disc and rn:
            sc[c + "*"] = (res | rres, n)
        # Attempt-1-only scoring: the COMPUTE-MATCHED contrast (amendment
        # Addendum 7). One rollout per instance, temp 0, no critic selection, in
        # every cell -- unlike the aggregated column, where the critic handed
        # base 2.32 rollouts/instance and the arm ~1.04.
        a1res, a1n = read_score(root, tagbase, "_a1")
        if a1n:
            sc[c + "@1"] = (a1res, a1n)
            a1col = f"{len(a1res):3} / {a1n:3}"
        else:
            a1col = "-"
        print(f"{c:4} {len(res):9} {n:7} {pct(len(res), n):>7} "
              f"{empty:11} {disc:10} {pair:>18} {a1col:>12}")
    print("  discarded = scored record has an empty patch though an earlier critic\n"
          "  attempt produced one (amendment Addendum 5). Measured, never assumed:\n"
          "  the harness tie-break is patch-blind and only bites models that retry,\n"
          "  so it is asymmetric between base and arm. Arm cells measured 0/400.")
    if warn:
        for c, d in warn:
            print(f"  !! CELL {c}: {d} instances had a good patch DISCARDED by the "
                  f"harness aggregator and no repaired scoring exists. This cell's "
                  f"score is a BIASED-LOW estimate; do not report it alone. Run "
                  f"repair_aggregate.py and score the repaired subset.")

    print()
    print("=" * 100)
    print("ADJACENT-RUNG DELTAS (attribution rule: assign an effect to the SMALLEST rung)")
    print("=" * 100)
    print("  These use the AS-HARNESS resolved sets (the pre-registered primary). Where\n"
          "  the SCORE table above shows a repaired pair for a base cell, read any rung\n"
          "  touching that cell alongside the repaired number.")
    print("  !! CROSS-MODEL RUNGS (E-A, F-B) ARE NOT COMPUTE-MATCHED. See the rollout\n"
          "     parity table above: the critic retried base 71/100 times and the arm\n"
          "     2-5/100, so base got ~2.2x the rollouts AND a best-of-3 selection at\n"
          "     temperature 0.1. That inflates base. The tie-break defect deflates base.\n"
          "     Biases run BOTH ways and one is unquantified, so F-B is a two-sided point\n"
          "     estimate, NOT a lower bound -- an earlier version of this script labelled\n"
          "     it [LB], which was wrong. The compute-matched contrast is the attempt-1\n"
          "     scoring (score_attempt1_ladder.sh); read that before quoting F-B.")
    rungs = [("B", "A", "stub removed"),
             ("C", "B", "wrapper & path matched"),
             ("D", "C", "prohibition & 5-phase list"),
             ("F", "E", "stub removed (base)"),
             ("C", "A", "L1: format rungs jointly"),
             ("E", "A", "base vs arm, both stock"),
             ("F", "B", "base vs arm, nostub [NOT compute-matched]")]
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

    # --- COMPUTE-MATCHED rungs (amendment Addendum 7). Same paired-intersection
    # rule, but over the attempt-1-only scorings, where every cell got exactly
    # one temperature-0 rollout per instance and no critic selection. This is the
    # contrast to quote for anything crossing the base/arm boundary.
    a1keys = [c for c in "ABCDEF" if (c + "@1") in sc]
    if a1keys:
        print()
        print("=" * 100)
        print("COMPUTE-MATCHED RUNGS (attempt 1 only: 1 rollout/instance, temp 0, no selection)")
        print("=" * 100)
        print(f"  cells with attempt-1 scoring: {' '.join(a1keys)}")

        # Validity gate, registered in the amendment before these scores existed.
        # Aggregate and attempt-1 differ only on retried instances, so a movement
        # larger than the retry count is impossible and indicts the scoring run.
        print()
        print("  VALIDITY GATE (|agg - a1| <= #retried is an identity, not a band):")
        bad = []
        for c in a1keys:
            ra, na = sc.get(c, (set(), 0))
            r1, n1 = sc.get(c + "@1", (set(), 0))
            if not na or not n1:
                continue
            retried = count_retried(root, _tag[c])
            move = abs(len(ra) - len(r1))
            ok = move <= len(retried)
            flag = "ok" if ok else "!! IMPOSSIBLE"
            if not ok:
                bad.append(c)
            print(f"    {c}: agg {len(ra):3} vs a1 {len(r1):3}  |move|={move:3} "
                  f" retried={len(retried):3}  {flag}")
            # On a NON-retried instance, agg and a1 score the byte-identical
            # rollout, so any disagreement is scoring non-reproducibility. A couple
            # of these are expected from genuinely flaky tests; many indicate the
            # sandbox collision. Reported either way, hard-failed only past
            # tolerance, since the benign explanation is real at small counts.
            TOL = 2
            disagree = ((ra - r1) | (r1 - ra)) - retried
            if disagree:
                mark = "!! ABOVE TOLERANCE" if len(disagree) > TOL else f"(<= tol {TOL}, flaky tests plausible)"
                if len(disagree) > TOL:
                    bad.append(c)
                print(f"       {len(disagree)} instance(s) disagree on an identical, "
                      f"never-retried rollout {mark}")
        if bad:
            print("  !! GATE FAILED for cells: " + " ".join(sorted(set(bad))))
            print("  !! Do NOT interpret the matched rungs below. The most likely cause is\n"
                  "  !! cross-cell sandbox collision (all cells share one instance set).\n"
                  "  !! Re-run the attempt-1 scoring serially and re-read.")
        else:
            print("  gate passed for all scored cells.")
        for hi, lo, what in [("B", "A", "stub removed"),
                             ("C", "B", "wrapper & path matched"),
                             ("D", "C", "prohibition & 5-phase list"),
                             ("C", "A", "L1: format rungs jointly"),
                             ("F", "E", "stub removed (base)"),
                             ("F", "B", "base vs arm, nostub [PRIMARY, matched]")]:
            rh, nh = sc.get(hi + "@1", (set(), 0))
            rl, nl = sc.get(lo + "@1", (set(), 0))
            if not nh or not nl:
                print(f"  {hi}-{lo:2} {what:40} (awaiting attempt-1 scoring)")
                continue
            a1h = read_attempt1(root, _tag[hi]); a1l = read_attempt1(root, _tag[lo])
            inter = a1h["all"] & a1l["all"]
            ph, pl = len(rh & inter), len(rl & inter)
            b = len((rh - rl) & inter); c_ = len((rl - rh) & inter)
            se = (b + c_) ** 0.5
            sig = f"{abs(ph - pl) / se:.2f}" if se else "inf"
            gate = "" if (se and abs(ph - pl) >= 2 * se) else "   (|net| < 2*SE: UNINFORMATIVE)"
            print(f"  {hi}-{lo:2} {what:40} {pl:3} -> {ph:3} on n={len(inter):3} "
                  f"net={ph - pl:+3}  b={b} c={c_}  SE={se:.2f}  {sig} sigma{gate}")
        print("  Compare each line with its aggregated counterpart above. Where the two\n"
              "  disagree, the aggregated number is the one carrying the unequal-rollout\n"
              "  confound, not this one.")

    if not stats:
        print("\n(no stats.jsonl supplied -- behavioural section skipped)")
        return

    print()
    print("=" * 100)
    print("BEHAVIOURAL, INTERSECTION-PAIRED (denominator printed; never assumed 100)")
    print("=" * 100)
    # NOTE: an earlier version of this list said "n_think_calls", which is not a
    # key emitted by extract_traj_stats.py (the key is "n_think"). The metric
    # loop skips absent keys silently, so S1's *registered primary outcome* was
    # being dropped without a warning. Names are now asserted against the data
    # below -- a typo here must fail loudly, not vanish.
    METRICS = ["n_think", "n_native_think", "n_reason_content", "n_any_reason",
               "n_turns_with_thought", "think_arg_chars_total",
               "verified_after_edit", "ran_any_test", "n_test_runs",
               "n_test_ok_runs", "n_edits", "empty_patch", "finished",
               "finish_claims_success", "n_actions", "n_terminal",
               "n_file_editor", "dup_action_frac"]
    # Per-action normalisation. A cell that contracts the whole trajectory makes
    # every raw COUNT move together, so a raw-count delta cannot distinguish "does
    # this less" from "does less of everything" -- measured on D-C, where 4 of the
    # metrics that cleared 2*SE as raw counts did not survive normalisation. Rates
    # and binary flags are already length-immune, so only counts are normalised.
    NOT_A_COUNT = {"dup_action_frac", "n_actions", "empty_patch", "finished",
                   "finish_claims_success", "verified_after_edit", "ran_any_test"}
    present = set()
    for m in stats.values():
        for row in m.values():
            present |= set(row)
    missing = [m for m in METRICS if m not in present]
    if missing:
        print(f"  !! METRICS not present in stats.jsonl (typo?): {missing}")
        METRICS = [m for m in METRICS if m in present]
    # C-A is the registered L1 rung and must appear behaviourally, not only on
    # score: length-normalising the counts shows the format rungs are NOT inert
    # on every channel (think rate per action moves ~4 sigma), even though score
    # is unmoved. Reading L1 off the score line alone would have missed it.
    for hi, lo, what in rungs[:4] + [("C", "A", "L1: format rungs jointly"),
                                     ("E", "A", "base vs arm, both stock"),
                                     ("F", "B", "base vs arm, nostub [NOT compute-matched]")]:
        sh_, sl_ = stats.get(hi, {}), stats.get(lo, {})
        inter = sorted(set(sh_) & set(sl_) & cov[hi]["have"] & cov[lo]["have"])
        if not inter:
            print(f"  {hi}-{lo}: no paired transcripts yet")
            continue
        cross_model = hi in ("E", "F") and lo in ("A", "B")
        print(f"\n  {hi} vs {lo} -- {what}   paired n={len(inter)} "
              f"(of {len(cov[hi]['have'])} / {len(cov[lo]['have'])} transcripts)")
        if cross_model:
            # Corrected once F completed. The cap-survivorship caveat belongs to
            # E (stub), which loses ~half its instances to the iteration cap, NOT
            # to F: F hit the cap 0/100 and has a transcript for all 100. So F-B
            # is a COMPLETE paired comparison, and its lower-bound direction comes
            # from the aggregator tie-break alone (15/100 good patches discarded,
            # arm 0/400) -- which the repaired column addresses directly.
            capfrac = 100.0 * len(cov[hi]["capped"]) / max(1, len(cov[hi]["all"]))
            if capfrac >= 5.0:
                print(f"      NOTE: cross-model AND survivorship-truncated -- {hi} "
                      f"lost {capfrac:.0f}% of instances to the\n            iteration "
                      f"cap (its LONGEST runs), so the surviving pairs are not a "
                      f"random subset.\n            Read as a bound, not a point "
                      f"estimate.")
            else:
                print(f"      NOTE: cross-model, but {hi} is COMPLETE "
                      f"({len(cov[hi]['have'])} transcripts, {capfrac:.0f}% cap-hit), "
                      f"so there is no\n            survivorship truncation here. The "
                      f"lower-bound direction comes only from the\n            "
                      f"aggregator tie-break, which depresses base and is 0/400 on "
                      f"the arm -- see the\n            repaired column.")
        def paired_delta(vals):
            """mean of hi, mean of lo, mean paired diff, and its SE.

            SE of the mean paired difference, computed from this run -- registered
            in place of an unavailable same-condition replicate."""
            n = len(vals)
            mh = sum(a for a, _ in vals)/n
            ml = sum(b for _, b in vals)/n
            d = [a - b for a, b in vals]
            md = sum(d)/n
            var = sum((x-md)**2 for x in d)/(n-1) if n > 1 else 0.0
            return mh, ml, md, (var/n) ** 0.5

        for m in METRICS:
            rows = [(sh_[i], sl_[i]) for i in inter]
            pairs, rates = [], []
            for rh_, rl_ in rows:
                a, b = rh_.get(m), rl_.get(m)
                if not (isinstance(a, (int, float, bool))
                        and isinstance(b, (int, float, bool))):
                    continue
                pairs.append((float(a), float(b)))
                na, nb = rh_.get("n_actions"), rl_.get("n_actions")
                if m not in NOT_A_COUNT and na and nb:
                    rates.append((float(a)/na, float(b)/nb))
            if not pairs:
                continue
            mh, ml, md, se = paired_delta(pairs)
            flag = "" if (se and abs(md) >= 2*se) else "   (|d| < 2*SE: UNINFORMATIVE)"
            print(f"      {m:24} {ml:8.3f} -> {mh:8.3f}   delta {md:+8.3f} "
                  f"+/- {se:.3f}{flag}")
            if len(rates) > 1:
                rh2, rl2, rd, rse = paired_delta(rates)
                rflag = ("" if (rse and abs(rd) >= 2*rse)
                         else "   (|d| < 2*SE: LENGTH ARTIFACT)")
                print(f"        {'per action':22} {rl2:8.4f} -> {rh2:8.4f}   "
                      f"delta {rd:+8.4f} +/- {rse:.4f}{rflag}")


if __name__ == "__main__":
    main()
