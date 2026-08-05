#!/usr/bin/env python3
"""The only sanctioned reader for SWE-bench-eval rollout directories.

WHY THIS EXISTS
---------------
The SDK harness scatters one logical eval cell across several jsonl files with
different completeness guarantees. Reading the wrong one, or combining them
naively, has silently produced *four* wrong published conclusions in this
campaign (see analysis/ANALYSIS_HOUSE_RULES.md, rule 1). Every failure mode is
silent: the file you opened parses fine and looks complete.

The three traps, all measured on real cells (2026-08-05):

  1. `output.jsonl` omits instances that ended in a terminal error. Those live
     in `output_errors.jsonl`. A `MaxIterationsReached` search over
     `output.jsonl` is *guaranteed* to return zero.
  2. The two files are NOT disjoint, contrary to the folklore. Cell
     E_base_stock_evalp had 3 overlapping instance_ids in shard 00 and 6 in
     shard 01. Concatenating them double-counts.
  3. `output.jsonl` is an APPEND LOG during the run, rewritten by
     `aggregate_results` only at the end. Mid-flight it contains duplicate
     instance_ids (observed: 1 in E/s00, 1 in E/s01, 4 in E/s01's error file).
     A cell whose job died on walltime is left as a raw append log forever.

`output.critic_attempt_N.jsonl` has none of these problems: it is frozen at
write time, carries exactly one row per instance, and is attempt-matched (which
you need anyway, because the harness spends more rollouts on whichever model its
critic dislikes -- see rule 2). Prefer it. This module does, automatically.

USAGE
-----
    from load_rollouts import load_cell

    cell = load_cell(out_dir, select="…/select/shard_00of10.txt")
    # raises RolloutIntegrityError unless the cell covers `select` exactly,
    # with one row per instance.

    cell.rows            # list[dict], one per instance, deterministic order
    cell.by_id           # dict[instance_id] -> row
    cell.taxonomy        # Counter: ok / cap500 / timeout / stuck / ENOSPC / other_err
    cell.gradeable       # ids whose test_result.git_patch is non-empty
    cell.transcript_ids  # ids with a non-empty history  (BEHAVIOURAL denominator)
    cell.source          # which file it actually read
    print(cell.report()) # one-screen provenance + integrity summary

Pure stdlib: runs under any venv on the cluster.
"""

from __future__ import annotations

import collections
import glob
import json
import os
from dataclasses import dataclass, field

__all__ = ["load_cell", "Cell", "RolloutIntegrityError", "read_jsonl", "classify"]


class RolloutIntegrityError(AssertionError):
    """A cell failed a completeness/uniqueness invariant. Do not compute on it."""


# --------------------------------------------------------------------------- io


def read_jsonl(path):
    """Tolerant jsonl reader. Returns (rows, n_unparseable)."""
    rows, bad = [], 0
    if not os.path.exists(path):
        return rows, bad
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                bad += 1
    return rows, bad


def _resolve_run_dir(out_dir):
    """out_par_X__s00/ -> …/<dataset>/<provider>/<run-tag>/ where the jsonls live."""
    if glob.glob(os.path.join(out_dir, "output*.jsonl")):
        return out_dir
    hits = sorted(glob.glob(os.path.join(out_dir, "*", "*", "*", "")))
    hits = [h for h in hits if glob.glob(os.path.join(h, "output*.jsonl"))]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise RolloutIntegrityError(f"no output*.jsonl anywhere under {out_dir}")
    raise RolloutIntegrityError(
        f"{len(hits)} candidate run dirs under {out_dir}; pass the leaf explicitly:\n  "
        + "\n  ".join(hits)
    )


# ---------------------------------------------------------------- classification


def classify(row):
    """Terminal-state taxonomy. `ok` means the harness reported no error.

    Deliberately coarse and string-based: the harness has no error enum, and the
    distinctions that matter (did it run out of iterations vs. did the node's
    disk fill up) are only recoverable from the message. Keep `cap500` separate
    from infra -- bundling them produces the "cell E fails half the time" quote,
    which is a disk-full node wearing a capped run's clothes.
    """
    err = str(row.get("error") or "")
    if not err or err == "None":
        return "ok"
    if "MaxIterationsReached" in err:
        return "cap500"
    if "No space left" in err or "Errno 28" in err:
        return "ENOSPC"
    if "timeout" in err.lower():
        return "timeout"
    if "stuck" in err.lower():
        return "stuck"
    return "other_err"


def _has_patch(row):
    tr = row.get("test_result") or {}
    return bool((tr.get("git_patch") or "").strip())


def _history(row):
    return row.get("history") or []


# ---------------------------------------------------------------------- the cell


@dataclass
class Cell:
    out_dir: str
    run_dir: str
    source: str  # basename of the file actually read
    attempt: int | None  # None => merged output/output_errors fallback
    rows: list = field(default_factory=list)
    select: set = field(default_factory=set)
    warnings: list = field(default_factory=list)
    dropped_duplicates: int = 0
    unparseable: int = 0

    # -- derived ------------------------------------------------------------
    @property
    def by_id(self):
        return {r.get("instance_id"): r for r in self.rows}

    @property
    def ids(self):
        return [r.get("instance_id") for r in self.rows]

    @property
    def taxonomy(self):
        return collections.Counter(classify(r) for r in self.rows)

    @property
    def gradeable(self):
        return [r.get("instance_id") for r in self.rows if _has_patch(r)]

    @property
    def transcript_ids(self):
        return [r.get("instance_id") for r in self.rows if _history(r)]

    def report(self):
        n = len(self.rows)
        tax = ", ".join(f"{k}={v}" for k, v in sorted(self.taxonomy.items()))
        lines = [
            f"cell     : {os.path.basename(self.out_dir.rstrip('/'))}",
            f"source   : {self.source}  (attempt={self.attempt})",
            f"n rows   : {n}" + (f"   select={len(self.select)}" if self.select else ""),
            f"taxonomy : {tax}",
            f"gradeable (has git_patch) : {len(self.gradeable)}/{n}",
            f"transcript-bearing        : {len(self.transcript_ids)}/{n}"
            "   <-- denominator for ALL behavioural stats",
        ]
        if self.dropped_duplicates:
            lines.append(f"dedup    : dropped {self.dropped_duplicates} duplicate row(s)")
        if self.unparseable:
            lines.append(f"unparseable lines: {self.unparseable}")
        for w in self.warnings:
            lines.append(f"WARNING  : {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------- loading


def _dedupe(rows):
    """One row per instance_id. Prefer a row with a transcript, then with a patch.

    Rationale: duplicates arise from the append-log behaviour, where a later
    write supersedes an earlier one, and from an instance appearing in both
    output.jsonl and output_errors.jsonl. In both cases the informative row is
    the one that carries evidence.
    """
    best, dropped = {}, 0
    for r in rows:
        iid = r.get("instance_id")
        prev = best.get(iid)
        if prev is None:
            best[iid] = r
            continue
        dropped += 1
        rank = lambda x: (bool(_history(x)), _has_patch(x), classify(x) == "ok")
        if rank(r) > rank(prev):
            best[iid] = r
    return list(best.values()), dropped


def load_cell(out_dir, select=None, attempt=1, strict=True, require_attempt=False):
    """Load one eval cell with its integrity invariants enforced.

    out_dir : the `out_<tag>` directory (or its leaf run dir).
    select  : path to the select shard txt, or an iterable of instance_ids, or
              None to skip the coverage assertion. PASS IT. Coverage is the
              invariant that catches the silent-truncation family.
    attempt : which frozen `output.critic_attempt_N.jsonl` to prefer. 1 is the
              compute-matched choice (one rollout, temp 0, no critic selection)
              and is what any cross-model comparison should use.
    strict  : raise RolloutIntegrityError on a violated invariant. Set False
              only for triage of a live/partial cell -- never for a number you
              intend to publish.
    require_attempt : refuse the output/output_errors fallback entirely.
    """
    run_dir = _resolve_run_dir(out_dir)

    if select is None:
        sel = set()
    elif isinstance(select, str):
        with open(select) as fh:
            sel = {ln.strip() for ln in fh if ln.strip()}
    else:
        sel = set(select)

    warnings = []

    a_path = os.path.join(run_dir, f"output.critic_attempt_{attempt}.jsonl")
    rows, bad = read_jsonl(a_path)
    if rows:
        source, used_attempt = os.path.basename(a_path), attempt
    else:
        if require_attempt:
            raise RolloutIntegrityError(
                f"{a_path} absent/empty and require_attempt=True. An unfrozen cell "
                "cannot be compute-matched; do not compare it across models."
            )
        o_rows, o_bad = read_jsonl(os.path.join(run_dir, "output.jsonl"))
        e_rows, e_bad = read_jsonl(os.path.join(run_dir, "output_errors.jsonl"))
        rows, bad = o_rows + e_rows, o_bad + e_bad
        source, used_attempt = "output.jsonl + output_errors.jsonl", None
        o_ids, e_ids = {r.get("instance_id") for r in o_rows}, {r.get("instance_id") for r in e_rows}
        warnings.append(
            "fell back to the append logs -- these are mutable mid-run and this "
            "measurement is not reproducible; prefer output.critic_attempt_N.jsonl"
        )
        if o_ids & e_ids:
            warnings.append(
                f"{len(o_ids & e_ids)} instance_id(s) appear in BOTH output.jsonl and "
                "output_errors.jsonl -- the files are not disjoint; naive concatenation "
                "would double-count them"
            )

    rows, dropped = _dedupe(rows)
    rows.sort(key=lambda r: str(r.get("instance_id")))

    cell = Cell(
        out_dir=out_dir, run_dir=run_dir, source=source, attempt=used_attempt,
        rows=rows, select=sel, warnings=warnings,
        dropped_duplicates=dropped, unparseable=bad,
    )

    problems = []
    if dropped:
        problems.append(
            f"{dropped} duplicate instance_id row(s) in {source} -- expected for a "
            "mid-flight append log, never for a frozen attempt file"
        )
    if bad:
        problems.append(f"{bad} unparseable line(s) in {source}")
    if sel:
        missing, extra = sel - set(cell.ids), set(cell.ids) - sel
        if missing:
            problems.append(
                f"{len(missing)} of {len(sel)} selected instances have NO row "
                f"(e.g. {sorted(missing)[:3]}) -- the cell is incomplete; any rate "
                "computed over it has a truncated denominator"
            )
        if extra:
            problems.append(f"{len(extra)} row(s) outside the select set (e.g. {sorted(extra)[:3]})")

    if problems:
        msg = f"{os.path.basename(out_dir.rstrip('/'))}: " + "; ".join(problems)
        if strict:
            raise RolloutIntegrityError(
                msg + "\n\nIf you are triaging a live cell, pass strict=False and say so "
                "in whatever you write down. Do not publish a number from a cell that "
                "fails this check."
            )
        cell.warnings.append(msg)

    return cell


# ------------------------------------------------------------------------- cli

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dirs", nargs="+")
    ap.add_argument("--select", default=None, help="select shard txt")
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--loose", action="store_true", help="strict=False (triage only)")
    args = ap.parse_args()

    for d in args.out_dirs:
        try:
            c = load_cell(d, select=args.select, attempt=args.attempt, strict=not args.loose)
            print(c.report())
        except RolloutIntegrityError as exc:
            print(f"cell     : {os.path.basename(d.rstrip('/'))}\nFAILED   : {exc}")
        print("-" * 72)
