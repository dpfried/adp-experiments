#!/usr/bin/env python3
"""Build a curated LLaMA-Factory training subset from an adp-v2 arm subset.

Motivation (ADP-v3 Lever A). Every v2 SWE subset is ~34-41% *context-condensation*
records -- 2-message (condenser-prompt -> prose state summary) pairs that train
"summarize and conclude" rather than "act". The v2 arms trained on the first 55k
records of a shuffled mixed file, so ~40% of their gradient was summarization. This
script builds the condensation-free counterfactual at a chosen record budget.

Design notes:
  * STREAMING and byte-preserving. Records are copied as the *original raw line*,
    never re-serialized, so a curated record is byte-identical to the one the v2 arm
    saw. (A json.loads/json.dumps round-trip would reorder keys and rewrite floats,
    which would make "same records, minus condensation" quietly untrue.)
  * `metadata` is a *stringified* JSON blob in these files, so it is parsed twice.
    Records whose metadata is missing/unparseable are counted and, by default,
    dropped (--on-bad-metadata).
  * Selection is "first N in file order". The v2 files are well-shuffled (verified in
    the v2 data audit), so first-N is a representative random subsample AND maximizes
    trajectory-record overlap with the matched mixed-55k arm -- the two arms then
    differ as close to *only* in composition as we can get.
  * Writes a manifest.json with the full generation/record_type census, so the
    composition claim for the resulting arm is file-backed rather than assumed.

Usage:
  python build_curated_subset.py \
      --src  /checkpoint/dpf/adp-data/v2_swe_subsets/<arm>/train.llamafactory.jsonl \
      --out-dir /checkpoint/dpf/adp-data/v3_curated/<name> \
      --drop-generation openhands_sdk_condensation_prompt \
      --max-records 55000

  # census only, writes nothing but the manifest (use this first):
  python build_curated_subset.py --src ... --out-dir ... --census-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path


def parse_metadata(rec: dict) -> tuple[dict | None, str | None]:
    """Return (metadata_dict, error). metadata is a stringified JSON blob."""
    md = rec.get("metadata")
    if md is None:
        return None, "missing"
    if isinstance(md, dict):
        return md, None
    if isinstance(md, str):
        try:
            parsed = json.loads(md)
        except (ValueError, TypeError) as e:
            return None, f"unparseable: {type(e).__name__}"
        if not isinstance(parsed, dict):
            return None, f"not-a-dict: {type(parsed).__name__}"
        return parsed, None
    return None, f"unexpected-type: {type(md).__name__}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True,
                    help="source train.llamafactory.jsonl")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="destination dir; gets train.llamafactory.jsonl + manifest.json")
    ap.add_argument("--drop-generation", action="append", default=None, metavar="VALUE",
                    help="drop records whose metadata.generation == VALUE (repeatable). "
                         "Default: openhands_sdk_condensation_prompt")
    ap.add_argument("--keep-record-type", action="append", default=None, metavar="VALUE",
                    help="if given, keep ONLY records whose metadata.record_type is one "
                         "of these (repeatable). Applied after --drop-generation.")
    ap.add_argument("--max-records", type=int, default=None,
                    help="stop after writing this many records (first-N in file order). "
                         "Default: write every surviving record.")
    ap.add_argument("--on-bad-metadata", choices=("drop", "keep", "fail"), default="drop",
                    help="what to do with records whose metadata is missing/unparseable")
    ap.add_argument("--census-only", action="store_true",
                    help="scan and write manifest.json, but do not write the subset")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow overwriting an existing train.llamafactory.jsonl")
    ap.add_argument("--progress-every", type=int, default=5000)
    args = ap.parse_args()

    drop_gen = set(args.drop_generation or ["openhands_sdk_condensation_prompt"])
    keep_rt = set(args.keep_record_type) if args.keep_record_type else None

    if not args.src.is_file():
        raise SystemExit(f"FATAL: --src not a file: {args.src}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "train.llamafactory.jsonl"
    if out_path.exists() and not (args.overwrite or args.census_only):
        raise SystemExit(f"FATAL: {out_path} exists (pass --overwrite to replace)")

    src_stat = args.src.stat()
    gen_census: Counter[str] = Counter()
    rt_census: Counter[str] = Counter()
    gen_rt_census: Counter[tuple[str, str]] = Counter()
    kept_gen: Counter[str] = Counter()
    n_total = n_written = n_dropped_gen = n_dropped_rt = n_bad_json = n_bad_md = 0
    src_traj_ids: set[str] = set()
    t0 = time.time()

    tmp_path = out_path.with_suffix(".jsonl.partial")
    out_fh = None if args.census_only else tmp_path.open("w", encoding="utf-8")
    try:
        with args.src.open("r", encoding="utf-8") as fh:
            for raw in fh:
                if not raw.strip():
                    continue
                n_total += 1
                try:
                    rec = json.loads(raw)
                except ValueError:
                    n_bad_json += 1
                    if args.on_bad_metadata == "fail":
                        raise SystemExit(f"FATAL: unparseable JSON at line {n_total}")
                    continue

                md, err = parse_metadata(rec)
                if md is None:
                    n_bad_md += 1
                    if args.on_bad_metadata == "fail":
                        raise SystemExit(
                            f"FATAL: bad metadata at line {n_total}: {err}")
                    if args.on_bad_metadata == "drop":
                        continue
                    md = {}

                gen = str(md.get("generation", "<none>"))
                rt = str(md.get("record_type", "<none>"))
                gen_census[gen] += 1
                rt_census[rt] += 1
                gen_rt_census[(gen, rt)] += 1

                if gen in drop_gen:
                    n_dropped_gen += 1
                    continue
                if keep_rt is not None and rt not in keep_rt:
                    n_dropped_rt += 1
                    continue

                # surviving record. Check the budget BEFORE accounting for it, so the
                # manifest never credits a record that was not actually written.
                if args.max_records is not None and n_written >= args.max_records:
                    # Budget reached. A census of a partially-scanned file would be
                    # misleading, so stop here and mark the census partial.
                    n_total -= 1  # this line was read but is not part of the subset
                    break

                kept_gen[gen] += 1
                tid = md.get("source_trajectory_id")
                if tid is not None:
                    src_traj_ids.add(str(tid))

                if out_fh is not None:
                    out_fh.write(raw if raw.endswith("\n") else raw + "\n")
                    n_written += 1

                if n_total % args.progress_every == 0:
                    el = time.time() - t0
                    print(f"  ..{n_total:,} scanned / {n_written:,} written "
                          f"({el:,.0f}s)", flush=True)
    finally:
        if out_fh is not None:
            out_fh.close()

    census_complete = (args.census_only
                       or args.max_records is None
                       or n_written < args.max_records)

    if not args.census_only:
        os.replace(tmp_path, out_path)

    manifest = {
        "generated_by": "build_curated_subset.py",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "src": str(args.src),
        "src_bytes": src_stat.st_size,
        "src_mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                       time.gmtime(src_stat.st_mtime)),
        "out": None if args.census_only else str(out_path),
        "out_bytes": None if args.census_only else out_path.stat().st_size,
        "filter": {
            "drop_generation": sorted(drop_gen),
            "keep_record_type": sorted(keep_rt) if keep_rt else None,
            "max_records": args.max_records,
            "on_bad_metadata": args.on_bad_metadata,
            "selection": "first-N in file order (source file is shuffled)",
        },
        "counts": {
            "records_scanned": n_total,
            "records_written": n_written,
            "dropped_by_generation": n_dropped_gen,
            "dropped_by_record_type": n_dropped_rt,
            "bad_json_lines": n_bad_json,
            "bad_metadata": n_bad_md,
            "distinct_source_trajectory_ids_kept": len(src_traj_ids),
        },
        "census_complete": census_complete,
        "census_note": ("full-file census" if census_complete else
                        "PARTIAL: scan stopped early once --max-records was reached; "
                        "these counts describe only the prefix that was read"),
        "census": {
            "generation": dict(gen_census.most_common()),
            "record_type": dict(rt_census.most_common()),
            "generation_x_record_type": {f"{g} | {r}": c
                                         for (g, r), c in gen_rt_census.most_common()},
            "kept_by_generation": dict(kept_gen.most_common()),
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps(manifest, indent=2))
    if not args.census_only:
        if args.max_records is not None and n_written < args.max_records:
            print(f"\n!! WARNING: budget {args.max_records:,} NOT met -- the source only "
                  f"yielded {n_written:,} surviving records. The arm will be "
                  f"compute-MISmatched vs a {args.max_records:,}-record mixed arm; "
                  f"report it as such or lower the budget on BOTH arms.", flush=True)
        print(f"\nWROTE {out_path} ({n_written:,} records)")
    print("BUILD_CURATED_OK")


if __name__ == "__main__":
    sys.exit(main())
