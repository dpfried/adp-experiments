#!/usr/bin/env python3
"""Merge sharded swebench-eval reports into one tally.

FAIR port of ../../swe-bench-babel-evals/scripts/merge_shard_reports.py — reads
from $SWEBENCH_ROOT/runs/score_<TAG> instead of the Babel home path.

Usage: python merge_shard_reports.py <TAG>
Reads  $SWEBENCH_ROOT/runs/score_<TAG>/shard_*of*.report.json (written by
swebench-eval next to each shard input) and prints a combined resolved/total
summary; writes merged.report.json alongside.
"""
import json
import os
import sys
from pathlib import Path


def main() -> None:
    tag = sys.argv[1]
    root = Path(os.environ.get("SWEBENCH_ROOT", "/checkpoint/dpf/swebench-eval"))
    score_dir = root / "runs" / f"score_{tag}"
    reports = sorted(score_dir.glob("shard_*.report.json"))
    if not reports:
        sys.exit(f"no shard reports found under {score_dir}")

    resolved, unresolved, total = [], [], 0
    for path in reports:
        r = json.loads(path.read_text())
        total += r.get("total", 0)
        resolved += r.get("resolved_ids", [])
        unresolved += r.get("unresolved_ids", [])
        print(f"{path.name}: {r.get('resolved')}/{r.get('total')} resolved")

    merged = {
        "tag": tag,
        "shards": len(reports),
        "total": total,
        "resolved": len(resolved),
        "unresolved": len(unresolved),
        "resolved_rate": round(len(resolved) / total, 4) if total else None,
        "resolved_ids": sorted(resolved),
        "unresolved_ids": sorted(unresolved),
    }
    out = score_dir / "merged.report.json"
    out.write_text(json.dumps(merged, indent=2))
    print(f"\n== {tag}: {merged['resolved']}/{total} resolved "
          f"({100 * (merged['resolved_rate'] or 0):.1f}%) — merged report: {out}")


if __name__ == "__main__":
    main()
