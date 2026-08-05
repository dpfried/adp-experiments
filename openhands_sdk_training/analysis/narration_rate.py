#!/usr/bin/env python3
"""Measure the plain-message narration channel in eval rollouts.

The channel: event-level `thought` on an `ActionEvent` -- prose the model emits
in the SAME completion as, and BEFORE, a tool call ("Now I can see the problem
clearly...", "BINGO! There's the bug!"). Because prose and call are one
completion with prose first, the action tokens are conditioned on it, which is
what makes it chain-of-thought by construction.

Do not confuse it with the two neighbours it is routinely conflated with:
  * `action.thought`      -- the ARGUMENT to the ADP `think` tool (a tool call).
  * `reasoning_content` /
    `thinking_blocks`     -- native CoT. Zero everywhere in this campaign.

`thought` is a LIST OF CONTENT BLOCKS, not a str. Calling .strip() on it raises;
that cost an hour once. Use txt().

Usage:
    python narration_rate.py <out_dir> [<out_dir> ...] \
        [--select .../shard_00of10.txt] [--attempt 1] [--loose]

Reads through load_rollouts.load_cell, so the denominators are enforced rather
than assumed -- see ANALYSIS_HOUSE_RULES.md rules 1 and 5. `--loose` permits an
incomplete/mid-flight cell for triage; it prints TRIAGE next to every number and
those numbers must not be published.
"""

from __future__ import annotations

import argparse
import collections
import os

from load_rollouts import RolloutIntegrityError, load_cell


def txt(value):
    """Flatten a content-block list (or str, or None) to plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(b.get("text", "") for b in value if isinstance(b, dict))
    return str(value)


def measure(cell):
    n_events = n_narrated = 0
    n_native = 0
    tools = collections.Counter()
    narrated_tools = collections.Counter()
    lengths = []

    for row in cell.rows:
        for ev in row.get("history") or []:
            if ev.get("kind") != "ActionEvent":
                continue
            n_events += 1
            tool = (ev.get("action") or {}).get("kind") or ev.get("tool_name") or "?"
            tools[tool] += 1
            prose = txt(ev.get("thought")).strip()
            if prose:
                n_narrated += 1
                narrated_tools[tool] += 1
                lengths.append(len(prose))
            # native CoT should be empty everywhere; assert it rather than assume
            if txt(ev.get("reasoning_content")).strip() or ev.get("thinking_blocks"):
                n_native += 1

    lengths.sort()
    return {
        "transcripts": len(cell.transcript_ids),
        "n_rows": len(cell.rows),
        "events": n_events,
        "narrated": n_narrated,
        "pct": 100.0 * n_narrated / n_events if n_events else 0.0,
        "median_chars": lengths[len(lengths) // 2] if lengths else 0,
        "native_cot_events": n_native,
        "tools": tools,
        "narrated_tools": narrated_tools,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dirs", nargs="+")
    ap.add_argument("--select", default=None)
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--loose", action="store_true",
                    help="allow an incomplete cell (triage only; never publish these)")
    args = ap.parse_args()

    for d in args.out_dirs:
        name = os.path.basename(d.rstrip("/"))
        try:
            cell = load_cell(d, select=args.select, attempt=args.attempt,
                             strict=not args.loose)
        except RolloutIntegrityError as exc:
            print(f"{name}\n  SKIPPED (integrity): {exc}\n")
            continue

        m = measure(cell)
        flag = "  [TRIAGE - incomplete cell, do not publish]" if cell.warnings else ""
        print(f"{name}   source={cell.source}{flag}")
        print(f"  transcripts {m['transcripts']}/{m['n_rows']}   ActionEvents {m['events']}")
        print(f"  NARRATION: {m['narrated']}/{m['events']} events = {m['pct']:.2f}%"
              f"   median {m['median_chars']} chars")
        if m["native_cot_events"]:
            print(f"  !! native CoT present on {m['native_cot_events']} events "
                  "-- it is supposed to be off everywhere; investigate before using this cell")
        top = ", ".join(f"{k} {v}" for k, v in m["tools"].most_common(4))
        print(f"  tools: {top}")
        nt = ", ".join(f"{k} {m['narrated_tools'][k]}/{v}" for k, v in m["tools"].most_common(3))
        print(f"  narrated-by-tool: {nt}")
        for w in cell.warnings:
            print(f"  WARNING: {w}")
        print()


if __name__ == "__main__":
    main()
