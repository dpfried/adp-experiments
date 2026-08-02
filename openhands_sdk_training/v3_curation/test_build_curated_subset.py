#!/usr/bin/env python3
"""Self-contained checks for build_curated_subset.py. Run: python test_build_curated_subset.py

No pytest dependency -- this has to be runnable on the FAIR login node with the bare
training venv, before we spend 8xA100-hours on whatever it produces.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "build_curated_subset.py"

CONDENSE = "openhands_sdk_condensation_prompt"
EVENTS = "openhands_sdk_events"


def rec(i: int, generation: str, record_type: str, tid: str,
        stringified: bool = True) -> str:
    md = {"generation": generation, "record_type": record_type,
          "source_trajectory_id": tid, "source_dataset": "unit-test"}
    r = {"id": f"r{i}", "messages": [{"role": "user", "content": f"m{i}"}],
         "tools": "[]", "metadata": json.dumps(md) if stringified else md}
    # deliberately NOT sorted/compact-normalized: we assert byte-preservation below
    return json.dumps(r)


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True)


def main() -> int:
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        if cond:
            print(f"  ok   {name}")
        else:
            print(f"  FAIL {name} {detail}")
            failures.append(name)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        src = tmp / "train.llamafactory.jsonl"

        # 10 records: alternating trajectory / condensation, 5 each
        lines = []
        for i in range(10):
            if i % 2 == 0:
                lines.append(rec(i, EVENTS, "trajectory", f"t{i // 2}"))
            else:
                lines.append(rec(i, CONDENSE, "condensation", f"t{i // 2}"))
        # plus one record with dict (not stringified) metadata, and one malformed
        lines.append(rec(99, EVENTS, "trajectory", "t99", stringified=False))
        lines.append('{"id":"broken","metadata":"{not json"}')
        src.write_text("\n".join(lines) + "\n")

        print("case 1: drop condensation, no budget")
        out1 = tmp / "out1"
        p = run("--src", str(src), "--out-dir", str(out1))
        check("exit 0", p.returncode == 0, p.stderr[-400:])
        check("stdout has BUILD_CURATED_OK", "BUILD_CURATED_OK" in p.stdout)
        got = (out1 / "train.llamafactory.jsonl").read_text().splitlines()
        check("wrote 6 trajectory records (5 stringified + 1 dict md)",
              len(got) == 6, f"got {len(got)}")
        check("no condensation survived",
              all(CONDENSE not in g for g in got))
        m1 = json.loads((out1 / "manifest.json").read_text())
        check("manifest dropped_by_generation == 5",
              m1["counts"]["dropped_by_generation"] == 5,
              str(m1["counts"]))
        check("manifest bad_metadata == 1",
              m1["counts"]["bad_metadata"] == 1, str(m1["counts"]))
        check("manifest census_complete", m1["census_complete"] is True)
        check("census sees both generations",
              m1["census"]["generation"].get(EVENTS) == 6
              and m1["census"]["generation"].get(CONDENSE) == 5,
              str(m1["census"]["generation"]))
        check("distinct trajectory ids kept == 6",
              m1["counts"]["distinct_source_trajectory_ids_kept"] == 6,
              str(m1["counts"]))

        # byte-preservation: every written line must appear verbatim in the source
        srclines = set(src.read_text().splitlines())
        check("written lines are byte-identical to source lines",
              all(g in srclines for g in got))

        print("case 2: budget smaller than supply -> exact budget, partial census")
        out2 = tmp / "out2"
        p = run("--src", str(src), "--out-dir", str(out2), "--max-records", "3")
        check("exit 0", p.returncode == 0, p.stderr[-400:])
        got2 = (out2 / "train.llamafactory.jsonl").read_text().splitlines()
        check("wrote exactly 3", len(got2) == 3, f"got {len(got2)}")
        m2 = json.loads((out2 / "manifest.json").read_text())
        check("records_written == 3", m2["counts"]["records_written"] == 3)
        check("census marked PARTIAL", m2["census_complete"] is False)
        check("kept_by_generation sums to written",
              sum(m2["census"]["kept_by_generation"].values()) == 3,
              str(m2["census"]["kept_by_generation"]))

        print("case 3: budget larger than supply -> warns, does not pad")
        out3 = tmp / "out3"
        p = run("--src", str(src), "--out-dir", str(out3), "--max-records", "1000")
        check("exit 0", p.returncode == 0, p.stderr[-400:])
        check("warns about unmet budget", "budget" in p.stdout and "NOT met" in p.stdout)
        got3 = (out3 / "train.llamafactory.jsonl").read_text().splitlines()
        check("wrote only what exists (6)", len(got3) == 6, f"got {len(got3)}")
        m3 = json.loads((out3 / "manifest.json").read_text())
        check("census_complete when supply exhausted", m3["census_complete"] is True)

        print("case 4: --census-only writes no subset")
        out4 = tmp / "out4"
        p = run("--src", str(src), "--out-dir", str(out4), "--census-only")
        check("exit 0", p.returncode == 0, p.stderr[-400:])
        check("no train file written",
              not (out4 / "train.llamafactory.jsonl").exists())
        check("manifest written", (out4 / "manifest.json").exists())

        print("case 5: refuses to clobber without --overwrite")
        p = run("--src", str(src), "--out-dir", str(out1))
        check("nonzero exit on existing output", p.returncode != 0)
        p = run("--src", str(src), "--out-dir", str(out1), "--overwrite")
        check("--overwrite succeeds", p.returncode == 0, p.stderr[-400:])

        print("case 6: --keep-record-type filters after --drop-generation")
        out6 = tmp / "out6"
        p = run("--src", str(src), "--out-dir", str(out6),
                "--keep-record-type", "condensation")
        check("exit 0", p.returncode == 0, p.stderr[-400:])
        got6p = out6 / "train.llamafactory.jsonl"
        got6 = got6p.read_text().splitlines() if got6p.exists() else []
        check("drop-generation wins: 0 records survive",
              len(got6) == 0, f"got {len(got6)}")

        print("case 7: --on-bad-metadata fail is strict")
        out7 = tmp / "out7"
        p = run("--src", str(src), "--out-dir", str(out7),
                "--on-bad-metadata", "fail")
        check("nonzero exit on bad metadata", p.returncode != 0)

    print()
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
