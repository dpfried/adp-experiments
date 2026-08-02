#!/usr/bin/env python3
"""Measure the SAME behavioural properties on SFT *training* trajectories that
extract_traj_stats.py measures on *eval* trajectories, so the two are directly
comparable.

Input is the ADP-converted OpenAI-chat format (`raw_train.openai.jsonl`):
  {id, messages:[{role, content, tool_calls?}], tools, metadata}
with assistant tool_calls naming the same four tools the eval harness exposes
(terminal / file_editor / think / finish).

Records are trajectory SEGMENTS: metadata carries source_trajectory_id and
trajectory_segment_index, and one source trajectory can be split across several
records. That distinction is the point of this script, so every metric is
reported twice:

  per-SEGMENT      what a training example actually looks like to the model
  per-TRAJECTORY   what the underlying agent episode looked like, segments
                   reassembled in order

If whole episodes verify but the segments the model trains on do not, the
conversion is what removes the verify-then-finish loop, not the source data.

Usage:
  python3 extract_train_stats.py OUT.json "name=/path/raw_train.openai.jsonl" [...]
"""
import json, sys, os, re, collections

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# Share the taxonomy with the eval-side extractor so the comparison is exact.
from extract_traj_stats import classify_cmd, SUCCESS_RE, ENVACT_RE, CORRUPT_RE  # noqa: E402

EDIT_SUBCMDS = ("create", "str_replace", "insert", "write", "undo_edit")


def _args(tc):
    fn = tc.get("function") or {}
    a = fn.get("arguments")
    if isinstance(a, str):
        try:
            return json.loads(a)
        except Exception:
            return {"_raw": a}
    return a if isinstance(a, dict) else {}


def calls_of(messages):
    """Flatten to an ordered list of (tool_name, args, result_text)."""
    results = {}
    for m in messages:
        if m.get("role") == "tool" and m.get("tool_call_id"):
            results[m["tool_call_id"]] = m.get("content") or ""
    out = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function") or {}
            out.append((fn.get("name") or "", _args(tc), results.get(tc.get("id"), "")))
    return out


ERR_RE = re.compile(r"error|no such file|not found|traceback|invalid|did not appear verbatim", re.I)


def measure(calls):
    """Same definitions as the eval-side extractor, over a list of tool calls."""
    tools = collections.Counter()
    test_turns, edit_turns = [], []
    cmdcat = collections.Counter()
    think_chars = 0
    env_act = False
    repro = False
    failed_edits = 0
    corrupt = 0
    finish_text = ""
    for i, (name, a, res) in enumerate(calls):
        tools[name] += 1
        blob = " ".join(str(v) for v in a.values())
        if CORRUPT_RE.search(blob):
            corrupt += 1
        if name == "terminal":
            c = str(a.get("command") or "")
            cats = classify_cmd(c)
            for x in cats:
                cmdcat[x] += 1
            if ENVACT_RE.search(c):
                env_act = True
            if "test" in cats or ("pyrun" in cats and re.search(r"repro|test", c)):
                test_turns.append(i)
        elif name == "file_editor":
            sub = str(a.get("command") or "?")
            path = str(a.get("path") or "")
            if sub in EDIT_SUBCMDS:
                edit_turns.append(i)
                if sub == "create" and re.search(r"repro|test_|/tmp/", path):
                    repro = True
                if res and ERR_RE.search(res[:400]):
                    failed_edits += 1
        elif name == "think":
            think_chars += len(str(a.get("thought") or ""))
        elif name == "finish":
            finish_text = blob

    last_edit = max(edit_turns) if edit_turns else None
    verified = bool(last_edit is not None and any(t > last_edit for t in test_turns))
    n = len(calls)
    tail = {t for t, _, _ in calls[-3:]} if calls else set()
    return dict(
        n_calls=n,
        n_terminal=tools.get("terminal", 0), n_file_editor=tools.get("file_editor", 0),
        n_think=tools.get("think", 0), n_finish=tools.get("finish", 0),
        n_test_runs=len(test_turns), n_edits=len(edit_turns),
        ran_any_test=bool(test_turns), verified_after_edit=verified,
        env_activated=env_act, repro_created=repro,
        think_chars=think_chars, failed_edits=failed_edits, corrupt_turns=corrupt,
        ends_on_finish=bool(calls and calls[-1][0] == "finish"),
        ends_on_edit=bool(calls and calls[-1][0] == "file_editor"
                          and str(calls[-1][1].get("command") or "") in EDIT_SUBCMDS),
        edit_in_last3=("file_editor" in tail),
        has_finish=bool(tools.get("finish")),
        finish_claims_success=bool(finish_text and SUCCESS_RE.search(finish_text)),
        cmd_test=cmdcat.get("test", 0), cmd_pyrun=cmdcat.get("pyrun", 0),
        cmd_search=cmdcat.get("search", 0), cmd_write=cmdcat.get("write", 0),
    )


def main():
    out_path = sys.argv[1]
    report = {}
    for spec in sys.argv[2:]:
        name, path = spec.split("=", 1)
        seg_rows = []
        by_traj = collections.defaultdict(list)   # source_trajectory_id -> [(idx, calls)]
        n = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                md = rec.get("metadata") or {}
                if isinstance(md, str):
                    try:
                        md = json.loads(md)
                    except Exception:
                        md = {}
                calls = calls_of(rec.get("messages") or [])
                seg_rows.append(measure(calls))
                sid = md.get("source_trajectory_id") or rec.get("id")
                idx = md.get("trajectory_segment_index") or 1
                by_traj[sid].append((idx, calls))
                n += 1
                if n % 10000 == 0:
                    print(f"[{name}] {n}", flush=True)

        traj_rows = []
        for sid, parts in by_traj.items():
            parts.sort(key=lambda t: t[0])
            merged = [c for _, cs in parts for c in cs]
            r = measure(merged)
            r["n_segments"] = len(parts)
            traj_rows.append(r)

        def agg(rows):
            if not rows:
                return {}
            keys = [k for k, v in rows[0].items() if isinstance(v, bool)]
            nums = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
            o = {"n": len(rows)}
            for k in keys:
                o[f"pct_{k}"] = round(100.0 * sum(1 for r in rows if r[k]) / len(rows), 1)
            for k in nums:
                vals = sorted(r[k] for r in rows)
                o[f"mean_{k}"] = round(sum(vals) / len(vals), 2)
                o[f"med_{k}"] = vals[len(vals) // 2]
            return o

        segdist = collections.Counter(len(v) for v in by_traj.values())
        report[name] = {
            "per_segment": agg(seg_rows),
            "per_trajectory": agg(traj_rows),
            "n_segments_total": len(seg_rows),
            "n_source_trajectories": len(by_traj),
            "segments_per_trajectory": dict(sorted(segdist.items())[:12]),
            "pct_records_that_are_continuations": round(
                100.0 * sum(1 for v in by_traj.values() for i, _ in v if i and i > 1) / max(1, len(seg_rows)), 1),
        }
        print(f"[{name}] DONE segments={len(seg_rows)} trajectories={len(by_traj)}", flush=True)

    json.dump(report, open(out_path, "w"), indent=1)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
