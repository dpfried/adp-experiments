#!/usr/bin/env python3
"""Export FULL trajectories -- thinking, tool calls, observations -- into one
normalized event schema, from either of the two shapes we have:

  eval   OpenHands SWE-bench `output.jsonl`   (rollouts by base / an SFT arm)
  train  LLaMA-Factory `train.llamafactory.jsonl` (the ADP demonstrations)

The point is that a demonstration and a rollout become the *same* object, so the
viewer can put "what we trained on" next to "what the model then did" without a
per-source special case.

Unlike `extract_traj_stats.py` (which truncates hard, because it keeps 500
instances x 2 models), this keeps text at readable length and is meant to be
pointed at a handful of trajectories.

Writes:
  OUT/traj/<label>/<id>.json   normalized trajectory
  OUT/index.json               manifest the viewer reads

Pure stdlib; run FAIR-side (eval records are ~3 MB each).
"""
import argparse, collections, glob, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_traj_stats import (          # one taxonomy, shared with the stats pass
    _s, tool_args, obs_text, obs_is_error, _blocks_to_text, patch_paths,
    classify_cmd, SUCCESS_RE, ENVACT_RE,
)

# ------------------------------------------------------------------ event schema
# Every trajectory is a flat, ordered list of these. `kind` is one of:
#   msg       system / user / assistant prose
#   think     explicit reasoning (the `think` tool's argument, or thinking blocks)
#   call      a tool call, with its FULL argument dict
#   obs       the tool result
#   condense  a context condensation (eval only)


def ev(kind, **kw):
    e = {"kind": kind}
    e.update(kw)
    return e


def cap(s, n, note="text"):
    """Truncate with an explicit marker -- never silently."""
    s = s if isinstance(s, str) else ""
    if n and len(s) > n:
        return s[:n] + f"\n\n[... {len(s) - n:,} more chars of {note} truncated ...]"
    return s


# ------------------------------------------------------------------ eval side
def events_from_eval(rec, max_text, max_arg):
    """OpenHands history -> events. Preserves original order."""
    hist = rec.get("history") or []
    out = []
    for e in hist:
        k = _s(e.get("kind"))
        if k == "ActionEvent":
            tn = _s(e.get("tool_name"))
            a = tool_args(e)
            # Reasoning can arrive three ways; only one is ever populated in this
            # data (the `think` tool argument), but read all three so the viewer
            # keeps working if a future run fills the standard fields.
            pre = _s(e.get("thought")) or _s(e.get("reasoning_content"))
            tb = e.get("thinking_blocks")
            if isinstance(tb, list):
                blocks = [_s(b.get("thinking")) or _s(b.get("text"))
                          for b in tb if isinstance(b, dict)]
                pre = (pre + "\n" + "\n".join(x for x in blocks if x)).strip()
            if pre:
                out.append(ev("think", source="thought", text=cap(pre, max_text)))
            if tn == "think":
                # The `think` tool IS the reasoning -- surface it as such rather
                # than as a tool call with a wall of JSON.
                out.append(ev("think", source="think-tool",
                              text=cap(_s(a.get("thought")) or _s(a.get("_raw")), max_text)))
            else:
                out.append(ev("call", tool=tn,
                              args={k2: cap(v if isinstance(v, str) else json.dumps(v, default=str),
                                            max_arg, "argument")
                                    for k2, v in a.items()},
                              cats=sorted(classify_cmd(_s(a.get("command"))))
                                   if tn == "terminal" else []))
        elif k == "ObservationEvent":
            out.append(ev("obs", tool=_s(e.get("tool_name")),
                          text=cap(obs_text(e), max_text, "output"),
                          err=obs_is_error(e)))
        elif k == "Condensation":
            summ = _s(e.get("summary")) or _blocks_to_text(e.get("content"))
            out.append(ev("condense", text=cap(summ, max_text),
                          forgotten=e.get("forgotten_event_ids") and
                                    len(e["forgotten_event_ids"]) or None))
        elif k in ("MessageEvent", "UserMessageEvent", "AgentMessageEvent"):
            m = e.get("llm_message") or e.get("message") or {}
            role = _s(m.get("role")) or _s(e.get("source")) or "message"
            txt = _blocks_to_text(m.get("content")) or _blocks_to_text(e.get("content"))
            if txt:
                out.append(ev("msg", role=role, text=cap(txt, max_text)))
        # other kinds (SystemPrompt, etc.) intentionally dropped -- noise
    return out


def traj_from_eval(rec, label, resolved_ids, max_text, max_arg):
    iid = rec.get("instance_id")
    evs = events_from_eval(rec, max_text, max_arg)
    gp = ((rec.get("test_result") or {}).get("git_patch")) or ""
    calls = [e for e in evs if e["kind"] == "call"]
    return {
        "id": iid, "label": label, "kind": "eval",
        "title": iid,
        "resolved": (iid in resolved_ids) if resolved_ids else None,
        "error": _s(rec.get("error"))[:300] or None,
        "instruction": cap(_s(rec.get("instruction")), 20000, "instruction"),
        "patch": cap(gp, 40000, "patch"),
        "meta": {
            "repo": (iid or "").split("__")[0] if iid else None,
            "n_events": len(evs), "n_calls": len(calls),
            "n_think": sum(1 for e in evs if e["kind"] == "think"),
            "n_obs": sum(1 for e in evs if e["kind"] == "obs"),
            "n_condense": sum(1 for e in evs if e["kind"] == "condense"),
            "tools": dict(collections.Counter(e.get("tool") for e in calls)),
            "empty_history": not bool(rec.get("history")),
        },
        "events": evs,
    }


# ------------------------------------------------------------------ train side
def events_from_train(rec, max_text, max_arg):
    """LLaMA-Factory / ADP messages -> the same events.

    Two shapes exist in this tree: role='function_call' whose content is a JSON
    list of {name, arguments} (what the pooled subsets ship), and the plainer
    role='assistant' + tool_calls form. Handle both.
    """
    msgs = rec.get("messages") or rec.get("conversations") or []
    out = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = _s(m.get("role")) or _s(m.get("from"))
        content = m.get("content")
        if content is None:
            content = m.get("value")

        if role in ("function_call", "tool_calls"):
            calls = content
            if isinstance(calls, str):
                try:
                    calls = json.loads(calls)
                except Exception:
                    calls = [{"name": "?", "arguments": {"_raw": content}}]
            if isinstance(calls, dict):
                calls = [calls]
            for c in calls or []:
                if not isinstance(c, dict):
                    continue
                name = _s(c.get("name")) or _s((c.get("function") or {}).get("name"))
                a = c.get("arguments")
                if a is None:
                    a = (c.get("function") or {}).get("arguments")
                if isinstance(a, str):
                    try:
                        a = json.loads(a)
                    except Exception:
                        a = {"_raw": a}
                if not isinstance(a, dict):
                    a = {"_raw": str(a)}
                if name == "think":
                    out.append(ev("think", source="think-tool",
                                  text=cap(_s(a.get("thought")) or _s(a.get("_raw")), max_text)))
                else:
                    out.append(ev("call", tool=name,
                                  args={k: cap(v if isinstance(v, str)
                                               else json.dumps(v, default=str), max_arg, "argument")
                                        for k, v in a.items()},
                                  cats=sorted(classify_cmd(_s(a.get("command"))))
                                       if name == "terminal" else []))
        elif role in ("tool", "function", "observation"):
            out.append(ev("obs", tool=None,
                          text=cap(_blocks_to_text(content) or _s(content), max_text, "output"),
                          err=False))
        else:
            txt = _blocks_to_text(content) or _s(content)
            if txt:
                out.append(ev("msg", role=role or "assistant", text=cap(txt, max_text)))
    return out


def train_meta(rec):
    """`metadata` ships as a JSON *string* in these files, not a dict."""
    md = rec.get("metadata")
    if isinstance(md, str):
        try:
            md = json.loads(md)
        except Exception:
            return {}
    return md if isinstance(md, dict) else {}


def traj_from_train(rec, label, max_text, max_arg, idx):
    evs = events_from_train(rec, max_text, max_arg)
    meta = train_meta(rec)
    rid = _s(rec.get("id")) or f"row{idx:05d}"
    calls = [e for e in evs if e["kind"] == "call"]
    # First user turn is the task statement; show it in the header like an
    # eval instruction so both sources read the same way.
    instr = ""
    for e in evs:
        if e["kind"] == "msg" and e.get("role") == "user":
            instr = e["text"]
            break
    # A third of these rows are not trajectories at all: they are condensation
    # *prompts* (summarise-the-history training examples), identifiable by
    # having condensation fields and no trajectory_segment_index.
    rtype = _s(meta.get("record_type")) or ("condensation" if "condensation_index" in meta else "?")
    return {
        "id": rid, "label": label, "kind": "train", "record_type": rtype,
        "title": rid,
        "resolved": None, "error": None,
        "instruction": cap(instr, 20000, "instruction"),
        "patch": "",
        "meta": {
            "n_events": len(evs), "n_calls": len(calls),
            "n_think": sum(1 for e in evs if e["kind"] == "think"),
            "n_obs": sum(1 for e in evs if e["kind"] == "obs"),
            "n_condense": 0,
            "tools": dict(collections.Counter(e.get("tool") for e in calls)),
            "record_type": rtype,
            "segment": meta.get("trajectory_segment_index"),
            "source_trajectory_id": meta.get("source_trajectory_id"),
            "source_dataset": meta.get("source_dataset"),
            "forgotten_event_count": meta.get("forgotten_event_count"),
            "condensation_index": meta.get("condensation_index"),
        },
        "events": evs,
    }


def traj_from_train_group(rows, label, max_text, max_arg):
    """Reassemble one source trajectory from its segments.

    Segments are shuffled across the file and each repeats the system prompt and
    task statement, so stitch them in segment order and drop the repeats -- what
    you want to read is the agent's arc, not the prompt three times.
    """
    rows = sorted(rows, key=lambda r: (train_meta(r).get("trajectory_segment_index") or 0))
    m0 = train_meta(rows[0])
    evs, seen_prompt = [], set()
    for k, rec in enumerate(rows):
        md = train_meta(rec)
        seg = md.get("trajectory_segment_index")
        part = events_from_train(rec, max_text, max_arg)
        if k:
            evs.append(ev("boundary", text=f"segment {seg} begins"))
        for e in part:
            if e["kind"] == "msg" and e.get("role") in ("system", "user"):
                key = (e["role"], e["text"][:400])
                if key in seen_prompt:
                    continue
                seen_prompt.add(key)
            evs.append(e)
    calls = [e for e in evs if e["kind"] == "call"]
    instr = next((e["text"] for e in evs if e["kind"] == "msg" and e.get("role") == "user"), "")
    tid = _s(m0.get("source_trajectory_id")) or _s(rows[0].get("id"))
    segs = [train_meta(r).get("trajectory_segment_index") for r in rows]
    return {
        "id": tid, "label": label, "kind": "train", "record_type": "trajectory",
        "title": f"{tid[:28]} ({len(rows)} segment{'s' if len(rows) > 1 else ''})",
        "resolved": None, "error": None,
        "instruction": cap(instr, 20000, "instruction"),
        "patch": "",
        "meta": {
            "n_events": len(evs), "n_calls": len(calls),
            "n_think": sum(1 for e in evs if e["kind"] == "think"),
            "n_obs": sum(1 for e in evs if e["kind"] == "obs"),
            "n_condense": 0,
            "tools": dict(collections.Counter(e.get("tool") for e in calls)),
            "record_type": "trajectory",
            "n_rows": len(rows), "segments": segs,
            "source_trajectory_id": tid,
            "source_dataset": m0.get("source_dataset"),
        },
        "events": evs,
    }


# ------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("specs", nargs="+", help=(
        "eval:<label>=<score_tag>=<output.jsonl glob>  |  train:<label>=<jsonl path>"))
    ap.add_argument("--iids", default="", help="comma list or @file: restrict eval to these instances")
    ap.add_argument("--train-n", type=int, default=6, help="trajectories to take per train spec")
    ap.add_argument("--train-mode", choices=("group", "rows"), default="group",
                    help="group: reassemble whole trajectories from segments (one pass "
                         "over the file). rows: take raw rows as-is, no reassembly.")
    ap.add_argument("--train-scan-head", type=int, default=800,
                    help="group mode: rows read to choose which trajectories to follow")
    ap.add_argument("--train-condensation", type=int, default=2,
                    help="group mode: condensation-prompt rows to also export (these are "
                         "~1/3 of the corpus and are not trajectories)")
    ap.add_argument("--train-min-events", type=int, default=0,
                    help="skip train rows shorter than this (segments can be tiny)")
    ap.add_argument("--max-text", type=int, default=12000, help="cap per message/observation")
    ap.add_argument("--max-arg", type=int, default=8000, help="cap per tool-call argument")
    args = ap.parse_args()

    want = set()
    if args.iids:
        if args.iids.startswith("@"):
            want = {l.strip() for l in open(args.iids[1:]) if l.strip()}
        else:
            want = {x.strip() for x in args.iids.split(",") if x.strip()}

    RUNS = os.environ.get("SWEBENCH_RUNS") or os.path.join(
        os.environ.get("SWEBENCH_ROOT", "."), "runs")

    os.makedirs(args.out, exist_ok=True)
    index = []

    for spec in args.specs:
        src, rest = spec.split(":", 1)
        if src == "eval":
            label, tag, pat = rest.split("=", 2)
            rep = os.path.join(RUNS, f"score_{tag}", "merged.report.json")
            resolved = set()
            if os.path.exists(rep):
                resolved = set(json.load(open(rep)).get("resolved_ids") or [])
            else:
                print(f"[{label}] WARNING no report at {rep} -- resolved labels unavailable",
                      file=sys.stderr)
                resolved = None
            files = sorted(glob.glob(pat, recursive=True))
            print(f"[{label}] eval, {len(files)} files, want={len(want) or 'all'}", flush=True)
            dd = os.path.join(args.out, "traj", label)
            os.makedirs(dd, exist_ok=True)
            seen, n = set(), 0
            for f in files:
                for line in open(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    iid = rec.get("instance_id")
                    if not iid or iid in seen:
                        continue
                    if want and iid not in want:
                        continue
                    seen.add(iid)
                    t = traj_from_eval(rec, label, resolved, args.max_text, args.max_arg)
                    with open(os.path.join(dd, f"{iid}.json"), "w") as g:
                        json.dump(t, g, default=str)
                    index.append({k: t[k] for k in ("id", "label", "kind", "title",
                                                    "resolved", "error", "meta")})
                    n += 1
                if want and len(seen & want) == len(want):
                    break     # every requested instance found; stop reading shards
            print(f"[{label}] wrote {n}", flush=True)

        elif src == "train":
            label, path = rest.split("=", 1)
            dd = os.path.join(args.out, "traj", label)
            os.makedirs(dd, exist_ok=True)

            def emit(t):
                fid = re.sub(r"[^A-Za-z0-9_.-]", "_", t["id"])[:120]
                t["id"] = fid
                with open(os.path.join(dd, f"{fid}.json"), "w") as g:
                    json.dump(t, g, default=str)
                index.append({k: t[k] for k in ("id", "label", "kind", "title",
                                                "resolved", "error", "meta")})

            if args.train_mode == "rows":
                n = 0
                with open(path) as fh:
                    for i, line in enumerate(fh):
                        if n >= args.train_n:
                            break
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        t = traj_from_train(rec, label, args.max_text, args.max_arg, i)
                        if t["meta"]["n_events"] < args.train_min_events:
                            continue
                        emit(t)
                        n += 1
                print(f"[{label}] train rows, wrote {n} from {path}", flush=True)
                continue

            # --- group mode: reassemble whole trajectories from their segments.
            # Segments of one trajectory are scattered across the file, so the
            # ids are chosen from the head and then the whole file is streamed
            # once to collect their other segments.
            head, want_ids, cond_rows = [], [], []
            with open(path) as fh:
                for i, line in enumerate(fh):
                    if i >= args.train_scan_head:
                        break
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    md = train_meta(rec)
                    if md.get("trajectory_segment_index") is not None:
                        tid = md.get("source_trajectory_id")
                        if tid and tid not in want_ids and len(want_ids) < args.train_n:
                            want_ids.append(tid)
                    elif len(cond_rows) < args.train_condensation:
                        cond_rows.append((i, rec))
            want_ids = set(want_ids)
            groups = collections.defaultdict(list)
            scanned = 0
            with open(path) as fh:
                for line in fh:
                    scanned += 1
                    # cheap prefilter -- avoid json.loads on 79k irrelevant rows
                    if not any(t in line for t in want_ids):
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    md = train_meta(rec)
                    tid = md.get("source_trajectory_id")
                    if tid in want_ids and md.get("trajectory_segment_index") is not None:
                        groups[tid].append(rec)
            n = 0
            for tid, rows in groups.items():
                t = traj_from_train_group(rows, label, args.max_text, args.max_arg)
                if t["meta"]["n_events"] < args.train_min_events:
                    continue
                emit(t)
                n += 1
            nc = 0
            for i, rec in cond_rows:
                t = traj_from_train(rec, label, args.max_text, args.max_arg, i)
                t["title"] = f"condensation prompt (row {i})"
                emit(t)
                nc += 1
            multi = sum(1 for v in groups.values() if len(v) > 1)
            print(f"[{label}] train groups: {n} trajectories ({multi} multi-segment) "
                  f"+ {nc} condensation rows, scanned {scanned:,} lines of {path}", flush=True)
        else:
            sys.exit(f"unknown spec source {src!r} (expected eval: or train:)")

    with open(os.path.join(args.out, "index.json"), "w") as g:
        json.dump(index, g, default=str)
    print(f"ALL DONE: {len(index)} trajectories -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
