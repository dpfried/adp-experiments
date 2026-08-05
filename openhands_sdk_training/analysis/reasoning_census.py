#!/usr/bin/env python3
"""Where does reasoning happen? Census over three populations x three channels.

Question (dpf, 2026-08-05): quantify how often ANY sort of reasoning happens in
(1) the base model, (2) the training data, (3) the arms -- across the three
channels (a) free text before a tool call, (b) the think() tool, (c) a <think> tag.

The three channels are NOT disjoint and the counts must not be presented as if
they were. Measured on this data: the <think> tag base emits sits at char 0 of the
`thought` field, so for base-nostub channel (c) IS channel (a) -- same text, two
descriptions. Reported nested: free text present, and of that, how much is
tag-wrapped. The think() tool is genuinely separate: it is an ActionEvent with
tool_name == "think", carrying its text in the tool arguments, not in `thought`.

The two populations also do not share a denominator, and forcing one would be
the misleading move rather than the rigorous one:

  rollouts       -- unit is an ActionEvent (one assistant turn that called a
                    tool). Free text = non-empty `thought` on that event.
  training data  -- unit is an assistant-authored message. The ADP ->
                    LLaMA-Factory conversion emits role "function_call" whose
                    content is a bare JSON tool-call array with NO text field,
                    so a training turn structurally CANNOT carry free text
                    alongside its tool call. Free text exists only as a separate
                    role "assistant" message. That asymmetry is a finding, not a
                    measurement artifact, so both are reported per
                    assistant-authored turn and the structural difference is
                    printed alongside.

Pure stdlib; streams. Usage:
  reasoning_census.py rollouts <swebench_root> [cell ...]
  reasoning_census.py train <v2_swe_subsets_dir> [--limit N]
"""
import json, os, sys, glob, collections, re

CELLS = [
    ("A", "par_A_arm_stock_evalp",   "arm",  "stock"),
    ("B", "par_B_arm_nostub_evalp",  "arm",  "nostub"),
    ("C", "par_C_arm_nostub_wrap",   "arm",  "nostub"),
    ("D", "par_D_arm_nostub_trainp", "arm",  "nostub"),
    ("E", "par_E_base_stock_evalp",  "base", "stock"),
    ("F", "par_F_base_nostub_evalp", "base", "nostub"),
    ("G", "par_G_base_prefill_evalp", "base", "stock+prefill"),
]


def _s(x):
    return x if isinstance(x, str) else ""


def thought_texts(ev):
    """`thought` is either a str or a list of content blocks."""
    th = ev.get("thought")
    if isinstance(th, list):
        return [_s(b.get("text")) for b in th if isinstance(b, dict)]
    if isinstance(th, str):
        return [th]
    return []


def strip_think_tag(s):
    """Return (had_tag, closed, text_outside_the_tag, text_inside_the_tag).

    Base's tags are unclosed, so "inside" runs to the end unless a markdown
    heading starts what is plainly prose again -- the same heuristic the turn
    extractor uses, kept identical so the two agree.
    """
    if "<think>" not in s:
        return False, False, s, ""
    before, after = s.split("<think>", 1)
    if "</think>" in after:
        inside, outside_tail = after.split("</think>", 1)
        return True, True, (before + outside_tail), inside
    m = re.search(r"\n\s*#{1,3}\s", after)
    if m:
        return True, False, before + after[m.start():], after[:m.start()]
    return True, False, before, after


def cell_instances(root, tag):
    """Instance ids present in a cell's rollout files, for intersection pairing."""
    ids = set()
    for sh in ("00", "01"):
        for p in sorted(glob.glob(os.path.join(
                root, "runs", f"out_{tag}__s{sh}", "**", "output.jsonl"),
                recursive=True)):
            with open(p) as f:
                for line in f:
                    try:
                        ids.add(json.loads(line).get("instance_id"))
                    except Exception:
                        pass
    return ids


def census_rollouts(root, only=None, restrict=None):
    """restrict: if given, count only these instance ids.

    Cells finish at different times (E lost a shard to walltime, G is still
    running), so the raw per-cell rates are computed over different instance
    sets. Any rate compared ACROSS cells has to be paired or it is measuring
    the instance mix as much as the condition.
    """
    rows = []
    for c, tag, model, tmpl in CELLS:
        if only and c not in only:
            continue
        d = collections.Counter()
        kinds = collections.Counter()
        insts = set()
        per_inst_actions = []
        # The harness buries output.jsonl several levels down, under
        # <benchmark>/<provider>/adp-eval-<tag>_sdk_<sha>_maxiter_<n>/. Globbing
        # for it rather than assuming a flat layout: an assumed path here returns
        # zero events, which reads identically to "this cell never reasoned".
        paths = []
        for sh in ("00", "01"):
            paths += sorted(glob.glob(os.path.join(
                root, "runs", f"out_{tag}__s{sh}", "**", "output.jsonl"),
                recursive=True))
        if not paths:
            print(f"  (no output.jsonl found for {tag})", file=sys.stderr)
        for p in paths:
            with open(p) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    iid = rec.get("instance_id")
                    if restrict is not None and iid not in restrict:
                        continue
                    if iid in insts:      # append-log dupes: first wins
                        continue
                    insts.add(iid)
                    hist = rec.get("history") or []
                    n_act_here = 0
                    for e in hist:
                        k = _s(e.get("kind"))
                        kinds[k] += 1
                        if k == "MessageEvent":
                            # assistant text with no tool call at all
                            src = _s(e.get("source") or e.get("role"))
                            if src in ("agent", "assistant"):
                                d["msg_only_turns"] += 1
                        if k != "ActionEvent":
                            continue
                        n_act_here += 1
                        d["actions"] += 1
                        tn = _s(e.get("tool_name"))
                        is_think_tool = tn == "think"
                        if is_think_tool:
                            d["think_tool"] += 1
                        raw = "\n".join(thought_texts(e))
                        had_tag = closed = False
                        outside = inside = ""
                        if raw:
                            had_tag, closed, outside, inside = strip_think_tag(raw)
                        if raw.strip():
                            d["free_text_any"] += 1
                        if had_tag:
                            d["think_tag"] += 1
                            d["think_tag_closed"] += int(closed)
                            d["tag_inner_chars"] += len(inside.strip())
                            if len(inside.strip()) > 0:
                                d["think_tag_nonempty"] += 1
                        if outside.strip():
                            d["free_text_outside_tag"] += 1
                        if raw.strip() or is_think_tool:
                            d["any_reasoning"] += 1
                        d["free_text_chars"] += len(raw.strip())
                    per_inst_actions.append(n_act_here)
        if not d["actions"]:
            continue
        rows.append((c, model, tmpl, len(insts), per_inst_actions, d, kinds))
    return rows


def census_train(root, limit=None):
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "*", "train.llamafactory.jsonl"))):
        name = os.path.basename(os.path.dirname(p))
        if name.startswith("_") or name.startswith("pooled"):
            continue
        d = collections.Counter()
        n_traj = 0
        with open(p) as f:
            for line in f:
                if limit and n_traj >= limit:
                    break
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                n_traj += 1
                ms = rec.get("messages") or []
                last_i = len(ms) - 1
                traj_think = 0
                for i, m in enumerate(ms):
                    role = _s(m.get("role"))
                    content = m.get("content")
                    txt = content if isinstance(content, str) else json.dumps(content)
                    if role == "function_call":
                        d["fc_turns"] += 1
                        try:
                            calls = json.loads(txt)
                        except Exception:
                            calls = []
                        if not isinstance(calls, list):
                            calls = [calls]
                        for call in calls:
                            if not isinstance(call, dict):
                                continue
                            d["tool_calls"] += 1
                            if call.get("name") == "think":
                                d["think_tool"] += 1
                                traj_think += 1
                                a = call.get("arguments") or {}
                                if isinstance(a, dict):
                                    d["think_chars"] += len(_s(a.get("thought")).strip())
                        if "<think>" in txt:
                            d["think_tag_fc"] += 1
                    elif role == "assistant":
                        d["asst_turns"] += 1
                        if txt.strip():
                            d["asst_turns_texty"] += 1
                        d["asst_chars"] += len(txt.strip())
                        if i == last_i:
                            d["asst_terminal"] += 1
                        if "<think>" in txt:
                            d["think_tag_asst"] += 1
                if traj_think:
                    d["traj_with_think"] += 1
        if n_traj:
            d["traj"] = n_traj
            rows.append((name, d))
    return rows


def pct(a, b):
    return f"{100.0 * a / b:5.1f}%" if b else "    -"


def main():
    mode = sys.argv[1]
    if mode == "rollouts":
        root = sys.argv[2]
        args = [a for a in sys.argv[3:] if a != "--intersect"]
        only = args or None
        restrict = None
        if "--intersect" in sys.argv:
            tags = {c[0]: c[1] for c in CELLS}
            sets = [cell_instances(root, tags[c]) for c in (only or list(tags))]
            sets = [s for s in sets if s]
            restrict = set.intersection(*sets) if sets else set()
        rows = census_rollouts(root, only, restrict)
        print("=" * 108)
        print("ROLLOUTS -- denominator is one ActionEvent (an assistant turn that called a tool)")
        if restrict is not None:
            print(f"PAIRED on the {len(restrict)} instances present in ALL of: "
                  f"{' '.join(only or [])}")
        print("=" * 108)
        print(f"{'cell':4} {'model':5} {'tmpl':13} {'inst':>5} {'actions':>8} {'act/inst':>9} "
              f"{'freeText':>9} {'ofWhich<think>':>15} {'textOutsideTag':>15} "
              f"{'think()':>8} {'anyReason':>10}")
        for c, model, tmpl, ninst, pia, d, kinds in rows:
            a = d["actions"]
            api = sum(pia) / len(pia) if pia else 0
            print(f"{c:4} {model:5} {tmpl:13} {ninst:5} {a:8} {api:9.1f} "
                  f"{pct(d['free_text_any'], a):>9} {pct(d['think_tag'], a):>15} "
                  f"{pct(d['free_text_outside_tag'], a):>15} "
                  f"{pct(d['think_tool'], a):>8} {pct(d['any_reasoning'], a):>10}")
        print()
        # RAW COUNTS. A rounded percent cannot distinguish "never" from "rarely",
        # and the difference matters here: cell A printed freeText 0.0% while
        # carrying a non-zero character total. Zero and 0.04% are different claims.
        print(f"{'cell':4} {'freeTextN':>10} {'think()N':>9} {'tagN':>7} "
              f"{'think()/inst':>13} {'freeText/inst':>14}")
        for c, model, tmpl, ninst, pia, d, kinds in rows:
            ni = ninst or 1
            print(f"{c:4} {d['free_text_any']:10} {d['think_tool']:9} {d['think_tag']:7} "
                  f"{d['think_tool'] / ni:13.2f} {d['free_text_any'] / ni:14.2f}")
        print()
        print(f"{'cell':4} {'tagClosed':>10} {'tagNonEmpty':>12} {'meanTagChars':>13} "
              f"{'meanFreeChars':>14} {'msgOnlyTurns':>13}")
        for c, model, tmpl, ninst, pia, d, kinds in rows:
            tt = d["think_tag"] or 1
            ft = d["free_text_any"] or 1
            print(f"{c:4} {pct(d['think_tag_closed'], d['think_tag']):>10} "
                  f"{pct(d['think_tag_nonempty'], d['think_tag']):>12} "
                  f"{d['tag_inner_chars'] // tt:13} {d['free_text_chars'] // ft:14} "
                  f"{d['msg_only_turns']:13}")
        print("  freeText      = ActionEvent with a non-empty `thought` (channel a)")
        print("  ofWhich<think>= those whose thought starts a <think> tag (channel c) -- a SUBSET of freeText")
        print("  textOutsideTag= free text that survives removing the tag body: reasoning that is")
        print("                  prose rather than tag-wrapped. Distinguishes 'moved channel' from 'more text'.")
        print("  think()       = ActionEvent whose tool_name is `think` (channel b)")
        print("  anyReason     = free text OR a think() call (the union, deduplicated)")
    elif mode == "train":
        root = sys.argv[2]
        limit = None
        if "--limit" in sys.argv:
            limit = int(sys.argv[sys.argv.index("--limit") + 1])
        rows = census_train(root, limit)
        print("=" * 108)
        print("TRAINING DATA -- denominator is one assistant-AUTHORED turn "
              "(role function_call, or role assistant)")
        print("=" * 108)
        print("  NOTE the structural asymmetry vs rollouts: a `function_call` turn is a bare JSON")
        print("  tool-call array with NO text field, so free text CANNOT accompany a tool call here.")
        print("  Free text exists only as a separate `assistant` message.")
        print()
        print(f"{'arm':44} {'traj':>6} {'asstTurns':>10} {'fcTurns':>8} {'freeText%':>10} "
              f"{'terminal':>9} {'think()%ofCalls':>16} {'trajWithThink':>14} {'<think>':>8}")
        for name, d in rows:
            turns = d["fc_turns"] + d["asst_turns"]
            print(f"{name:44} {d['traj']:6} {d['asst_turns']:10} {d['fc_turns']:8} "
                  f"{pct(d['asst_turns_texty'], turns):>10} "
                  f"{pct(d['asst_terminal'], d['asst_turns']):>9} "
                  f"{pct(d['think_tool'], d['tool_calls']):>16} "
                  f"{pct(d['traj_with_think'], d['traj']):>14} "
                  f"{d['think_tag_fc'] + d['think_tag_asst']:8}")
        print()
        print(f"{'arm':44} {'toolCalls':>10} {'calls/traj':>11} {'think/traj':>11} "
              f"{'meanThinkChars':>15} {'meanAsstChars':>14}")
        for name, d in rows:
            tt = d["think_tool"] or 1
            at = d["asst_turns"] or 1
            print(f"{name:44} {d['tool_calls']:10} {d['tool_calls'] / d['traj']:11.1f} "
                  f"{d['think_tool'] / d['traj']:11.2f} {d['think_chars'] // tt:15} "
                  f"{d['asst_chars'] // at:14}")
        print("  freeText%  = share of ALL assistant-authored turns that carry prose (channel a)")
        print("  terminal   = share of those prose turns that are the LAST message, i.e. a closing")
        print("               summary rather than reasoning that precedes an action")
        print("  <think>    = literal tag occurrences anywhere in assistant-side content (channel c)")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
