#!/usr/bin/env python3
"""Extract per-instance trajectory statistics + compact turn digests from
OpenHands SWE-bench eval output.jsonl files.

One streaming pass per model. Emits:
  <out>/stats.jsonl      one row per (model, instance_id)  -- small, pulled to devserver
  <out>/digest/<model>/<instance_id>.json   per-turn digest -- pulled selectively

Pure stdlib. Designed to run FAIR-side under sbatch (records are ~3MB each).
"""
import json, os, sys, glob, collections, re

# ---------------------------------------------------------------- helpers
def _s(x):
    return x if isinstance(x, str) else ""

def tool_args(ev):
    """Best-effort extraction of the tool-call argument dict."""
    tc = ev.get("tool_call")
    if isinstance(tc, dict):
        fn = tc.get("function")
        if isinstance(fn, dict):
            a = fn.get("arguments")
            if isinstance(a, str):
                try:
                    return json.loads(a)
                except Exception:
                    return {"_raw": a}
            if isinstance(a, dict):
                return a
        a = tc.get("arguments")
        if isinstance(a, str):
            try:
                return json.loads(a)
            except Exception:
                return {"_raw": a}
        if isinstance(a, dict):
            return a
    act = ev.get("action")
    if isinstance(act, dict):
        return act
    return {}

def act_signature(ev):
    """Canonical string identifying 'the same action taken again'."""
    a = tool_args(ev)
    tn = _s(ev.get("tool_name"))
    for k in ("command", "path", "file_text", "old_str", "new_str", "view_range", "_raw"):
        if k in a:
            v = a[k]
            return f"{tn}|{k}={str(v)[:300]}"
    return f"{tn}|{json.dumps(a, default=str, sort_keys=True)[:300]}"

def _blocks_to_text(v):
    """content may be a str, a list of {type:text,text:...} blocks, or nested."""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        parts = []
        for p in v:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                t = p.get("text") or p.get("content")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts)
    return ""

def obs_text(ev):
    """ObservationEvent.observation = {content:[{text}], is_error, kind, ...}"""
    o = ev.get("observation")
    if isinstance(o, str) and o:
        return o
    if isinstance(o, dict):
        for kk in ("content", "text", "output", "stdout", "result"):
            t = _blocks_to_text(o.get(kk))
            if t:
                return t
    for k in ("content", "text", "result", "output"):
        t = _blocks_to_text(ev.get(k))
        if t:
            return t
    return ""

def obs_is_error(ev):
    o = ev.get("observation")
    return bool(isinstance(o, dict) and o.get("is_error"))

def thinking_len(ev):
    tb = ev.get("thinking_blocks")
    n = 0
    if isinstance(tb, list):
        for b in tb:
            if isinstance(b, dict):
                n += len(_s(b.get("thinking")) or _s(b.get("text")))
            elif isinstance(b, str):
                n += len(b)
    return n

# A test command whose output shows the interpreter/env is broken never produced
# a real verification signal, however many times it was issued.
BROKEN_ENV_RE = re.compile(
    r"ModuleNotFoundError|ImportError|No module named|command not found|"
    r"cannot import name|ERROR: file or directory not found|No such file or directory|"
    r"Bus error|Segmentation fault|core dumped|Killed|MemoryError", re.I)
# Repo test files the task instructions tell the agent not to modify.
TESTPATH_RE = re.compile(r"(^|/)(tests?|testing)/")   # repo test trees only
SCRATCH_TEST_RE = re.compile(r"^(test_[^/]*|.*_test)\.py$")  # root-level scratch
# Files a patch can contain that the agent never authored: core dumps from its own
# crashes, and untracked build trees already present in the image.
DEBRIS_RE = re.compile(r"(^|/)core\.\d+$|(^|/)build/|\.pyc$|\.so$|\.egg-info/|(^|/)\.pytest_cache/")
# The eval image can expose the repo at two paths; editing the non-graded one
# means a correct fix never reaches the patch.
TESTBED_RE = re.compile(r"^/testbed/")
WORKSPACE_RE = re.compile(r"^/workspace/")


def patch_paths(p):
    return re.findall(r"^diff --git a/(\S+)", p or "", re.M)


def patch_stats(p):
    if not p:
        return dict(patch_len=0, patch_files=0, patch_added=0, patch_removed=0, patch_hunks=0,
                    patch_touches_tests=False, patch_test_files=0, patch_root_scratch=0,
                    patch_debris_files=0, patch_source_files=0)
    files = len(re.findall(r"^diff --git ", p, re.M))
    add = len([l for l in p.splitlines() if l.startswith("+") and not l.startswith("+++")])
    rem = len([l for l in p.splitlines() if l.startswith("-") and not l.startswith("---")])
    hunks = len(re.findall(r"^@@ ", p, re.M))
    paths = patch_paths(p)
    tf = [x for x in paths if TESTPATH_RE.search(x)]
    scratch = [x for x in paths if "/" not in x and x.endswith(".py")]
    # Not everything in a patch was written by the agent: `git add -A` sweeps up
    # crash dumps and pre-existing untracked build trees from the image.
    debris = [x for x in paths if DEBRIS_RE.search(x)]
    src = [x for x in paths if x not in set(debris) and x not in set(scratch)
           and x.endswith((".py", ".pyx", ".rst", ".txt", ".cfg", ".toml"))]
    return dict(patch_len=len(p), patch_files=files, patch_added=add, patch_removed=rem, patch_hunks=hunks,
                patch_touches_tests=bool(tf), patch_test_files=len(tf), patch_root_scratch=len(scratch),
                patch_debris_files=len(debris), patch_source_files=len(src))

TOOLS = ("terminal", "file_editor", "think", "finish")

# ---- shell command taxonomy -------------------------------------------------
TEST_RE = re.compile(r"\b(pytest|py\.test|tox|nosetests|unittest|runtests|test_runner|manage\.py\s+test|bin/test)\b")
PYRUN_RE = re.compile(r"\bpython[0-9.]*\s+(-c\b|-m\b|[\w./\-]+\.py)")
SEARCH_RE = re.compile(r"\b(grep|rg|ag|find|locate|ack)\b")
READ_RE = re.compile(r"\b(cat|head|tail|less|more|sed\s+-n|awk)\b")
NAV_RE = re.compile(r"^\s*(ls|cd|pwd|tree|du|df|wc)\b")
GIT_RE = re.compile(r"\bgit\b")
WRITE_RE = re.compile(r"(>\s*\S+|<<\s*'?EOF|tee\s+\S+)")
PKG_RE = re.compile(r"\b(pip|conda|apt-get|npm)\b")

# Degeneration signatures (quantifying what qualitative reading surfaced)
SUCCESS_RE = re.compile(
    r"successfully (implemented|fixed|resolved|completed|added)|I have (successfully|implemented|fixed|resolved)"
    r"|the (issue|bug|problem) (is|has been) (now )?(fixed|resolved)|this (fix|change) (resolves|fixes)"
    r"|## Validation|works correctly now|now works correctly", re.I)
# Non-ASCII outside ordinary typography — CJK / Cyrillic / etc. leaking into shell text
CORRUPT_RE = re.compile(r"[Ѐ-ӿ֐-ࣿ　-鿿가-힯]")
ENVACT_RE = re.compile(r"conda activate|source activate|\.\s*venv/bin/activate")

def classify_cmd(c):
    """Return the set of categories a shell command belongs to."""
    cats = set()
    if not c:
        return cats
    if TEST_RE.search(c):
        cats.add("test")
    if PYRUN_RE.search(c):
        cats.add("pyrun")
    if SEARCH_RE.search(c):
        cats.add("search")
    if READ_RE.search(c):
        cats.add("read")
    if NAV_RE.search(c):
        cats.add("nav")
    if GIT_RE.search(c):
        cats.add("git")
    if WRITE_RE.search(c):
        cats.add("write")
    if PKG_RE.search(c):
        cats.add("pkg")
    if not cats:
        cats.add("other")
    return cats

# ---------------------------------------------------------------- per record
def analyse(rec, model, resolved_ids):
    iid = rec.get("instance_id")
    hist = rec.get("history") or []
    met = rec.get("metrics") or {}
    tu = met.get("token_usages") or []
    acc = met.get("accumulated_token_usage") or {}

    actions = [e for e in hist if e.get("kind") == "ActionEvent"]
    obs = [e for e in hist if e.get("kind") == "ObservationEvent"]
    cond = [e for e in hist if e.get("kind") == "Condensation"]

    toolc = collections.Counter(_s(e.get("tool_name")) for e in actions)

    thought_lens, reason_lens, think_lens = [], [], []
    think_arg_lens = []          # the `think` tool's argument = the model's explicit reasoning
    # A model can reason on two different channels and `n_think` only sees one of
    # them. The base model emits native <think>/reasoning blocks and never calls
    # the think tool; the SFT arm is the exact mirror. Comparing `n_think` across
    # the two is apples-to-oranges, so count the channels separately and also
    # count "reasoned at all, either way" for the cross-model comparison.
    n_native_think = n_reason_content = n_any_reason = 0
    for e in actions:
        thought_lens.append(len(_s(e.get("thought"))))
        reason_lens.append(len(_s(e.get("reasoning_content"))))
        think_lens.append(thinking_len(e))
        is_tool = _s(e.get("tool_name")) == "think"
        if is_tool:
            a = tool_args(e)
            think_arg_lens.append(len(_s(a.get("thought")) or _s(a.get("_raw"))))
        native = think_lens[-1] > 0
        rc = reason_lens[-1] > 0
        n_native_think += native
        n_reason_content += rc
        n_any_reason += bool(native or rc or is_tool)

    # --- repetition / looping
    sigs = [act_signature(e) for e in actions]
    sig_counts = collections.Counter(sigs)
    max_repeat = max(sig_counts.values()) if sig_counts else 0
    dup_frac = 1.0 - (len(sig_counts) / len(sigs)) if sigs else 0.0
    run, best_run = 1, (1 if sigs else 0)
    for i in range(1, len(sigs)):
        run = run + 1 if sigs[i] == sigs[i - 1] else 1
        best_run = max(best_run, run)

    # --- errors surfaced in observations
    err_obs = 0
    flagged_err = 0
    obs_chars = 0
    for e in obs:
        t = obs_text(e)
        obs_chars += len(t)
        if obs_is_error(e):
            flagged_err += 1
        if re.search(r"Traceback \(most recent call last\)|SyntaxError|command not found|No such file or directory|ERROR:", t[:4000]):
            err_obs += 1

    # --- shell / editor behaviour ------------------------------------------
    obs_for = {}
    for e in obs:
        if e.get("tool_call_id"):
            obs_for[("tc", e["tool_call_id"])] = e
        if e.get("action_id"):
            obs_for[("act", e["action_id"])] = e

    cmdcat = collections.Counter()
    edcat = collections.Counter()
    test_turns, edit_turns, view_turns = [], [], []
    test_ok_turns = []          # test runs that actually executed (env not broken)
    files_edited, files_viewed = set(), set()
    edited_testbed, edited_workspace = set(), set()
    repro_created = False
    for i, e in enumerate(actions):
        tn = _s(e.get("tool_name"))
        a = tool_args(e)
        if tn == "terminal":
            c = _s(a.get("command"))
            cats = classify_cmd(c)
            for x in cats:
                cmdcat[x] += 1
            if "test" in cats or ("pyrun" in cats and re.search(r"repro|test", c)):
                test_turns.append(i)
                o = obs_for.get(("tc", e.get("tool_call_id"))) or obs_for.get(("act", e.get("id")))
                t = obs_text(o) if o is not None else ""
                # counts only if something actually ran: non-empty output with no
                # broken-interpreter signature
                if t and not BROKEN_ENV_RE.search(t[:3000]):
                    test_ok_turns.append(i)
        elif tn == "file_editor":
            sub = _s(a.get("command")) or "?"
            edcat[sub] += 1
            path = _s(a.get("path"))
            if sub in ("create", "str_replace", "insert", "write", "undo_edit"):
                edit_turns.append(i)
                if path:
                    files_edited.add(path)
                    if TESTBED_RE.match(path):
                        edited_testbed.add(path)
                    elif WORKSPACE_RE.match(path):
                        edited_workspace.add(path)
                if sub == "create" and re.search(r"repro|test_|/tmp/", path):
                    repro_created = True
            elif sub == "view":
                view_turns.append(i)
                if path:
                    files_viewed.add(path)

    last_edit = max(edit_turns) if edit_turns else None
    verified_after_edit = bool(last_edit is not None and any(t > last_edit for t in test_turns))
    # the same check, but only crediting test runs that actually executed
    verified_ok_after_edit = bool(last_edit is not None and any(t > last_edit for t in test_ok_turns))

    # --- degeneration signatures ------------------------------------------
    obs_by_tc = {e.get("tool_call_id"): e for e in obs if e.get("tool_call_id")}
    obs_by_act = {e.get("action_id"): e for e in obs if e.get("action_id")}
    failed_edits = failed_cmds = 0
    corrupt_turns = 0
    env_activated = False
    finish_text = ""
    for i, e in enumerate(actions):
        tn = _s(e.get("tool_name"))
        a = tool_args(e)
        blob = " ".join(str(v) for v in a.values())
        if CORRUPT_RE.search(blob):
            corrupt_turns += 1
        if tn == "terminal" and ENVACT_RE.search(blob):
            env_activated = True
        if tn == "finish":
            finish_text = blob
        o = obs_by_tc.get(e.get("tool_call_id")) or obs_by_act.get(e.get("id"))
        if o is not None and obs_is_error(o):
            if tn == "file_editor":
                failed_edits += 1
            elif tn == "terminal":
                failed_cmds += 1
    finish_claims_success = bool(finish_text and SUCCESS_RE.search(finish_text))

    comp = [u.get("completion_tokens") or 0 for u in tu]
    prom = [u.get("prompt_tokens") or 0 for u in tu]

    last_tool = _s(actions[-1].get("tool_name")) if actions else ""

    # wall-clock
    def ts(e):
        return _s(e.get("timestamp"))
    dur = None
    if hist:
        try:
            import datetime as _dt
            a = _dt.datetime.fromisoformat(ts(hist[0]))
            b = _dt.datetime.fromisoformat(ts(hist[-1]))
            dur = (b - a).total_seconds()
        except Exception:
            dur = None

    gp = ((rec.get("test_result") or {}).get("git_patch")) or ""
    row = dict(
        model=model, instance_id=iid, attempt=rec.get("attempt"),
        resolved=(iid in resolved_ids),
        error=_s(rec.get("error"))[:200] or None,
        n_events=len(hist), n_actions=len(actions), n_obs=len(obs),
        n_llm_calls=len(tu), n_condensation=len(cond),
        last_tool=last_tool, finished=(last_tool == "finish"),
        used_finish=bool(toolc.get("finish")),
        n_terminal=toolc.get("terminal", 0), n_file_editor=toolc.get("file_editor", 0),
        n_think=toolc.get("think", 0), n_finish=toolc.get("finish", 0),
        n_other_tool=sum(v for k, v in toolc.items() if k not in TOOLS),
        thought_chars_total=sum(thought_lens), thought_chars_mean=(sum(thought_lens) / len(thought_lens)) if thought_lens else 0,
        thought_chars_max=max(thought_lens) if thought_lens else 0,
        n_turns_with_thought=sum(1 for x in thought_lens if x > 0),
        n_native_think=n_native_think, n_reason_content=n_reason_content,
        n_any_reason=n_any_reason,
        reasoning_chars_total=sum(reason_lens), thinking_chars_total=sum(think_lens),
        think_arg_chars_total=sum(think_arg_lens),
        think_arg_chars_mean=(sum(think_arg_lens) / len(think_arg_lens)) if think_arg_lens else 0,
        think_arg_chars_max=max(think_arg_lens) if think_arg_lens else 0,
        obs_chars_total=obs_chars, flagged_err_obs=flagged_err,
        cmd_test=cmdcat.get("test", 0), cmd_pyrun=cmdcat.get("pyrun", 0),
        cmd_search=cmdcat.get("search", 0), cmd_read=cmdcat.get("read", 0),
        cmd_nav=cmdcat.get("nav", 0), cmd_git=cmdcat.get("git", 0),
        cmd_write=cmdcat.get("write", 0), cmd_pkg=cmdcat.get("pkg", 0),
        cmd_other=cmdcat.get("other", 0),
        ed_view=edcat.get("view", 0), ed_create=edcat.get("create", 0),
        ed_str_replace=edcat.get("str_replace", 0), ed_insert=edcat.get("insert", 0),
        ed_undo=edcat.get("undo_edit", 0),
        n_test_runs=len(test_turns), n_edits=len(edit_turns),
        n_files_edited=len(files_edited), n_files_viewed=len(files_viewed),
        n_edited_testbed=len(edited_testbed), n_edited_workspace=len(edited_workspace),
        edited_wrong_tree=bool(edited_testbed and not edited_workspace),
        first_edit_turn=(min(edit_turns) if edit_turns else None),
        last_edit_turn=last_edit,
        verified_after_edit=verified_after_edit,
        ran_any_test=bool(test_turns), repro_created=repro_created,
        n_test_ok_runs=len(test_ok_turns), ran_any_test_ok=bool(test_ok_turns),
        verified_ok_after_edit=verified_ok_after_edit,
        failed_edits=failed_edits, failed_cmds=failed_cmds,
        corrupt_turns=corrupt_turns, env_activated=env_activated,
        finish_claims_success=finish_claims_success,
        finish_text=finish_text[:600],
        completion_tokens_total=sum(comp), completion_tokens_mean=(sum(comp) / len(comp)) if comp else 0,
        completion_tokens_max=max(comp) if comp else 0,
        prompt_tokens_total=sum(prom), prompt_tokens_max=max(prom) if prom else 0,
        acc_prompt_tokens=acc.get("prompt_tokens"), acc_completion_tokens=acc.get("completion_tokens"),
        max_repeat=max_repeat, dup_action_frac=round(dup_frac, 4), max_consec_repeat=best_run,
        n_distinct_actions=len(sig_counts), err_obs=err_obs,
        duration_s=dur,
    )
    row.update(patch_stats(gp))
    row["empty_patch"] = (row["patch_len"] == 0)
    return row, actions, obs


def digest(rec, actions, obs, model):
    """Compact per-turn digest for qualitative reading."""
    turns = []
    by_tcid, by_actid = {}, {}
    for e in obs:
        if e.get("tool_call_id"):
            by_tcid[e["tool_call_id"]] = e
        if e.get("action_id"):
            by_actid[e["action_id"]] = e
    for i, e in enumerate(actions):
        a = tool_args(e)
        o = by_tcid.get(e.get("tool_call_id")) or by_actid.get(e.get("id"))
        if o is None and i < len(obs):
            o = obs[i]          # positional fallback
        turns.append(dict(
            i=i, tool=_s(e.get("tool_name")),
            args={k: (str(v)[:900]) for k, v in list(a.items())[:8]},
            obs=obs_text(o)[:1500] if o is not None else "",
            obs_err=obs_is_error(o) if o is not None else False,
        ))
    return dict(
        instance_id=rec.get("instance_id"), model=model,
        instruction=_s(rec.get("instruction"))[:4000],
        git_patch=((rec.get("test_result") or {}).get("git_patch") or "")[:8000],
        n_turns=len(actions), turns=turns,
    )


def main():
    out = sys.argv[1]
    spec = sys.argv[2:]  # model=glob pairs
    os.makedirs(out, exist_ok=True)
    statsf = open(os.path.join(out, "stats.jsonl"), "w")
    # Where score_<tag>/merged.report.json lives. Override with SWEBENCH_RUNS
    # (or SWEBENCH_ROOT, matching the eval kit's own env var).
    RUNS = os.environ.get("SWEBENCH_RUNS") or os.path.join(
        os.environ.get("SWEBENCH_ROOT", "."), "runs")

    for s in spec:
        model, tag, pat = s.split("=", 2)
        rep = os.path.join(RUNS, f"score_{tag}", "merged.report.json")
        resolved = set()
        if os.path.exists(rep):
            resolved = set(json.load(open(rep)).get("resolved_ids") or [])
        print(f"[{model}] tag={tag} resolved={len(resolved)}", flush=True)
        dd = os.path.join(out, "digest", model)
        os.makedirs(dd, exist_ok=True)
        files = sorted(glob.glob(pat, recursive=True))
        print(f"[{model}] {len(files)} files", flush=True)
        seen = set()
        n = 0
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
                seen.add(iid)
                row, actions, obs = analyse(rec, model, resolved)
                statsf.write(json.dumps(row, default=str) + "\n")
                with open(os.path.join(dd, f"{iid}.json"), "w") as g:
                    json.dump(digest(rec, actions, obs, model), g, default=str)
                n += 1
                if n % 100 == 0:
                    print(f"[{model}] {n}", flush=True)
        print(f"[{model}] DONE {n} instances", flush=True)
    statsf.close()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
