#!/usr/bin/env python3
"""Sample adp-v2 sft_openhands_24k records per config and analyze final-turn behavior.

Streams two windows (head + mid-file) of the first shard of each config's 24k
condenser SFT JSONL over HTTP, then reports per config:
  - final assistant action taxonomy (finish / edit / exec / browser / message-only ...)
  - fraction of records with any file-edit evidence, and edit in last 3 assistant turns
  - test/verification evidence, `diff --git` presence
  - condensation segment stats (segment index > 0 == continuation records)
Only needs the Python stdlib.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request


def urlopen_retry(req, timeout=180, tries=5):
    for attempt in range(tries):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < tries - 1:
                wait = 30 * (attempt + 1)
                print(f"429 rate-limited, sleeping {wait}s", flush=True)
                time.sleep(wait)
                continue
            raise
from collections import Counter
from pathlib import Path

BASE = "https://huggingface.co/datasets/neulab/adp-v2/resolve/main"
API_TREE = "https://huggingface.co/api/datasets/neulab/adp-v2/tree/main?recursive=true"

OLD_MIX = [
    "swe-smith",
    "swe-gym_openhands_sampled_trajectories",
    "nebius_SWE-agent-trajectories",
    "openhands",
    "codeactinstruct",
    "code_feedback",
    "orca_agentinstruct",
    "synatra",
]
NEW_SWE = [
    "nvidia_SWE-Zero-openhands-trajectories",
    "coderforge_preview",
    "scale_swe_distilled",
    "logicstar_swe-star",
    "nebius_SWE-rebench-openhands-trajectories",
    "mini-coder",
    "kwai-klear_swe-smith-mini_swe_agent_plus-trajectories-66k",
    "SALT-NLP_SWE-chat",
    "swe-play-trajectories",
    "hybrid-gym",
    "codescout",
    "openthoughts_agent_sft",
]

EDIT_TOOL_RE = re.compile(r"edit|str_replace|apply_patch|write_file|create_file", re.I)
EXEC_TOOL_RE = re.compile(r"terminal|bash|execute|ipython|jupyter|run_command|shell|cmd_run", re.I)
BROWSER_TOOL_RE = re.compile(r"browser|goto|click|scroll|navigate|type_text|hover|tab", re.I)
FINISH_TOOL_RE = re.compile(r"finish|submit|complete|done|send_msg_to_user|report", re.I)

EDIT_IN_ARGS_RE = re.compile(
    r"str_replace|\"command\":\s*\"(create|insert)\"|sed\s+-i|git\s+apply|patch\s+-p\d"
    r"|\btee\s|cat\s*<<|cat\s*>|echo\s+.{0,200}?>>?\s*\S+\.(py|txt|js|ts|json|c|cpp|go|rs|java|md|cfg|toml|yaml|yml)"
    # SWE-agent ACI editing syntax passed inside the `terminal` tool's command arg:
    #   edit <start>:<end>\n...new lines...\nend_of_edit   |   create <path>   |   insert <line>
    r"|end_of_edit|\bedit\s+\d+:\d+|\bcreate\s+\S+\.(py|txt|js|ts|json|c|cpp|go|rs|java|md|cfg|toml|yaml|yml)",
    re.I | re.S,
)
TEST_IN_ARGS_RE = re.compile(
    r"pytest|python\s+-m\s+unittest|npm\s+test|\btox\b|make\s+test|cargo\s+test|go\s+test|reproduce", re.I
)
DIFF_RE = re.compile(r"diff --git ")


def classify_tool(name: str) -> str:
    if FINISH_TOOL_RE.search(name):
        return "finish"
    if EDIT_TOOL_RE.search(name):
        return "edit"
    if EXEC_TOOL_RE.search(name):
        return "exec"
    if BROWSER_TOOL_RE.search(name):
        return "browser"
    if name == "think":
        return "think"
    return "other"


def analyze_record(rec: dict) -> dict | None:
    msgs = rec.get("messages")
    if not msgs:
        return None
    assistant_turns = [m for m in msgs if m.get("role") == "assistant"]
    if not assistant_turns:
        return None

    def turn_info(m):
        tcs = m.get("tool_calls") or []
        names = []
        args = []
        for t in tcs:
            fn = t.get("function", {})
            names.append(fn.get("name", "?"))
            args.append(str(fn.get("arguments", "")))
        return names, " ".join(args)

    final_names, final_args = turn_info(assistant_turns[-1])
    if not final_names:
        final_class = "message_only"
        final_tool = "-"
    else:
        final_tool = final_names[-1]
        final_class = classify_tool(final_tool)
        # a "finish" whose args are trivial vs substantive doesn't change class here

    any_edit = False
    edit_last3 = False
    any_test = False
    n_tool_calls = 0
    for i, m in enumerate(assistant_turns):
        names, args = turn_info(m)
        n_tool_calls += len(names)
        is_edit = any(classify_tool(n) == "edit" for n in names) or bool(EDIT_IN_ARGS_RE.search(args))
        if is_edit:
            any_edit = True
            if i >= len(assistant_turns) - 3:
                edit_last3 = True
        if TEST_IN_ARGS_RE.search(args):
            any_test = True

    all_text = " ".join(str(m.get("content") or "")[:20000] for m in msgs)
    meta = rec.get("metadata") or {}
    return {
        "final_class": final_class,
        "final_tool": final_tool,
        "any_edit": any_edit,
        "edit_last3": edit_last3,
        "any_test": any_test,
        "has_diff": bool(DIFF_RE.search(all_text) or DIFF_RE.search(final_args)),
        "n_messages": len(msgs),
        "n_assistant": len(assistant_turns),
        "n_tool_calls": n_tool_calls,
        "chars": sum(len(str(m.get("content") or "")) for m in msgs),
        "segment_index": meta.get("trajectory_segment_index"),
        "tools_declared": [t.get("function", {}).get("name", "?") for t in rec.get("tools") or []],
    }


def fetch_window(url: str, start: int, max_records: int, max_bytes: int) -> list[dict]:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + max_bytes - 1}"})
    out = []
    with urlopen_retry(req, timeout=180) as resp:
        buf = b""
        skipped_partial = start == 0
        while len(out) < max_records:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for ln in lines:
                if not skipped_partial:
                    skipped_partial = True  # first line at a mid-file offset is partial
                    continue
                if not ln.strip():
                    continue
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue
                if len(out) >= max_records:
                    break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-window", type=int, default=250)
    ap.add_argument("--max-window-bytes", type=int, default=80 * (1 << 20))
    ap.add_argument("--out", type=Path, default=Path("analysis/v2_sft_action_analysis.json"))
    ap.add_argument("--configs", nargs="*", default=None)
    args = ap.parse_args()

    import os
    if os.path.exists('/tmp/adp_v2_tree.json'):
        tree = json.load(open('/tmp/adp_v2_tree.json'))
    else:
        with urlopen_retry(API_TREE, timeout=120) as r:
            tree = json.load(r)
    files = {f["path"]: f["size"] for f in tree if f["type"] == "file"}

    configs = args.configs or (OLD_MIX + NEW_SWE)
    results = {}
    for cfg in configs:
        shards = sorted(p for p in files if p.startswith(cfg + "/24k/full_sft/") and p.endswith(".jsonl"))
        if not shards:
            print(f"{cfg}: NO 24k SFT FILE FOUND", flush=True)
            continue
        shard = shards[0]
        size = files[shard]
        url = f"{BASE}/{shard}"

        manifest = {}
        mpath = f"{cfg}/24k/manifest.json"
        if mpath in files:
            try:
                with urlopen_retry(f"{BASE}/{mpath}", timeout=60) as r:
                    manifest = json.load(r)
            except Exception as e:  # noqa: BLE001
                print(f"{cfg}: manifest fetch failed: {e}", flush=True)

        recs = []
        for label, start in (("head", 0), ("mid", max(0, size // 2))):
            try:
                got = fetch_window(url, start, args.per_window, args.max_window_bytes)
                print(f"{cfg}: {label} window -> {len(got)} records", flush=True)
                recs.extend(got)
            except Exception as e:  # noqa: BLE001
                print(f"{cfg}: {label} window failed: {e}", flush=True)

        infos = [i for i in (analyze_record(r) for r in recs) if i]
        if not infos:
            print(f"{cfg}: no analyzable records", flush=True)
            continue
        n = len(infos)
        final_classes = Counter(i["final_class"] for i in infos)
        final_tools = Counter(i["final_tool"] for i in infos)
        seg_idx = [i["segment_index"] for i in infos if isinstance(i["segment_index"], int)]
        tool_inventories = Counter(tuple(i["tools_declared"]) for i in infos)
        results[cfg] = {
            "group": "old_mix" if cfg in OLD_MIX else "new_swe",
            "n_sampled": n,
            "n_shards": len(shards),
            "shard_bytes_total": sum(files[s] for s in shards),
            "manifest_std_lines": manifest.get("std_lines"),
            "manifest_condensation_lines": manifest.get("condensation_lines"),
            "manifest_llm_model": manifest.get("llm_model"),
            "final_class_pct": {k: round(100 * v / n, 1) for k, v in final_classes.most_common()},
            "final_tool_top": {k: round(100 * v / n, 1) for k, v in final_tools.most_common(8)},
            "pct_any_edit": round(100 * sum(i["any_edit"] for i in infos) / n, 1),
            "pct_edit_last3": round(100 * sum(i["edit_last3"] for i in infos) / n, 1),
            "pct_any_test": round(100 * sum(i["any_test"] for i in infos) / n, 1),
            "pct_has_diff": round(100 * sum(i["has_diff"] for i in infos) / n, 1),
            "pct_continuation_segment": round(100 * sum(1 for s in seg_idx if s and s > 1) / max(1, len(seg_idx)), 1),
            "segment_multiplier": round(manifest["condensation_lines"] / manifest["std_lines"], 2)
            if manifest.get("std_lines") and manifest.get("condensation_lines")
            else None,
            "mean_messages": round(sum(i["n_messages"] for i in infos) / n, 1),
            "mean_assistant_turns": round(sum(i["n_assistant"] for i in infos) / n, 1),
            "mean_chars": int(sum(i["chars"] for i in infos) / n),
            "distinct_tool_inventories": len(tool_inventories),
            "most_common_tools": list(tool_inventories.most_common(1)[0][0]) if tool_inventories else [],
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {args.out}", flush=True)

    hdr = f"{'config':<44} {'grp':<7} {'n':>4} {'finish%':>7} {'edit%':>6} {'ed<3%':>6} {'test%':>6} {'diff%':>6} {'cont%':>6} {'msgs':>5}"
    print(hdr)
    print("-" * len(hdr))
    for cfg, s in results.items():
        print(
            f"{cfg:<44} {s['group'][:7]:<7} {s['n_sampled']:>4} "
            f"{s['final_class_pct'].get('finish', 0):>7} {s['pct_any_edit']:>6} {s['pct_edit_last3']:>6} "
            f"{s['pct_any_test']:>6} {s['pct_has_diff']:>6} {s['pct_continuation_segment']:>6} {s['mean_messages']:>5}"
        )


if __name__ == "__main__":
    main()
