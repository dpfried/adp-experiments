# traj_compare — paired agent-trajectory comparison

Tools for asking *why* one model's SWE-bench agent runs differ from another's,
rather than just *by how much*. Two layers:

**Population view** — rates and distributions over all 500 instances:

```
extract_traj_stats.py   OpenHands output.jsonl  ->  stats.jsonl + per-turn digests
build_dashboard.py      stats.jsonl             ->  one standalone .html
```

**Step-by-step view** — read individual trajectories end to end, including
training demonstrations (see [Reading whole trajectories](#reading-whole-trajectories)):

```
export_trajectories.py  output.jsonl | train.llamafactory.jsonl  ->  normalized events
build_traj_viewer.py    exported/                                ->  one standalone .html
```

Both are pure stdlib. The extractor is written to run cluster-side (eval records
are ~3 MB each, so a 500-instance run is a few GB); the dashboard builder runs
anywhere and needs only the small `stats.jsonl` it produces.

## 1. Extract

Runs one streaming pass per model. `SPEC` is `label=score_tag=glob`, where
`score_tag` names a `runs/score_<tag>/merged.report.json` (used for the
resolved/unresolved labels) and the glob points at the inference `output.jsonl`s.

```bash
export SWEBENCH_ROOT=/path/to/swebench-eval   # or SWEBENCH_RUNS=<...>/runs
R=$SWEBENCH_ROOT/runs

python3 extract_traj_stats.py OUTDIR \
  "base=v2_init_singlerun_4b=$R/out_v2_init_singlerun_4b/combined/output.jsonl" \
  "swezero=v2_swezero_4b=$R/out_v2_swezero_4b__s*of10/**/output.jsonl"
```

`SWEBENCH_ROOT`/`SWEBENCH_RUNS` is where the extractor looks for
`score_<tag>/merged.report.json`; the `output.jsonl` glob is given per spec.

Writes:

- `OUTDIR/stats.jsonl` — one row per `(model, instance_id)`, ~1 kB each. This is
  the file you copy back; everything downstream reads it.
- `OUTDIR/digest/<model>/<instance_id>.json` — per-turn digest
  (`{i, tool, args, obs, obs_err}`) for qualitative reading and for the
  fingerprint strips. Truncated per field, so it stays readable.

Because it is a few GB of JSON, submit it rather than running it on a login node:

```bash
sbatch --partition=scavenge --cpus-per-task=2 --mem=24G --time=01:00:00 \
       --wrap "python3 extract_traj_stats.py ..."
```

### What it measures

Per trajectory: turn and LLM-call counts; the tool histogram
(`terminal`/`file_editor`/`think`/`finish`); context condensations; reasoning
length (the `think` tool's argument — note that OpenHands leaves the
`thought`/`reasoning_content` fields empty for these runs, so that argument is
where the model's reasoning actually lives); token usage; patch shape; wall time.

Plus a behavioural layer, which is what makes the comparison interesting:

| field | meaning |
|---|---|
| `ran_any_test`, `n_test_runs` | did it ever execute a test command |
| `repro_created` | did it write a reproduction/test script |
| `verified_after_edit` | did a test run occur **after** the last edit |
| `env_activated` | did it activate the test environment |
| `first_edit_turn` | localization speed |
| `failed_edits` | `file_editor` calls the tool rejected (a proxy for `str_replace` against hallucinated file text) |
| `finish_claims_success` | does the final message assert success |
| `dup_action_frac`, `max_consec_repeat` | looping / degeneration |
| `corrupt_turns` | non-ASCII (CJK/Cyrillic) leaking into shell text |

`finish_claims_success && !verified_after_edit` — success asserted with nothing
run to back it — turned out to be the single most discriminating measure.

## 2. Build the dashboard

```bash
python3 build_dashboard.py \
  --stats stats.jsonl --digest digest/ --pairs selection.json \
  --a base --b swezero \
  --name-a "Qwen3.5-4B instruct (base)" --name-b "swezero SFT arm" \
  --out dashboard.html
```

`--pairs` is optional: `[{"tag": "...", "iid": "..."}]`, the instances to feature
as side-by-side **fingerprint strips** — one cell per turn, coloured by tool,
faded where the tool errored. A policy's shape is legible at a glance: long
blue-heavy shell exploration with test re-runs looks nothing like a short strip
that ends in an edit.

Panels: hero tiles · per-repo paired dumbbell · discordance (lost vs gained,
flagged symmetric-churn vs systematic) · behavioural rates · turn distribution ·
fingerprints · table view.

### A caveat the tool surfaces for you

Records whose rollout errored can carry an **empty `history`** while still
holding a patch. In the v2 base run that is 200/500 rollouts, so any per-turn
statistic for that model is conditioned on the 300 that stored a trajectory —
and those exclude its longest runs (131 of the 200 hit the 500-iteration cap).
The dashboard states this denominator in its subtitle rather than silently
averaging over 500. Check `error` and `n_actions == 0` before comparing means.

## Reading whole trajectories

The dashboard tells you *that* the arm stopped verifying. To see what a rollout
actually did — reasoning, the tool call with its full arguments, the observation
that came back — use the reader. It puts eval rollouts and the training
demonstrations they were fine-tuned on into one event schema, so a demonstration
and a rollout can sit side by side.

```bash
export SWEBENCH_RUNS=/path/to/swebench-eval/runs
python3 export_trajectories.py exported \
  "eval:base-instruct=v2_init_singlerun_4b=$R/out_v2_init_singlerun_4b/combined/output.jsonl" \
  "eval:swezero-SFT=v2_swezero_4b=$R/out_v2_swezero_4b__s*of10/**/output.jsonl" \
  "train:train-swezero=$D/nvidia_SWE-Zero-openhands-trajectories/train.llamafactory.jsonl" \
  --iids @iids.txt --train-n 4 --train-condensation 2 --max-text 6000

python3 build_traj_viewer.py --in exported --out trajectories.html
```

`--iids` takes the instances to pull from every eval spec, so the same instance
lands once per model and can be compared turn by turn. Submit the export rather
than running it on a login node (the base run's `output.jsonl` is ~1.5 GB, and
group mode streams the whole 4.6 GB training file).

Every trajectory becomes an ordered list of five event kinds — `think`, `call`
(with full arguments), `obs`, `msg`, `condense` — plus a `boundary` marker where
reassembled training segments join. In the viewer: click to open a trajectory,
**shift-click a second one to compare**, `j`/`k` to step, `/` to search, and
per-kind checkboxes to hide observations when you only want the plan.

### Three things this surfaced that the aggregate view could not

- **`metadata` in the training files is a JSON *string*, not an object.** Read it
  with `json.loads` or you will silently get nothing (`train_meta()` handles it).
- **~1/3 of the training rows are not trajectories.** They are condensation
  *prompts* — summarise-the-history examples carrying `forgotten_event_count`
  and no `trajectory_segment_index`. The exporter labels them `record_type:
  condensation` and keeps them separate rather than mixing them into the
  demonstration count.
- **Trajectories are stored chopped into segments, and our sampled subsets do
  not contain all of them.** Segments share a `source_trajectory_id` but are
  scattered across the file, and reassembly routinely yields gaps — `[1,3]`,
  `[1,2,5]`, or a lone segment 2. Row-level reservoir sampling in
  `build_v2_swe_subsets.py` selects rows, not trajectories, so the model largely
  sees scattered middles rather than complete arcs. (This is a property of the
  sampled subset, not necessarily of the upstream ADP corpus.)

Also visible, on the eval side: base rollouts contain **malformed tool calls** —
hallucinated names (`thing`, `run_shell`, `invoke_skill`) and cases where the
model broke its own tool-call syntax mid-name, embedding literal `</think>` and
`<tool_call>` text into the function name. The viewer flags these
(`malformed tool name`) instead of rendering them as ordinary calls. Tool names
are model output, so they are slugged before touching a class attribute and
escaped before display — do not remove that.

## Notes

- Colours come from the house data-viz palette, slots 1 and 2, validated in both
  light and dark with `validate_palette.js`. Deltas are coloured by **valence**,
  not by arithmetic sign.
- Tooltips are native SVG `<title>`, so the file works with no JS and survives
  being copied around.
- Output is one self-contained file with no network dependencies.
- `example_dashboard.html` is committed because it holds only aggregates. A
  rendered **trajectory reader is not committed**: it embeds raw observation
  text, which in these runs contains absolute cluster paths. Build it locally.
