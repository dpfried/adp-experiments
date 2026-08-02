# traj_compare — paired agent-trajectory comparison

Two small tools for asking *why* one model's SWE-bench agent runs differ from
another's, rather than just *by how much*.

```
extract_traj_stats.py   OpenHands output.jsonl  ->  stats.jsonl + per-turn digests
build_dashboard.py      stats.jsonl             ->  one standalone .html
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

## Notes

- Colours come from the house data-viz palette, slots 1 and 2, validated in both
  light and dark with `validate_palette.js`. Deltas are coloured by **valence**,
  not by arithmetic sign.
- Tooltips are native SVG `<title>`, so the file works with no JS and survives
  being copied around.
- Output is one self-contained file with no network dependencies.
