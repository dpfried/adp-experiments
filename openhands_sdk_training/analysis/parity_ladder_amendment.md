# Parity ladder — amendment merging the stub and prompt probes

**Status: filed 2026-08-05 ~03:00 UTC. Both SLURM arrays were submitted at
02:58–03:00 UTC and were still `PENDING` when this was written — no rollout
output existed. This is an amendment to a design, not a reading of a result.**

Supersedes the *designs* in `think_stub_prereg.md` §2 and
`prompt_parity_prereg.md` §2. Both of those documents stay as filed; their
findings (§1) and predictions (§3) are unchanged and are not restated here.

## Why the two probes merged

They were written as separate GPU spends, with the prompt probe explicitly
deferred (`think_stub_prereg.md` §5). dpf asked to remove every format confound
at once rather than one at a time: *"remove all possible confounding vars and
just see if we can have the arm eval match the way the arms were trained."*

That request is well-posed for the format mismatches and needs a caveat for the
semantic one:

| mismatch | kind | removing it… |
|---|---|---|
| `<think>\n\n</think>` stub, 4 tokens per assistant turn | format | removes a confound |
| `<uploaded_files>` header; `django` vs `/workspace/django/` | format | removes a confound |
| prohibition on running python/tests; 5 phases vs 8 | **semantic** | **removes capability** |

The prohibition cannot be "controlled away." Restoring it tells the model not to
run tests in a container where tests run fine. So a single all-changes-at-once
"full parity" cell would confound three edits and could not be attributed. The
merged design is therefore a ladder in which each rung adds exactly one change.

## Design as launched

Runner `swe-bench-fair-evals/scripts/run_parity_ladder.sbatch`, 6 cells × 2
shards (`select/shard_{00,01}of10.txt`), n=100/cell, 12 jobs, one pass.

| cell | model | chat template | task statement | rung adds |
|---|---|---|---|---|
| **A** | arm | stock (stub) | harness `default.j2` | — control, as the board was measured |
| **B** | arm | `nostub.jinja` | harness `default.j2` | stub removed |
| **C** | arm | `nostub.jinja` | `train_wrapper.j2` | + header & path style matched |
| **D** | arm | `nostub.jinja` | `train_swezero_noenv.j2` | + prohibition & 5-phase list |
| **E** | base | stock (stub) | harness `default.j2` | — base control |
| **F** | base | `nostub.jinja` | harness `default.j2` | stub removed |

- arm = `v2_swezero_inst_4b_a100/output`; base = the cached `Qwen/Qwen3.5-4B`
  snapshot. Both paths taken from the existing shard-00/01 runs' `llm_config.json`
  so the comparators are the same weights the board used.
- Base gets the stub axis only. The swezero training statement is arm-specific
  and means nothing for a model that never saw it.
- Cells A/B/E/F are exactly the 8-job stub design in `think_stub_prereg.md` §2;
  C/D are exactly the T1/T2 idea from `prompt_parity_prereg.md` §2, moved onto a
  no-stub base so they test the task statement and not the stub again.
- **A and E are fresh controls, not reused board numbers.** vLLM prefix caching
  plus continuous batching make greedy decoding non-reproducible.
- Held constant: harness commit, temperature 0.0, condenser settings, tool
  preset, `NUM_WORKERS=16`, `max_input_tokens` 28000.

Submitted as two arrays only because the 12-job launch needed a second
authorization: `753617` (indices 0,1,4,5,6,7,10,11) and `753633` (2,3,8,9).
Same script, same environment, same pass — the split is administrative.

## Pre-launch verification

Done before submitting, recorded so a null is not later blamed on plumbing:

| check | result |
|---|---|
| all three task templates render under Jinja `StrictUndefined` | pass — no silently-empty variables |
| `workspace_dir_name` is in the harness render context | yes, `swebench/run_infer.py:get_instruction` |
| eval phase count | 7 `Phase N` + `8. FINAL REVIEW` = **8** |
| swezero phase count | 4 `Phase N` + `5. FINAL REVIEW` = **5** |
| the 3 phases eval adds | RUNNING, TEST CREATION, VERIFICATION |
| `train_wrapper.j2` carries no prohibition | confirmed by render probe |
| `train_swezero_noenv.j2` carries the prohibition | confirmed by render probe |
| 12-index cell mapping | dry-run enumerated, all 12 distinct and correct |

One template fix was made pre-launch: `train_wrapper.j2` line 14 used
`{{ instance.repo_path }}` (renders `/workspace/django/`) where the training
statement uses `/workspace/{{ workspace_dir_name }}` (`/workspace/django`).
Aligned, so cell C is exactly "training's wrapper and path style, eval's phase
content." Deployed sha256 `980ff5975d95698f…`.

## Added predictions for the new rungs

`think_stub_prereg.md` S1–S4 and `prompt_parity_prereg.md` P1–P4 stand as filed
and carry over: S1–S3 read on A/B/E/F, P1 on C, P2–P4 on D. Two predictions are
new because they only exist once the rungs are chained.

**L1 — the format rungs are jointly inert on score.** Registered:
|C − A| ≤ 6/100. *Rationale:* B and C between them remove every non-semantic
train/eval difference for this arm. If format mismatch were costing the arm real
points, this is where it would show, and it is the cleanest available test of
"the arms are being scored unfairly."
*Falsified if* C − A ≥ +6/100 → the board is confounded by serving format and
every arm must be re-measured before any further curation work.

**L2 — the ladder is monotone in nothing.** No directional prediction for
D relative to C. D adds capability-removing instructions on top of a
format-matched baseline, so it can move either way, and I decline to predict a
sign. Recorded so that whichever direction it lands is not narrated as expected.

**Attribution rule, fixed in advance.** With four arm cells, a score change is
attributed to the *smallest* rung that produces it, not to the largest change
present. B − A is the stub; C − B is the wrapper and path; D − C is the
prohibition and phase list. A difference that appears only at D − A and is
absent at every adjacent rung is noise, not a three-way interaction.

## Reading order, fixed in advance

1. **S1 first** (manipulation check: does removing the stub change `think`
   usage at all?). If nothing moves behaviourally in B vs A, the format half of
   the ladder is inert and B/C score deltas are uninterpretable noise.
2. **E vs F** before believing anything about the arm — it is the first
   empirical test of the never-measured assumption that base is stub-native.
3. Then the arm rungs, adjacent pairs only, per the attribution rule.
4. Behavioural extraction for all six cells in **one pass** with a pinned
   `traj_compare/extract_traj_stats.py`, matching the same-scoring-pass
   discipline used for the A1 readout.

Primary endpoint remains **behavioural** (`verified_after_edit`, `ran_any_test`,
`think`-call rate) at n=100 population proportions. **Score is secondary and
underpowered**: at a 10–16% base rate and n=100, sd ≈ 3.0–3.7, so only swings
≳6–7/100 clear 2sd. Stated again here so a null on score is not read as
evidence of absence.
