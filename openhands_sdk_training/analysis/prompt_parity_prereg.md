# Train/eval prompt parity — finding + pre-registered probe

**Status: filed 2026-08-05 ~02:40 UTC, before any rollout of the probe exists.**
Nothing in §3 has been computed. §1–2 are measurements on data already on disk.

## 1. The finding: swezero's training data forbids running code

Question asked: *are our system prompts in eval not matching training?*

**System prompt: matches.** sha `ae9d8c7556`, 14,089 chars, identical in 100% of
trajectory records across all four v2 arms and the eval harness.

**Task statement: does not match** — and the interesting part is not the wrapper.

Eval builds its task statement from
`benchmarks/benchmarks/swebench/prompts/default.j2` (selectable at runtime via
`--prompt-path`, which accepts an absolute path — so a swap needs no harness
patch). The swezero training statement differs in three ways, of which only the
first is cosmetic:

1. **Wrapper** — `<uploaded_files>…</uploaded_files>` + "I've uploaded a python
   code repository in the directory `<bare-dir>`" vs eval's "I have access to a
   python code repository in the directory `/workspace/<dir>/` .".
2. **An explicit prohibition on executing anything**, present in swezero only:
   > "The development environment is unavailable. This means you CANNOT RUN
   > PYTHON CODE for any purpose. Do not write or execute any tests. Do not use
   > Python to check your work in any way. Do not install any packages."
   > "…you must not use any of these commands: python, pytest, mypy, pip, apt,
   > apt-get."
   > "Remember that you MUST NOT write or execute any tests, or run Python code
   > at all."
3. **Phase list** — swezero ships **5** phases (READING, EXPLORATION, FIX
   ANALYSIS, FIX IMPLEMENTATION, FINAL REVIEW). Eval ships **8**, adding exactly
   the three swezero omits: Phase 2 RUNNING (`conda activate testbed`), Phase 4
   TEST CREATION, Phase 7 VERIFICATION.

So eval instructs the precise behaviour swezero's data forbids.

Scanned across all six training subsets (49k–136k task statements each):

| arm | prohibition | RUNNING/VERIFY phases | eval verify% | eval any-test% | score |
|---|---|---|---|---|---|
| swezero | **99.9%** | **0.0%** | **6.2%** | **16.6%** | 77 |
| rebench | 0.0% | 99.2% | 67.9% | 100% | 76 |
| coderforge | 0.0% | 99.8% | 72.8% | 99.8% | 50 |
| scale | 0.0% | 94.7% | 70.3% | 99.8% | 36 |
| pooled55k | 26.2% | 72.1% | 67.1% | 100% | 51 |
| pooled220k | 26.4% | 72.0% | 66.6% | 100% | 57 |
| *(base, no SFT)* | — | — | 74.3% | 90.3% | 119 |

The four single-source arms split 4/4 exactly: the one arm whose data forbids
running tests is the one arm that does not run tests.

**Tool names are NOT a mismatch.** Training tools are
`terminal`/`file_editor`/`think`/`finish` (+`task_tracker` for rebench/pooled),
matching eval — the ADP conversion normalized them. The swezero prompt's literal
reference to `execute_bash` is stale text *inside the training data*, not a
train/eval gap.

### What this costs us

The behavioural signature the whole v3 A1 line was built to explain
(`verified_after_edit` 6.2%, `ran_any_test` 16.6%) is substantially
**instruction-induced**, not an emergent property of demonstration style.

`adp_v3_a1_behavioural_prereg.md` concluded "curation did not restore
verification, it halved what remained." That reading needs qualifying: swezero
data **cannot** demonstrate verification, because 99.9% of its task statements
prohibit it. Lever B ("curate for verifying trajectories") is not runnable on
this source at all.

### The pooled arms falsify the naive dose-response

At 26% prohibition a linear mixture predicts ~53% verify. Observed: 66.6% /
67.1% — indistinguishable from the 0%-prohibition arms. The naive
"prohibition dose → suppression" model is **wrong**.

The account that survives: the prohibition is **prompt-conditional**. Pooled saw
both regimes, learned the conditional mapping, and correctly ignores the
prohibition when the eval prompt omits it. swezero saw one regime only, so
"don't run tests" generalized **unconditionally**. This is a mechanism, not a
correlation, and §3 tests one arm of it.

## 2. Probe design

One arm (swezero, the `v2_swezero_inst_4b_a100` checkpoint), three prompt cells,
the same instances, all six jobs in one pass.

| cell | template | changes vs control |
|---|---|---|
| **C** | `default.j2` | — (fresh control) |
| **T1** | `train_wrapper.j2` | cosmetic header only; eval's 8 phases kept |
| **T2** | `train_swezero_noenv.j2` | full training statement (header + prohibition + 5 phases) |

- **Instances:** `select/shard_{00,01}of10.txt`, n=100/cell. Prior swezero rate
  on shard 0 is 5/50 (base 8/50); at n=50 the score endpoint resolves only a
  doubling. Cells are compared within shard, so shards pool.
- **Fresh control is mandatory.** vLLM `--enable-prefix-caching` + continuous
  batching make greedy decoding non-reproducible; the existing swezero run is
  not a parity control.
- **T2 is verified byte-identical** to a real swezero training task statement
  (rendered with that record's dir/commit/issue, diffed — exact match).
- **T1 exists because T2 alone is confounded**: T2 forbids python in a container
  where python works, so a T2 score drop cannot distinguish "the mismatch was
  harmless" from "we handicapped the model." T1 carries the cosmetic mismatch
  with no capability change.
- Everything else held: same checkpoint, temperature 0.0, same condenser
  settings, same tool preset, same 16 workers, same harness commit.

**Primary endpoint is behavioural** (`verified_after_edit`, `ran_any_test`) —
n=100 population proportions, immune to grader flake. **Score is secondary and
underpowered**: at a 10% base rate and n=100, sd ≈ 3.0, so only swings ≳6/100
are outside 2sd. Stated up front so a null on score is not read as evidence of
absence.

## 3. Predictions

Registered before any rollout exists.

**P1 — the cosmetic wrapper is inert.** |T1 − C| ≤ 4/100 on score and ≤ 10pp on
`verified_after_edit`. *Rationale:* three lines of boilerplate that change no
affordance. The board also already argues against a wrapper effect — the two
BARE-dir arms rank 1–2 and the two ABS-dir arms rank 5–6, i.e. the mismatch
anti-correlates with score.
*Falsified if* T1 − C ≥ +5/100 → the board is confounded by prompt formatting
and **every arm must be re-measured** under a matched wrapper before any further
curation work.

**P2 (manipulation check) — regime match suppresses execution.** T2
`ran_any_test` < 10% and T2 `verified_after_edit` < C. *Rationale:* swezero
already complies at 6.2%/16.6% under a prompt that *encourages* testing;
removing the encouragement and adding the prohibition should push it toward its
0.1%/0.7% training rate.
*If falsified* (T2 does not reduce execution) the model is ignoring the task
statement altogether — which makes the entire mismatch question moot, and is
itself the answer.

**P3 — matching the prompt does not rescue the score.** T2 − C < +5/100.
*Rationale:* prompt matching adds no capability, and T2 removes verification
from the ~6% of runs that used it.
*Falsified if* T2 − C ≥ +5/100 → prompt-regime match outweighs execution access;
would be genuinely surprising and would force re-running the board under matched
prompts.

**P4 (directional, no threshold).** `finish_claims_success && !verified_after_edit`
rises in T2 from swezero's 91.8%.

## 4. Decision rules, fixed in advance

1. **P1 and P3 both hold** → the task-statement mismatch does not explain
   swezero's deficit vs base. Report as a null that *hardens* the SFT-lift
   inversion verdict, and stop.
2. **P1 falsified** → halt curation work; the board is confounded. Re-measure
   all arms under a matched wrapper first.
3. **P2 falsified** → report that the behavioural signature is weight-encoded
   rather than instruction-following, which strengthens the "swezero learned an
   unconditional policy" account in §1.
4. **P3 falsified** → escalate to the mechanism test: pooled220k × {C, T2}.
   The conditional-acquisition account predicts an **interaction** — pooled's
   verify rate should collapse under T2 (it learned the conditional mapping)
   while swezero's barely moves (it never had a choice). That is the sharpest
   available test and is worth its own GPU spend only if P3 fails.

## 5. Pinned

- Templates: `swe-bench-fair-evals/prompts/{eval_default,train_wrapper,train_swezero_noenv}.j2`
  (`eval_default.j2` is a verbatim copy of the deployed `default.j2`,
  md5 `fd0b60d1f0c622f71a0e6be9289e9371`, kept so the control is auditable if
  upstream changes).
- Runner: `swe-bench-fair-evals/scripts/run_promptmatch.sbatch`. It derives
  `run_full_infer_prompt.sbatch` from the deployed `run_full_infer.sbatch`
  rather than editing it, so the shared script is never mutated.
- Metrics: `traj_compare/extract_traj_stats.py` on branch `traj-compare-viz` —
  same definitions as the v2 board and the A1 readout.
- Instance sets: `select/shard_00of10.txt`, `select/shard_01of10.txt`.
- Comparators: swezero full-500 (77, verify 6.2%, any-test 16.6%) and its
  shard-0 prior 5/50; base 119 and its shard-0 prior 8/50.
