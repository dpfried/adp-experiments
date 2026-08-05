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

## Addendum — the base iteration cap (filed 03:35 UTC, still pre-rollout)

Raised by agent-b17cac1e on the coordination channel after submission but
before any cell produced output. Independently re-verified here against the
existing board runs rather than taken on report:

| | instances with a transcript | only in `output_errors.jsonl` | of those, `MaxIterationsReached` |
|---|---|---|---|
| base `v2_init_4b` | **300 / 500** | **200 (40.0%)** | 155 |
| arm `v2_swezero_4b` | 500 / 500 | 0 | 0 |

The dropped rows carry `history_len: 0`, `git_patch: None`, and
`error: "…MaxIterationsReached: Agent reached maximum iterations limit (500)"`.
They are not blank *rows in* `output.jsonl` — they are **absent from it** and
present only in the error file. Scoring already reads both files
(`run_score_shards.sbatch` concatenates them), so **score is over 500 while
every behavioural statistic is over ~300**. That asymmetry applies to every
base number in the campaign, not just to this probe.

Consequences for cells E and F, registered now:

1. **Base behavioural n is ~60/cell, not 100.** At n=100 expect ~40 instances
   with no transcript. Any E/F behavioural proportion must state its surviving
   denominator; comparing a base proportion against an arm proportion without
   doing so compares ~60 against 100.
2. **The survivors are top-truncated.** The dropped instances are base's
   *longest* runs by construction — they were dropped for running too long. So
   base's tool-share and depth profiles are conditioned on "did not exhaust the
   budget," and are biased toward shorter trajectories. This is the sharpest
   available caveat on agent-b's item 5 (base 45.8% `terminal` / 16.2% message
   content vs arms' `file_editor` dominance): that contrast is measured on
   base's shorter 60%.
3. **E vs F must be compared on the intersection.** Only instances with a
   transcript in *both* E and F enter any behavioural diff, and the intersection
   size is reported alongside every number. Comparing E's survivors against F's
   survivors as two independent samples would let a shift in *which* instances
   survive masquerade as a shift in behaviour.
4. **Cap-hit rate becomes a registered secondary outcome per cell**, reported
   for all six cells, not only base.

**New prediction L3 (placebo / plumbing check).** The 500-iteration cap counts
**agent iterations, not tokens**, so trimming 4 tokens per assistant turn has no
mechanism by which it should move the cap-hit rate. Registered: |cap-hit(F) −
cap-hit(E)| ≤ 8pp.
*If falsified* — the cap-hit rate moves substantially between E and F — then
something larger than a 4-token trim changed, and the correct response is to
**inspect the rendered prompts and the vLLM logs before interpreting any E/F
result**. In that world an E−F resolve-rate delta is confounded by how much
budget each condition got, not by serving format, and must not be reported as a
format effect. This is the cheapest available guard against the ladder silently
measuring budget instead of parity.

Note this cuts against a reading of the existing board: base's 119/500 is
achieved while 40% of its instances never finish. Whether that makes base's
score an under- or over-statement is not something this probe measures, and I
am not claiming it does.

## Addendum 2 — reasoning-channel accounting and the F−B differential

Filed 04:00 UTC, still pre-rollout. Raised by devils-advocate reviewing the
three prereg docs.

### The objection

S1 reads the stub's effect on the arm's `think`-**tool** rate (B vs A). In cell
A the stub pre-closes an empty `<think></think>`, so the tool is the only
reasoning channel available. In cell B the assistant prefix is bare
`<|im_start|>assistant\n`, which in principle permits a **native** `<think>…</think>`
block. If nostub unlocks native blocks, a rise in tool rate at B could be
reasoning *relocating between channels* (a format shift) rather than
*un-suppressing* (a rate change) — identical S1 signature, different meaning.

### Resolved for the arm, live for base — measured, not assumed

Scanned assistant-generated content in the existing board runs, both served
with the **stock** (stub) template:

| | records with a native `<think>` block | total blocks |
|---|---|---|
| arm `v2_swezero_4b` | **0 / 500** | 0 |
| base `v2_init_4b` | **5 / 324 (1.5%)** | 5 |

The arm never emits a native block, consistent with 0 `<think>` occurrences in
6,508,136 training tokens. Base does, at a low rate, *despite* the stub already
being closed — so base retains the capability and the stub does not fully
suppress it.

Consequences:

1. **S1 (B−A) stands as written.** For the arm the relocation channel is
   empirically absent at n=500 under stock. It is still measured in B/C/D
   rather than assumed, because nostub changes the prefix and could in
   principle unlock what stock did not.
2. **Base cells require combined-channel accounting.** Any E/F reasoning
   comparison is reported as **native-block tokens + `think`-tool tokens**, and
   the two components are also reported separately so a pure relocation is
   visible as a compositional shift at constant total.
3. **Native-`<think>` rate becomes a registered secondary outcome for all six
   cells.** If it is non-zero in B/C/D, S1 is re-scored on the combined channel
   before being read.

### F vs B — the matched base-vs-arm differential

devils-advocate correctly notes that the base-vs-arm tool-profile question
(base 45.8% `terminal` / 16.2% message content vs arms' `file_editor`
dominance, previously n=1) is answered not by E/F but by **F vs B**: base-nostub
vs arm-nostub, same harness, same template, same shards, differing only in
model. This is already in the ladder and is now named as an explicit readout.

**Registered as a directional lower bound, not a point estimate.** The cap
asymmetry is ~40% (base) vs ~0% (arm), and base's dropped instances are its
longest by construction, so F and B cannot be intersected across models. Base's
terminal-heavy profile is therefore measured on its shorter-finishing runs, and
the true base-vs-arm terminal gap is if anything **larger** than F−B shows.
Reported with that direction stated.

### Noise band for E−F, and what L3 can and cannot do

No same-condition replicate exists, so an empirical run-noise band cannot be
pre-registered from data. Instead, registered procedure: because E and F are
compared **on the intersection**, each instance yields a paired difference, so
the standard error of the mean paired difference is computable directly from
this run. **Rule: treat |mean paired Δ| < 2·SE as uninformative** and say so,
rather than narrating a small tool-share move as a serving artifact.

Recorded explicitly, per devils-advocate: L3's 8pp threshold is ≈1.6·SE at
n=100, so it is deliberately insensitive — it will not false-alarm on decode
noise, and correspondingly **cannot detect a real sub-8pp cap-hit effect**. L3
is a plumbing guard, not a test of the stub's effect on budget, and must not be
reported as the latter.

---

## Addendum 3 — served-prompt verification (S0), measured on the live servers

The chat-template override was verified three ways before any rollout finished,
in increasing strength:

1. **Launcher echo** — each cell logs the template path it was handed, plus the
   `sha256` of `nostub.jinja` (`603813c5466bbf49`, identical across cells).
2. **vLLM arg dump** — the nostub cells list
   `'chat_template': '<...>/prompts_adp/nostub.jinja'` under `non-default args`;
   the stock cells do not. This proves vLLM *accepted* the flag.
3. **Rendered-token probe** — the decisive one. `POST /tokenize` on each live
   server with the same one-message conversation, `add_generation_prompt=true`
   **and `chat_template_kwargs={"enable_thinking": false}`** (the kwarg the
   harness always sends):

   | cell | model | template | tokens | tail |
   |---|---|---|---|---|
   | A | arm  | stock  | 13 | `… 248068, 271, 248069, 271` |
   | B | arm  | nostub |  9 | `… 74455, 198` |
   | E | base | stock  | 13 | `… 248068, 271, 248069, 271` |
   | F | base | nostub |  9 | `… 74455, 198` |

   The 4-token difference is exactly `<think>` `\n\n` `</think>` `\n\n`
   (`[248068, 271, 248069, 271]`) and nothing else — the prefix is
   byte-identical. The manipulation is confirmed **in the string the model
   actually conditions on**, on all four cells, not merely in a launcher flag.

**Recorded trap for anyone re-running this probe:** omitting
`chat_template_kwargs` makes stock and nostub render *identically* (both 11
tokens, ending `<think>` `\n`). That is correct behaviour, not a failed
override — `nostub.jinja` only edits the `enable_thinking=false` branch, and the
default branch opens a real `<think>` for the model to fill. A probe without the
kwarg tests the wrong branch and would wrongly read as "the override did
nothing."
