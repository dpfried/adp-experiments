# Infra errors in the SWE-bench Verified runs, and what they did to Qwen3.5-4B Instruct

Written 2026-07-26. Source data: `output.critic_attempt_1.jsonl` per arm, snapshotted before
the retry purge as `output.critic_attempt_1.jsonl.prepurge-20260726-164127`.

## TL;DR

The `rawinstruct4b` arm (raw `Qwen/Qwen3.5-4B`, the instruct model) **has never actually been
measured on SWE-bench Verified.** It attempted all 500 instances, but 431 of them (86%) ended in
an error row rather than a rollout. 78% of those errors are infrastructure, not model behaviour,
and they trace almost entirely to one event: the shared-NFS brownout on tir1 on 2026-07-16/17.

The published-looking number for that arm — 13 resolved — is out of a scored denominator of 53,
not 500. Reading it as `13/500 = 2.6%` is wrong, and reading it as `13/53 = 24.5%` is also wrong
(survivorship bias). The honest statement is that the arm is unmeasured.

Worse, the eval harness's resume logic treats an error row as a *completed* instance, so every
resubmit skipped exactly the instances that had failed. The damage was sticky: it never healed
on its own, and it was distributed unevenly across arms, which biased cross-arm comparison.

This is not only a baselines problem. The headline **init ablation is contaminated too** —
`swesmithinstruct540` lost 61 instances to infra against `swesmith540`'s 9, all 61 of them
infrastructure. That plausibly halves the measured instruct-init advantage (§4b).

## 1. What the errors were

`rawinstruct4b`, all 431 error rows in attempt 1:

| Error | n | Attribution |
|---|---:|---|
| `Command '['git','rev-parse','--show-toplevel']' returned non-zero exit status 128` | 191 | infra — NFS brownout |
| `MaxIterationsReached: Agent reached maximum iterations limit (500)` | 70 | **model** |
| `cp_testbed_repo failed: Command timed out after 600.0 seconds` | 67 | infra — NFS brownout |
| `Instance did not complete within 14400s timeout` | 25 | mixed — model verbosity × node contention |
| `git reset failed: Command timed out after 300.0 seconds` | 22 | infra — NFS brownout |
| `Remote conversation got stuck` | 17 | infra |
| `Remote conversation ended ...` | 14 | infra |
| `[Errno 4] Interrupted system call` | 8 | infra |
| `Run timed out after Ns` (LLM request) | 7 | infra |
| `HTTPSConnectionPool(host='ghcr.io'): Max retries exceeded` | 5 | infra — registry |
| `Container failed to become healthy in time` | 3 | infra |
| `git diff failed: fatal: bad object ...` | 1 | infra |
| `git diff failed: can't cd to /workspace/astropy/...` | 1 | infra |

**336 infra (78%) · 70 model (16%) · 25 mixed (6%).**

Every arm, for comparison — error rows in attempt 1, and how unevenly they landed:

| Arm | attempted | error rows | % errored |
|---|---:|---:|---:|
| rawinstruct4b (Qwen3.5-4B instruct) | 500 | 431 | 86% |
| base4b (Qwen3.5-4B-Base) | 500 | 279 | 56% |
| v2coderforge1719 | 500 | 108 | 22% |
| v2rebench1719 | 212 | 32 | 15% |
| swesmithinstruct540 | 500 | 61 | 12% |
| v2scale1719 | 188 | 12 | 6% |
| papernonweb1154 | 500 | 16 | 3% |
| swesmith540 | 500 | 9 | 2% |
| v2swezero1719 | 358 | 0 | 0% |

That spread — 86% to 0% — is the whole problem. It is not a property of the models.

## 2. Root causes

**(a) `git rev-parse --show-toplevel` exit 128 — 191 rows, the single largest class.**
`build_base_images._get_repo_root()` shelled out to `git rev-parse --show-toplevel` **with no
`cwd` pinned**, and it was called *per instance* via `get_phased_image_tag_prefix`. Under NFS
brownout that subprocess returns 128, and since it runs on every `prepare_workspace`, entire
runs mass-failed in seconds. Fixed in benchmarks commit `b0cf90dd` (2026-07-23) by computing the
root statically from `Path(__file__)` — no subprocess. The sbatch also now exports
`IMAGE_TAG_PREFIX` to short-circuit the call.

Timing: `b0cf90dd` landed 2026-07-23, a week *after* rawinstruct4b finished on 07-16. So all 191
of these are fixed going forward.

**(b) Filesystem op timeouts — 89 rows — are NOT an unfixed-timeout bug.**
This is the part that is easy to get wrong. `ebe771e5` (2026-07-13 23:12) had already raised
`cp_testbed_repo` from 30s → 600s and `git reset` → 300s. The rawinstruct4b failures are all at
the **raised** limits (`600.0 seconds`, `300.0 seconds`) — i.e. copying a testbed repo took more
than ten minutes and still didn't finish. That's a sick filesystem, not a tight timeout.

The repo distribution proves it: 86 of the 89 FS timeouts are `django` (61) and `matplotlib`
(25) — the two largest working trees in the benchmark. Nothing else is meaningfully affected.

By contrast, base4b's 137 `cp_testbed_repo` failures are all at `30.0 seconds` — that arm ran
*before* `ebe771e5`, so for base4b this genuinely was the unfixed-timeout bug.

**(c) The rest** (remote-conversation stalls, interrupted syscalls, ghcr.io retry exhaustion,
container health) are the usual symptoms of the same degraded-node/degraded-NFS window.

**Timeline.** rawinstruct4b ran across four legs, 2026-07-13T23:00 → 2026-07-16T18:30
(jobs 9278509, 9283538, 9309527/8/9, 9283539). Its final legs on 07-16 coincide exactly with the
NFS brownout that `b0cf90dd`'s docstring records as costing ">2500 instance attempts" on
2026-07-16/17.

## 3. Why the damage never healed

`get_completed_instances()` (`benchmarks/utils/critics.py:148`) returns every `instance_id`
present in the attempt file — explicitly "regardless of success/failure". `_get_instances_for_attempt()`
(`benchmarks/utils/evaluation.py:455`) then builds the to-do list as *all instances minus completed*.

So an instance that errored is recorded as done, and **every subsequent resubmit skips it**. This
is noted in `RESULTS.md` as an operational gotcha ("erroring instances exhaust 3 in-run attempts
and are NOT retried across chained legs"), but its consequence for cross-arm comparison wasn't
followed through: the arms did not fail at equal rates, so the surviving denominators were not
comparable.

## 4. The effect on Qwen3.5-4B Instruct

Scored result on disk: **13 resolved / 53 scored**. Two ways to read it, both wrong:

- `13/500 = 2.6%` — treats 431 infra failures as model failures. Severe underestimate.
- `13/53 = 24.5%` — conditions on surviving instances, which are not a random sample.

To test the second, here is how the *other* arms did on exactly those 53 instances versus their
own overall rate:

| Arm | overall | on the rawinstruct-53 subset | Δ |
|---|---:|---:|---:|
| swesmithinstruct540 | 82/500 (16.4%) | 14/53 (26.4%) | **+10.0pp** |
| papernonweb1154 | 52/500 (10.4%) | 10/53 (18.9%) | **+8.5pp** |
| swesmith540 | 74/500 (14.8%) | 8/53 (15.1%) | +0.3pp |
| base4b | 53/379 (14.0%) | 5/48 (10.4%) | −3.6pp |

Two of four arms find the surviving subset substantially easier; one is neutral, one slightly
harder. With n=53 the binomial σ is ≈4.9pp, so +10.0pp and +8.5pp are ≈2.0σ and ≈1.7σ —
suggestive rather than conclusive, but pointing the same direction. The subset skews easy, so
24.5% is if anything an *optimistic* ceiling.

The mechanism is intuitive: the instances that survived are disproportionately the ones whose
testbed repos copy fast (small repos), and django — 176 of the 500, and the repo the agent
struggles most with — is exactly where the failures concentrated.

**Conclusion: the arm is unmeasured, not bad.** It should not appear in any comparison table
until re-run. Note also that this is the correct zero-shot baseline for all four adp-v2 arms,
since those are instruct-init — so its absence is not a minor gap.

### The one real signal in the wreckage

The 70 `MaxIterationsReached` rows *are* model behaviour: on ~14% of attempted instances the raw
instruct model burned all 500 iterations without converging, and 64 of those 70 are django. That
is a genuine finding about the untrained instruct model, and it is the one error class we are
deliberately **not** retrying — at temp 0 it would reproduce identically.

## 4b. The same bias also distorts the init ablation

This is not confined to the wrecked baselines. The headline init ablation — same SWE-smith data,
base-init vs instruct-init — is contaminated in a way that is easy to miss because both arms look
"complete" at 500/500:

| Arm | resolved | errored | of which infra |
|---|---:|---:|---:|
| swesmith540 (base-init) | 74 | 9 | 8 |
| swesmithinstruct540 (instruct-init) | 82 | **61** | **61 (all of them)** |

The instruct arm lost 6.8× more instances, and *every single one* was infrastructure — 43
`cp_testbed_repo` timeouts, 14 remote-conversation failures, 3 `git reset` timeouts, 1 LLM request
timeout, and zero `MaxIterations`. The two obvious estimators disagree, and they bracket the truth
from opposite sides:

- **Raw rate** (errors counted as unresolved): 16.4% vs 14.8% → **+1.6pp** for instruct-init.
  Understates it, because 61 instruct-arm instances were scored as failures without being run.
- **Conditional rate** (errors dropped from the denominator): 18.7% vs 15.1% → **+3.6pp**.
  Overstates it, because errors concentrate in the largest and hardest repos, so dropping them
  removes hard work from the denominator — and removes 6.8× more of it from the instruct arm.

So the reported +1.6pp init gap, which sits inside the ±1–2pp noise band and reads as "suggestive
only", may well be a real ~2–4pp effect. The re-run is needed to tell. This is the clearest
demonstration of why the bias matters: it did not just add noise, it systematically shrank the
effect the ablation exists to measure.

## 5. What was done (2026-07-26)

1. `full/purge_error_rows.sh` drops retryable error rows from `output.critic_attempt_1.jsonl` so
   the harness's own resume logic re-runs exactly those instances, uniformly, at temp 0 in
   attempt 1. `MaxIterations` rows are kept as legitimate results. Originals are backed up to
   `*.prepurge-<stamp>`.
2. Applied to all nine arms (`*.prepurge-20260726-164127` and `-170855`):

   | Arm | attempt-1 before | after | error rows dropped | instances the resubmit will run |
   |---|---:|---:|---:|---:|
   | rawinstruct4b | 500 | 139 | 361 | 361 |
   | v2rebench1719 | 212 | 183 | 29 | 317 |
   | v2scale1719 | 188 | 176 | 12 | 324 |
   | base4b | 500 | 224 | 276 | 276 |
   | v2swezero1719 | 358 | 358 | 0 | 142 |
   | v2coderforge1719 | 500 | 401 | 99 | 99 |
   | swesmithinstruct540 | 500 | 439 | 61 | 61 |
   | papernonweb1154 | 500 | 484 | 16 | 16 |
   | swesmith540 | 500 | 492 | 8 | 8 |
   | **total** | | | **862** | **1604** |

3. All nine resubmitted as chained 2-day pairs — jobs 9508098–9508109 (six arms) and
   9509365–9509370 (the three older arms) — on the sbatch with `IMAGE_TAG_PREFIX` pinned,
   workspaces on node-local `/scratch`, `--time` 2d and `--mem` 350G.

Note that the existing `score_*` directories are now stale for every re-run arm; they will need
re-scoring via `run_score_shards.sbatch` once inference lands.

## 5b. Progress and a second failure mode (2026-07-27)

Four arms have landed clean at 500/500 with `output.jsonl` aggregated:
**v2swezero1719, swesmith540, swesmithinstruct540, papernonweb1154.** v2coderforge1719 is at
499/500 and still running; base4b and rawinstruct4b are in progress.

A separate, unrelated failure surfaced during the re-run: **vLLM intermittently hangs at TP2
startup**, immediately after `vLLM is using nccl==2.28.9`, and never serves — despite
`NCCL_NVLS_ENABLE=0` already being set. Healthy startup is 5–8 minutes, so the 30-minute health
wait expiring means the server is wedged, not slow. It hit four job launches (9508102, 9508103,
9508104, 9508106) across three different nodes, so it is flaky rather than node-specific. The
chained-pair design absorbed it for rebench and rawinstruct, but **v2scale1719 drew it on both
legs and died without running a single instance**.

Two fixes to the sbatch:

1. **In-job vLLM startup retry** (`VLLM_TRIES=3`, 15-minute health window each) instead of
   forfeiting an entire 2-day allocation to a wedged server.
2. **Merged the EXIT traps.** The vLLM-kill trap was silently *replacing* the workspace-cleanup
   trap — bash allows one EXIT handler, and the second `trap ... EXIT` overwrites the first — so
   the `/scratch` workspace cleanup that the ENOSPC fix introduced had never actually run. Both
   now live in a single `cleanup()`.

Slurm snapshots the batch script at submit time, so these apply to submissions after 2026-07-27
only. v2scale1719 was resubmitted (9562263/4) with the hardened script and the two nodes it hung
on excluded.

## 6. What to watch

- **Re-check the error rate per arm when these land.** The purge fixes the *recorded* bias; it
  does not prevent a fresh brownout from re-creating it. If any arm comes back with a materially
  higher error count than the others, the comparison is contaminated again.
- **`14400s` per-instance timeouts are contention-dependent** and will vary with how busy the
  node is. They are being retried, but they are the class most likely to differ between arms for
  reasons unrelated to the model.
- **`RESULTS.md` numbers for `base4b` and `rawinstruct4b` are not usable** and are marked as such.
- Per RESULTS.md's own note: at p≈0.1, n=500, binomial σ≈1.3pp — treat ±1–2pp between-arm gaps in
  single greedy runs as noise regardless of any of the above.
