# Investigation handoff: provenance of the "base Qwen3.5-4B ≈ 25/500 (5%)" SWE-bench anchor

**For:** an agent running in the Babel environment (where the original ADP SWE-bench evals were run).
**Please:** investigate, then report findings back to dpf (this is a fact-finding task, not a code change).

---

## The ask (one sentence)
Track down where the baseline anchor **"base Qwen3.5-4B ≈ 25/500 (≈5%)"** (and **"paper-nonweb ≈ 52/500 ≈ 10.4%"**)
came from — was it a real in-house SWE-bench eval, and if so **on which model, harness, and split** — because a
fresh rerun measured a very different number and it inverts the campaign's headline.

## Why this matters
The adp-v2 campaign trained 4 SFT "arms" from a starting model θ₀ and reported an "SFT lift" of θ₀ ≈5% → arms
≈14–15%. That lift was computed against the **≈5% anchor above**, which was carried in as a cross-campaign
"sanity anchor," not re-measured. A fresh A100 rerun measured the SFT starting point **θ₀ = Qwen3.5-4B *instruct*,
single-run pass@1, SWE-bench Verified** at **≥70/500 (≥14%)** — roughly **3× the 5% anchor**. If ≥14% is the right
baseline, the "SFT lift" **inverts**: the arms (77/70/48/35 on the same 500) no longer beat base, and two are
below it. So the exact provenance and comparability of the 5% number is now load-bearing.

## Leading hypothesis (please test this FIRST — it's the cheapest resolution)
**Base-vs-instruct model mismatch.** The anchor literally says "***base*** Qwen3.5-4B." Two distinct models exist:
`Qwen3.5-4B-Base` (pretrained, non-instruct) and `Qwen3.5-4B` (instruct). The arms were fine-tuned **from the
instruct model**, so the correct θ₀ baseline is *instruct*. If the 25/500 (5%) was measured on
**Qwen3.5-4B-Base** (non-instruct — which plausibly does score ~5% agentically), then **both numbers are real and
correct** — they're just *different models*. The campaign would have mistakenly used the *base* model's 5% where
it should have used the *instruct* model's ~14%. That makes the "SFT-lift" a model-mismatch artifact, not a
fabrication, and it cleanly explains the 5% vs 14% gap.

## What to search for on Babel
1. Any SWE-bench eval (**Verified / Lite / full**) of Qwen3.5-4B (base **or** instruct) used as an *untrained
   baseline*: result JSON/reports, run logs, eval output dirs, wandb runs, launch/eval scripts, notebooks.
2. For each baseline eval found, record precisely:
   - **Which model** — `Qwen3.5-4B-Base` vs `Qwen3.5-4B` (instruct). **This is the crux.**
   - **Exact resolved count + denominator** — is "25/500" real? is "52/500" real?
   - **Harness / scaffold** — OpenHands (or other) version, `max_iterations`, temperature, single-sample vs
     pass@k, max input/output tokens, the tool/patch parser.
   - **Which SWE-bench split** — Verified (500) vs Lite vs full.
3. Provenance of **"paper-nonweb ≈ 52/500 (10.4%)"** — is it from the ADP paper (which table/setting), or an
   in-house Babel run?

## Hypotheses to distinguish (report which one holds, with evidence)
- **H1 — model mismatch (leading):** 5% was `Qwen3.5-4B-Base` (non-instruct); the instruct θ₀ was never compared.
- **H2 — harness mismatch:** 5% used a weaker/different scaffold, sampling, or split than the arms' eval → not
  comparable to the arms' or the rerun's numbers.
- **H3 — paper number:** 5% ≈ a published baseline, never run in-house.
- **H4 — real & comparable:** 5% was `Qwen3.5-4B` *instruct* on the same harness/split → then the rerun's ≥70/500
  is the anomaly and there is a genuine harness discrepancy to reconcile (diff the two configs).

## What to report back to dpf
For each baseline number (the 5% and the 10.4%): **model · resolved/denominator · harness/scaffold · split ·
file path**, and a one-line verdict on which hypothesis (H1–H4) holds. That tells us whether the campaign's
"SFT-lift" was a model/harness-mismatch artifact or a real (now-inverted) comparison.

---

## Reference: the A100 rerun numbers (for comparison — you do NOT need any FAIR paths)
- **SFT arms, single-run pass@1, SWE-bench Verified 500:** swezero 77, rebench 70, coderforge 48, scale 35.
- **θ₀ = Qwen3.5-4B *instruct*, single-run pass@1:** **≥70/500 certain**, bound [70,145] (clean number pending).
  Scaffold byte-identical to the arms (temp 0, `max_iterations` 500, max_input 28000, max_output 2047, same
  condenser + parser). So the rerun's arms-vs-θ₀ comparison is apples-to-apples *within the rerun*.
- **Oracle union of the 4 arms = 131/500.**
- **Caveat:** the SWE-bench `sphinx-doc/sphinx` repo (44 instances) is mis-scored 0/44 by the official pytest
  parser (sphinx's `tox` output has no per-test `PASSED` markers, so genuine passes are recorded as failures) —
  when comparing absolute rates, drop sphinx and use the 456-instance board.

_Created 2026-07-25 by the FAIR-side main-worker session for a Babel-side fact-finding pass._

---

# Findings — Babel-side response (2026-07-26)

## Verdict in one line
**H2 (harness mismatch) is the answer.** H1 is *nominally* true — the anchor really is the Base
model — but it is **not** what explains 5% vs 14%: the **same Base model, re-evaluated in-house
under this campaign's own harness, scores ≥10.6% and likely ~15%**. Babel already had that
disconfirming run on disk; it was never written up. **H3 and H4 are false.**

So: the ≥70/500 rerun is **not** the anomaly. The 5% anchor is a *harness* artifact of Graham's
June 2026 run and should not anchor anything.

## Provenance table

| Anchor | Model | Resolved / denom | Harness | Split | Path |
|---|---|---:|---|---|---|
| **25/500 = 5.0%** | `Qwen/Qwen3.5-4B-Base` (weights) served with the **FT ckpt-2000 tokenizer/chat template** | 25/500 all-patch; 25/96 of non-empty | Graham, June 2026: benchmarks PRs #745/#743/#751 + SDK #3641, Apptainer, TP2 2×L40S, `NUM_WORKERS=4`, temp 0, 28000/2047, condenser 28k, `qwen3_coder` native parser, Epoch GHCR images | Verified `test`, 500 | `swe-bench-full-4b/README.md:240` (raw-count table); source dirs `/home/gneubig/exp/adp/evals/full-swebench/q35_base_swe_epoch_tp2_…_r4/apptainer_patch_eval_rerun_20260613_080646/`; Slurm 8331642 infer, 8331644 + 8331751 score |
| **52/500 = 10.4%** | `papernonweb1154` — **Base-init SFT ckpt-1154** on ADP-paper-style `openhands_nonweb` (~40k). Paper-*style data*; in-house *model and eval* | **52 / 500 unique** (exact) | This campaign's harness (identical `llm_config` to the arms) | Verified `test`, 500 | `~/exp/adp-smoke/swebench/full/score_papernonweb1154/` |

**Both numbers are real in-house Babel runs. Neither is a published/paper number → H3 is false for
both.** The 10.4% is *not* the ADP paper's own baseline; it is an eval of a checkpoint trained here
on paper-style data. (`merged.report.json` reports `total: 515` from duplicate shard rows;
deduplicating by instance gives a clean 500 unique / 52 resolved, matching `RESULTS.md:23`.)

Caveat on the 5%: I could **not** re-verify it at source — `/home/gneubig` is permission-denied for
this account. It rests on Graham's checked-in runbook, which does include full raw counts, Slurm job
IDs, and a `0 scoring errors` column, so it is well-evidenced but second-hand.

## The disconfirming run that was already on disk

`RESULTS.md:22` lists `base4b` as *"running — Graham measured 25/500 (5.0%) on his infra"*. That run
**was** in fact scored, and it re-measures the **same base model** under this campaign's harness:

| Run | Model | Resolved | Non-empty patch rate |
|---|---|---:|---:|
| Graham, June harness | Qwen3.5-4B-**Base** | 25/500 = **5.0%** | 96/500 = **19.2%** |
| In-house `base4b` | Qwen3.5-4B-**Base** | 53 on 362 scored → **≥10.6%** of 500, **14.6%** of scored | 281/404 rows = **69.6%** |
| In-house `rawinstruct4b` | Qwen3.5-4B **instruct** | 13 on 47 scored — *uninformative, see caveats* | 41/53 = 77.4% |

Provenance verified from the scored shard metadata, not just the dir name —
`score_base4b254/shard_0of8.jsonl` carries
`metadata.llm.custom_tokenizer = Qwen/Qwen3.5-4B-Base`, `metadata.llm.model = adp-eval-base4b`,
`metadata.eval_output_dir = …/out_base4b/`. Its `llm_config.json` is byte-comparable to the arms'
(temp 0, `max_input_tokens` 28000, `max_output_tokens` 2047, `native_tool_calling: true`, thinking
off), and the tokenizer path is the `Qwen3.5-4B-Base` HF snapshot — while `out_rawinstruct4b`'s
points at the `Qwen3.5-4B` (instruct) snapshot. The two models are cleanly distinguishable on disk.

**The ≥10.6% figure is robust to every bias below.** It counts all 138 never-scored instances as
unresolved, so it is a hard floor on the full-500 board — already **2.1× Graham's 5.0%**.

## Why H1 is not the explanation

H1 is right that the anchor is literally `Qwen3.5-4B-Base`, and right that this was the wrong θ₀ for
instruct-init arms — Graham's `EXPERIMENT_SUMMARY.md:6-14` is explicit that **every** one of his runs
started from a base checkpoint. So the campaign did compare instruct-init arms against a
*base-model* baseline, and that is a real methodological defect worth fixing.

But base-vs-instruct **cannot** account for 5% → 14%, because the base model *itself* clears 10.6%
here. The gap is in the **harness**, and specifically in **patch production**: 69.6% vs 19.2%
non-empty patches on the same weights. Graham's dominant failure mode was 404/500 *empty* patches
(his own note: "81% of base outputs produced no patch at all"), while non-empty patches applied
essentially perfectly in both setups. The bottleneck was never diff quality — it was the agent not
emitting a patch at all.

Leading mechanism, stated as a hypothesis rather than a proven cause: `RESULTS.md:70-72` records this
campaign's fix for `workspace.execute_command`'s **30 s default timeout** on `cp_testbed_repo`
(raised to 600 s / 300 s), described as "the #1 infra error fleet-wide (70–87 error rows per arm)."
Full-repo copies over loaded NFS blow through 30 s, the agent loses its workspace, and the row lands
as an empty patch. Graham's June run predates that fix. I can demonstrate the 3.6× patch-rate gap;
I cannot attribute all of it to that one fix without his raw error rows, which I can't read.
Note this is a *different* failure from the 2026-07-16/17 tir1 NFS brownout, which post-dates both.

## Caveats that bound these numbers

1. **`base4b`'s 14.6% is survivorship-biased upward.** SWE-bench resume permanently skips errored
   instances (`get_completed_instances` counts an instance done "regardless of success/failure"), so
   the 362-instance board is the subset that ran *cleanly*. Use **≥10.6%** for any claim that has to
   hold; treat 14.6% as an optimistic point estimate. The 362 board is at least broadly
   representative — all 12 repos present at ~72% of their full-500 counts (django 158/231,
   sympy 58/75, sphinx 30/44), with matplotlib the most depleted (17/34).
2. **`rawinstruct4b` was never measured — do not use it.** 13 resolved on 47 scored. Both readings
   are wrong: 2.6% charges infra failures to the model, 27.7% is survivorship bias on n=47 (other
   arms resolve that same subset ~8–10 pp above their own overall rate). Its one real signal is 70
   `MaxIterations` rows, 64 of them django. **Babel has no valid untrained-instruct anchor**, which
   is exactly the gap the rerun's θ₀ fills.
3. **sphinx.** Confirmed on this side: `base4b` scored **0/30** sphinx instances, consistent with the
   brief's parser bug. Excluding sphinx, `base4b` is 53/332 = **16.0%** of scored.
4. **Existing `score_*` dirs are stale** as of 2026-07-26 — 862 retryable error rows were purged
   from attempt_1 and 1604 instances are re-running, so every arm needs re-scoring. The numbers here
   are as-of the 2026-07-15 scoring snapshot.

## Reconciliation with the A100 rerun

θ₀ = instruct at **≥70/500 (≥14%)** is **consistent with** Babel's own evidence, not in conflict
with it: base clears ≥10.6% here, and instruct sits above base on every partial signal available.
There is no harness discrepancy left to reconcile — the rerun and this campaign's harness agree; it
is Graham's June harness that is the outlier.

**Consequence for the headline, unchanged by any of the above:** the "SFT lift" does invert. Arms at
77/70/48/35 are being compared against a θ₀ worth ≥53/500 and plausibly ~73/500, so two arms are
below baseline. But the inversion is a **harness artifact plus a base/instruct mix-up**, not a
fabricated or dishonest number — every individual figure in the chain is a real measurement of
*something*; they were just never comparable.

## Bonus finding

`swesmithinstruct540` has a **final** score on disk that never made it into the docs:
**82/500 = 16.4%** (`RESULTS.md:25` still shows the partial "20/96 (~21%)"). Treat with caveat 1 —
it lost 61 instances to infra vs `swesmith540`'s 9, so the init ablation brackets rather than
resolves: raw rate +1.6 pp for instruct-init, conditional rate +3.6 pp.

## Recommended doc fixes (not made — this was scoped fact-finding only)

Four places still carry the dead 25/500 anchor:
- `v2_arms_a100_rerun/README.md:41` — "Babel reference points: untrained base = 25/500"
- `analysis/adp_v2_data_analysis_report.md:12`, `:18`, `:154`
- `swe-bench-babel-evals/RESULTS.md:22` — replace *"running"* with 53 on 362 scored (≥10.6%)

Note `adp_v2_data_analysis_report.md:154-155` shows the campaign *knew* it needed a matched instruct
anchor and launched one ("a raw-instruct baseline eval is in flight"); it died at 47 scored
instances and the stale 25/500 was never revisited.

## Reproducing

```bash
# resolved counts, deduplicated by instance (merged.report.json over-counts duplicate shard rows)
cd ~/exp/adp-smoke/swebench/full
python3 - <<'PY'
import json, glob, os
for d in ['score_base4b254','score_rawinstruct4b53','score_papernonweb1154',
          'score_swesmith540','score_swesmithinstruct540']:
    per = {}
    for f in sorted(glob.glob(d + '/reports_*of8/*/report.json')):
        r = json.load(open(f))
        for k, v in r.items():
            if isinstance(v, dict) and 'resolved' in v:
                per[k] = bool(v['resolved']); break
        else:
            per[os.path.basename(os.path.dirname(f))] = bool(r.get('resolved', False))
    print(d, 'unique:', len(per), 'resolved:', sum(per.values()))
PY
```
This reproduces the two independently documented figures exactly — papernonweb **52/500** and
swesmith **74/492**, both matching `RESULTS.md` — which is what validates the method for the two
baseline runs that were never written up.

_Investigated 2026-07-26 on Babel (login2) by a Babel-side agent session. `/data/tir` and
`/home/gneubig` are not readable from a login node, so all evidence above comes from
`~/exp/adp-smoke/swebench/full/` plus the checked-in runbooks._
