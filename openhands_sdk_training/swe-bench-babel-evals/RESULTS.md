# ADP 4B training + SWE-bench Verified eval campaign (Babel)

Status as of **2026-07-27** — all five arms have final numbers (scoring snapshot
2026-07-15). Follow-up to Graham's experiments (`../EXPERIMENT_SUMMARY.md`), which found
his condenser-data fine-tune *regressed* vs raw base (14 vs 25 resolved / 500, with the
caveat of a ~38%-of-epoch checkpoint). This campaign trains Qwen3.5-4B to completion on
different data recipes and evaluates all arms under Graham's exact protocol for
comparability.

⚠️ **Read ["The 25/500 baseline anchor"](#the-25500-baseline-anchor-why-it-is-not-usable)
below before quoting any lift over base.** An earlier version of this file concluded the
fine-tunes "beat base by 2–3×" against Graham's 25/500 anchor. That anchor turns out not
to be comparable to anything measured here, and the conclusion is retracted.

## Protocol (identical across arms, = Graham's)

default.j2 prompt, max-iterations 500, temp 0, `max_input_tokens` 28000 /
`max_output_tokens` 2047, thinking off, condenser on (28k threshold), vLLM
`qwen3_coder` native tool parsing, TP2 on 2×L40S, apptainer workspaces,
patches from errored rollouts captured and scored (PR #751). Training: LLaMA-Factory
full-parameter SFT, seq 32768, 4×L40S + FA2 + Liger, lr 1e-5, 1 epoch,
in-training eval every 50 steps. Scripts in `scripts/`, configs in `configs/`.

## Results (SWE-bench Verified, 500 instances, temp 0, single greedy run)

Resolved counts are **deduplicated by instance** — `merged.report.json` over-counts
duplicate shard rows (it reports 515 for papernonweb's 500). Reproduce with the snippet at
the bottom of this section. Denominators differ by arm: SWE-bench resume permanently skips
instances that errored (`get_completed_instances` marks an instance done regardless of
success), so an arm's *scored* board is the subset that ran cleanly, and "% of 500" charges
every unscored instance as unresolved. Full per-arm error-rate breakdown: [`INFRA_ERROR_ANALYSIS.md`](INFRA_ERROR_ANALYSIS.md).

| Arm | Init | Training data | Resolved / scored | % of 500 | Notes |
|---|---|---|---:|---:|---|
| base4b | Qwen3.5-4B-Base (raw) | — | **53 / 362** | **≥10.6%** | 138 never scored → the % is a hard floor; 14.6% *of scored* is biased up. 69.6% non-empty patches. **Not** Graham's 25/500 — see below |
| papernonweb1154 | Base | ADP-paper openhands_nonweb (~40k) | **52 / 500** | **10.4%** | complete board; 69% non-empty patches; train_loss 0.247, eval_loss 0.281 |
| swesmith540 | Base | SWE-smith condenser SFT | **74 / 492** | **14.8%** | 15.0% of scored; 93% non-empty patches |
| swesmithinstruct540 | **Instruct** | SWE-smith condenser SFT (same as above) | **82 / 500** | **16.4%** | complete board; init ablation vs swesmith540; eval_loss 0.126 vs 0.131 |
| rawinstruct4b | Qwen3.5-4B (instruct, raw) | — | *13 / 47* | **unusable** | arm died at 47 scored. 2.6% charges infra failures to the model; 27.7% is survivorship bias at n=47. **Babel has no valid untrained-instruct anchor** |

**Ordering among the fine-tunes** (unchanged, and the part that is well-supported):
instruct-init + SWE-smith (16.4%) > base-init + SWE-smith (14.8%) > base-init +
paper-nonweb (10.4%). Training data is the larger effect — SWE-smith beats paper-nonweb by
**4.4 pp** at identical init and hyperparameters, outside the ≈1.9 pp σ on a between-arm
difference (σ≈1.3 pp per arm at p≈0.1, n=500). The init effect is
**+1.6 pp** raw, inside noise, and the two arms' boards are not like-for-like (492 vs 500
scored, with instruct carrying more infra-degraded error rows in its denominator);
conditioning on cleanly-run instances puts it at +3.6 pp. The ablation brackets the init
effect rather than resolving it.

**Ordering against base is not established.** Against base4b's hard floor (≥10.6%) the
SWE-smith arms are +4.2/+5.8 pp; against its optimistic point estimate (14.6% of scored)
they are indistinguishable. **papernonweb1154 at 10.4% sits at or below base on both
readings — this campaign shows no measurable lift from the paper-nonweb recipe.** Note
also that base4b scores 0/30 on sphinx (the official pytest parser mis-grades sphinx's
`tox` output); excluding sphinx, base4b is 53/332 = 16.0% of scored.

The claim in the previous revision that the fine-tunes "beat base by 2–3×" rested entirely
on the 25/500 anchor and does not survive re-measurement.

```bash
# resolved counts, deduplicated by instance
cd ~/exp/adp-smoke/swebench/full
python3 - <<'PY'
import json, glob
for d in ['score_base4b254','score_rawinstruct4b53','score_papernonweb1154',
          'score_swesmith540','score_swesmithinstruct540']:
    per = {}
    for f in sorted(glob.glob(d + '/reports_*of8/*/report.json')):
        for k, v in json.load(open(f)).items():
            if isinstance(v, dict) and 'resolved' in v:
                per[k] = bool(v['resolved'])
    print(d, 'unique:', len(per), 'resolved:', sum(per.values()))
PY
```

### The 25/500 baseline anchor: why it is not usable

The number **"untrained base Qwen3.5-4B = 25/500 (5.0%)"** appears throughout this repo and
seeded the campaign's headline SFT lift. It is **a real in-house measurement, but it is not
comparable to any arm in the table above, and it should not be quoted as a baseline.**

**What it actually is.** Graham's June 2026 run on **Babel**: `Qwen/Qwen3.5-4B-Base` weights
served with the *fine-tuned* ckpt-2000 tokenizer/chat template, SWE-bench Verified `test`
(500), his harness (benchmarks PRs #745/#743/#751 + SDK #3641, Apptainer, TP2 2×L40S,
`NUM_WORKERS=4`, temp 0, 28000/2047, condenser 28k, `qwen3_coder` parser, Epoch GHCR
images). Raw counts in `../swe-bench-full-4b/README.md:240`; Slurm 8331642 (infer),
8331644 + 8331751 (score). Not a published/paper number.

**Why it is wrong as a baseline.** The *same base weights* re-evaluated under *this
campaign's* harness score **53 / 362 = ≥10.6% of the 500 board** — 2.1× Graham's figure.
The gap is harness, not model: **non-empty patch rate 69.6% vs 19.2%** on identical
weights. Graham's dominant failure mode was 404/500 *empty* patches ("81% of base outputs
produced no patch at all"); non-empty patches applied essentially perfectly in both setups,
so the bottleneck was never diff quality — the agent simply never emitted a patch. Leading
mechanism (hypothesis, not proven): the 30 s `workspace.execute_command` default timeout on
`cp_testbed_repo`, fix #3 under *Infra fixes* below, raised to 600 s here and recorded as
"the #1 infra error fleet-wide (70–87 error rows per arm)". Full-repo copies over loaded
NFS blow past 30 s, the agent loses its workspace, and the row lands as an empty patch.
Graham's run predates that fix. This cannot be fully attributed without his raw error rows,
which are unreadable from this account (`/home/gneubig` is permission-denied).

**The corrected number and its provenance.** **53 resolved / 362 scored — a floor of
≥10.6% on the full 500 board, 14.6% of the scored subset, 16.0% excluding sphinx.**

- **Cluster: Babel** — the *same* cluster as the 25/500. The two numbers differ by harness
  and date, **not** by machine. This is an in-house re-measurement, not a paper number.
- **Model verified from shard metadata, not the directory name:**
  `score_base4b254/shard_0of8.jsonl` carries `metadata.llm.custom_tokenizer =
  Qwen/Qwen3.5-4B-Base` and `metadata.eval_output_dir = …/out_base4b/`; its
  `llm_config.json` is byte-comparable to the arms'. (`out_rawinstruct4b`'s tokenizer path
  points at the *instruct* snapshot — the two models are cleanly distinguishable on disk.)
- **Path:** `~/exp/adp-smoke/swebench/full/score_base4b254/`, scoring snapshot 2026-07-15.
- **Use ≥10.6% for any claim that has to hold.** 14.6% of scored is survivorship-biased
  upward (the 362-instance board is the cleanly-run subset), though broadly representative
  — all 12 repos present at ~72% of their full-500 counts.

**On the 10.4% too:** `papernonweb1154` is likewise an **in-house Babel eval** — of a
checkpoint trained *here* on ADP-paper-*style* `openhands_nonweb` data. It is not a number
reported by the ADP paper.

**The right θ₀ for instruct-init arms is not on Babel.** Graham's runs all started from a
base checkpoint, so instruct-init arms were being compared against a *base-model* baseline.
A matched untrained-**instruct** θ₀ was measured only on the **separate A100 cluster** (the
v2 A100 rerun, scaffold byte-identical to its arms): **≥70/500 (≥14%)**. `rawinstruct4b`
was this campaign's attempt at the same anchor on Babel and it died at 47 scored instances.

Full investigation, including the hypotheses ruled out: `../BABEL_baseline_provenance_investigation.md`
(PR #5). Three other files still carry the dead anchor: `../v2_arms_a100_rerun/README.md:41`
and `../analysis/adp_v2_data_analysis_report.md:12,:18,:154`.

**Staleness note.** These are the 2026-07-15 scoring snapshot. On 2026-07-26, 862 retryable
error rows were purged and ~1604 instances were re-queued to fix the infra-error bias
(PR #4); every arm needs re-scoring afterwards, which should raise the low-denominator arms
(`base4b`, `rawinstruct4b`) most.

## Prefix caching + temperature A/B (20 fixed instances, `scripts/run_smoke_ab.sbatch`)

Qwen3.5's GDN hybrid architecture disables vLLM automatic prefix caching by default,
so every agent turn re-prefills ~28k tokens (0% hit rate; prompt throughput dominates
GPU time ~100:1 vs generation). vLLM ≥0.24 supports
`--enable-prefix-caching --mamba-cache-mode align` (PR #30877, 2026-01, experimental).

| Config | Resolved /20 | Non-empty patches |
|---|---:|---:|
| full-run baseline rows (no cache, temp 0) | 3 | 13 |
| cache, temp 0 | 1 | 12 |
| cache, temp 1.0 / top_p 0.95 | 2 | 9 |
| no-cache control, temp 0 (same day) | 1 | 13 |

- Cache verdict: **exonerated** — the no-cache control matches the cached run (1/20,
  resolving an instance no other config resolved). Same-config variance spans the whole
  observed range; greedy agent rollouts on this 4B are extremely high-variance.
  Cache gives **13.6× less prefill compute per instance** (89–93% hit rate).
  Recommended ON for future runs; scored arms so far all ran uncached.
- Temperature: temp 1.0/top_p 0.95 (public Qwen3.6/Coder-Next SWE-bench setting)
  showed no benefit for this small fine-tune (fewer patches). Stay at temp 0.
- Methodology: at p≈0.1, n=500, binomial σ≈1.3pp — treat ±1–2pp between-arm gaps
  in single greedy runs as noise.

## Infra fixes made along the way (patched in the benchmarks checkout)

Local commits in `~/exp/adp-smoke/swebench/benchmarks` (branch `babel-scoring-fixes`;
candidates for upstream PRs to OpenHands/benchmarks):

1. `apptainer_eval.py`: subprocess decoding `errors="backslashreplace"`; sanitize
   `test_output.txt` to valid UTF-8 before grading (django test output contains raw
   bytes → `UnicodeDecodeError` killed 3 of 8 scoring shards); per-instance try/except
   so one bad instance can't kill a shard.
2. `eval_infer.py`: `generate_cost_report` `sys.exit(1)`s on shard files (no
   output.jsonl in dir) and `SystemExit` bypasses `except Exception` — scoring "failed"
   after full success, which also skipped sandbox pruning. Now warn-and-continue.
3. `run_infer.py`: `workspace.execute_command` defaults to a 30s timeout; full-repo
   copies over loaded NFS exceed it. This was the #1 infra error fleet-wide
   (70–87 error rows per arm). Now 600s for `cp_testbed_repo`, 300s for `git reset`.

Operational gotchas (details in the sbatch headers): partial scoring-array resubmits
need an explicit TOTAL_SHARDS arg; erroring instances exhaust 3 in-run attempts and are
NOT retried across chained legs; HF_HOME override relocates the HF token path
(`$HF_HOME/token`) — anonymous requests get 429'd from the cluster's shared IP, so
serve local snapshot paths, never hub ids; sick nodes to exclude are listed in the
sbatch `--exclude` lines.
