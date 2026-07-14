# ADP 4B training + SWE-bench Verified eval campaign (Babel)

Status as of **2026-07-13 late**. Follow-up to Graham's experiments
(`../EXPERIMENT_SUMMARY.md`), which found his condenser-data fine-tune *regressed*
vs raw base (14 vs 25 resolved / 500, with the caveat of a ~38%-of-epoch checkpoint).
This campaign trains Qwen3.5-4B to completion on different data recipes and evaluates
all arms under Graham's exact protocol for comparability.

## Protocol (identical across arms, = Graham's)

default.j2 prompt, max-iterations 500, temp 0, `max_input_tokens` 28000 /
`max_output_tokens` 2047, thinking off, condenser on (28k threshold), vLLM
`qwen3_coder` native tool parsing, TP2 on 2×L40S, apptainer workspaces,
patches from errored rollouts captured and scored (PR #751). Training: LLaMA-Factory
full-parameter SFT, seq 32768, 4×L40S + FA2 + Liger, lr 1e-5, 1 epoch,
in-training eval every 50 steps. Scripts in `scripts/`, configs in `configs/`.

## Results (SWE-bench Verified, 500 instances, temp 0, single greedy run)

| Arm | Init | Training data | Resolved | Notes |
|---|---|---|---:|---|
| base4b | Qwen3.5-4B-Base (raw) | — | *running* | 393/500 attempted; Graham measured 25/500 (5.0%) on his infra |
| papernonweb1154 | Base | ADP-paper openhands_nonweb (~40k) | **52/500 (10.4%)** | final; 69% non-empty patches; train_loss 0.247, eval_loss 0.281 |
| swesmith540 | Base | SWE-smith condenser SFT | **74/491 (15.1%)** | 9 instances still scoring; 93% non-empty patches |
| swesmithinstruct540 | **Instruct** | SWE-smith condenser SFT (same as above) | *scoring; 20/96 (~21%) partial* | init ablation vs swesmith540; eval_loss 0.126 vs 0.131 |
| rawinstruct4b | Qwen3.5-4B (instruct, raw) | — | *running* | decomposes instruct-init gains; native chat template |

Early ordering: **instruct-init + SWE-smith > base-init + SWE-smith > base-init +
paper-nonweb > raw base** — i.e. data quality first, then init, each roughly stacking.
Both fine-tunes clearly reverse Graham's regression finding: his checkpoint was
mid-cosine-decay at 38% of an epoch; trained to completion, the same 4B + agentic SFT
data beats base by 2–3×.

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
