# ADP-v3 — data-curation kit

Tooling for the ADP-v3 campaign, which asks the one question the v2 souping campaign
left open: **does quality-filtered ADP data lift over the measured base?**

Background, in one screen (all file-backed, single-run temp-0 pass@1 on SWE-bench
Verified / 500, OpenHands scaffold):

> base θ₀ **119** ≫ swezero 77 > rebench 70 > top2-soup 62 > α0.7 54 ≈ pooled220k 54 >
> uniform4 50 > coderforge 48 ≈ pooled55k 46 > α1.55 38 > scale 35 > α2.0 19

No SFT arm and no combination method (weight-soup or joint-train, any budget) beats the
base. The cause is **data content/objective, not a pipeline bug** — format, chat
template, tool-call parser, loss masking, LR, cutoff and truncation were each checked
and refuted. On the 82 base-solved/arm-failed instances, 82/82 of the arm's patches are
valid and well-formed but **misdiagnose the bug**; the arms are *more* scaffold-compliant
than base (swezero finishes 500/500 vs base's 207, emits 0 empty patches vs 138, uses
~half the actions). SFT **traded depth for compliance**.

Two candidate data defects drive that trade (correlational until this campaign tests
them):

1. **~34–41% of every subset is context-condensation records** — 2-message
   (condenser-prompt → prose state summary) pairs that train *summarize-and-conclude*
   rather than *act*.
2. **No success/outcome filtering** — the records carry no resolved/reward field, so
   unresolved and looping trajectories are imitated wholesale.

This kit builds the counterfactual datasets and the matching training runs.

See `../analysis/adp_v3_data_curation_campaign.md` for the full campaign design
(levers A/B/C, hypotheses, falsifiers, sequencing).

---

## What's here

| file | purpose |
| --- | --- |
| `build_curated_subset.py` | Stream a v2 arm subset → a curated subset (drop condensation records, cap at a record budget), plus a `manifest.json` census. |
| `test_build_curated_subset.py` | Self-contained checks for the above. No pytest needed. Run it before spending GPU-hours. |
| `build_curated.sbatch` | CPU SLURM job wrapper for the builder (the sources are 23–27 GB). |
| `generate_v3_runs.py` | Emit `pretok.yaml` + `train.yaml` + `submit.sbatch` per v3 arm. Fork of `../v2_arms_a100_rerun/generate_arm_runs.py` with the **recipe unchanged**. |

## Design invariants (do not "improve" these)

* **The training recipe is frozen to match v2.** Qwen3.5-4B, `qwen3_5_nothink`,
  `cutoff_len 32768`, LR 1e-5, cosine, warmup 0.03, 1 epoch, **global batch 32**,
  seed 42, ZeRO-2, liger + fa2. v3 arms are read against already-trained v2 arms; any
  recipe drift turns a data result into an uninterpretable joint change.
* **`generate_v3_runs.py` reuses the v2 kit's `pretokenize.py` and both `patch_*.py`
  scripts unmodified** (`--v2-kit-dir`), for the same reason.
* **Records are copied byte-for-byte.** The builder never re-serializes a record, so a
  curated record is identical to the one the v2 arm saw. A `json.loads`/`json.dumps`
  round-trip would reorder keys and rewrite floats, quietly making "the same records,
  minus condensation" false.
* **Selection is first-N in file order.** The v2 sources are well-shuffled (verified in
  the v2 audit), so first-N is a representative random subsample *and* maximizes
  trajectory-record overlap with the matched mixed arm.
* **The eval carve-out is shared with the matched v2 arm** (pass an absolute path to
  `--eval-set`), so in-training val-loss curves are directly comparable.

## Two things that will bite you

* **CUDA_HOME.** FAIR compute nodes have no system `nvcc`, and DeepSpeed needs
  `CUDA_HOME` at runtime. The v2 sbatch relied on the *submitting* shell exporting it
  (via sbatch's default `--export=ALL`), so submitting without sourcing `env.sh` first
  failed at startup and cost a full resubmit. **The v3 sbatch sources `$ENV_ROOT/env.sh`
  itself** and hard-fails early if `nvcc` is still missing. Source it anyway out of habit.
* **`sbatch --wrap "source ... && ..."` runs under dash**, where `source` does not
  exist → exit 127. This silently ate a merge step in the v2 eval kit. Use a real
  sbatch file (as here), or `bash -lc "..."`.

Also note: the v2 generator (`../v2_arms_a100_rerun/generate_arm_runs.py`) interpolates
`args.max_samples` at line ~341 but never registers the flag, so it raises
`AttributeError` if you run it today. `generate_v3_runs.py` registers `--max-samples`
properly; the v2 script is left untouched so the v2 runs stay reproducible as-run.

---

## Usage

### 0. Test the builder

```bash
python test_build_curated_subset.py       # expect: ALL CHECKS PASSED
```

### 1. Census a source first (cheap, writes no data)

```bash
sbatch --partition=learnfair \
  --output=$SCRATCH/logs/census-%j.out --error=$SCRATCH/logs/census-%j.err \
  --export=ALL,SRC=$V2/nvidia_SWE-Zero-openhands-trajectories/train.llamafactory.jsonl,\
OUT_DIR=$V3/census_swezero,EXTRA_ARGS=--census-only \
  build_curated.sbatch
```

Read `manifest.json` → `census.generation` / `census.record_type`. This tells you the
exact pure-trajectory pool size, i.e. whether the intended record budget is even
reachable from one source.

### 2. Build the curated subset

```bash
sbatch --partition=learnfair \
  --output=$SCRATCH/logs/build-%j.out --error=$SCRATCH/logs/build-%j.err \
  --export=ALL,SRC=$V2/nvidia_SWE-Zero-openhands-trajectories/train.llamafactory.jsonl,\
OUT_DIR=$V3/a1_traj_swezero,MAX_RECORDS=55000 \
  build_curated.sbatch
```

If the budget can't be met the builder **warns and does not pad** — read the warning,
then either lower the budget on *both* arms or accept and report a compute mismatch.

### 3. Generate the run dir

```bash
python generate_v3_runs.py \
  --env-root  /checkpoint/dpf/adp-env \
  --out-root  /checkpoint/dpf/adp-runs \
  --runs-root /checkpoint/dpf/adp-runs \
  --partition learnfair --gres gpu:8 --constraint ampere80gb \
  --arm a1_traj_swezero=$V3/a1_traj_swezero \
  --eval-set arm_eval=$V2/nvidia_SWE-Zero-openhands-trajectories/eval.llamafactory.jsonl \
  --max-samples 55000
```

### 4. Smoke, then launch

Always smoke first — it exercises pretokenize, the FA2/liger patches, and one eval step
(the eval path OOMs at 32k without the liger `skip_logits` patch):

```bash
python generate_v3_runs.py ... --smoke
sbatch /checkpoint/dpf/adp-runs/v3_<arm>_inst_4b_a100_smoke/submit.sbatch
# then, once the smoke run is green:
sbatch /checkpoint/dpf/adp-runs/v3_<arm>_inst_4b_a100/submit.sbatch
```

A full 55k arm is ~14 h on 8×A100 (the v2 arms ran 49,432 s / 1719 steps).

### 5. Eval

Use the shared v2 eval kit at `/checkpoint/dpf/swebench-eval` (10×50 shards → aggregate
→ score → merge). Announce on the coordination channel before queueing shards. Report
**all three boards** (full-500 / 456-ex-sphinx / clean-301) plus McNemar vs base(119)
and vs the matched raw arm, and the **depth diagnostics** (actions/instance,
finish rate, empty-patch rate) — those test the mechanism, not just the score.
