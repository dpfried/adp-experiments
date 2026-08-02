#!/usr/bin/env python3
"""Generate run dirs (pretok.yaml + train.yaml + submit.sbatch) for ADP-v3 curated arms.

Fork of ../v2_arms_a100_rerun/generate_arm_runs.py. The TRAINING RECIPE IS UNCHANGED
from v2 on purpose -- v3 arms are compared against the already-trained v2 arms, so the
model, template, cutoff, LR, schedule, seed, global batch, ZeRO stage and patches must
all stay byte-for-byte equivalent. The only thing v3 changes is the DATA.

Differences from the v2 generator (all mechanical):
  * Arms are declared on the CLI (--arm NAME=DIR) instead of a hardcoded 6-entry dict,
    because v3 arms are built ad hoc by build_curated_subset.py.
  * `--max-samples` is now an actual registered flag. The v2 generator interpolates
    `args.max_samples` into the pretok template but never registered the argument, so
    running it raises AttributeError; the flag is also how you keep pretokenize from
    silently truncating to a default N. Omitting it now emits no max_samples line
    (= use the whole file) rather than crashing.
  * The sbatch sources `$ENV_ROOT/env.sh` itself when present. DeepSpeed needs
    CUDA_HOME at runtime and FAIR compute nodes have no system nvcc; the v2 sbatch
    relied on CUDA_HOME being exported by the *submitting* shell and inherited via
    sbatch's default --export=ALL, which fails silently-at-startup whenever someone
    submits without sourcing env.sh first. Sourcing in-script makes it order-independent.
  * Run/output dirs and the wandb run name are v3_/adp-v3- prefixed so nothing can
    collide with or overwrite a v2 arm.

Usage:
  python generate_v3_runs.py \
      --env-root  /checkpoint/dpf/adp-env \
      --out-root  /checkpoint/dpf/adp-runs \
      --runs-root /checkpoint/dpf/adp-runs \
      --partition learnfair --gres gpu:8 --constraint ampere80gb \
      --arm a1_traj_swezero=/checkpoint/dpf/adp-data/v3_curated/a1_traj_swezero \
      --eval-set arm_eval=/checkpoint/dpf/adp-data/v2_swe_subsets/nvidia_SWE-Zero-openhands-trajectories/eval.llamafactory.jsonl \
      --max-samples 55000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DI_ENTRY = {
    "formatting": "openai",
    "columns": {"messages": "messages", "tools": "tools"},
    "tags": {
        "role_tag": "role", "content_tag": "content", "user_tag": "user",
        "assistant_tag": "assistant", "observation_tag": "tool",
        "function_tag": "function_call", "system_tag": "system",
    },
}

PRETOK_YAML = """\
model_name_or_path: Qwen/Qwen3.5-4B
trust_remote_code: true
stage: sft
do_train: true
finetuning_type: full
dataset: {ds_train}
eval_dataset: {ds_eval}
eval_on_each_dataset: true
dataset_dir: {data_dir}
template: qwen3_5_nothink
cutoff_len: 32768
{max_samples_line}overwrite_cache: true
preprocessing_num_workers: 8
tokenized_path: {tok_path}
output_dir: {out_dir}
per_device_train_batch_size: {per_device_bs}
"""

TRAIN_YAML = """\
### model
model_name_or_path: Qwen/Qwen3.5-4B
trust_remote_code: true
enable_liger_kernel: true
flash_attn: fa2

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: {env_root}/LLaMA-Factory/examples/deepspeed/ds_z{zero_stage}_config.json

### dataset (loads the pre-built tokenized cache from phase 1; eval sets are baked
### into the cache at pretokenize time -- changing --eval-set changes tokenized_path
### and forces a rebuild)
dataset: {ds_train}
eval_dataset: {ds_eval}
eval_on_each_dataset: true
dataset_dir: {data_dir}
template: qwen3_5_nothink
cutoff_len: 32768
overwrite_cache: false
preprocessing_num_workers: 8
dataloader_num_workers: 2
tokenized_path: {tok_path}

### output -- FULL-STATE checkpoints (LR-integrity fix; ~70GB each, limit 2)
output_dir: {out_dir}
logging_steps: 5
save_steps: {save_steps}
save_total_limit: 2
save_only_model: false
plot_loss: true
overwrite_output_dir: false
report_to: wandb
run_name: adp-v3-{arm}-4b-a100{run_suffix}

### train -- DO NOT CHANGE (must match the v2 arms exactly; global batch 32)
seed: 42
per_device_train_batch_size: {per_device_bs}
gradient_accumulation_steps: {grad_accum}
learning_rate: 1.0e-5
num_train_epochs: 1
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
fp16: false
gradient_checkpointing: true
ddp_timeout: 180000000
resume_from_checkpoint: null
{smoke_line}
### eval (in-training val loss, one eval_<name>_loss metric per eval set; safe at 32k
### ONLY because the sbatch applies patch_llamafactory_liger_eval_skip_logits.py.
### prediction_loss_only: true and per_device_eval_batch_size: 1 are REQUIRED
### pairings -- do not remove either.)
eval_strategy: steps
eval_steps: {eval_steps}
per_device_eval_batch_size: 1
prediction_loss_only: true
"""

SBATCH = """\
#!/usr/bin/env bash
#SBATCH --job-name=adp-v3-{arm}-4b-a100{run_suffix}
#SBATCH --partition={partition}
{account_line}#SBATCH --gres={gres}
{constraint_line}#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time={time}
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output={run}/logs/%x-%j.out
#SBATCH --error={run}/logs/%x-%j.err

set -uo pipefail
RUN={run}
PRETOK_CFG=$RUN/pretok.yaml
TRAIN_CFG=$RUN/train.yaml
mkdir -p "$RUN/logs"

# DeepSpeed needs CUDA_HOME at runtime and FAIR compute nodes have no system nvcc.
# Source it HERE rather than relying on the submitting shell's exported environment
# (sbatch --export=ALL): submitting without sourcing env.sh first is a recurring
# startup failure that costs a full resubmit.
if [ -f "{env_root}/env.sh" ]; then
  # shellcheck disable=SC1091
  . "{env_root}/env.sh"
fi
: "${{CUDA_HOME:=/public/apps/cuda/12.4}}"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:{env_root}/.venv/bin:$PATH"
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  echo "FATAL: no nvcc at $CUDA_HOME/bin/nvcc -- DeepSpeed will fail to build ops."
  exit 1
fi

export HF_HOME={env_root}/hf_cache
export HF_HUB_OFFLINE=1
export WANDB_PROJECT={wandb_project}
export WANDB_DIR=$RUN
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

echo "== node=$(hostname) job=${{SLURM_JOB_ID:-manual}} arm={arm} started=$(date -Is) =="
echo "== CUDA_HOME=$CUDA_HOME  data={data_dir} =="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

# fla/causal-conv1d mandatory (Qwen3.5 linear-attn); re-apply FA2 s_aux patch (idempotent)
python -c "import fla, causal_conv1d, flash_attn" \\
  && python {v2_kit_dir}/patch_transformers_fa2_s_aux.py \\
  || {{ echo "FATAL: fla/causal_conv1d/flash_attn missing or patch failed"; exit 1; }}
# in-training eval at 32k OOMs without Liger skip_logits on loss-only eval steps
python {v2_kit_dir}/patch_llamafactory_liger_eval_skip_logits.py \\
  || {{ echo "FATAL: liger eval skip_logits patch failed"; exit 1; }}

OUT={out_dir}
mkdir -p "$OUT" || {{ echo "FATAL: bulk storage unavailable at $OUT"; exit 1; }}
echo "== output_dir=$OUT =="

# --- resume picker (LR-integrity hardened) ---
# highest-step checkpoint that has trainer_state.json (skips partial mid-save kills);
# HARD-FAIL if it lacks optimizer state (model-only) -- resuming would silently
# restart warmup+cosine from the resume step.
# Parse the numeric suffix directly (NOT `sort -t- -kN`): any '-' in $OUT breaks
# field-based sorting and picks the wrong checkpoint.
CKPT=""; BEST=-1
for c in "$OUT"/checkpoint-*; do
  [ -d "$c" ] && [ -f "$c/trainer_state.json" ] || continue
  n=${{c##*checkpoint-}}
  case $n in ''|*[!0-9]*) continue;; esac
  [ "$n" -gt "$BEST" ] && {{ BEST=$n; CKPT=$c; }}
done
RESUME_ARG=""
if [ -n "$CKPT" ]; then
  if ! ls "$CKPT"/global_step* >/dev/null 2>&1; then
    echo "FATAL: $CKPT has no optimizer state (model-only). Refusing to resume with a"
    echo "fresh LR schedule. Quarantine the checkpoint or clear $OUT to restart cleanly."
    exit 1
  fi
  RESUME_ARG="resume_from_checkpoint=$CKPT"; echo "== resuming (full state) from $CKPT =="
fi

echo "== Phase 1: pre-tokenize (single rank; multi-rank tokenization deadlocks) =="
torchrun --nnodes 1 --nproc_per_node 1 --master_port $((29500 + ${{SLURM_JOB_ID:-0}} % 400)) \\
  {v2_kit_dir}/pretokenize.py "$PRETOK_CFG"
RC=$?; if [ $RC -ne 0 ]; then echo "pretokenize FAILED rc=$RC"; exit $RC; fi

echo "== Phase 2: training on {nproc} GPUs =="
export FORCE_TORCHRUN=1
export NPROC_PER_NODE={nproc}
export NCCL_DEBUG=WARN
export TRITON_CACHE_DIR=${{TMPDIR:-/tmp}}/$USER-triton-${{SLURM_JOB_ID:-manual}}
mkdir -p "$TRITON_CACHE_DIR"
export MASTER_PORT=$((30000 + ${{SLURM_JOB_ID:-0}} % 20000))

( while true; do date -Is; nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits; sleep 30; done ) \\
  > "$RUN/logs/gpu_monitor_${{SLURM_JOB_ID:-manual}}.log" 2>&1 &
MON=$!; trap 'kill $MON 2>/dev/null || true' EXIT

llamafactory-cli train "$TRAIN_CFG" output_dir="$OUT" $RESUME_ARG
RC=$?
echo "== finished=$(date -Is) exit=$RC =="
exit $RC
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env-root", type=Path, required=True,
                    help="dir holding .venv/, LLaMA-Factory/, hf_cache/, env.sh")
    ap.add_argument("--out-root", type=Path, required=True,
                    help="BULK storage root for checkpoints (~150GB headroom per arm)")
    ap.add_argument("--runs-root", type=Path, required=True,
                    help="shared-FS dir for run dirs (configs, logs, wandb)")
    ap.add_argument("--v2-kit-dir", type=Path, default=None,
                    help="dir holding pretokenize.py + the two patch_*.py scripts. "
                         "Default: ../v2_arms_a100_rerun next to this file. The v3 kit "
                         "deliberately reuses them unmodified so the recipe matches.")
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=DIR",
                    help="arm as NAME=DATA_DIR (repeatable). DATA_DIR must contain "
                         "train.llamafactory.jsonl (built by build_curated_subset.py).")
    ap.add_argument("--partition", required=True)
    ap.add_argument("--account", default=None)
    ap.add_argument("--gres", default="gpu:8")
    ap.add_argument("--constraint", default=None,
                    help="sbatch --constraint, e.g. ampere80gb for FAIR A100s")
    ap.add_argument("--time", default="2-00:00:00")
    ap.add_argument("--gpus-per-node", type=int, default=8)
    ap.add_argument("--per-device-bs", type=int, default=1,
                    help="micro-batch/GPU; grad_accum derived to keep global batch 32")
    ap.add_argument("--zero-stage", type=int, default=2, choices=(2, 3))
    ap.add_argument("--wandb-project", default="adp-v3-a100")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="cap rows fed to pretokenize (first-N). Omit to use the whole "
                         "file. Set this to the intended training budget so the arm's "
                         "step count is what you think it is.")
    ap.add_argument("--eval-set", action="append", metavar="NAME=FILE", default=None,
                    help="eval set as NAME=FILE; repeatable. FILE relative to the arm's "
                         "data dir, or absolute (one shared file for all arms -- this is "
                         "the usual v3 case: point at the matched v2 arm's carve-out so "
                         "the val-loss curves are directly comparable). NAME becomes the "
                         "eval_<NAME>_loss metric key, so keep it uniform across arms.")
    ap.add_argument("--smoke", action="store_true",
                    help="30-step smoke variant (separate _smoke run/output dirs)")
    args = ap.parse_args()

    v2_kit_dir = (args.v2_kit_dir
                  or Path(__file__).resolve().parent.parent / "v2_arms_a100_rerun")
    for needed in ("pretokenize.py", "patch_transformers_fa2_s_aux.py",
                   "patch_llamafactory_liger_eval_skip_logits.py"):
        if not (v2_kit_dir / needed).exists():
            raise SystemExit(f"FATAL: {v2_kit_dir / needed} missing "
                             f"(pass --v2-kit-dir)")

    arms: list[tuple[str, Path]] = []
    for spec in args.arm:
        name, sep, d = spec.partition("=")
        if not sep or not name or not d:
            raise SystemExit(f"--arm must be NAME=DIR, got {spec!r}")
        if not name.replace("_", "").isalnum():
            raise SystemExit(f"--arm name {name!r} must be [A-Za-z0-9_]")
        arms.append((name, Path(d)))
    if len({n for n, _ in arms}) != len(arms):
        raise SystemExit("--arm names must be unique")

    eval_sets: list[tuple[str, str]] = []
    for spec in (args.eval_set or ["arm_eval=eval.llamafactory.jsonl"]):
        name, sep, fn = spec.partition("=")
        if not sep or not name or not fn:
            raise SystemExit(f"--eval-set must be NAME=FILE, got {spec!r}")
        if not name.replace("_", "").isalnum():
            raise SystemExit(f"--eval-set name {name!r} must be [A-Za-z0-9_]")
        eval_sets.append((name, fn))
    if len({n for n, _ in eval_sets}) != len(eval_sets):
        raise SystemExit("--eval-set names must be unique")
    ds_eval = ",".join(n for n, _ in eval_sets)
    tok_suffix = "_ev_" + "-".join(n for n, _ in eval_sets)

    micro = args.gpus_per_node * args.per_device_bs
    if 32 % micro:
        raise SystemExit(f"global batch 32 must be divisible by nproc*per_device_bs "
                         f"({args.gpus_per_node}*{args.per_device_bs}={micro})")
    grad_accum = 32 // micro

    run_suffix = "-smoke" if args.smoke else ""
    smoke_line = "max_steps: 30\n" if args.smoke else ""
    save_steps = 25 if args.smoke else 100
    eval_steps = 25 if args.smoke else 50
    max_samples_line = (f"max_samples: {args.max_samples}\n"
                        if args.max_samples is not None else "")

    for arm, data_dir in arms:
        data_dir = data_dir.resolve()
        train_file = data_dir / "train.llamafactory.jsonl"
        if not train_file.exists():
            raise SystemExit(f"[{arm}] missing {train_file} "
                             f"-- build it with build_curated_subset.py first")
        for name, fn in eval_sets:
            f = Path(fn) if Path(fn).is_absolute() else data_dir / fn
            if not f.exists():
                raise SystemExit(f"[{arm}] eval set {name!r}: missing {f}")

        run = args.runs_root / f"v3_{arm}_inst_4b_a100{run_suffix.replace('-', '_')}"
        out_dir = args.out_root / \
            f"v3_{arm}_inst_4b_a100{run_suffix.replace('-', '_')}/output"
        tok_path = data_dir / f"tokenized_qwen35_4b_inst_seq32768{tok_suffix}"
        ds_train = f"v3_{arm}_train"
        (run / "logs").mkdir(parents=True, exist_ok=True)

        # merge (not overwrite) dataset_info.json: absolute eval file_names win over
        # dataset_dir in LLaMA-Factory's os.path.join, so shared eval files need no copy
        di_path = data_dir / "dataset_info.json"
        di = json.loads(di_path.read_text()) if di_path.exists() else {}
        di_missing = {
            name: {**DI_ENTRY, "file_name": fn}
            for name, fn in [(ds_train, "train.llamafactory.jsonl"), *eval_sets]
            if name not in di or di[name].get("file_name") != fn
        }
        if di_missing:
            di.update(di_missing)
            di_path.write_text(json.dumps(di, indent=2) + "\n")

        fmt = dict(env_root=args.env_root, v2_kit_dir=v2_kit_dir, run=run, arm=arm,
                   data_dir=data_dir, tok_path=tok_path, out_dir=out_dir,
                   ds_train=ds_train, ds_eval=ds_eval, partition=args.partition,
                   gres=args.gres, time=args.time, nproc=args.gpus_per_node,
                   grad_accum=grad_accum, per_device_bs=args.per_device_bs,
                   zero_stage=args.zero_stage, wandb_project=args.wandb_project,
                   max_samples_line=max_samples_line, run_suffix=run_suffix,
                   smoke_line=smoke_line, save_steps=save_steps, eval_steps=eval_steps,
                   account_line=(f"#SBATCH --account={args.account}\n"
                                 if args.account else ""),
                   constraint_line=(f"#SBATCH --constraint={args.constraint}\n"
                                    if args.constraint else ""))
        (run / "pretok.yaml").write_text(PRETOK_YAML.format(**fmt))
        (run / "train.yaml").write_text(TRAIN_YAML.format(**fmt))
        (run / "submit.sbatch").write_text(SBATCH.format(**fmt))
        print(f"[{arm}] wrote {run}  (data={data_dir}, out={out_dir}, "
              f"bs={args.per_device_bs} ga={grad_accum} z{args.zero_stage}, "
              f"max_samples={args.max_samples})")
    print("V3_RUNS_GENERATED")


if __name__ == "__main__":
    main()
