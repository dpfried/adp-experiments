#!/usr/bin/env python3
"""Generate run dirs (pretok.yaml + train.yaml + submit.sbatch) for the 4 adp-v2 arms.

Clean-LR rerun of the Babel 2026-07 campaign (see README.md). Key differences from the
original scripts/generate_v2_arm_configs.py:
  * save_only_model: false from step 0 (full optimizer+scheduler state in checkpoints)
  * sbatch resume picker skips partial checkpoints (no trainer_state.json) and HARD-FAILS
    on model-only checkpoints instead of silently resetting the LR schedule
  * 8 GPUs x bs1 x ga4 = global batch 32 (identical schedule/steps to the 4-GPU x ga8
    Babel runs), explicit seed
  * all cluster-specific values are CLI flags

Usage (see README.md §6):
  python generate_arm_runs.py --env-root ... --data-root ... --out-root ... \
      --runs-root ... --partition ... [--account ...] [--gres gpu:a100:8] [--smoke]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# arm short-name -> adp-v2 config name (= subset dir name under --data-root)
ARMS = {
    "coderforge": "coderforge_preview",
    "scale": "scale_swe_distilled",
    "rebench": "nebius_SWE-rebench-openhands-trajectories",
    "swezero": "nvidia_SWE-Zero-openhands-trajectories",
}

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
dataset_dir: {data_dir}
template: qwen3_5_nothink
cutoff_len: 32768
max_samples: 55000
overwrite_cache: true
preprocessing_num_workers: 8
tokenized_path: {tok_path}
output_dir: {out_dir}
per_device_train_batch_size: 1
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
deepspeed: {env_root}/LLaMA-Factory/examples/deepspeed/ds_z3_config.json

### dataset (loads pre-built tokenized cache from phase 1)
dataset: {ds_train}
dataset_dir: {data_dir}
template: qwen3_5_nothink
cutoff_len: 32768
overwrite_cache: false
preprocessing_num_workers: 8
dataloader_num_workers: 2
tokenized_path: {tok_path}

### output — FULL-STATE checkpoints (LR-integrity fix; ~70GB each, limit 2)
output_dir: {out_dir}
logging_steps: 5
save_steps: {save_steps}
save_total_limit: 2
save_only_model: false
plot_loss: true
overwrite_output_dir: false
report_to: wandb
run_name: adp-v2-{arm}-4b-a100{run_suffix}

### train — DO NOT CHANGE (matched schedule across arms; global batch 32)
seed: 42
per_device_train_batch_size: 1
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
### eval (disabled: Liger 32k eval OOMs; comparison metric is SWE-bench, not eval loss)
eval_strategy: "no"
"""

SBATCH = """\
#!/usr/bin/env bash
#SBATCH --job-name=adp-v2-{arm}-4b-a100{run_suffix}
#SBATCH --partition={partition}
{account_line}#SBATCH --gres={gres}
#SBATCH --nodes=1
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

export PATH="{env_root}/.venv/bin:$PATH"
export HF_HOME={env_root}/hf_cache
export HF_HUB_OFFLINE=1
export WANDB_PROJECT={wandb_project}
export WANDB_DIR=$RUN
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

echo "== node=$(hostname) job=${{SLURM_JOB_ID:-manual}} arm={arm} started=$(date -Is) =="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

# fla/causal-conv1d mandatory (Qwen3.5 linear-attn); re-apply FA2 s_aux patch (idempotent)
python -c "import fla, causal_conv1d, flash_attn" \\
  && python {kit_dir}/patch_transformers_fa2_s_aux.py \\
  || {{ echo "FATAL: fla/causal_conv1d/flash_attn missing or patch failed"; exit 1; }}

OUT={out_dir}
mkdir -p "$OUT" || {{ echo "FATAL: bulk storage unavailable at $OUT"; exit 1; }}
echo "== output_dir=$OUT =="

# --- resume picker (LR-integrity hardened) ---
# newest checkpoint that has trainer_state.json (skips partial mid-save kills);
# HARD-FAIL if it lacks optimizer state (model-only) — resuming would silently
# restart warmup+cosine from the resume step, which is the confound this rerun fixes.
CKPT=""
for c in $(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n -r); do
  [ -f "$c/trainer_state.json" ] && CKPT=$c && break
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
  {kit_dir}/pretokenize.py "$PRETOK_CFG"
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env-root", type=Path, required=True,
                    help="dir holding .venv/, LLaMA-Factory/, hf_cache/ (from setup_env.sh)")
    ap.add_argument("--data-root", type=Path, required=True,
                    help="dir holding the 4 v2_swe_subsets config dirs (README §5)")
    ap.add_argument("--out-root", type=Path, required=True,
                    help="BULK storage root for checkpoints (~150GB headroom per arm)")
    ap.add_argument("--runs-root", type=Path, required=True,
                    help="shared-FS dir for run dirs (configs, logs, wandb)")
    ap.add_argument("--partition", required=True)
    ap.add_argument("--account", default=None, help="sbatch --account, if the cluster needs one")
    ap.add_argument("--gres", default="gpu:8", help="e.g. gpu:8 or gpu:a100:8")
    ap.add_argument("--time", default="2-00:00:00")
    ap.add_argument("--gpus-per-node", type=int, default=8)
    ap.add_argument("--wandb-project", default="adp-v2-a100")
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--smoke", action="store_true",
                    help="30-step smoke variant (separate _smoke run/output dirs)")
    args = ap.parse_args()

    if 32 % args.gpus_per_node:
        raise SystemExit("global batch 32 must be divisible by --gpus-per-node")
    grad_accum = 32 // args.gpus_per_node
    kit_dir = Path(__file__).resolve().parent
    run_suffix = "-smoke" if args.smoke else ""
    smoke_line = "max_steps: 30\n" if args.smoke else ""
    save_steps = 25 if args.smoke else 100

    for arm in args.arms:
        cfg = ARMS[arm]
        data_dir = args.data_root / cfg
        if not (data_dir / "train.llamafactory.jsonl").exists():
            raise SystemExit(f"[{arm}] missing {data_dir}/train.llamafactory.jsonl — see README §5")
        run = args.runs_root / f"v2_{arm}_inst_4b_a100{run_suffix.replace('-', '_')}"
        out_dir = args.out_root / f"v2_{arm}_inst_4b_a100{run_suffix.replace('-', '_')}/output"
        tok_path = data_dir / "tokenized_qwen35_4b_inst_seq32768"
        ds_train = f"{cfg}_train"
        (run / "logs").mkdir(parents=True, exist_ok=True)

        di = {ds_train: {**DI_ENTRY, "file_name": "train.llamafactory.jsonl"}}
        di_path = data_dir / "dataset_info.json"
        if not di_path.exists():
            di_path.write_text(json.dumps(di, indent=2) + "\n")

        fmt = dict(env_root=args.env_root, kit_dir=kit_dir, run=run, arm=arm,
                   data_dir=data_dir, tok_path=tok_path, out_dir=out_dir,
                   ds_train=ds_train, partition=args.partition, gres=args.gres,
                   time=args.time, nproc=args.gpus_per_node, grad_accum=grad_accum,
                   wandb_project=args.wandb_project, run_suffix=run_suffix,
                   smoke_line=smoke_line, save_steps=save_steps,
                   account_line=f"#SBATCH --account={args.account}\n" if args.account else "")
        (run / "pretok.yaml").write_text(PRETOK_YAML.format(**fmt))
        (run / "train.yaml").write_text(TRAIN_YAML.format(**fmt))
        (run / "submit.sbatch").write_text(SBATCH.format(**fmt))
        print(f"[{arm}] wrote {run}  (data={data_dir}, out={out_dir}, ga={grad_accum})")
    print("ARMS_GENERATED")


if __name__ == "__main__":
    main()
