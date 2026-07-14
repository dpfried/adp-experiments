#!/usr/bin/env python3
"""Generate run dirs + pretok/train yaml + sbatch for the 4 adp-v2 SWE training arms.

Templatized from the proven swesmith_4b_4gpu_fa2 recipe. Each arm trains
Qwen3.5-4B-Base full-SFT, seq 32768, 4xL40S, ZeRO-3 + FA2 + Liger, 1 epoch,
openai/tools format (matches the OpenHands SDK eval harness). Data comes from
build_v2_swe_subsets.py (LF jsonl + dataset_info under tir1 v2_swe_subsets/<cfg>).
"""
from __future__ import annotations

import json
from pathlib import Path

HOME = Path("~/exp/adp-smoke").expanduser()
TIR = Path("/data/tir/projects/tir1/users/dfried")
SUBSETS = TIR / "adp-smoke/datasets/v2_swe_subsets"
RUNS = HOME / "runs"

# arm short-name -> adp-v2 config name
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
deepspeed: {home}/.cache/LLaMA-Factory/examples/deepspeed/ds_z3_config.json

### dataset (loads pre-built tokenized cache)
dataset: {ds_train}
dataset_dir: {data_dir}
template: qwen3_5_nothink
cutoff_len: 32768
overwrite_cache: false
preprocessing_num_workers: 8
dataloader_num_workers: 2
tokenized_path: {tok_path}

### output
output_dir: {out_dir}
logging_steps: 5
save_steps: 100
save_total_limit: 3
save_only_model: true
plot_loss: true
overwrite_output_dir: false
report_to: wandb
run_name: adp-v2-{arm}-instruct-4b-seq32768

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-5
num_train_epochs: 1
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
fp16: false
gradient_checkpointing: true
ddp_timeout: 180000000
resume_from_checkpoint: null

### eval (disabled: Liger 32k eval OOMs; comparison metric is SWE-bench, not eval loss)
eval_strategy: "no"
"""

SBATCH = """\
#!/usr/bin/env bash
#SBATCH --job-name=adp-v2-{arm}-inst-4b
#SBATCH --partition={partition}
#SBATCH --gres=gpu:L40S:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
{requeue_line}#SBATCH --exclude=babel-o5-28,babel-n5-32,babel-p5-28,babel-s5-24
#SBATCH --open-mode=append
#SBATCH --output={run}/logs/%x-%j.out
#SBATCH --error={run}/logs/%x-%j.err

set -uo pipefail
RUN={run}
PRETOK_CFG=$RUN/pretok.yaml
TRAIN_CFG=$RUN/train.yaml
mkdir -p "$RUN/logs"

export PATH="{home}/.venv/bin:$PATH"
export HF_HOME={home}/hf_cache
export HF_HUB_OFFLINE=1
export WANDB_PROJECT=adp-smoke
export WANDB_DIR=$RUN
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

echo "== node=$SLURMD_NODENAME job=$SLURM_JOB_ID arm={arm} started=$(date -Is) =="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

python -c "import fla, causal_conv1d, flash_attn" && python {home}/patch_transformers_fa2_s_aux.py || {{ echo "FATAL: fla/causal_conv1d/flash_attn missing"; exit 1; }}

OUT={out_dir}
mkdir -p "$OUT" || {{ echo "FATAL: tir1 unavailable at $OUT"; exit 1; }}
echo "== output_dir=$OUT =="

# Resume from latest checkpoint if present (preempt requeue / manual restart). overwrite_output_dir:false.
CKPT=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
RESUME_ARG=""
if [ -n "$CKPT" ]; then RESUME_ARG="resume_from_checkpoint=$CKPT"; echo "== resuming from $CKPT =="; fi

echo "== Phase 1: pre-tokenize (single rank) =="
torchrun --nnodes 1 --nproc_per_node 1 --master_port $((29500 + SLURM_JOB_ID % 400)) \\
  {home}/pretokenize.py "$PRETOK_CFG"
RC=$?; if [ $RC -ne 0 ]; then echo "pretokenize FAILED rc=$RC"; exit $RC; fi

echo "== Phase 2: training on 4 GPUs =="
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=4
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN
export TRITON_CACHE_DIR=/scratch/dfried/triton-cache-${{SLURM_JOB_ID}}
mkdir -p "$TRITON_CACHE_DIR"
export MASTER_PORT=$((30000 + SLURM_JOB_ID % 20000))

( while true; do date -Is; nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits; sleep 30; done ) \\
  > "$RUN/logs/gpu_monitor_${{SLURM_JOB_ID}}.log" 2>&1 &
MON=$!; trap 'kill $MON 2>/dev/null || true' EXIT

llamafactory-cli train "$TRAIN_CFG" output_dir="$OUT" $RESUME_ARG
RC=$?
echo "== finished=$(date -Is) exit=$RC =="
exit $RC
"""


def main() -> None:
    # Headline verification pair (coderforge=verified vs swezero=unverified) on guaranteed general;
    # scale + rebench on preempt (requeue + resume-from-checkpoint).
    PARTITION = {"coderforge": "general", "swezero": "general", "scale": "preempt", "rebench": "preempt"}
    for arm, cfg in ARMS.items():
        partition = PARTITION[arm]
        requeue_line = "#SBATCH --requeue\n" if partition == "preempt" else ""
        run = RUNS / f"v2_{arm}_inst_4b"
        (run / "logs").mkdir(parents=True, exist_ok=True)
        data_dir = SUBSETS / cfg
        tok_path = data_dir / "tokenized_qwen35_4b_inst_seq32768"
        out_dir = TIR / f"exp/adp-smoke/runs/v2_{arm}_inst_4b/output"
        ds_train = f"{cfg}_train"

        # merged dataset_info.json (train only; eval disabled)
        (data_dir).mkdir(parents=True, exist_ok=True)
        di = dict(DI_ENTRY)
        di_full = {ds_train: {**di, "file_name": "train.llamafactory.jsonl"}}
        (data_dir / "dataset_info.json").write_text(json.dumps(di_full, indent=2) + "\n")

        fmt = dict(home=HOME, run=run, arm=arm, cfg=cfg, data_dir=data_dir,
                   tok_path=tok_path, out_dir=out_dir, ds_train=ds_train,
                   partition=partition, requeue_line=requeue_line)
        (run / "pretok.yaml").write_text(PRETOK_YAML.format(**fmt))
        (run / "train.yaml").write_text(TRAIN_YAML.format(**fmt))
        sb = run / "submit.sbatch"
        sb.write_text(SBATCH.format(**fmt))
        print(f"[{arm}] wrote {run} (data_dir={data_dir})")
    print("ARMS_GENERATED")


if __name__ == "__main__":
    main()
