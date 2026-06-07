# Current OpenHands SDK Experiments

This file tracks the current experiments under `~/exp/adp` as of 2026-06-05.
The original 0.8B reproduction flow remains in `README.md`; the current main
work is the 32k Qwen3.5 full-parameter SFT line on OpenHands SDK condenser data.

## Data

Current all-records adapted split:

```text
root: ~/exp/adp/runs/openhands_sdk_training/full_condenser_24k_all_records_adapted
manifest: ~/exp/adp/runs/openhands_sdk_training/full_condenser_24k_all_records_adapted/manifest.json
train: full_condenser_24k_all_records_train.openai.jsonl
eval: full_condenser_24k_all_records_eval.openai.jsonl
llamafactory train: full_condenser_24k_all_records_train.llamafactory.jsonl
llamafactory eval: full_condenser_24k_all_records_eval.llamafactory.jsonl
tokenized 4B path: tokenized_qwen35_4b_seq32768_all_records
```

Manifest counts:

```text
total_records: 170958
canonical_train_records: 170458
canonical_eval_records: 500
unique_source_trajectories: 107322
selection: all currently available OpenHands SDK condenser SFT records; eval split is 500 records with lowest sha1(record_id)
```

## Current 32k Run Family

Common settings:

```text
trainer: LLaMA-Factory SFT
finetuning_type: full
template: qwen3_5_nothink
cutoff_len: 32768
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-5
max_steps: 5322
save_steps: 500
save_total_limit: 3
eval_strategy: steps
eval_steps: 100
bf16: true
gradient_checkpointing: true
save_only_model: true
```

Configs in `~/exp/adp/runs/openhands_sdk_training/full_condenser_24k_all_records_adapted`:

| Config | Model | Variant | Output |
| --- | --- | --- | --- |
| `qwen35_4b_full_condenser_24k_all_records_seq32768_zero3_one_epoch.yaml` | `Qwen/Qwen3.5-4B-Base` | ZeRO-3 baseline | `output_qwen35_4b_seq32768_zero3_one_epoch` |
| `qwen35_4b_full_condenser_24k_all_records_seq32768_zero3_one_epoch_flashattn.yaml` | `Qwen/Qwen3.5-4B-Base` | ZeRO-3 + FlashAttention-2 | `output_qwen35_4b_seq32768_zero3_one_epoch_flashattn` |
| `qwen35_4b_full_condenser_24k_all_records_seq32768_zero3_one_epoch_muon.yaml` | `Qwen/Qwen3.5-4B-Base` | ZeRO-3 + Muon | `output_qwen35_4b_seq32768_zero3_one_epoch_muon` |
| `qwen35_4b_full_condenser_24k_all_records_seq32768_z2_one_epoch_muon.yaml` | `Qwen/Qwen3.5-4B-Base` | ZeRO-2 + Muon | `output_qwen35_4b_seq32768_z2_one_epoch_muon` |
| `qwen35_9b_full_condenser_24k_all_records_seq32768_zero3_one_epoch_a100.yaml` | `Qwen/Qwen3.5-9B` | ZeRO-3 + FlashAttention-2 for A100 | `output_qwen35_9b_seq32768_zero3_one_epoch_a100_flashattn` |

Run a config directly:

```bash
cd ~/work/adp/adp-experiments/openhands_sdk_training

llamafactory-cli train \
  ~/exp/adp/runs/openhands_sdk_training/full_condenser_24k_all_records_adapted/qwen35_4b_full_condenser_24k_all_records_seq32768_zero3_one_epoch_flashattn.yaml
```

For Slurm runs, capture the job metadata, `nvidia-smi`, flash-attn check, and
GPU monitor path in the `.out` file before launching `llamafactory-cli train`.
The current successful flash-attn run did this and used one 4x L40S node.

## FlashAttention-2 Setup

The working flash-attn run used:

```text
flash-attn: 2.8.3
torch CUDA arch list includes sm_80
is_flash_attn_2_available: True
hardware: 4x NVIDIA L40S, 46068 MiB each
```

Verify before launching:

```bash
python - <<'PY'
import flash_attn
from transformers.utils import is_flash_attn_2_available

print("flash_attn", flash_attn.__version__)
print("is_flash_attn_2_available", is_flash_attn_2_available())
PY
```

The LLaMA-Factory log should include:

```text
Using FlashAttention-2 for faster training and inference.
Fine-tuning method: Full
```

## Status Snapshot

Observed status on 2026-06-05:

```text
active Slurm job: 8254382
run: adp-full-condenser-24k-all-records-qwen35-4b-seq32768-zero3-one-epoch-flashattn
config: qwen35_4b_full_condenser_24k_all_records_seq32768_zero3_one_epoch_flashattn.yaml
latest log step: 1697 / 5322
latest saved checkpoint: checkpoint-1500
checkpoint-1500 eval_loss: 0.2295125424861908
```

Find the latest checkpoint dynamically instead of hard-coding a step:

```bash
find ~/exp/adp/runs/openhands_sdk_training/full_condenser_24k_all_records_adapted/output_qwen35_4b_seq32768_zero3_one_epoch_flashattn \
  -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' |
  sed 's/checkpoint-//' |
  sort -n |
  tail -1
```

List current checkpoints with timestamps:

```bash
find ~/exp/adp/runs/openhands_sdk_training/full_condenser_24k_all_records_adapted/output_qwen35_4b_seq32768_zero3_one_epoch_flashattn \
  -maxdepth 1 -type d -name 'checkpoint-*' -printf '%T@ %p\n' |
  sort -nr
```

## Loading A Checkpoint

The saved 4B checkpoints are full-model Hugging Face checkpoints, not LoRA
adapters and not ZeRO shards. They contain `model.safetensors`, `config.json`,
tokenizer files, `processor_config.json`, and `generation_config.json`.

Example:

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

ckpt = (
    "/home/gneubig/exp/adp/runs/openhands_sdk_training/"
    "full_condenser_24k_all_records_adapted/"
    "output_qwen35_4b_seq32768_zero3_one_epoch_flashattn/"
    "checkpoint-1500"
)

processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    ckpt,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
```

Compatibility note: these checkpoints have `model_type: qwen3_5` and were saved
with a Transformers build that reports `transformers_version: 5.6.0`. Use a
Transformers version or local branch that knows the Qwen3.5 classes.

## Hugging Face Uploads

The private Hub repo `gneubig/adp-qwen35-4b-flashattn-ckpt1000` contains the
older `checkpoint-1000`. The current local latest checkpoint is newer
(`checkpoint-1500` as of 2026-06-05), so upload scripts should discover the
latest checkpoint before publishing.

For checkpoints larger than 5 GiB, enable large-file support and use Git LFS.
The Python `huggingface_hub.upload_folder` path can exceed memory limits on
large `model.safetensors` files.

```bash
hf lfs-enable-largefiles /path/to/local/repo
git lfs track '*.safetensors' 'tokenizer.json' '*.bin'
git add .gitattributes .
git commit -m "Upload flash-attn checkpoint"
git push origin main
```
