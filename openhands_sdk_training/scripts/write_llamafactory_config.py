#!/usr/bin/env python3
"""Write LLaMA-Factory dataset metadata and a Qwen3.5 0.8B SFT config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASET_INFO_ENTRY = {
    "formatting": "sharegpt",
    "columns": {
        "messages": "conversations",
        "system": "system",
    },
    "tags": {
        "role_tag": "from",
        "content_tag": "value",
        "user_tag": "human",
        "assistant_tag": "gpt",
        "system_tag": "system",
    },
}


def add_dataset_info(dataset_dir: Path, train_file: str, eval_file: str) -> None:
    dataset_info_path = dataset_dir / "dataset_info.json"
    if dataset_info_path.exists():
        dataset_info = json.loads(dataset_info_path.read_text())
    else:
        dataset_info = {}

    for name, file_name in {
        "paper_openhands_nonweb_train_clean_mm_safe": train_file,
        "paper_openhands_nonweb_eval_clean_mm_safe": eval_file,
    }.items():
        entry = dict(DATASET_INFO_ENTRY)
        entry["file_name"] = file_name
        dataset_info[name] = entry

    dataset_info_path.write_text(json.dumps(dataset_info, indent=2) + "\n")
    print(f"Wrote {dataset_info_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="adp-openhands-nonweb-qwen35-0.8b-full-10k-bs1-seq2048-mm-safe")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--cutoff-len", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", default="2.0e-5")
    parser.add_argument("--eval-steps", type=int, default=500)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = "paper_openhands_nonweb_train_clean_mm_safe.jsonl"
    eval_file = "paper_openhands_nonweb_eval_clean_mm_safe.jsonl"
    add_dataset_info(dataset_dir, train_file, eval_file)

    config_path = dataset_dir / "qwen35_0_8b_openhands_nonweb_full_10k_bs1_seq2048_mm_safe.yaml"
    config = f"""### model
model_name_or_path: {args.model}
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full

### dataset
dataset: paper_openhands_nonweb_train_clean_mm_safe
eval_dataset: paper_openhands_nonweb_eval_clean_mm_safe
dataset_dir: {dataset_dir}
template: qwen3_5_nothink
cutoff_len: {args.cutoff_len}
overwrite_cache: true
preprocessing_num_workers: 1
dataloader_num_workers: 0

### output
output_dir: {output_dir}
logging_steps: 10
save_steps: 1000
save_total_limit: 2
plot_loss: true
overwrite_output_dir: true
save_only_model: true
report_to: wandb
run_name: {args.run_name}

### train
per_device_train_batch_size: {args.batch_size}
gradient_accumulation_steps: 1
learning_rate: {args.learning_rate}
max_steps: {args.max_steps}
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
fp16: false
gradient_checkpointing: true
ddp_timeout: 180000000
resume_from_checkpoint: null

### eval
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: {args.eval_steps}
"""
    config_path.write_text(config)
    print(f"Wrote {config_path}")


if __name__ == "__main__":
    main()

