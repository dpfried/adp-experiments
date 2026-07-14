"""Standalone, single-process pre-tokenization for LLaMA-Factory.

Builds the `tokenized_path` Arrow cache using ONLY the tokenizer (no model load,
no distributed init), so the subsequent distributed training run just mmaps the
cache via `has_tokenized_data -> load_from_disk` and never tokenizes under
torchrun (which is where the per-rank preprocessing deadlock happens).

Usage:  python pretokenize.py <pretokenize_config.yaml>
"""
import sys

from llamafactory.hparams import get_train_args
from llamafactory.model import load_tokenizer
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer


def main() -> None:
    model_args, data_args, training_args, finetuning_args, _ = get_train_args()
    if data_args.tokenized_path is None:
        raise SystemExit("pretokenize.py requires `tokenized_path` to be set in the config")
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    print(f"PRETOKENIZE_DONE: {data_args.tokenized_path}")


if __name__ == "__main__":
    main()
