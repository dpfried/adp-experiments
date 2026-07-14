"""Offline validation loss for a checkpoint, using the pre-tokenized eval split.

Runs on a single GPU with plain bf16 weights (no ZeRO/optimizer state), so the
Liger 32k eval-logits OOM that forces eval_strategy:"no" during training does
not apply. Loss is token-weighted (sum of per-token CE over all valid label
tokens / total valid label tokens), matching HF Trainer's eval_loss.

Usage: python eval_checkpoint_loss.py --checkpoint <dir> --tokenized <tokenized_path> [--out csv]
"""
import argparse, csv, json, os, time
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForImageTextToText


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenized", required=True, help="tokenized_path dir with a 'validation' split")
    ap.add_argument("--out", default=None, help="append result row to this CSV")
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--wandb-run", default=None,
                    help="log eval_loss to this wandb run name/id (companion run, resume='allow', project from WANDB_PROJECT)")
    args = ap.parse_args()

    ds = load_from_disk(args.tokenized)["validation"]
    if args.max_examples:
        ds = ds.select(range(min(args.max_examples, len(ds))))

    model = AutoModelForImageTextToText.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, trust_remote_code=True
    ).cuda().eval()

    total_loss, total_tokens = 0.0, 0
    t0 = time.time()
    with torch.inference_mode():
        for i, ex in enumerate(ds):
            input_ids = torch.tensor([ex["input_ids"]], device="cuda")
            labels = torch.tensor([ex["labels"]], device="cuda")
            n_valid = int((labels[:, 1:] != -100).sum())
            if n_valid == 0:
                continue
            # No labels kwarg: HF's loss path upcasts ALL logits to fp32 (27GiB at 32k x 248k vocab).
            # Keep bf16 logits and compute CE in sequence chunks, upcasting ~2GB at a time.
            logits = model(input_ids=input_ids).logits[0, :-1]   # [L-1, V] bf16
            tgt = labels[0, 1:]                                   # [L-1]
            loss_sum = 0.0
            for s in range(0, tgt.shape[0], 4096):
                loss_sum += float(torch.nn.functional.cross_entropy(
                    logits[s:s + 4096].float(), tgt[s:s + 4096],
                    ignore_index=-100, reduction="sum"))
            del logits
            total_loss += loss_sum
            total_tokens += n_valid
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(ds)} examples, running loss {total_loss/total_tokens:.4f}", flush=True)

    loss = total_loss / total_tokens
    step = Path(args.checkpoint).name.replace("checkpoint-", "")
    result = {
        "checkpoint": args.checkpoint, "step": step, "eval_loss": round(loss, 6),
        "eval_tokens": total_tokens, "n_examples": len(ds), "seconds": round(time.time() - t0, 1),
    }
    print("RESULT " + json.dumps(result), flush=True)
    if args.wandb_run:
        import wandb
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "adp-smoke"),
                         name=args.wandb_run, id=args.wandb_run, resume="allow")
        # log train/global_step alongside so panels keyed on it align with the
        # training run's curves (wandb's default "Step" axis counts log calls, not trainer steps)
        run.define_metric("eval_loss", step_metric="train/global_step")
        run.log({"eval_loss": loss, "train/global_step": int(step)})
        run.finish()
    if args.out:
        new = not os.path.exists(args.out)
        with open(args.out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(result.keys()))
            if new:
                w.writeheader()
            w.writerow(result)


if __name__ == "__main__":
    main()
