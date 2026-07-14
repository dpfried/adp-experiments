#!/usr/bin/env python3
"""Download + uniformly subsample adp-v2 SWE configs, then convert to LLaMA-Factory openai format.

For each target config:
  1. list its 24k sft shards, shuffle order (fixed seed);
  2. download shards to /scratch one at a time (native hf_xet), reservoir-sample records
     toward TARGET_RAW, delete each shard after (bounded disk);
     for huge configs, stop once we've seen >= SEEN_MULT * TARGET_RAW across the shards;
  3. write the sampled raw OpenAI-format jsonl + a small eval carve-out to tir1;
  4. run the ADP `sft_to_llamafactory` adapter -> LLaMA-Factory openai/tools jsonl + dataset_info.

Record-level uniform reservoir (Algorithm R). Each condensed record is a self-contained
trainable OpenAI conversation, so record-level sampling is the right training unit.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("HF_TOKEN", Path("~/.cache/huggingface/token").expanduser().read_text().strip())
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

REPO = "neulab/adp-v2"
CONFIGS = [
    "coderforge_preview",
    "scale_swe_distilled",
    "nebius_SWE-rebench-openhands-trajectories",
    "nvidia_SWE-Zero-openhands-trajectories",
]
ADP_REPO = Path("~/exp/adp-smoke/agent-data-protocol").expanduser()
PYTHON = Path("~/exp/adp-smoke/.venv/bin/python").expanduser()


def reservoir_from_shards(shards, scratch: Path, target: int, seen_mult: float, max_shards: int, rng):
    """Download shards in given order, reservoir-sample `target` records, bounded disk."""
    reservoir: list[str] = []
    seen = 0
    for i, (fname, _size) in enumerate(shards):
        if i >= max_shards or (seen >= seen_mult * target and i >= 3):
            break
        local = hf_hub_download(REPO, fname, repo_type="dataset", local_dir=str(scratch))
        with open(local) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                seen += 1
                if len(reservoir) < target:
                    reservoir.append(line)
                else:
                    j = rng.randint(0, seen - 1)
                    if j < target:
                        reservoir[j] = line
        os.remove(local)
        print(f"    shard {i} done: seen={seen}, reservoir={len(reservoir)}", flush=True)
    return reservoir, seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-raw", type=int, default=80000)
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--seen-mult", type=float, default=2.5)
    ap.add_argument("--max-shards", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", type=Path,
                    default=Path("/data/tir/projects/tir1/users/dfried/adp-smoke/datasets/v2_swe_subsets"))
    ap.add_argument("--scratch", type=Path, default=Path(f"/scratch/{os.environ.get('USER','dfried')}/adp_v2_dl"))
    ap.add_argument("--configs", nargs="*", default=CONFIGS)
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    info = api.repo_info(REPO, repo_type="dataset", files_metadata=True)

    for cfg in args.configs:
        out_dir = args.out_root / cfg
        done_marker = out_dir / "PREP_DONE"
        if done_marker.exists():
            print(f"[{cfg}] already done, skipping", flush=True)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(args.seed)
        shards = sorted(
            (s.rfilename, s.size) for s in info.siblings
            if s.rfilename.startswith(f"{cfg}/24k/full_sft/") and s.rfilename.endswith(".jsonl")
        )
        rng.shuffle(shards)
        print(f"[{cfg}] {len(shards)} shards; sampling {args.target_raw} records "
              f"(<= {args.max_shards} shards, seen_mult {args.seen_mult})", flush=True)

        reservoir, seen = reservoir_from_shards(
            shards, args.scratch, args.target_raw, args.seen_mult, args.max_shards, rng)
        rng.shuffle(reservoir)
        eval_lines = reservoir[: args.eval_n]
        train_lines = reservoir[args.eval_n:]
        raw_train = out_dir / "raw_train.openai.jsonl"
        raw_eval = out_dir / "raw_eval.openai.jsonl"
        raw_train.write_text("\n".join(train_lines) + "\n")
        raw_eval.write_text("\n".join(eval_lines) + "\n")
        print(f"[{cfg}] sampled seen={seen} -> train={len(train_lines)} eval={len(eval_lines)}", flush=True)

        # Convert with the ADP adapter (openai/tools format, SDK tool names preserved).
        for split, src in (("train", raw_train), ("eval", raw_eval)):
            out_jsonl = out_dir / f"{split}.llamafactory.jsonl"
            di = out_dir / f"dataset_info_{split}.json"
            subprocess.run(
                [str(PYTHON), "-m", "agents.openhands_sdk.sft_to_llamafactory",
                 "--input", str(src), "--output", str(out_jsonl),
                 "--dataset-info", str(di), "--dataset-name", f"{cfg}_{split}",
                 "--trim-to-trainable", "--skip-untrainable"],
                cwd=str(ADP_REPO), check=True,
            )
        done_marker.write_text("ok\n")
        print(f"[{cfg}] DONE -> {out_dir}", flush=True)

    print("ALL_PREP_DONE", flush=True)


if __name__ == "__main__":
    main()
