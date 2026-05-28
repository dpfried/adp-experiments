#!/usr/bin/env python3
"""Create deterministic ADP paper-style balanced train/eval SFT splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Iterable


WEIGHTS = {
    "agenttuning_alfworld": 2,
    "agenttuning_db": 2,
    "agenttuning_kg": 2,
    "agenttuning_mind2web": 2,
    "agenttuning_os": 2,
    "agenttuning_webshop": 2,
    "code_feedback": 0.1,
    "codeactinstruct": 1,
    "go-browse-wa": 1,
    "mind2web": 1,
    "nebius_SWE-agent-trajectories": 0.2,
    "nnetnav-live": 1,
    "nnetnav-wa": 1,
    "openhands": 1,
    "orca_agentinstruct": 0.001,
    "swe-gym_openhands_sampled_trajectories": 3,
    "swe-smith": 1,
    "synatra": 0.01,
}

MIXTURES = {
    "openhands_nonweb": [
        "agenttuning_alfworld",
        "agenttuning_db",
        "agenttuning_kg",
        "agenttuning_mind2web",
        "agenttuning_os",
        "agenttuning_webshop",
        "code_feedback",
        "codeactinstruct",
        "nebius_SWE-agent-trajectories",
        "openhands",
        "orca_agentinstruct",
        "swe-gym_openhands_sampled_trajectories",
        "swe-smith",
    ],
    "agentlab_web": [
        "go-browse-wa",
        "mind2web",
        "nnetnav-live",
        "nnetnav-wa",
        "synatra",
    ],
}
MIXTURES["all_weighted_union"] = MIXTURES["openhands_nonweb"] + MIXTURES["agentlab_web"]


def source_path(root: Path, subset: str) -> Path:
    return root / subset / "full_sft" / "full_sft_openhands.jsonl"


def count_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for count, _ in enumerate(handle, start=1):
            pass
    return count


def eval_count(raw_count: int) -> int:
    previous = min(100, max(10, math.ceil(0.02 * raw_count)))
    return max(1, previous // 2)


def line_hash(line: bytes) -> str:
    return hashlib.sha1(line).hexdigest()


def select_train_indices(raw_count: int, eval_indices: set[int], weight: float, rng: random.Random) -> list[int]:
    remaining = [idx for idx in range(raw_count) if idx not in eval_indices]
    if weight == 1:
        return remaining

    target_count = math.ceil(len(remaining) * weight)
    if weight < 1:
        return rng.sample(remaining, target_count)

    return rng.choices(remaining, k=target_count)


def write_selected(path: Path, selected_indices: Iterable[int], out_handle, eval_hashes: set[str] | None = None) -> int:
    selected = set(selected_indices)
    written = 0
    with path.open("rb") as in_handle:
        for idx, line in enumerate(in_handle):
            if idx not in selected:
                continue
            if eval_hashes is not None and line_hash(line) in eval_hashes:
                continue
            out_handle.write(line)
            written += 1
    return written


def collect_eval_hashes(path: Path, eval_indices: set[int], eval_handle) -> tuple[int, set[str]]:
    hashes: set[str] = set()
    written = 0
    with path.open("rb") as in_handle:
        for idx, line in enumerate(in_handle):
            if idx not in eval_indices:
                continue
            eval_handle.write(line)
            hashes.add(line_hash(line))
            written += 1
    return written, hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mixture", choices=sorted(MIXTURES), default="openhands_nonweb")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=4242)
    args = parser.parse_args()

    args.input_root = args.input_root.expanduser()
    args.output_dir = args.output_dir.expanduser()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    name = f"paper_{args.mixture}"
    train_path = args.output_dir / f"{name}_train.jsonl"
    eval_path = args.output_dir / f"{name}_eval.jsonl"
    manifest_path = args.output_dir / f"{name}.manifest.json"

    eval_rng = random.Random(args.eval_seed)
    train_rng = random.Random(args.train_seed)

    stats = {}
    total_eval = 0
    total_train_before_decontamination = 0
    skipped_train_hash_matches = 0

    with train_path.open("wb") as train_out, eval_path.open("wb") as eval_out:
        for subset in MIXTURES[args.mixture]:
            path = source_path(args.input_root, subset)
            if not path.exists():
                raise FileNotFoundError(path)

            raw_count = count_lines(path)
            heldout_count = eval_count(raw_count)
            eval_indices = set(eval_rng.sample(range(raw_count), heldout_count))
            train_indices = select_train_indices(raw_count, eval_indices, WEIGHTS[subset], train_rng)

            written_eval, eval_hashes = collect_eval_hashes(path, eval_indices, eval_out)
            written_train = write_selected(path, train_indices, train_out, eval_hashes)

            sampled_train_count = len(train_indices)
            skipped = sampled_train_count - written_train
            skipped_train_hash_matches += skipped
            total_eval += written_eval
            total_train_before_decontamination += sampled_train_count

            if WEIGHTS[subset] == 1:
                mode = "all_train_remaining"
            elif WEIGHTS[subset] < 1:
                mode = "downsample_without_replacement_from_train_remaining"
            else:
                mode = "upsample_with_replacement_from_train_remaining"

            stats[subset] = {
                "raw_count": raw_count,
                "heldout_eval_count": heldout_count,
                "train_remaining_count": raw_count - heldout_count,
                "weight": WEIGHTS[subset],
                "sampled_train_count": sampled_train_count,
                "mode": mode,
                "written_train_count": written_train,
                "written_eval_count": written_eval,
            }

    total_train = sum(item["written_train_count"] for item in stats.values())
    manifest = {
        "name": name,
        "source_repo": "neulab/agent-data-collection",
        "source_format": "full_sft/full_sft_openhands.jsonl",
        "train_seed": args.train_seed,
        "eval_seed": args.eval_seed,
        "eval_policy": "half-size deterministic per-source holdout before train sampling: floor(previous eval policy / 2), minimum 1 row/source; eval is not upsampled",
        "previous_eval_policy": "min(100, max(10, ceil(0.02 * raw_count))) rows/source",
        "train_policy": "paper appendix multipliers applied to remaining train source lines after removing eval holdout",
        "datasets": MIXTURES[args.mixture],
        "total_train": total_train,
        "total_eval": total_eval,
        "stats": stats,
        "content_hash_decontamination": {
            "hash": "sha1(raw_jsonl_line_bytes)",
            "skipped_train_rows_matching_eval_content": skipped_train_hash_matches,
        },
        "total_train_before_content_hash_decontamination": total_train_before_decontamination,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote {train_path} ({total_train} rows)")
    print(f"Wrote {eval_path} ({total_eval} rows)")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()

