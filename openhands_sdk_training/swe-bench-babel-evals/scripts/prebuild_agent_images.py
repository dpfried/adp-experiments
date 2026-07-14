#!/usr/bin/env python3
"""Prebuild per-instance Apptainer agent-server SIFs for SWE-bench Verified.

Mirrors the lazy build path in benchmarks/swebench/run_infer.py
prepare_workspace(), so inference finds every SIF already cached in
OPENHANDS_APPTAINER_BUILD_ROOT and never builds on the GPU node.

Usage (one shard of an array job):
  python prebuild_agent_images.py --shard-index $SLURM_ARRAY_TASK_ID \
      --num-shards $SLURM_ARRAY_TASK_COUNT
"""
import argparse
import sys
import traceback

from datasets import load_dataset

from benchmarks.swebench import constants
from benchmarks.swebench.apptainer_build import ensure_apptainer_agent_image
from benchmarks.swebench.build_images import (
    extract_custom_tag,
    get_official_docker_image,
    should_wrap_instance_id,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    ap.add_argument("--split", default="test")
    ap.add_argument("--shard-index", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    ids = sorted(row["instance_id"] for row in ds)
    shard = [iid for i, iid in enumerate(ids) if i % args.num_shards == args.shard_index]
    print(f"shard {args.shard_index}/{args.num_shards}: {len(shard)} instances", flush=True)

    failures = []
    for k, iid in enumerate(shard, 1):
        official = get_official_docker_image(iid)
        try:
            path = ensure_apptainer_agent_image(
                base_image=official,
                custom_tag=extract_custom_tag(official),
                target=constants.DEFAULT_BUILD_TARGET,
                wrap_swebench_deps=should_wrap_instance_id(iid),
            )
            print(f"[{k}/{len(shard)}] {iid} -> {path}", flush=True)
        except Exception:
            failures.append(iid)
            print(f"[{k}/{len(shard)}] {iid} FAILED", flush=True)
            traceback.print_exc()

    print(f"done: {len(shard) - len(failures)} ok, {len(failures)} failed", flush=True)
    if failures:
        print("failed instances:", *failures, sep="\n  ", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
