#!/usr/bin/env python3
"""Download ADP full_sft OpenHands-format JSONL files from Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


OPENHANDS_NONWEB = [
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
]

AGENTLAB_WEB = [
    "go-browse-wa",
    "mind2web",
    "nnetnav-live",
    "nnetnav-wa",
    "synatra",
]

PRESETS = {
    "openhands_nonweb": OPENHANDS_NONWEB,
    "agentlab_web": AGENTLAB_WEB,
    "all": OPENHANDS_NONWEB + AGENTLAB_WEB,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="neulab/agent-data-collection")
    parser.add_argument("--out", type=Path, default=Path("~/exp/adp/datasets/hf_release").expanduser())
    parser.add_argument("--preset", choices=sorted(PRESETS), default="all")
    parser.add_argument("--subset", action="append", help="Additional or replacement subset. May be repeated.")
    args = parser.parse_args()

    subsets = args.subset if args.subset else PRESETS[args.preset]
    allow_patterns = [f"{subset}/full_sft/full_sft_openhands.jsonl" for subset in subsets]

    args.out.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        local_dir=args.out,
    )

    print(f"Downloaded to: {path}")
    for subset in subsets:
        file_path = args.out / subset / "full_sft" / "full_sft_openhands.jsonl"
        print(f"{subset}: {file_path}")


if __name__ == "__main__":
    main()
