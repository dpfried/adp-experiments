#!/usr/bin/env python3
"""Write a stride-sampled instance_id list for a SWE-bench Verified subset.

Same idea as ../../swe-bench-babel-evals/scripts/smoke20_ids.txt (stride over the
sorted split for repo diversity), but generated so the proof-of-port smoke can
pick any N. Deterministic: same N always yields the same ids.

Usage:
  python make_select_ids.py --n 10 --out $SWEBENCH_ROOT/select/smoke10_ids.txt
"""
import argparse
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ids = sorted(row["instance_id"] for row in load_dataset(args.dataset, split=args.split))
    stride = max(1, len(ids) // args.n)
    picked = ids[::stride][: args.n]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(picked) + "\n")
    print(f"wrote {len(picked)} ids (stride {stride} over {len(ids)}) -> {out}")


if __name__ == "__main__":
    main()
