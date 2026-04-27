#!/usr/bin/env python3
"""Generate a 15% demo subset of the training data for conference talks."""

import argparse
import json
import random
from pathlib import Path

DEMO_FRACTION = 0.15
SEED = 42


def main():
    parser = argparse.ArgumentParser(
        description="Sample a class-balanced demo subset from training data"
    )
    parser.add_argument(
        "--input", type=Path, default=Path("datasets/splits/train.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("datasets/splits/train_demo.jsonl")
    )
    parser.add_argument("--fraction", type=float, default=DEMO_FRACTION)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    with open(args.input) as f:
        examples = [json.loads(line) for line in f]

    by_type: dict[str, list[dict]] = {}
    for ex in examples:
        st = ex["metadata"]["screen_type"]
        by_type.setdefault(st, []).append(ex)

    demo = []
    for st, pool in by_type.items():
        n = max(1, int(len(pool) * args.fraction))
        demo.extend(pool[:n])

    random.shuffle(demo)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for record in demo:
            f.write(json.dumps(record) + "\n")

    print(
        f"Wrote {len(demo)} demo samples to {args.output}"
        f" ({args.fraction:.0%} of {len(examples)})"
    )
    for st, pool in sorted(by_type.items()):
        n = max(1, int(len(pool) * args.fraction))
        print(f"  {st}: {n}/{len(pool)}")


if __name__ == "__main__":
    main()
