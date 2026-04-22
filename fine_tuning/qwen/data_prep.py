#!/usr/bin/env python3
"""Prepare training splits for Phase 1 (tool selection) fine-tuning.

Reads synthetic conversations, transforms them into 3-message sequences
(system + user image + assistant tool call), filters out eval-set images,
and writes train/val/tiny JSONL splits.

Usage:
    python fine_tuning/qwen/data_prep.py
    python fine_tuning/qwen/data_prep.py --output-dir datasets/splits
"""

import argparse
import json
import logging
import random
from pathlib import Path

from evaluation.prompt import NO_TOOL_RESPONSE, SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SCREEN_TYPES = ["caught_fish", "no_tools", "pierre_shop", "tv_dialog"]

TOOL_FOR_SCREEN = {
    "caught_fish": "crop_caught_fish_notification",
    "pierre_shop": "crop_pierres_detail_panel",
    "tv_dialog": "crop_tv_dialog",
}

TINY_PER_CLASS = 10
SEED = 42
VAL_FRACTION = 0.15


def load_eval_image_paths(eval_set_path: Path) -> set[str]:
    with open(eval_set_path) as f:
        eval_set = json.load(f)
    return {item["image_path"] for item in eval_set}


def format_tool_call(tool_name: str) -> str:
    call = json.dumps({"name": tool_name, "arguments": {}})
    return f"<tool_call>\n{call}\n</tool_call>"


def transform_conversation(record: dict) -> dict | None:
    """Transform a synthetic conversation into a Phase 1 training example.

    Returns a 3-message conversation: system, user (image+text), assistant (tool call or refusal).
    """
    messages = record["messages"]
    screen_type = record["metadata"]["screen_type"]

    user_msg = messages[0]
    if user_msg["role"] != "user":
        logger.warning("Unexpected first message role: %s", user_msg["role"])
        return None

    if screen_type in TOOL_FOR_SCREEN:
        assistant_content = format_tool_call(TOOL_FOR_SCREEN[screen_type])
    else:
        assistant_content = NO_TOOL_RESPONSE

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            user_msg,
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": record["metadata"],
    }


def load_and_transform(
    synthetic_dir: Path, eval_images: set[str]
) -> list[dict]:
    examples = []
    skipped_eval = 0

    for screen_type in SCREEN_TYPES:
        conv_file = synthetic_dir / screen_type / "conversations.jsonl"
        if not conv_file.exists():
            logger.warning("Missing: %s", conv_file)
            continue

        count = 0
        with open(conv_file) as f:
            for line in f:
                record = json.loads(line)
                image_path = record["messages"][0]["content"][0]["image"]
                # Strip file:// prefix for comparison
                clean_path = image_path.removeprefix("file://")

                if clean_path in eval_images:
                    skipped_eval += 1
                    continue

                transformed = transform_conversation(record)
                if transformed:
                    examples.append(transformed)
                    count += 1

        logger.info("%-15s: %d examples loaded", screen_type, count)

    logger.info("Skipped %d eval-set images", skipped_eval)
    return examples


def write_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    logger.info("Wrote %d records to %s", len(records), path)


def make_tiny_split(examples: list[dict], per_class: int) -> list[dict]:
    by_type: dict[str, list[dict]] = {}
    for ex in examples:
        st = ex["metadata"]["screen_type"]
        by_type.setdefault(st, []).append(ex)

    tiny = []
    for st in SCREEN_TYPES:
        pool = by_type.get(st, [])
        n = min(per_class, len(pool))
        tiny.extend(pool[:n])

    random.shuffle(tiny)
    return tiny


def main():
    parser = argparse.ArgumentParser(description="Prepare Phase 1 training splits")
    parser.add_argument(
        "--synthetic-dir",
        type=Path,
        default=Path("datasets/synthetic"),
    )
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=Path("datasets/eval_set.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/splits"),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--tiny-per-class", type=int, default=TINY_PER_CLASS)
    args = parser.parse_args()

    random.seed(args.seed)

    eval_images = load_eval_image_paths(args.eval_set)
    logger.info("Eval set: %d images to exclude", len(eval_images))

    examples = load_and_transform(args.synthetic_dir, eval_images)
    random.shuffle(examples)

    split_idx = int(len(examples) * (1 - args.val_fraction))
    train = examples[:split_idx]
    val = examples[split_idx:]

    logger.info("Split: %d train, %d val (%.0f%%/%.0f%%)",
                len(train), len(val),
                100 * len(train) / len(examples),
                100 * len(val) / len(examples))

    tiny = make_tiny_split(train, args.tiny_per_class)

    write_jsonl(train, args.output_dir / "train.jsonl")
    write_jsonl(val, args.output_dir / "val.jsonl")
    write_jsonl(tiny, args.output_dir / "train_tiny.jsonl")

    # Summary by class
    for split_name, split_data in [("train", train), ("val", val), ("tiny", tiny)]:
        counts = {}
        for ex in split_data:
            st = ex["metadata"]["screen_type"]
            counts[st] = counts.get(st, 0) + 1
        logger.info("%s breakdown: %s", split_name, counts)


if __name__ == "__main__":
    main()
