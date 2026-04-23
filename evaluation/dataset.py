#!/usr/bin/env python3
"""Load the fixed evaluation set from datasets/eval_set.json."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SCREEN_TYPE_TO_TOOL = {
    "tv_dialog": "crop_tv_dialog",
    "caught_fish": "crop_caught_fish_notification",
    "pierre_shop": "crop_pierres_detail_panel",
    "no_tools": None,
}

SCREEN_TYPES = list(SCREEN_TYPE_TO_TOOL.keys())

EVAL_SET_FILE = "datasets/eval_set.json"


@dataclass
class TestSample:
    image_path: str
    screen_type: str
    expected_tool: str | None


def load_test_set(datasets_dir: str = "datasets") -> list[TestSample]:
    """Load the fixed evaluation set from eval_set.json.

    This is a curated set of 100 images (25 per screen type) that must
    never be used for training.
    """
    eval_file = Path(datasets_dir) / "eval_set.json"
    if not eval_file.exists():
        logger.error(f"Eval set file not found: {eval_file}")
        return []

    with open(eval_file) as f:
        records = json.load(f)

    samples = []
    missing_images = []

    for record in records:
        image_path = record["image_path"]
        if not Path(image_path).exists():
            missing_images.append(image_path)
            continue

        screen_type = record["screen_type"]
        expected_tool = record.get("expected_tool", SCREEN_TYPE_TO_TOOL.get(screen_type))
        samples.append(TestSample(
            image_path=image_path,
            screen_type=screen_type,
            expected_tool=expected_tool,
        ))

    if missing_images:
        logger.warning(f"{len(missing_images)} images not found on disk, skipped")

    per_type = {}
    for s in samples:
        per_type[s.screen_type] = per_type.get(s.screen_type, 0) + 1
    for screen_type, count in sorted(per_type.items()):
        logger.info(f"  {screen_type}: {count} samples")

    logger.info(f"Total eval samples: {len(samples)}")
    return samples
