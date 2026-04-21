#!/usr/bin/env python3
"""
Run the synthetic data generation pipeline for Stardew Vision.

Generates synthetic screenshots with varied text content using
clean backgrounds and the bitmap font renderer. No LLM required —
all text comes from game data files.

Usage:
    python synthetic_data/run_pipeline.py --screen-type tv_dialog --num 150
    python synthetic_data/run_pipeline.py --screen-type caught_fish --num 150
    python synthetic_data/run_pipeline.py --screen-type pierre_shop --num 150
    python synthetic_data/run_pipeline.py --all --num 150
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

# Import blocks to trigger registration
import synthetic_data.blocks  # noqa: F401

from synthetic_data.blocks.seed_data import SeedDataBlock
from synthetic_data.blocks.caught_fish_sampler import CaughtFishSamplerBlock
from synthetic_data.blocks.tv_dialog_sampler import TVDialogSamplerBlock
from synthetic_data.blocks.pierre_shop_sampler import PierreShopSamplerBlock
from synthetic_data.blocks.render_screenshot import RenderScreenshotBlock
from synthetic_data.blocks.build_chatml import BuildChatMLBlock

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SCREEN_TYPES = ["tv_dialog", "caught_fish", "pierre_shop"]


def run_pipeline(screen_type: str, num_samples: int, output_dir: str | None = None):
    """Run the synthetic data pipeline for a single screen type."""
    if output_dir is None:
        output_dir = f"datasets/{screen_type}/synthetic_images"
    output_jsonl = f"datasets/{screen_type}/conversations_synthetic.jsonl"

    logger.info(f"Generating {num_samples} synthetic {screen_type} examples")

    # Step 1: Create seed DataFrame
    seed = SeedDataBlock(block_name="seed", num_samples=num_samples)
    df = seed.generate(pd.DataFrame())
    logger.info(f"  Seed: {len(df)} rows")

    # Step 2: Sample content (screen-type specific)
    if screen_type == "tv_dialog":
        sampler = TVDialogSamplerBlock(block_name="tv_sampler")
        df = sampler.generate(df)
        logger.info(f"  Sampled {len(df)} TV dialogs")
    elif screen_type == "caught_fish":
        sampler = CaughtFishSamplerBlock(block_name="fish_sampler")
        df = sampler.generate(df)
        # Assign random backgrounds from caught_fish images
        import random
        bg_dir = Path("datasets/caught_fish/backgrounds")
        backgrounds = list(bg_dir.glob("*.PNG")) + list(bg_dir.glob("*.jpg"))
        df["background_path"] = [str(random.choice(backgrounds)) for _ in range(len(df))]
        logger.info(f"  Sampled {len(df)} caught fish")
    elif screen_type == "pierre_shop":
        sampler = PierreShopSamplerBlock(block_name="shop_sampler")
        df = sampler.generate(df)
        logger.info(f"  Sampled {len(df)} Pierre's shop items")

    # Step 3: Render screenshots
    renderer = RenderScreenshotBlock(
        block_name="renderer", screen_type=screen_type, output_dir=output_dir,
    )
    df = renderer.generate(df)
    logger.info(f"  Rendered {len(df)} screenshots to {output_dir}")

    # Step 4: Build ChatML conversations
    builder = BuildChatMLBlock(block_name="chatml", screen_type=screen_type)
    df = builder.generate(df)
    logger.info(f"  Built {len(df)} ChatML conversations")

    # Step 5: Save conversations to JSONL
    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w") as f:
        for conv in df["conversation"]:
            f.write(conv + "\n")
    logger.info(f"  Saved to {output_jsonl}")

    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--screen-type",
        choices=SCREEN_TYPES,
        help="Screen type to generate",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate for all screen types",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=150,
        help="Number of synthetic examples per screen type (default: 150)",
    )
    args = parser.parse_args()

    if args.all:
        for st in SCREEN_TYPES:
            run_pipeline(st, args.num)
    elif args.screen_type:
        run_pipeline(args.screen_type, args.num)
    else:
        parser.error("Specify --screen-type or --all")


if __name__ == "__main__":
    main()
