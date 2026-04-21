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

SCREEN_TYPES = ["tv_dialog", "caught_fish", "pierre_shop", "no_tools"]


def run_pipeline(screen_type: str, num_samples: int, output_dir: str | None = None):
    """Run the synthetic data pipeline for a single screen type."""
    if output_dir is None:
        output_dir = f"datasets/synthetic/{screen_type}/images"
    output_jsonl = f"datasets/synthetic/{screen_type}/conversations.jsonl"

    logger.info(f"Generating {num_samples} synthetic {screen_type} examples")

    if screen_type == "no_tools":
        _run_no_tools_pipeline(num_samples, output_dir, output_jsonl)
        return

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
        # Assign random backgrounds from caught_fish backgrounds
        import random
        bg_dir = Path("datasets/caught_fish/backgrounds")
        backgrounds = list(bg_dir.glob("*.png"))
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


def _run_no_tools_pipeline(num_samples: int, output_dir: str, output_jsonl: str):
    """Generate no_tools examples from original screenshots.

    No compositing needed — uses original screenshots directly.
    The VLM should learn NOT to call any extraction tool for these.
    """
    import json
    import random
    import shutil

    images_dir = Path("datasets/no_tools/images")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    backgrounds = list(images_dir.glob("*.PNG")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    if not backgrounds:
        raise FileNotFoundError(f"No images found in {images_dir}")

    # Use each image once if we have enough, otherwise sample with replacement
    if num_samples <= len(backgrounds):
        selected = random.sample(backgrounds, num_samples)
    else:
        # Use all images once, then fill remainder randomly
        selected = list(backgrounds)
        random.shuffle(selected)
        selected += [random.choice(backgrounds) for _ in range(num_samples - len(backgrounds))]

    conversations = []
    for i, bg in enumerate(selected):
        # Copy image to output dir
        out_name = f"synth_no_tools_{i:04d}.png"
        out_path = out_dir / out_name
        from PIL import Image
        Image.open(bg).save(out_path)

        # Build 2-turn ChatML (no extraction tool, just text_to_speech)
        conversation = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{out_path}"},
                        {"type": "text", "text": "What's on this screen?"},
                    ],
                },
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "text_to_speech",
                                "arguments": json.dumps({
                                    "text": "No tool available for this screen type.",
                                }),
                            }
                        }
                    ],
                },
            ],
            "metadata": {
                "screen_type": "no_tools",
                "synthetic": True,
            },
        }
        conversations.append(json.dumps(conversation))

    # Save JSONL
    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w") as f:
        for conv in conversations:
            f.write(conv + "\n")

    logger.info(f"  Copied {num_samples} no_tools images to {output_dir}")
    logger.info(f"  Saved {len(conversations)} conversations to {output_jsonl}")


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
