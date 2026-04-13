#!/usr/bin/env python3
"""
Generate synthetic training data variations from real annotations.

Uses LLM (Claude/GPT-4) to generate plausible variations while preserving
domain knowledge (fish names, item IDs, etc.).

Usage:
    python synthetic_data/generate_variations.py \
        --screen-type tv_dialog \
        --num-variations 100 \
        --output datasets/tv_dialog/conversations_synthetic.jsonl
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List

from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_real_annotations(screen_type: str) -> List[dict]:
    """Load real annotations from datasets/{screen_type}/annotations.jsonl"""
    annotations_file = Path(f"datasets/{screen_type}/annotations.jsonl")
    annotations = []
    with open(annotations_file) as f:
        for line in f:
            annotations.append(json.loads(line))
    return annotations


def load_domain_knowledge(screen_type: str) -> dict:
    """
    Load domain-specific knowledge for synthetic generation.

    For caught_fish: Load fish names from item_manifest.json
    For pierre_shop: Load item names and prices
    etc.
    """
    if screen_type == "caught_fish":
        manifest_file = Path("datasets/assets/item_manifest.json")
        with open(manifest_file) as f:
            manifest = json.load(f)
        # Extract fish names
        fish = [
            item["name"]
            for item in manifest.values()
            if item.get("type") == "Fish"
        ]
        return {"fish_names": fish}

    elif screen_type == "pierre_shop":
        manifest_file = Path("datasets/assets/item_manifest.json")
        with open(manifest_file) as f:
            manifest = json.load(f)
        # Extract items with prices
        items = [
            {
                "name": item["name"],
                "price": item.get("price", 0),
                "description": item.get("description", ""),
            }
            for item in manifest.values()
            if item.get("price", 0) > 0
        ]
        return {"items": items}

    # For other screen types, return empty
    return {}


def generate_synthetic_variation(
    client: OpenAI,
    screen_type: str,
    real_example: dict,
    domain_knowledge: dict,
) -> dict:
    """
    Generate a synthetic variation using LLM.

    Prompt LLM to create plausible variation while preserving:
    - Field structure
    - Domain knowledge (fish names, item IDs)
    - Narration style
    """
    # TODO: Construct prompt for LLM
    # TODO: Call LLM API
    # TODO: Parse response into ChatML format
    # TODO: Return synthetic conversation

    # Placeholder
    synthetic_conversation = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "data:image/png;base64,..."},
                    {"type": "text", "text": "What's on this screen?"}
                ]
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": f"crop_{screen_type}",
                            "arguments": "{\"image_b64\": \"...\"}"
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "name": f"crop_{screen_type}",
                "content": json.dumps(real_example.get("ocr_fields", {}))
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "text_to_speech",
                            "arguments": json.dumps({
                                "text": real_example.get("narration", "")
                            })
                        }
                    }
                ]
            }
        ],
        "metadata": {
            "synthetic": True,
            "source_image_id": real_example.get("image_id"),
            "variation_type": "llm_generated"
        }
    }

    return synthetic_conversation


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument(
        "--screen-type",
        type=str,
        required=True,
        choices=["tv_dialog", "caught_fish", "pierre_shop", "quest_board", "game_letter", "level_up"],
        help="Screen type to generate variations for"
    )
    parser.add_argument(
        "--num-variations",
        type=int,
        default=100,
        help="Number of synthetic variations to generate"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSONL file (default: datasets/{screen_type}/conversations_synthetic.jsonl)"
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic"],
        help="LLM provider for synthetic generation"
    )
    args = parser.parse_args()

    output_file = args.output or f"datasets/{args.screen_type}/conversations_synthetic.jsonl"

    # Load real annotations
    logger.info(f"Loading real annotations for {args.screen_type}")
    real_annotations = load_real_annotations(args.screen_type)
    logger.info(f"Found {len(real_annotations)} real examples")

    # Load domain knowledge
    logger.info("Loading domain knowledge")
    domain_knowledge = load_domain_knowledge(args.screen_type)

    # Initialize LLM client
    if args.llm_provider == "openai":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    else:
        # Anthropic client (OpenAI-compatible)
        client = OpenAI(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            base_url="https://api.anthropic.com/v1"
        )

    # Generate variations
    logger.info(f"Generating {args.num_variations} synthetic variations")
    synthetic_conversations = []

    for i in range(args.num_variations):
        # Sample a random real example as template
        real_example = real_annotations[i % len(real_annotations)]

        # Generate synthetic variation
        synthetic = generate_synthetic_variation(
            client,
            args.screen_type,
            real_example,
            domain_knowledge
        )

        synthetic_conversations.append(synthetic)

        if (i + 1) % 10 == 0:
            logger.info(f"Generated {i + 1}/{args.num_variations} variations")

    # Save to JSONL
    logger.info(f"Saving to {output_file}")
    with open(output_file, "w") as f:
        for conversation in synthetic_conversations:
            f.write(json.dumps(conversation) + "\n")

    logger.info(f"Synthetic data generation complete: {len(synthetic_conversations)} variations")


if __name__ == "__main__":
    main()
