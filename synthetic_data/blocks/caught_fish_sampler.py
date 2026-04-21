"""
CaughtFishSamplerBlock: Sample random fish + length from the item manifest.

No LLM needed — fish names come from the manifest and lengths are
randomly generated within realistic ranges.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
from sdg_hub import BaseBlock, BlockRegistry

_MANIFEST_PATH = Path("datasets/assets/item_manifest_game.json")
_SPRITES_DIR = Path("datasets/assets/sprites")

# Length ranges by category (inches)
_LENGTH_RANGES = {
    "fish": (5, 50),
    "trash": None,  # no length for trash items
}

# Items that don't display length (trash, algae, seaweed, joja cola, jellies)
_NO_LENGTH_IDS = {
    "152", "153", "157",                        # Seaweed, Green Algae, White Algae
    "167", "168", "169", "170", "171", "172",   # Joja Cola, Trash, Driftwood, etc.
    "SeaJelly", "CaveJelly", "RiverJelly",      # Jellies
}

# Minimum fraction of samples that should be non-fish catchables
_MIN_JUNK_FRACTION = 0.10


@BlockRegistry.register(
    "CaughtFishSamplerBlock", "transform",
    "Sample random fish species and lengths from the item manifest",
)
class CaughtFishSamplerBlock(BaseBlock):
    manifest_path: str = str(_MANIFEST_PATH)

    def generate(self, samples: pd.DataFrame, **kwargs) -> pd.DataFrame:
        with open(self.manifest_path) as f:
            manifest = json.load(f)

        # Split catchable items into fish (with length) and junk (no length)
        real_fish = []
        junk_items = []
        for item_id, item in manifest.items():
            if item.get("type") != "Fish":
                continue
            entry = {
                "item_id": item_id,
                "name": item.get("name", ""),
                "sprite_file": item.get("sprite_file", ""),
            }
            if item_id in _NO_LENGTH_IDS:
                junk_items.append(entry)
            else:
                real_fish.append(entry)

        n = len(samples)
        min_junk = max(1, int(n * _MIN_JUNK_FRACTION))

        results = []
        for i in range(n):
            if i < min_junk:
                fish = random.choice(junk_items)
            else:
                fish = random.choice(real_fish + junk_items)

            item_id = fish["item_id"]
            length = None if item_id in _NO_LENGTH_IDS else random.randint(*_LENGTH_RANGES["fish"])

            results.append({
                "fish_name": fish["name"],
                "fish_item_id": item_id,
                "length_inches": length,
                "sprite_path": str(_SPRITES_DIR / f"sprite_{item_id}.png"),
            })

        result_df = pd.DataFrame(results)
        return pd.concat([samples.reset_index(drop=True), result_df], axis=1)
