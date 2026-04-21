"""
PierreShopSamplerBlock: Sample items with descriptions for Pierre's shop.

Uses the resolved item descriptions, item manifest (for edibility),
and real annotation patterns to generate realistic shop panel data.
At least 25% of samples include energy/health values. No LLM needed.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
from sdg_hub import BaseBlock, BlockRegistry

_DESCRIPTIONS_PATH = Path("datasets/assets/item_descriptions_resolved.json")
_MANIFEST_PATH = Path("datasets/assets/item_manifest_game.json")
_BACKGROUNDS_DIR = Path("datasets/pierre_shop/images")

# Quantity distribution: mostly small, occasionally large
_QUANTITY_WEIGHTS = [(1, 40), (5, 15), (10, 10), (25, 8), (50, 5),
                     (100, 3), (250, 2), (500, 1), (999, 1)]

# Minimum fraction of samples that should have energy/health values
_MIN_EDIBLE_FRACTION = 0.25


@BlockRegistry.register(
    "PierreShopSamplerBlock", "transform",
    "Sample random items, quantities, and prices for Pierre's shop",
)
class PierreShopSamplerBlock(BaseBlock):
    descriptions_path: str = str(_DESCRIPTIONS_PATH)
    manifest_path: str = str(_MANIFEST_PATH)
    backgrounds_dir: str = str(_BACKGROUNDS_DIR)

    def generate(self, samples: pd.DataFrame, **kwargs) -> pd.DataFrame:
        with open(self.descriptions_path) as f:
            items = json.load(f)

        with open(self.manifest_path) as f:
            manifest = json.load(f)

        # Build item lists split by edibility
        edible_items = []
        non_edible_items = []
        for item_id, item in items.items():
            if item.get("price", 0) <= 0 or not item.get("description"):
                continue
            edibility = manifest.get(item_id, {}).get("edibility", -300)
            entry = dict(item)
            if edibility > 0:
                entry["energy"] = f"+{int(edibility * 2.5)}"
                entry["health"] = f"+{int(edibility * 1.125)}"
                edible_items.append(entry)
            else:
                entry["energy"] = ""
                entry["health"] = ""
                non_edible_items.append(entry)

        # Collect backgrounds
        bg_dir = Path(self.backgrounds_dir)
        backgrounds = list(bg_dir.glob("*.PNG")) + list(bg_dir.glob("*.jpg"))

        # Build quantity distribution
        quantities, weights = zip(*_QUANTITY_WEIGHTS)

        # Ensure at least 25% of samples are edible items
        n = len(samples)
        min_edible = int(n * _MIN_EDIBLE_FRACTION)

        results = []
        for i in range(n):
            # Force edible items for the first min_edible samples
            if i < min_edible:
                item = random.choice(edible_items)
            else:
                item = random.choice(edible_items + non_edible_items)

            quantity = random.choices(quantities, weights=weights, k=1)[0]
            price = item["price"]
            total = price * quantity

            bg = random.choice(backgrounds) if backgrounds else ""

            results.append({
                "item_name": item["name"],
                "description": item["description"],
                "price_per_unit": price,
                "quantity_selected": quantity,
                "total_cost": total,
                "energy": item.get("energy", ""),
                "health": item.get("health", ""),
                "background_path": str(bg),
            })

        result_df = pd.DataFrame(results)
        return pd.concat([samples.reset_index(drop=True), result_df], axis=1)
