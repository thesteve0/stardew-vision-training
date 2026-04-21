#!/usr/bin/env python3
"""
Parse resolved item descriptions from SMAPI-exported Objects strings.

Reads Objects.json (localized strings) and the item manifest to produce
a clean mapping of item_id -> {name, description} for Pierre's shop
synthetic data generation.

Usage:
    python scripts/parse_item_descriptions.py
"""

import json
import re
from pathlib import Path

STRINGS_PATH = Path("datasets/assets/game_files/strings/Objects.json")
MANIFEST_PATH = Path("datasets/assets/item_manifest_game.json")
OUTPUT = Path("datasets/assets/item_descriptions_resolved.json")


def main():
    # Load Objects strings
    with open(STRINGS_PATH) as f:
        strings = json.load(f)["content"]

    # Build name -> description lookup from the strings file
    # Keys are like "Parsnip_Name", "Parsnip_Description"
    name_to_desc = {}
    for key, value in strings.items():
        if key.endswith("_Name"):
            base = key[: -len("_Name")]
            desc_key = base + "_Description"
            desc = strings.get(desc_key, "")
            if desc and desc != "?":
                name_to_desc[value] = desc

    # Load the item manifest to get item IDs and match with descriptions
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    # Match manifest items to resolved descriptions
    resolved = {}
    matched = 0
    unmatched = 0

    for item_id, item in manifest.items():
        raw_name = item.get("name", "")

        # Extract the internal name from localization key
        # e.g. "[LocalizedText Strings\Objects:Parsnip_Name]" -> "Parsnip"
        m = re.search(r":(\w+?)_Name\]", raw_name)
        if m:
            internal_name = m.group(1)
            # Convert CamelCase to lookup key
            # Try direct lookup in the strings
            name_key = internal_name + "_Name"
            desc_key = internal_name + "_Description"
            display_name = strings.get(name_key, "")
            description = strings.get(desc_key, "")
        else:
            # Plain name (some items have non-localized names)
            display_name = raw_name
            description = ""
            # Try matching by name
            if raw_name in name_to_desc:
                description = name_to_desc[raw_name]

        if display_name and description and description != "?":
            resolved[item_id] = {
                "name": display_name,
                "description": description,
                "type": item.get("type", ""),
                "price": item.get("price", 0),
            }
            matched += 1
        else:
            unmatched += 1

    print(f"Resolved: {matched} items with descriptions")
    print(f"Unmatched: {unmatched} items (no description found)")

    # Show some examples
    print("\nSample resolved items:")
    for item_id in list(resolved.keys())[:10]:
        r = resolved[item_id]
        print(f"  {item_id}: {r['name']} - {r['description'][:60]}...")

    # Save
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(resolved, f, indent=2)
    print(f"\nSaved {len(resolved)} items to {OUTPUT}")


if __name__ == "__main__":
    main()
