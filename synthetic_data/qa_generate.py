#!/usr/bin/env python3
"""
Generate QA samples with human-readable chat files.

Produces synthetic screenshots + readable chat dialog files for
manual quality review. One .txt chat file per image.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from synthetic_data.compositor import (
    composite_caught_fish,
    composite_pierre_shop,
    composite_tv_dialog,
)


def write_chat_file(path: Path, turns: list[dict]):
    """Write a human-readable chat dialog file."""
    with open(path, "w") as f:
        for turn in turns:
            role = turn["role"].upper()
            f.write(f"{'='*60}\n")
            f.write(f"[{role}]\n")
            f.write(f"{'='*60}\n\n")

            if role == "USER":
                for item in turn.get("content", []):
                    if item["type"] == "image":
                        f.write(f"  Image: {item['image']}\n\n")
                    elif item["type"] == "text":
                        f.write(f"  {item['text']}\n\n")

            elif role == "ASSISTANT":
                for tc in turn.get("tool_calls", []):
                    fn = tc["function"]
                    f.write(f"  Tool call: {fn['name']}\n")
                    # Pretty-print arguments
                    try:
                        args = json.loads(fn["arguments"])
                        for k, v in args.items():
                            val = str(v)
                            if len(val) > 80:
                                val = val[:80] + "..."
                            f.write(f"    {k}: {val}\n")
                    except (json.JSONDecodeError, TypeError):
                        f.write(f"    {fn['arguments']}\n")
                    f.write("\n")

            elif role == "TOOL":
                f.write(f"  Tool: {turn.get('name', '?')}\n")
                try:
                    content = json.loads(turn["content"])
                    f.write(f"  Response:\n")
                    for k, v in content.items():
                        val = str(v)
                        if len(val) > 100:
                            val = val[:100] + "..."
                        f.write(f"    {k}: {val}\n")
                except (json.JSONDecodeError, TypeError):
                    f.write(f"  {turn.get('content', '')}\n")
                f.write("\n")

            f.write("\n")


def generate_tv_dialog(n: int, out_dir: Path):
    """Generate TV dialog QA samples."""
    corpus_path = Path("datasets/assets/tv_dialog_corpus.json")
    bg_dir = Path("datasets/tv_dialog/backgrounds")

    with open(corpus_path) as f:
        corpus = json.load(f)

    show_types = ["weather_forecasts", "fortune_teller",
                  "livin_off_the_land", "queen_of_sauce"]
    dialogs_by_type = {st: corpus.get(st, []) for st in show_types if corpus.get(st)}
    available_types = list(dialogs_by_type.keys())

    backgrounds = list(bg_dir.glob("*.png"))

    for i in range(n):
        # Even distribution across show types (round-robin)
        show_type = available_types[i % len(available_types)]
        dialog_text = random.choice(dialogs_by_type[show_type])
        # Cap at ~200 chars (game breaks long text across multiple pages)
        from synthetic_data.blocks.tv_dialog_sampler import _truncate_to_page
        dialog_text = _truncate_to_page(dialog_text)
        bg = random.choice(backgrounds)

        img = composite_tv_dialog(str(bg), dialog_text)
        img_path = out_dir / f"tv_dialog_{i+1:02d}.png"
        img.save(img_path)

        narration = f"TV {show_type}: {dialog_text}"
        turns = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(img_path)},
                    {"type": "text", "text": "What's on this screen?"},
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [{"function": {
                    "name": "crop_tv_dialog",
                    "arguments": json.dumps({"image_b64": "..."}),
                }}],
            },
            {
                "role": "tool",
                "name": "crop_tv_dialog",
                "content": json.dumps({
                    "screen_type": "tv_dialog",
                    "dialog_text": dialog_text,
                }),
            },
            {
                "role": "assistant",
                "tool_calls": [{"function": {
                    "name": "text_to_speech",
                    "arguments": json.dumps({"text": narration}),
                }}],
            },
        ]

        chat_path = out_dir / f"tv_dialog_{i+1:02d}.txt"
        write_chat_file(chat_path, turns)
        print(f"  {img_path.name}: [{show_type}] {dialog_text[:50]}...")


def generate_caught_fish(n: int, out_dir: Path):
    """Generate caught fish QA samples."""
    manifest_path = Path("datasets/assets/item_manifest_game.json")
    sprites_dir = Path("datasets/assets/sprites_game")
    images_dir = Path("datasets/caught_fish/backgrounds")

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Items that show notification WITHOUT length
    no_length_ids = {
        "167", "168", "169", "170", "171", "172",  # Joja Cola, Trash, Driftwood, etc.
    }
    # Items that NEVER show the caught fish notification (exclude entirely)
    exclude_ids = {
        "152", "153", "157",                    # Seaweed, Green Algae, White Algae
        "SeaJelly", "CaveJelly", "RiverJelly",  # Jellies
    }

    # Split into fish (with length) and junk (no length)
    real_fish = []
    junk_items = []
    for item_id, item in manifest.items():
        if item.get("type") != "Fish":
            continue
        if item_id in exclude_ids:
            continue
        if item_id in no_length_ids:
            junk_items.append((item_id, item))
        else:
            real_fish.append((item_id, item))

    backgrounds = list(images_dir.glob("*.png")) + list(images_dir.glob("*.PNG")) + list(images_dir.glob("*.jpg"))
    backgrounds = [b for b in backgrounds if b.name != "positions.json"]

    # Ensure at least 10% are non-fish catchables
    min_junk = max(1, int(n * 0.10))

    for i in range(n):
        if i < min_junk:
            # Force junk/non-fish item
            item_id, item = random.choice(junk_items)
            length = None
        else:
            item_id, item = random.choice(real_fish + junk_items)
            length = None if item_id in no_length_ids else random.randint(5, 50)

        fish_name = item["name"]
        bg = random.choice(backgrounds)
        sprite_path = sprites_dir / f"sprite_{item_id}.png"

        img = composite_caught_fish(
            str(bg),
            length_inches=length,
            fish_sprite_path=str(sprite_path) if sprite_path.exists() else None,
        )
        img_path = out_dir / f"caught_fish_{i+1:02d}.png"
        img.save(img_path)

        if length:
            narration = f"You caught a {fish_name}! It's {length} inches long."
        else:
            narration = f"You caught {fish_name}!"

        extraction = {
            "screen_type": "caught_fish",
            "fish_name": fish_name,
            "length_inches": length,
        }

        turns = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(img_path)},
                    {"type": "text", "text": "What's on this screen?"},
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [{"function": {
                    "name": "crop_caught_fish_notification",
                    "arguments": json.dumps({"image_b64": "..."}),
                }}],
            },
            {
                "role": "tool",
                "name": "crop_caught_fish_notification",
                "content": json.dumps(extraction),
            },
            {
                "role": "assistant",
                "tool_calls": [{"function": {
                    "name": "text_to_speech",
                    "arguments": json.dumps({"text": narration}),
                }}],
            },
        ]

        chat_path = out_dir / f"caught_fish_{i+1:02d}.txt"
        write_chat_file(chat_path, turns)
        print(f"  {img_path.name}: {fish_name} ({length or 'no length'})")


def generate_pierre_shop(n: int, out_dir: Path):
    """Generate Pierre's shop QA samples.

    At least 25% of samples include energy/health values.
    """
    desc_path = Path("datasets/assets/item_descriptions_resolved.json")
    manifest_path = Path("datasets/assets/item_manifest_game.json")
    images_dir = Path("datasets/pierre_shop/images")

    with open(desc_path) as f:
        items = json.load(f)

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Split items by edibility
    edible_items = []
    non_edible_items = []
    for item_id, item in items.items():
        if item.get("price", 0) <= 0 or not item.get("description"):
            continue
        edibility = manifest.get(item_id, {}).get("edibility", -300)
        entry = dict(item)
        entry["item_id"] = item_id
        if edibility > 0:
            entry["energy"] = f"+{int(edibility * 2.5)}"
            entry["health"] = f"+{int(edibility * 1.125)}"
            edible_items.append(entry)
        else:
            entry["energy"] = ""
            entry["health"] = ""
            non_edible_items.append(entry)

    backgrounds = list(images_dir.glob("*.PNG")) + list(images_dir.glob("*.jpg"))
    qty_choices = [1, 1, 1, 5, 10, 25, 50, 100]
    min_edible = int(n * 0.25)

    for i in range(n):
        # Force edible items for the first 25%
        if i < min_edible:
            item = random.choice(edible_items)
        else:
            item = random.choice(edible_items + non_edible_items)

        bg = random.choice(backgrounds)
        quantity = random.choice(qty_choices)
        price = item["price"]
        total = price * quantity
        energy = item.get("energy", "")
        health = item.get("health", "")

        img = composite_pierre_shop(
            str(bg),
            item_name=item["name"],
            description=item["description"],
            price_per_unit=price,
            quantity=quantity,
            total_cost=total,
            energy=energy,
            health=health,
        )
        img_path = out_dir / f"pierre_shop_{i+1:02d}.png"
        img.save(img_path)

        narration = f"Pierre's shop: {item['name']}. {item['description']}. {price}g each"
        if quantity > 1:
            narration += f", buying {quantity} for {total}g total."
        else:
            narration += "."

        extraction = {
            "screen_type": "pierre_shop",
            "name": item["name"],
            "description": item["description"],
            "price_per_unit": price,
            "quantity_selected": quantity,
            "total_cost": total,
            "energy": energy,
            "health": health,
        }

        turns = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(img_path)},
                    {"type": "text", "text": "What's on this screen?"},
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [{"function": {
                    "name": "crop_pierres_detail_panel",
                    "arguments": json.dumps({"image_b64": "..."}),
                }}],
            },
            {
                "role": "tool",
                "name": "crop_pierres_detail_panel",
                "content": json.dumps(extraction),
            },
            {
                "role": "assistant",
                "tool_calls": [{"function": {
                    "name": "text_to_speech",
                    "arguments": json.dumps({"text": narration}),
                }}],
            },
        ]

        chat_path = out_dir / f"pierre_shop_{i+1:02d}.txt"
        write_chat_file(chat_path, turns)
        print(f"  {img_path.name}: {item['name']} x{quantity} = {total}g")


def generate_no_tools(n: int, out_dir: Path):
    """Generate no_tools QA samples.

    These use the original no_tools screenshots unchanged.
    The VLM should learn to NOT call any extraction tool for these.
    """
    images_dir = Path("datasets/no_tools/images")
    backgrounds = list(images_dir.glob("*.PNG")) + list(images_dir.glob("*.jpg"))

    if len(backgrounds) < n:
        # Repeat backgrounds if needed
        backgrounds = backgrounds * (n // len(backgrounds) + 1)

    selected = random.sample(backgrounds, n)

    for i, bg in enumerate(selected):
        # Copy the original screenshot to the QA output dir
        from PIL import Image
        img = Image.open(bg)
        img_path = out_dir / f"no_tools_{i+1:02d}.png"
        img.save(img_path)

        narration = "No tool available for this screen type."

        turns = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(img_path)},
                    {"type": "text", "text": "What's on this screen?"},
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [{"function": {
                    "name": "text_to_speech",
                    "arguments": json.dumps({
                        "text": narration,
                    }),
                }}],
            },
        ]

        chat_path = out_dir / f"no_tools_{i+1:02d}.txt"
        write_chat_file(chat_path, turns)
        print(f"  {img_path.name}: {bg.name}")


if __name__ == "__main__":
    n = 15

    print("=== TV Dialog ===")
    generate_tv_dialog(n, Path("tmp/qa_tv_dialog"))

    print("\n=== Caught Fish ===")
    generate_caught_fish(n, Path("tmp/qa_caught_fish"))

    print("\n=== Pierre's Shop ===")
    generate_pierre_shop(n, Path("tmp/qa_pierre_shop"))

    print("\n=== No Tools ===")
    generate_no_tools(n, Path("tmp/qa_no_tools"))

    print(f"\nDone. Generated {n * 4} image + chat pairs across 4 screen types.")
