#!/usr/bin/env python3
"""
Parse TV dialog text from SMAPI-exported game files into a clean corpus.

Reads TipChannel.json, CookingChannel.json, and StringsFromCSFiles.json
and outputs a unified tv_dialog_corpus.json with all possible dialog text
organized by show type.

Usage:
    python scripts/parse_tv_dialog_corpus.py
"""

import json
from pathlib import Path

TV_DIR = Path("datasets/assets/game_files/tv")
OUTPUT = Path("datasets/assets/tv_dialog_corpus.json")


def parse_weather_forecasts(strings: dict) -> list[str]:
    """Extract weather forecast dialog strings."""
    forecasts = []
    # Weather report intro
    intro_key = "TV.cs.13136"
    if intro_key in strings:
        # The intro has "^" as a line break marker in the game
        intro = strings[intro_key].replace("^", "\n")
        forecasts.append(intro)

    # Individual weather forecasts
    weather_keys = [
        "TV.cs.13180",  # Bundle up, snow
        "TV.cs.13181",  # Expect snow
        "TV.cs.13182",  # Beautiful sunny day
        "TV.cs.13183",  # Clear and sunny
        "TV.cs.13184",  # Rain all day
        "TV.cs.13185",  # Storm approaching
        "TV.cs.13187",  # Partially cloudy
        "TV.cs.13189",  # Cloudy with wind
        "TV.cs.13190",  # Snow all day
    ]
    for key in weather_keys:
        if key in strings:
            forecasts.append(strings[key])

    return forecasts


def parse_fortune_teller(strings: dict) -> list[str]:
    """Extract fortune teller dialog strings."""
    messages = []

    # Fortune teller intros
    intro_keys = [
        "TV.cs.13128",  # New viewer (male)
        "TV.cs.13130",  # New viewer (female)
        "TV.cs.13132",  # Spirits whispering
        "TV.cs.13133",  # Welcome back to Welwick's Oracle
        "TV.cs.13134",  # Glimmer in scrying orb
        "TV.cs.13135",  # Welcome to Welwick's Oracle
    ]
    for key in intro_keys:
        if key in strings:
            messages.append(strings[key])

    # Fortune results
    fortune_keys = [
        "TV.cs.13191",  # Spirits furious
        "TV.cs.13192",  # Spirits very displeased
        "TV.cs.13193",  # Spirits somewhat annoyed
        "TV.cs.13195",  # Spirits somewhat mildly perturbed
        "TV.cs.13197",  # Spirits joyous
        "TV.cs.13198",  # Spirits very happy
        "TV.cs.13199",  # Spirits in good humor
        "TV.cs.13200",  # Spirits feel neutral
        "TV.cs.13201",  # Spirits absolutely neutral
    ]
    for key in fortune_keys:
        if key in strings:
            messages.append(strings[key])

    return messages


def parse_livin_off_the_land(tips_data: dict) -> list[str]:
    """Extract Livin' Off The Land tip texts."""
    tips = []
    for key in sorted(tips_data.keys(), key=lambda x: int(x)):
        text = tips_data[key]
        # Tips may have "/" separating multi-page dialog
        # Split on "/" and take meaningful parts
        parts = text.split("/")
        for part in parts:
            part = part.strip()
            if part and len(part) > 10:
                tips.append(part)
    return tips


def parse_queen_of_sauce(cooking_data: dict) -> list[str]:
    """Extract Queen of Sauce recipe dialog texts."""
    recipes = []
    for key in sorted(cooking_data.keys(), key=lambda x: int(x)):
        text = cooking_data[key]
        # Format: "RecipeName/Dialog text about the recipe"
        parts = text.split("/")
        if len(parts) >= 2:
            recipe_name = parts[0].strip()
            dialog = parts[1].strip()
            if dialog:
                recipes.append(dialog)
    return recipes


def main():
    # Load game files
    with open(TV_DIR / "StringsFromCSFiles.json") as f:
        strings = json.load(f)["content"]

    with open(TV_DIR / "TipChannel.json") as f:
        tips_data = json.load(f)["content"]

    with open(TV_DIR / "CookingChannel.json") as f:
        cooking_data = json.load(f)["content"]

    # Parse each show type
    corpus = {
        "weather_forecasts": parse_weather_forecasts(strings),
        "fortune_teller": parse_fortune_teller(strings),
        "livin_off_the_land": parse_livin_off_the_land(tips_data),
        "queen_of_sauce": parse_queen_of_sauce(cooking_data),
        "show_intros": {
            "weather": strings.get("TV.cs.13136", "").replace("^", "\n"),
            "livin_off_the_land": strings.get("TV.cs.13124", ""),
            "queen_of_sauce": strings.get("TV.cs.13127", ""),
        },
    }

    # Summary
    print("TV Dialog Corpus:")
    for category, items in corpus.items():
        if isinstance(items, list):
            print(f"  {category}: {len(items)} entries")
        elif isinstance(items, dict):
            print(f"  {category}: {len(items)} entries")

    # Save
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(corpus, f, indent=2)
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
