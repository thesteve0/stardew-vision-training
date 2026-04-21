"""
TVDialogSamplerBlock: Sample dialog text from the TV dialog corpus.

Randomly selects from the game's actual TV dialog strings across all
four show types: weather forecasts, fortune teller, Livin' Off The
Land tips, and Queen of Sauce recipes.

No LLM needed — all text comes from the game data.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
from sdg_hub import BaseBlock, BlockRegistry

_CORPUS_PATH = Path("datasets/assets/tv_dialog_corpus.json")
_BACKGROUNDS_DIR = Path("datasets/tv_dialog/backgrounds")

# Max chars per dialog page (based on real screenshot analysis: max 194 chars)
_MAX_PAGE_CHARS = 200


def _truncate_to_page(text: str, max_chars: int = _MAX_PAGE_CHARS) -> str:
    """Truncate text to fit a single dialog page.

    Cuts at the last sentence boundary (. ! ?) before max_chars.
    If no sentence boundary found, cuts at the last space.
    """
    if len(text) <= max_chars:
        return text

    # Find the last sentence-ending punctuation before max_chars
    truncated = text[:max_chars]
    for end_char in [". ", "! ", "? "]:
        last_pos = truncated.rfind(end_char)
        if last_pos > max_chars * 0.4:  # don't cut too short
            return truncated[: last_pos + 1].rstrip()

    # No sentence boundary — cut at last space
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.4:
        return truncated[:last_space].rstrip() + "..."

    return truncated.rstrip() + "..."


@BlockRegistry.register(
    "TVDialogSamplerBlock", "transform",
    "Sample TV dialog text from the game corpus and assign backgrounds",
)
class TVDialogSamplerBlock(BaseBlock):
    corpus_path: str = str(_CORPUS_PATH)
    backgrounds_dir: str = str(_BACKGROUNDS_DIR)

    def generate(self, samples: pd.DataFrame, **kwargs) -> pd.DataFrame:
        with open(self.corpus_path) as f:
            corpus = json.load(f)

        # Group dialogs by show type for even distribution
        show_types = ["weather_forecasts", "fortune_teller",
                      "livin_off_the_land", "queen_of_sauce"]
        dialogs_by_type = {}
        for show_type in show_types:
            texts = corpus.get(show_type, [])
            if texts:
                dialogs_by_type[show_type] = texts

        available_types = list(dialogs_by_type.keys())

        # Collect available backgrounds
        bg_dir = Path(self.backgrounds_dir)
        backgrounds = list(bg_dir.glob("*.png"))
        if not backgrounds:
            raise FileNotFoundError(f"No backgrounds found in {bg_dir}")

        results = []
        for i in range(len(samples)):
            # Pick show type evenly (round-robin)
            show_type = available_types[i % len(available_types)]
            dialog_text = random.choice(dialogs_by_type[show_type])
            # The game breaks long text across multiple dialog pages.
            # Cap at ~200 chars, truncating at a sentence boundary.
            dialog_text = _truncate_to_page(dialog_text, max_chars=200)
            bg = random.choice(backgrounds)
            results.append({
                "show_type": show_type,
                "dialog_text": dialog_text,
                "background_path": str(bg),
            })

        result_df = pd.DataFrame(results)
        return pd.concat([samples.reset_index(drop=True), result_df], axis=1)
