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
            bg = random.choice(backgrounds)
            results.append({
                "show_type": show_type,
                "dialog_text": dialog_text,
                "background_path": str(bg),
            })

        result_df = pd.DataFrame(results)
        return pd.concat([samples.reset_index(drop=True), result_df], axis=1)
