"""
RenderScreenshotBlock: Render synthetic screenshots using the compositor.

Takes sampled text/field data and a background image path, renders the
appropriate screen type using the bitmap font compositor, and saves
the result to disk.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
from sdg_hub import BaseBlock, BlockRegistry

from synthetic_data.compositor import (
    composite_caught_fish,
    composite_pierre_shop,
    composite_tv_dialog,
)


@BlockRegistry.register(
    "RenderScreenshotBlock", "transform",
    "Render synthetic screenshots using bitmap font compositor",
)
class RenderScreenshotBlock(BaseBlock):
    screen_type: str
    output_dir: str

    def generate(self, samples: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        image_paths = []
        for _, row in samples.iterrows():
            img_id = str(uuid.uuid4())[:8]
            out_path = out_dir / f"synth_{self.screen_type}_{img_id}.png"

            if self.screen_type == "tv_dialog":
                img = composite_tv_dialog(
                    row["background_path"],
                    row["dialog_text"],
                )
            elif self.screen_type == "caught_fish":
                length = row.get("length_inches")
                length = int(length) if pd.notna(length) else None
                sprite = row.get("sprite_path", None)
                if length is not None:
                    img = composite_caught_fish(
                        row["background_path"],
                        length_inches=length,
                        fish_sprite_path=sprite,
                    )
                else:
                    # Trash items — render without length
                    img = composite_caught_fish(
                        row["background_path"],
                        length_inches=0,
                        fish_sprite_path=sprite,
                    )
            elif self.screen_type == "pierre_shop":
                img = composite_pierre_shop(
                    row["background_path"],
                    item_name=row["item_name"],
                    description=row["description"],
                    price_per_unit=int(row["price_per_unit"]),
                    quantity=int(row.get("quantity_selected", 1)),
                    total_cost=int(row.get("total_cost", row["price_per_unit"])),
                )
            else:
                raise ValueError(f"Unknown screen type: {self.screen_type}")

            img.save(out_path)
            image_paths.append(str(out_path))

        samples = samples.copy()
        samples["synthetic_image_path"] = image_paths
        return samples
