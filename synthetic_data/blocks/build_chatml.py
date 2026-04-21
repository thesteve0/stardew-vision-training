"""
BuildChatMLBlock: Package synthetic data into ChatML training conversations.

Creates the 4-turn multi-turn conversation format expected by the
Qwen2.5-VL fine-tuning pipeline:
  1. User: image + "What's on this screen?"
  2. Assistant: tool_call to crop_{screen_type}
  3. Tool: extraction JSON response
  4. Assistant: tool_call to text_to_speech with narration
"""

from __future__ import annotations

import json

import pandas as pd
from sdg_hub import BaseBlock, BlockRegistry

# Tool names per screen type
_TOOL_NAMES = {
    "tv_dialog": "crop_tv_dialog",
    "caught_fish": "crop_caught_fish_notification",
    "pierre_shop": "crop_pierres_detail_panel",
}


def _build_extraction(screen_type: str, row: pd.Series) -> dict:
    """Build the expected_extraction JSON from the row data."""
    if screen_type == "tv_dialog":
        return {
            "screen_type": "tv_dialog",
            "dialog_text": row["dialog_text"],
        }
    elif screen_type == "caught_fish":
        length = row.get("length_inches")
        return {
            "screen_type": "caught_fish",
            "fish_name": row["fish_name"],
            "length_inches": int(length) if pd.notna(length) else None,
        }
    elif screen_type == "pierre_shop":
        return {
            "screen_type": "pierre_shop",
            "name": row["item_name"],
            "description": row["description"],
            "price_per_unit": int(row["price_per_unit"]),
            "quantity_selected": int(row.get("quantity_selected", 1)),
            "total_cost": int(row.get("total_cost", row["price_per_unit"])),
        }
    return {}


def _build_narration(screen_type: str, row: pd.Series) -> str:
    """Generate the natural language narration from the extraction data."""
    if screen_type == "tv_dialog":
        show = row.get("show_type", "tv")
        return f"TV {show}: {row['dialog_text']}"
    elif screen_type == "caught_fish":
        name = row["fish_name"]
        length = row.get("length_inches")
        if pd.notna(length) and int(length) > 0:
            return f"You caught a {name}! It's {int(length)} inches long."
        return f"You caught {name}!"
    elif screen_type == "pierre_shop":
        name = row["item_name"]
        desc = row["description"]
        price = int(row["price_per_unit"])
        qty = int(row.get("quantity_selected", 1))
        total = int(row.get("total_cost", price))
        text = f"Pierre's shop: {name}. {desc}. {price}g each"
        if qty > 1:
            text += f", buying {qty} for {total}g total."
        else:
            text += "."
        return text
    return ""


@BlockRegistry.register(
    "BuildChatMLBlock", "transform",
    "Build ChatML multi-turn conversations from synthetic data",
)
class BuildChatMLBlock(BaseBlock):
    screen_type: str

    def generate(self, samples: pd.DataFrame, **kwargs) -> pd.DataFrame:
        tool_name = _TOOL_NAMES.get(self.screen_type, f"crop_{self.screen_type}")

        conversations = []
        for _, row in samples.iterrows():
            image_path = row.get("synthetic_image_path", "")
            extraction = _build_extraction(self.screen_type, row)
            narration = _build_narration(self.screen_type, row)

            conversation = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"file://{image_path}"},
                            {"type": "text", "text": "What's on this screen?"},
                        ],
                    },
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(
                                        {"image_b64": "..."}
                                    ),
                                }
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "name": tool_name,
                        "content": json.dumps(extraction),
                    },
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "text_to_speech",
                                    "arguments": json.dumps({"text": narration}),
                                }
                            }
                        ],
                    },
                ],
                "metadata": {
                    "screen_type": self.screen_type,
                    "synthetic": True,
                },
            }
            conversations.append(json.dumps(conversation))

        samples = samples.copy()
        samples["conversation"] = conversations
        return samples
