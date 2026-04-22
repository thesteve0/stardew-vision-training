#!/usr/bin/env python3
"""Tool definitions and message construction for Qwen2.5-VL evaluation.

The Qwen2.5-VL chat template does NOT handle tools (unlike the non-VL
Qwen2.5 template). We bake tool definitions directly into the system
prompt using the same format the non-VL template uses: tools in <tools>
XML tags, with instructions to respond using <tool_call> tags.
"""

import json

from PIL import Image

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "crop_tv_dialog",
            "description": "Extract TV show type and dialog text from a TV screen",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crop_caught_fish_notification",
            "description": (
                "Extract fish name and length from a caught fish notification"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crop_pierres_detail_panel",
            "description": (
                "Extract item name, description, price, and quantity "
                "from Pierre's shop detail panel"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

EXTRACTION_TOOLS = {
    "crop_tv_dialog",
    "crop_caught_fish_notification",
    "crop_pierres_detail_panel",
}

NO_TOOL_RESPONSE = "I don't have a tool to handle that screen"

_TOOLS_JSON = "\n".join(json.dumps(t) for t in TOOL_DEFINITIONS)

SYSTEM_PROMPT = (
    "You are a Stardew Valley accessibility assistant. "
    "Analyze the screenshot and call the appropriate extraction tool. "
    "If no extraction tool matches the screen, respond with exactly: "
    f'"{NO_TOOL_RESPONSE}"'
    "\n\n# Tools\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    f"{_TOOLS_JSON}\n"
    "</tools>\n\n"
    "For each function call, return a json object with function name and "
    "arguments within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>\n\n"
    "If no tool matches the screen, respond with only: "
    f"{NO_TOOL_RESPONSE}"
)


def build_messages(image: Image.Image) -> list[dict]:
    """Build the chat messages list for a single screenshot evaluation."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What's on this screen?"},
            ],
        },
    ]