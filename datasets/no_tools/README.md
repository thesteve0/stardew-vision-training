# No Tools Screen Dataset

## Overview

This dataset contains screenshots of Stardew Valley screens for which no extraction tool exists. These examples train the model to recognize when it should NOT call a tool and instead respond with a message indicating no tool is available for the current screen.

## Purpose

A robust VLM must distinguish between screens it can handle (pierre_shop, tv_dialog, caught_fish) and screens it cannot. Without negative examples, the model may hallucinate tool calls for unsupported screens.

**Expected model behavior:**
- Classify the screen as unsupported
- Return a message like: "I don't have a tool for this screen type yet."
- Do NOT call any extraction tool

## Screen Types to Include

Any screen that is NOT one of the supported types:
- Inventory / backpack
- Map screen
- Calendar
- Relationships / social tab
- Crafting menu
- Shipping bin summary
- Dialogue with NPCs (non-TV)
- Combat / mine screens
- Fishing minigame (the active bobber game, not the caught-fish result)
- Community center bundles
- Any other miscellaneous screen

## Annotation Schema

Each screenshot is annotated with:
- `image_id`: UUID
- `image_path`: Relative path to screenshot
- `screen_type`: "no_tools"
- `tool_call`: `null` (no tool should be called)
- `ocr_fields`: `{}` (no extraction performed)
- `narration`: Expected natural language response explaining no tool is available

## Collection Guidelines

**Screenshot criteria:**
- Capture a variety of unsupported screen types
- Ensure screenshots are clear and representative
- Include screens that could be visually confused with supported types

**Diversity targets:**
- Menu screens: 3-5 examples (inventory, crafting, social)
- Dialog screens: 2-3 examples (NPC conversations, not TV)
- Gameplay screens: 3-5 examples (combat, fishing minigame, map)
- Summary screens: 2-3 examples (shipping, end of day)

## Current Status

**Images collected**: 0/15
**Images annotated**: 0
**Quality check**: Not started
**Priority**: Important for model robustness — prevents hallucinated tool calls
