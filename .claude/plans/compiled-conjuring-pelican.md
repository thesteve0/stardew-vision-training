# Plan: Synthetic Data Generation with SDG Hub

## Context

We have ~120 real annotated screenshots across 3 screen types (tv_dialog: 30, caught_fish: 33, pierre_shop: 26) but need ~150 synthetic examples per type (~450 total) for VLM fine-tuning. The user wants to generate simulated screenshots by rendering game text onto real screenshot backgrounds using Stardew Valley's bitmap fonts, orchestrated through Red Hat's SDG Hub framework with a local teacher model.

**OCR fine-tuning is NOT a blocker.** The synthetic data trains the VLM (Qwen2.5-VL), not the OCR engine. Training uses ground-truth text fields. PaddleOCR fine-tuning is a separate, independent workstream.

## Architecture Overview

```
Game Assets (fonts, sprites, text corpus, item manifest)
        ↓
SDG Hub Pipeline (YAML-defined flows with custom blocks)
        ↓
┌──────────────────────────────────────────────────────┐
│  Seed Data → Text Variation → Screenshot Render → ChatML │
│  (manifest,    (local LLM     (bitmap font      (4-turn   │
│   corpus)      or sampling)    compositor)       format)   │
└──────────────────────────────────────────────────────┘
        ↓
datasets/{screen_type}/conversations_synthetic.jsonl
        ↓
Train/Eval/Test splits → Qwen2.5-VL LoRA fine-tuning
```

## Prerequisites (Before Pipeline)

### P1: Extract TV dialog text from game files

The user needs to extract TV text data using SMAPI on the game machine:
```
patch export Data/TV/TipChannel
patch export Data/TV/CookingChannel
patch export Strings/StringsFromCSFiles
```

Create `scripts/extract_tv_dialog_text.py` to parse the exports into:
```json
// datasets/assets/tv_dialog_corpus.json
{
  "weather_forecasts": ["It's going to be a beautiful, sunny day tomorrow!", ...],
  "fortune_teller": ["The spirits feel neutral today...", ...],
  "livin_off_the_land": ["This one's for all you greenhorns...", ...],
  "queen_of_sauce": ["Today we're going to learn how to make Fried Egg...", ...]
}
```

**Fallback**: If game files are unavailable, the LLM teacher can generate variations from the 14 unique dialog texts already in our annotations.

### P2: Resolve item descriptions

The manifest has unresolved localization keys (`[LocalizedText ...]`). Create `scripts/resolve_item_descriptions.py` that:
1. Uses the 26 real `pierre_shop` annotations as known description pairs
2. For remaining items, uses the LLM teacher to generate plausible descriptions
3. Outputs `datasets/assets/item_descriptions_resolved.json`

**Alternative**: Extract `Strings/Objects` from the game via SMAPI (same approach as P1).

### P3: Install SDG Hub

```bash
uv add sdg-hub
```

## Implementation Steps

### Step 1: Bitmap Font Renderer

**File**: `synthetic_data/font_renderer.py`

Parse XNA SpriteFont JSON (SmallFont.json, SpriteFont1.json) into a renderer:

- **Character lookup**: `characterMap[i]` → `glyphs[i]` (x,y,w,h rectangle in sprite sheet) + `kerning[i]` (x=left padding, y=char width, z=right padding)
- **Advance width**: `kern.x + kern.y + kern.z + horizontalSpacing`
- **Line height**: `verticalLineSpacing` (33 for SmallFont, 50 for SpriteFont1)
- **Color**: Font atlas is white-on-transparent. Apply target color by setting RGB to target and preserving alpha channel.

Key functions:
```python
class SpriteFont:
    def measure_text(text: str) -> tuple[int, int]
    def render_text(text: str, color=(86, 22, 12), scale=1) -> Image
    def wrap_text(text: str, max_width_px: int) -> str
```

Font selection by screen type:
- `tv_dialog` → SpriteFont1 (larger dialog font)
- `pierre_shop` → SmallFont (smaller UI font)
- `caught_fish` → SpriteFont1 (notification text)

### Step 2: Screenshot Compositor

**File**: `synthetic_data/compositor.py`

Template-based approach: take a real screenshot, clear the text region, render new text.

**tv_dialog compositor:**
1. Pick random real screenshot as background donor
2. Crop dialog box region using `tv_dialog_layout.json` (x=0.1025, y=0.71, w=0.68, h=0.28)
3. Sample background color from dialog box edges (tan/wooden)
4. Fill text area with background color
5. Render new dialog text with SpriteFont1 in dark brown, word-wrapped to fit
6. Paste rendered text with game-appropriate padding

**caught_fish compositor:**
1. Pick random real caught fish screenshot
2. Clear length text area in notification region
3. Optionally swap fish sprite: scale 16x16 sprite from `sprites_game/` into the fish_sprite sub-region
4. Render new length text
5. Paste back

**pierre_shop compositor:**
1. Pick random real Pierre's shop screenshot
2. Clear text sub-regions in detail panel (name, description, price)
3. Render new item name (SpriteFont1), description (SmallFont), price line (SmallFont)
4. Paste back

### Step 3: SDG Hub Custom Blocks

**Directory**: `synthetic_data/blocks/`

| Block | Category | Purpose |
|---|---|---|
| `LoadAnnotationsBlock` | transform | Load real annotations + domain knowledge (manifest, corpus) as seed data |
| `CaughtFishSamplerBlock` | transform | Sample random fish + length from manifest (no LLM needed) |
| `PierreShopSamplerBlock` | transform | Sample random items + quantities from manifest |
| `RenderScreenshotBlock` | transform | Render synthetic screenshot using compositor |
| `BuildChatMLBlock` | transform | Package screenshot + fields into 4-turn ChatML conversation |

All blocks operate on pandas DataFrames. Image data flows as file paths, not pixel arrays.

For **tv_dialog text variation**, use SDG Hub's built-in `PromptBuilderBlock` + `LLMChatBlock` chain with a local teacher model.

For **caught_fish** and **pierre_shop**, no LLM is needed for field variation - the manifest provides all fish names/items/prices, and lengths/quantities are randomly sampled.

### Step 4: YAML Flow Definitions

**Directory**: `synthetic_data/flows/`

One flow per screen type. Example for `caught_fish` (simplest - no LLM):

```yaml
blocks:
  - block_type: "LoadAnnotationsBlock"
    block_config:
      screen_type: "caught_fish"
      annotations_path: "datasets/caught_fish/annotations.jsonl"
      manifest_path: "datasets/assets/item_manifest_game.json"

  - block_type: "CaughtFishSamplerBlock"
    block_config:
      manifest_path: "datasets/assets/item_manifest_game.json"

  - block_type: "RenderScreenshotBlock"
    block_config:
      screen_type: "caught_fish"
      images_dir: "datasets/caught_fish/images"
      layout_path: "datasets/assets/templates/caught_fish_layout.json"
      font_dir: "datasets/assets/game_files/unpacked"
      output_dir: "datasets/caught_fish/synthetic_images"

  - block_type: "BuildChatMLBlock"
    block_config:
      screen_type: "caught_fish"
```

### Step 5: Pipeline Runner + Local LLM Setup

**File**: `synthetic_data/run_pipeline.py`

```bash
# Start local teacher model
ollama run llama3.1:8b

# Run pipeline
python synthetic_data/run_pipeline.py --screen-type caught_fish --num-variations 150
python synthetic_data/run_pipeline.py --screen-type tv_dialog --num-variations 150
python synthetic_data/run_pipeline.py --screen-type pierre_shop --num-variations 150
```

Teacher model config in `synthetic_data/config.yaml`:
```yaml
teacher_model:
  provider: "ollama"
  model: "ollama/llama3.1:8b"
  api_base: "http://localhost:11434/v1"
```

### Step 6: Validation + Training Data Assembly

**Validation** (`synthetic_data/validate_screenshots.py`):
- Visual: display grid of 10 random synthetic screenshots per type
- Format: verify 4-turn ChatML structure, correct tool names, non-empty fields
- Diversity: unique text count, fish species coverage, item coverage

**Merge** (`scripts/data_prep/merge_training_data.py`):
- Combine real annotations (converted to ChatML) + synthetic conversations
- Split 80/10/10 train/eval/test, stratified by screen type
- Output: `datasets/splits/{train,eval,test}.jsonl`

## File Organization

```
synthetic_data/
  run_pipeline.py              # Main runner
  config.yaml                  # Teacher model config
  font_renderer.py             # XNA SpriteFont bitmap renderer
  compositor.py                # Screenshot compositor (per screen type)
  validate_screenshots.py      # Visual + format validation
  blocks/
    __init__.py
    load_annotations.py        # LoadAnnotationsBlock
    caught_fish_sampler.py     # CaughtFishSamplerBlock
    pierre_shop_sampler.py     # PierreShopSamplerBlock
    render_screenshot.py       # RenderScreenshotBlock
    build_chatml.py            # BuildChatMLBlock
  flows/
    tv_dialog_synthetic.yaml
    caught_fish_synthetic.yaml
    pierre_shop_synthetic.yaml
  prompts/
    tv_dialog_variation.yaml   # LLM prompt template for dialog text
scripts/
  extract_tv_dialog_text.py    # Game data extraction (prerequisite)
  resolve_item_descriptions.py # Description resolution (prerequisite)
  data_prep/
    merge_training_data.py     # Combine real + synthetic → splits
```

## Implementation Sequence

| Phase | Work | LLM Required? |
|---|---|---|
| **Phase 1**: Font renderer | `font_renderer.py` with SmallFont + SpriteFont1 | No |
| **Phase 2**: Compositor | `compositor.py` for tv_dialog first, then caught_fish + pierre_shop | No |
| **Phase 3**: SDG Hub blocks | Custom blocks wrapping font renderer + compositor | No |
| **Phase 4**: Flows + prompts | YAML flows, LLM prompt templates, runner script | Setup only |
| **Phase 5**: Generate | Run caught_fish first (no LLM), then tv_dialog + pierre_shop | Yes (local) |
| **Phase 6**: Validate + merge | Validation scripts, training data assembly | No |

Start with **caught_fish** (no LLM needed) to validate the full pipeline end-to-end before adding LLM complexity for tv_dialog.

## Verification

1. Render "Hello World" with SmallFont — verify pixel dimensions match expected (sum of glyph widths + kerning)
2. Generate 1 synthetic caught_fish screenshot — visually confirm text placement and fish sprite
3. Dry-run caught_fish flow with `sample_size=2` — verify ChatML output format
4. Generate 150 caught_fish examples — check diversity (fish species coverage)
5. Generate 150 tv_dialog examples — check LLM output quality
6. Generate 150 pierre_shop examples — verify price×quantity=total constraint
7. Merge all + split — verify final train/eval/test counts and stratification
