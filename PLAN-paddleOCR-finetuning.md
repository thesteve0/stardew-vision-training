# Plan: Fine-Tune PaddleOCR Recognition Model on Stardew Valley Game Fonts

## Context

PaddleOCR (PP-OCRv5) is used across all extraction tools to read text from Stardew Valley screenshots. The user reports OCR errors on Pierre's shop, TV dialog, and other screen types. All game screens use the same bitmap fonts (SpriteFont1 for dialog/descriptions, SmallFont for UI quantities). The project has the original game font sprite sheets with full glyph metadata, enabling synthetic training data generation. Fine-tuning the recognition model on these exact fonts should improve accuracy across every tool with a single integration change.

**Goal**: Fine-tune PaddleOCR's recognition model, measure before/after accuracy with a formal rubric, and integrate the improved model so all tools benefit automatically.

---

## Phase 0: Baseline Measurement

**Create**: `evaluation/eval_ocr_accuracy.py`

Measure current OCR accuracy on all 89 annotated samples (30 tv_dialog, 26 pierre_shop, 33 caught_fish) before any changes.

For each sample:
1. Load image, apply crop regions using `common.py:crop_region()` with the appropriate layout JSON
2. Run `common.py:run_ocr()` on the cropped region (same params the tools use)
3. Compare OCR output to `expected_extraction` ground truth from `annotations.jsonl`

**Metrics computed** (see Evaluation Rubric below):
- CER (Character Error Rate) per text field
- WER (Word Error Rate) per text field
- Exact match rate for numeric fields and short text fields
- Aggregate per screen type and overall

**Per-screen-type comparison logic**:
- `tv_dialog`: concatenate OCR text blocks → compare to `expected_extraction.dialog_text`
- `pierre_shop`: crop the panel region (`pierre_panel_layout.json`, key `panel_rel`) → OCR → compare to `name`, `description` (CER/WER), and `price_per_unit`, `quantity_selected`, `total_cost` (exact numeric match)
- `caught_fish`: crop notification region (`caught_fish_layout.json`) → OCR → extract length via regex → compare `length_inches` (exact match)

**Output**: `evaluation/ocr_baseline_results.json`

**Reuses**: `common.py` functions: `load_layout()`, `crop_region()`, `run_ocr()`, `decode_image_b64()` / `load_image_from_path()`

---

## Phase 1: Font Verification + Renderer

**Create**: `fine_tuning/paddleocr/font_renderer.py`
**Create**: `fine_tuning/paddleocr/verify_font.py`
**Create**: `tests/test_font_renderer.py`

### Step 1a: Font Verification

Before committing to SpriteFont1 as the dialog font, verify it by:
1. Build a minimal glyph renderer (just enough to render a word)
2. Render a known string (e.g., "Welcome to KOZU 5") using SpriteFont1 glyphs with dialog brown color on tan background
3. Crop the same text from a real TV dialog screenshot (e.g., `IMG_0002.PNG`)
4. Visual side-by-side comparison — do letterforms, spacing, and proportions match?
5. Repeat with SmallFont to check quantity overlay numbers
6. If neither matches, investigate tinyFont or whether the game uses a system font

Output: `fine_tuning/paddleocr/verify_font.py` — a small script that produces comparison images for manual inspection. This gates the rest of Phase 1.

### Step 1b: Full Font Renderer

Build a `StardewFontRenderer` class that renders arbitrary text using the verified game font sprite sheets. This is reusable for VLM synthetic data generation later.

**Inputs**: Verified font .json + .png files (expected: SpriteFont1 for dialog, SmallFont for quantities)
- Located at: `datasets/assets/game_files/unpacked/`

**Rendering algorithm** (matching XNA SpriteFont behavior):
1. Parse JSON: `characterMap[]` → char-to-index lookup, `glyphs[]` → source rects in atlas, `cropping[]` → placement offsets, `kerning[]` → advance widths
2. For each character: advance by `kerning[i].x` (left bearing), composite glyph from atlas at `(cursor + cropping.x, cropping.y)`, advance by `kerning[i].y + kerning[i].z + horizontalSpacing`
3. Color tinting: atlas is white+alpha — apply target color via alpha mask
4. Render onto specified background color

**Color presets** (sampled from screenshots):
- Dialog: dark brown ~`#5B3A1D` on tan ~`#F4E4C1`
- Notification: dark ~`#56160C` on white `#FFFFFF`
- Shop panel: dark brown on cream ~`#FFEFCE`

**Unit tests**: verify glyph dimensions, multi-char advance width, line wrapping, all printable ASCII render without error

**Can run in parallel with Phase 0.**

---

## Phase 2: Text Corpus Assembly

**Create**: `fine_tuning/paddleocr/text_corpus.py`
**Output**: `fine_tuning/paddleocr/corpus/corpus.txt`, `fine_tuning/paddleocr/corpus/dict.txt`

**Prerequisite**: User provides game text data. Sources:
- Unpack Content XNB files: `Data/TV/`, `Strings/StringsFromCSFiles` (weather, fortune teller, tips), `Data/Objects` (item descriptions), NPC dialog
- The existing `item_manifest.json` has item **names** but descriptions are unresolved localized references — need the actual strings from Content files
- The 89 annotation ground truth texts (supplement, not primary source)

The script:
1. Loads text from game data files
2. Splits long strings into single lines (PaddleOCR rec model expects single-line crops)
3. Deduplicates
4. Generates `dict.txt` (one char per line, all unique characters — must be compatible with pretrained model's dict)

**Target**: 5,000-10,000 unique text lines covering dialog, item names, descriptions, prices, numeric patterns

**Can run in parallel with Phase 0 and Phase 1.**

---

## Phase 3: Synthetic Training Data Generation

**Create**: `fine_tuning/paddleocr/generate_training_data.py`
**Output**: `fine_tuning/paddleocr/dataset/` (PaddleX MSTextRecDataset format)

**Depends on**: Phase 1 (font renderer) + Phase 2 (text corpus)

**Dataset structure** (PaddleX expected format):
```
fine_tuning/paddleocr/dataset/
├── train/           # ~8,000-10,000 single-line text images
├── val/             # ~1,000-1,500 images
├── train.txt        # image_path\tlabel (tab-separated)
├── val.txt
└── dict.txt         # character dictionary
```

**Generation pipeline**:
1. For each corpus line, render using `StardewFontRenderer` as a single-line text crop
2. Randomize: background color (dialog/notification/shop variants ±5 per channel), scale (2x-4x), padding (5-20px)
3. Apply augmentations to simulate real screenshot artifacts: JPEG re-encode (quality 60-95), Gaussian noise (sigma 0-3), slight blur (0-1px kernel)
4. 90/10 split by **text content** (all augmentation variants of a line stay in the same split)
5. Generate both SpriteFont1 and SmallFont variants

**Important**: Each image = one line of text (not multi-line). This matches what PaddleOCR's recognition model receives after the detection stage crops text regions.

---

## Phase 4: Fine-Tuning

**Create**: `fine_tuning/paddleocr/train_config.yaml`
**Create**: `fine_tuning/paddleocr/train.py`
**Output**: `experiments/paddleocr-stardew-v1/`

**Depends on**: Phase 3 (training data)

**Model**: `en_PP-OCRv5_mobile_rec` (already cached, ~10M params, fast on CPU)

**Training config**:
```yaml
Global:
  model: en_PP-OCRv5_mobile_rec
  mode: train
  dataset_dir: fine_tuning/paddleocr/dataset
  device: cpu
  output: experiments/paddleocr-stardew-v1

Train:
  epochs_iters: 50        # bitmap fonts converge fast; can reduce if needed
  batch_size: 32
  learning_rate: 0.0005   # conservative for fine-tuning
  log_interval: 50
  eval_interval: 1
  save_interval: 5
```

**Smoke test first**: Run 1 epoch on 10 images to verify PaddleX training works with PaddlePaddle 3.2.0 before generating the full dataset.

**CPU time estimate**: ~4-8 hours for 10K images x 50 epochs. Can reduce to 20 epochs if val accuracy plateaus early.

**Export**: After training, export inference model to `experiments/paddleocr-stardew-v1/inference/` (contains `inference.pdmodel`, `inference.pdiparams`, `inference.json`)

---

## Phase 5: Integration

**Modify**: `tools-code/common.py` (~5 lines changed in `load_ocr()`)

```python
# Add near module-level constants
_FINETUNED_REC_MODEL = Path(
    os.getenv(
        "STARDEW_REC_MODEL_DIR",
        str(Path(__file__).resolve().parent.parent / "experiments"
            / "paddleocr-stardew-v1" / "inference"),
    )
)

# In load_ocr(), add rec_model_dir if fine-tuned model exists:
rec_kwargs = {}
if _FINETUNED_REC_MODEL.exists():
    rec_kwargs["rec_model_dir"] = str(_FINETUNED_REC_MODEL)

_OCR_INSTANCE = PaddleOCR(
    ...,
    **rec_kwargs,
)
```

**Design**: Falls back to default model if fine-tuned model doesn't exist. Environment variable override for switching models. Zero changes needed in any tool script.

---

## Phase 6: Post-Training Evaluation

**Reuse**: `evaluation/eval_ocr_accuracy.py` (same script from Phase 0)
**Output**: `evaluation/ocr_finetuned_results.json`

**Create**: `evaluation/compare_ocr_models.py`
- Loads baseline and fine-tuned results
- Produces side-by-side comparison report
- Flags regressions (any screen type where CER increased > 0.02)
- Outputs `evaluation/ocr_comparison_report.json`

---

## Evaluation Rubric

### Metrics

| Metric | Definition | Applied To |
|--------|-----------|------------|
| **CER** | `edit_distance(predicted, expected) / len(expected)` | Text fields (dialog_text, name, description) |
| **WER** | Word-level edit distance / word count in expected | Text fields |
| **Exact Match** | `1 if predicted == expected else 0` | Numeric fields (price, quantity, total, length) and short text (name) |

Both CER and WER normalize: lowercase, collapse whitespace, strip leading/trailing whitespace.

### Pass/Fail Thresholds

| Screen Type | Metric | Fine-Tuned Target |
|------------|--------|-------------------|
| tv_dialog | CER | <= 0.05 |
| tv_dialog | WER | <= 0.08 |
| pierre_shop name | Exact Match | >= 90% |
| pierre_shop description | CER | <= 0.05 |
| pierre_shop numerics | Exact Match | >= 95% |
| caught_fish length | Exact Match | >= 95% |
| **Overall CER** | macro-avg | <= 0.05 |

### Rules

1. **No regression**: Fine-tuned model must not increase CER on any screen type by > 0.02 vs baseline
2. **Improvement required**: Must show meaningful improvement on >= 2 of 3 screen types
3. **Test set integrity**: The 89 annotated samples are test-only — never used for training or validation
4. **Same preprocessing**: Both models evaluated with identical `run_ocr()` params (2x upscale, 0.5 min_confidence)
5. **Error analysis**: For any sample with CER > 0.10, log predicted vs expected text for manual review

### Report Format

```json
{
  "baseline_model": "en_PP-OCRv5_mobile_rec (default)",
  "finetuned_model": "paddleocr-stardew-v1",
  "summary": {
    "overall_cer": {"baseline": "...", "finetuned": "...", "delta": "..."},
    "pass": true
  },
  "by_screen_type": { "... per-type metrics ..." : ""},
  "regressions": [ "... any screen types that got worse ..." ],
  "error_analysis": [ "... samples with CER > 0.10 ..." ]
}
```

---

## Files Created/Modified

| # | Path | Action |
|---|------|--------|
| 1 | `evaluation/eval_ocr_accuracy.py` | Create — baseline + post-training eval |
| 2a | `fine_tuning/paddleocr/verify_font.py` | Create — font verification comparison script |
| 2b | `fine_tuning/paddleocr/font_renderer.py` | Create — reusable SpriteFont renderer |
| 3 | `fine_tuning/paddleocr/text_corpus.py` | Create — corpus assembly from game data |
| 4 | `fine_tuning/paddleocr/generate_training_data.py` | Create — synthetic training image generation |
| 5 | `fine_tuning/paddleocr/train_config.yaml` | Create — PaddleX training config |
| 6 | `fine_tuning/paddleocr/train.py` | Create — training launch script |
| 7 | `evaluation/compare_ocr_models.py` | Create — before/after comparison |
| 8 | `tests/test_font_renderer.py` | Create — font renderer unit tests |
| 9 | `tools-code/common.py` | Modify — ~5 lines in `load_ocr()` |
| 10 | `fine_tuning/paddleocr/README.md` | Update — replace placeholder |

**Generated artifacts** (not committed, add to .gitignore):
- `fine_tuning/paddleocr/dataset/` — synthetic training images
- `fine_tuning/paddleocr/corpus/` — text corpus files
- `experiments/paddleocr-stardew-v1/` — model checkpoints and inference model

---

## Execution Order and Parallelism

```
Phase 0 (Baseline Eval)  ──────────────┐
Phase 1 (Font Renderer)  ──────────┐   │   <- These 3 phases can run in parallel
Phase 2 (Text Corpus)  ────────┐   │   │
                               v   v   │
Phase 3 (Training Data Gen)  ──────┐   │
                                   v   │
Phase 4 (Fine-Tuning + Export) ────┐   │
                                   v   v
Phase 5 (Integration)  ───────────────┐
                                      v
Phase 6 (Post-Training Eval + Compare)
```

---

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| CPU training too slow (8+ hrs) | Medium | Start with 20 epochs; mobile model is small (~10M params); bitmap fonts converge fast |
| Synthetic-to-real domain gap | Medium | Augmentations (JPEG, noise, blur) target screenshot artifacts; can add a few real text crops to training if needed |
| PaddlePaddle 3.2.0 training API issues | Low-Med | Smoke test with 10 images / 1 epoch before full data generation |
| Baseline already good enough | Low-Med | Phase 0 quantifies this upfront; if CER < 0.05 already, document and stop |
| dict.txt mismatch breaks transfer learning | Low | Use pretrained model's dict as base; only add chars if in game font but missing from dict |

---

## Verification

1. **Phase 0**: Run `python evaluation/eval_ocr_accuracy.py` — produces baseline JSON, visually spot-check a few error cases
2. **Phase 1**: Run `pytest tests/test_font_renderer.py` — verify glyph rendering; visually compare rendered text to a screenshot crop
3. **Phase 3**: Spot-check 10-20 generated training images — text readable, augmentations look realistic
4. **Phase 4**: Monitor training loss/accuracy curves; check val accuracy converges
5. **Phase 5**: Run `python -c "from tools_code.common import load_ocr; ocr = load_ocr(); print('loaded')"` — verify fine-tuned model loads
6. **Phase 6**: Run eval again, run comparison script, verify all rubric thresholds met and no regressions

---

## Prerequisite (User Action)

Before Phase 2 can start, the user needs to unpack game Content XNB files (same workflow used for fonts):
- `Content/Data/TV/` — all TV channel text (weather, fortune teller, tips, cooking show)
- `Content/Strings/StringsFromCSFiles` — general game strings (weather phrases, etc.)
- `Content/Data/Objects` — resolved item names and descriptions (the current `item_manifest.json` only has unresolved `[LocalizedText ...]` references)
- Any other text-heavy Content files (letters, quest text, NPC dialog) for corpus breadth
- Place unpacked JSON files in `datasets/assets/game_files/unpacked/` or a new `datasets/assets/game_text/` directory
