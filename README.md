# Stardew Vision Training Repository

**Model fine-tuning and dataset preparation for Stardew Vision accessibility project**

This repository contains all training, evaluation, and dataset preparation code for fine-tuning VLMs (Vision-Language Models) and OCR models for the [Stardew Vision](https://github.com/yourusername/stardew-vision) accessibility application.

---

## Purpose

1. **Fine-tune Qwen2.5-VL-7B** to recognize Stardew Valley UI screens and generate natural language narrations
2. **Prepare datasets** with real screenshots + synthetic augmentation
3. **Evaluate model performance** on tool calling accuracy and narration quality
4. **Support conference talks/workshops** on VLM fine-tuning for accessibility

---

## Repository Structure

```
stardew-vision-training/
├── datasets/                   # Screenshots, annotations, sprites, templates
│   ├── pierre_shop/           # Pierre's shop item detail panel
│   ├── tv_dialog/             # TV shows (weather, fortune, cooking, tips)
│   ├── caught_fish/           # Fish caught notifications
│   ├── quest_board/           # Help Wanted quests
│   ├── game_letter/           # In-game letters
│   ├── level_up/              # Skill level-up notifications
│   └── assets/                # Sprites, item manifest, templates
│
├── fine_tuning/
│   ├── qwen/                  # Qwen2.5-VL LoRA fine-tuning
│   │   ├── train.py
│   │   ├── lora_config.yaml
│   │   └── data_prep.py
│   └── paddleocr/             # OCR fine-tuning (if needed)
│
├── evaluation/
│   ├── eval_tool_calling.py   # Tool selection accuracy
│   ├── eval_narration.py      # Narration quality metrics
│   └── metrics.yaml
│
├── synthetic_data/
│   ├── generate_variations.py # LLM-based synthetic data generation
│   └── templates/             # Narration templates per screen type
│
├── scripts/
│   ├── annotation/            # Annotation tools (organize_screenshot.py, etc.)
│   └── data_prep/             # Dataset creation, splitting, validation
│
├── experiments/               # MLFlow tracking, checkpoints
│   └── mlruns/
│
├── docs/                      # ADRs, collection plan, OCR analysis
│
└── configs/                   # Training configs, output schemas
```

---

## Quick Start

### 1. Clone and Setup

```bash
# Clone this repo
git clone https://github.com/yourusername/stardew-vision-training.git
cd stardew-vision-training

# Open in devcontainer (VS Code)
# Or manually install dependencies:
uv sync
source .venv/bin/activate
```

### 2. Collect Screenshots

See [`docs/phase2-collection-plan.md`](docs/phase2-collection-plan.md) for collection guidelines.

```bash
# Organize a screenshot
python scripts/annotation/organize_screenshot.py \
  /path/to/screenshot.jpg \
  --screen-type tv_dialog \
  --name tv_weather_01
```

### 3. Fine-Tune Qwen

```bash
# Prepare training data (real + synthetic)
python synthetic_data/generate_variations.py \
  --screen-type tv_dialog \
  --num-variations 100

# Train with LoRA
python fine_tuning/qwen/train.py \
  --config fine_tuning/qwen/lora_config.yaml \
  --output-dir experiments/qwen-tv-fish-v1
```

### 4. Evaluate

```bash
# Evaluate tool calling accuracy
python evaluation/eval_tool_calling.py \
  --model experiments/qwen-tv-fish-v1 \
  --test-set datasets/splits/test.jsonl

# Evaluate narration quality
python evaluation/eval_narration.py \
  --model experiments/qwen-tv-fish-v1 \
  --test-set datasets/splits/test.jsonl
```

### 5. Push to HuggingFace Hub

```bash
python scripts/upload_to_hub.py \
  --model-path experiments/qwen-tv-fish-v1 \
  --repo-id yourusername/stardew-vision-vlm \
  --version v1
```

---

## Baseline Evaluation Results

Baseline evaluation of the untuned Qwen2.5-VL-7B-Instruct model on tool selection accuracy (100 images, 25 per screen type):

| Screen Type  | Count | Correct | Accuracy |
|-------------|-------|---------|----------|
| tv_dialog   | 25    | 23      | 92.0%    |
| caught_fish | 25    | 13      | 52.0%    |
| pierre_shop | 25    | 25      | 100.0%   |
| no_tools    | 25    | 9       | 36.0%    |
| **Overall** | **100** | **70** | **70.0%** |

**Macro F1: 69.7%**

Results saved to:
- `experiments/eval-baseline-v1/results.json` — aggregate metrics
- `experiments/eval-baseline-v1/sample_outputs.jsonl` — per-sample predictions and raw model output
- MLflow experiment `qwen-tool-selection-eval`, run `baseline`

The fixed evaluation set (never use for training): `datasets/eval_set.json`

---

## Current Status

**Phase 2 Focus**: tv_dialog + caught_fish

- **tv_dialog**: 45 screenshots collected
- **caught_fish**: 33 screenshots collected
- **pierre_shop**: 26 screenshots collected
- **no_tools**: 173 screenshots collected

**Next Steps**:
1. Generate synthetic training data (real → synthetic variations per type)
2. Fine-tune Qwen on pierre_shop + tv_dialog + caught_fish + no_tools
3. Re-evaluate against baseline using `datasets/eval_set.json`

---

## Training Data Format

### ChatML Conversation Example

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "data:image/png;base64,..."},
        {"type": "text", "text": "What's on this screen?"}
      ]
    },
    {
      "role": "assistant",
      "tool_calls": [
        {
          "function": {
            "name": "crop_tv_dialog",
            "arguments": "{\"image_b64\": \"...\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "name": "crop_tv_dialog",
      "content": "{\"dialog_text\": \"...\"}"
    },
    {
      "role": "assistant",
      "tool_calls": [
        {
          "function": {
            "name": "text_to_speech",
            "arguments": "{\"text\": \"TV weather forecast: ...\"}"
          }
        }
      ]
    }
  ]
}
```

This trains Qwen to:
1. Classify screen type → call correct extraction tool
2. Parse OCR results → generate natural narration

---

## Key Technologies

- **VLM**: Qwen2.5-VL-7B-Instruct (7B parameters, FP16 only on ROCm)
- **Fine-tuning**: LoRA via PEFT + TRL SFTTrainer
- **OCR**: PaddleOCR (PP-OCRv5, CPU-only)
- **Distributed Training**: Ray Train on OpenShift AI
- **Experiment Tracking**: MLFlow
- **Synthetic Data**: LLM-based (Claude/GPT-4) variation generation

---

## ROCm Constraints

**CRITICAL**: This project runs on AMD ROCm 7.2 (Strix Halo gfx1151).

- **FP16 only** — No BF16, no INT4, no INT8
- **PaddlePaddle 3.2.0** — Version 3.3.0 has OneDNN PIR bug
- **Never `pip install torch`** — Use ROCm-provided build
- **Use `uv` for package management** — Protects ROCm packages

See `template_docs/notesOnRocm72.md` for details.

---

## Documentation

- **[Phase 2 Collection Plan](docs/phase2-collection-plan.md)** — Screenshot collection guidelines
- **[Screen OCR Analysis](docs/screen-ocr-analysis.md)** — Extraction tool requirements
- **[ADR-001: VLM Selection](docs/adr/001-vlm-selection.md)** — Why Qwen2.5-VL
- **[ADR-010: Screen Extraction](docs/adr/010-screen-region-extraction.md)** — OCR strategy

---

## Related Repositories

- **[stardew-vision](https://github.com/yourusername/stardew-vision)** — Production application (FastAPI, KServe, deployment)

---

## License

MIT License (same as main application)

---

## Contact

For questions about this training repository, see the main [Stardew Vision](https://github.com/yourusername/stardew-vision) project.
