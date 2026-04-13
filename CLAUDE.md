# CLAUDE.md — Training Repository Context

This file provides context to Claude Code when working on the training repository.

---

## Project Overview

**Purpose**: Fine-tune Qwen2.5-VL-7B to recognize Stardew Valley UI screens, call appropriate extraction tools, and generate natural language narrations for visually impaired players. This repository handles model training, dataset preparation, and evaluation — separate from the production application.

**Parent Project**: [stardew-vision](https://github.com/yourusername/stardew-vision) — Production FastAPI application with KServe serving

**Problem Domain**: VLM fine-tuning for UI accessibility, tool calling, synthetic data augmentation

**Key Technologies**:
- PyTorch 2.9.1 (ROCm 7.2, AMD Strix Halo gfx1151)
- **FP16 only** (no BF16, no INT4 on this hardware)
- HuggingFace: transformers, peft, trl, datasets, evaluate
- VLM: Qwen/Qwen2.5-VL-7B-Instruct
- Distributed training: Ray Train on OpenShift AI
- Experiment tracking: MLFlow
- OCR: PaddleOCR (CPU-only, for annotation tools)

---

## Repository Structure

```
stardew-vision-training/
├── datasets/              # Screenshots, annotations, sprites
├── fine_tuning/
│   ├── qwen/             # VLM LoRA training
│   └── paddleocr/        # OCR fine-tuning (future)
├── evaluation/           # Metrics, eval scripts
├── synthetic_data/       # LLM-based data generation
├── scripts/
│   ├── annotation/       # organize_screenshot.py, etc.
│   └── data_prep/        # Dataset creation, splitting
├── experiments/          # MLFlow runs, checkpoints
└── docs/                 # ADRs, collection plan, OCR analysis
```

---

## Current Focus (Phase 2)

**Screen types to fine-tune**:
1. **pierre_shop** (baseline — 8 examples collected)
2. **tv_dialog** (priority #2 — 1/15 collected)
3. **caught_fish** (priority #3 — 1/15 collected, sprite matching challenge)

**Training strategy**:
- Collect 15-20 real screenshots per screen type
- Generate synthetic variations (15 → 150 examples per type)
- Fine-tune Qwen on ~450 total ChatML conversations
- Each conversation: screenshot → tool call → OCR response → narration

**Current task**: Collecting screenshots for tv_dialog and caught_fish

---

## Training Data Format

Multi-turn ChatML conversations that teach:
1. **Screen classification** → correct tool call
2. **Narration generation** → natural language from OCR JSON

Example:
```json
{
  "messages": [
    {"role": "user", "content": [{"type": "image", ...}, {"type": "text", "text": "What's on this screen?"}]},
    {"role": "assistant", "tool_calls": [{"function": {"name": "crop_tv_dialog", ...}}]},
    {"role": "tool", "name": "crop_tv_dialog", "content": "{...OCR JSON...}"},
    {"role": "assistant", "tool_calls": [{"function": {"name": "text_to_speech", "arguments": "{\"text\": \"TV weather forecast: ...\"}"}}]}
  ]
}
```

---

## ROCm Constraints (CRITICAL)

This project runs on AMD ROCm 7.2 (gfx1151). **ALWAYS follow these rules**:

1. **FP16 only** — No BF16, no INT4, no INT8
2. **Never `pip install torch`** — ROCm-provided build MUST NOT be replaced
3. **Use `uv` for package management** — `pyproject.toml` has `exclude-dependencies` to protect ROCm packages
4. **PaddlePaddle 3.2.0 required** — Version 3.3.0 has OneDNN PIR bug (SIGTERM on CPU inference)
5. **Environment variables** (already set in devcontainer):
   - `ROCBLAS_USE_HIPBLASLT=1`
   - `ROCM_HOME=/opt/rocm`
   - `HIP_VISIBLE_DEVICES=0`

See `template_docs/notesOnRocm72.md` (from main repo) for full details.

---

## Development Workflow

### 1. Collecting Screenshots

```bash
# Organize a new screenshot
python scripts/annotation/organize_screenshot.py \
  /path/to/screenshot.jpg \
  --screen-type tv_dialog \
  --name tv_weather_01
```

Adds to `datasets/{screen_type}/images/` and creates stub annotation.

### 2. Running Annotation Tools

(Note: Extraction tools live in main `stardew-vision` repo. For training, we use simplified OCR wrappers.)

```bash
# Annotate a screenshot (once tool is built)
python scripts/annotation/annotate_{screen_type}.py \
  datasets/{screen_type}/images/example.jpg
```

### 3. Generating Synthetic Data

```bash
python synthetic_data/generate_variations.py \
  --screen-type tv_dialog \
  --num-variations 100 \
  --output datasets/tv_dialog/conversations_synthetic.jsonl
```

### 4. Fine-Tuning

```bash
python fine_tuning/qwen/train.py \
  --config fine_tuning/qwen/lora_config.yaml \
  --output-dir experiments/qwen-tv-fish-v1
```

Uses LoRA (rank=16, alpha=32) via PEFT. Checkpoints saved to experiments/.

### 5. Evaluation

```bash
# Tool calling accuracy
python evaluation/eval_tool_calling.py \
  --model experiments/qwen-tv-fish-v1 \
  --test-set datasets/splits/test.jsonl

# Narration quality
python evaluation/eval_narration.py \
  --model experiments/qwen-tv-fish-v1 \
  --test-set datasets/splits/test.jsonl
```

### 6. Uploading to HuggingFace Hub

```bash
python scripts/upload_to_hub.py \
  --model-path experiments/qwen-tv-fish-v1 \
  --repo-id yourusername/stardew-vision-vlm \
  --version v1
```

---

## Important Patterns

### Package Management (CRITICAL)

- **ALWAYS use `uv add <package>`** to add dependencies
- **ALWAYS use `uv sync`** to install from lockfile
- **NEVER use `pip install`** — it silently overwrites ROCm-provided packages

Exception: `pip install uv` is acceptable only as Dockerfile bootstrap.

### Dataset Annotation Schema

All annotations follow this format (Feast-compatible):

```json
{
  "image_id": "uuid",
  "image_path": "images/example.jpg",
  "screen_type": "tv_dialog",
  "timestamp": "2026-04-13T...",
  "ocr_fields": {...},
  "narration": "...",
  "tool_call": "crop_tv_dialog",
  "metadata": {}
}
```

Stored as JSONL in `datasets/{screen_type}/annotations.jsonl`.

### LoRA Configuration

LoRA hyperparameters (see `fine_tuning/qwen/lora_config.yaml`):
- Rank: 16
- Alpha: 32
- Dropout: 0.05
- Target modules: q_proj, v_proj, k_proj, o_proj
- Bias: none

### MLFlow Experiment Naming

Format: `{model_short_name}-{run_type}-v{N}`

Example: `qwen-tv-fish-v1`

---

## Known Issues and Gotchas

1. **PaddlePaddle 3.3.0 bug** — MUST use 3.2.0. Version 3.3.0 crashes with OneDNN PIR error.
2. **ROCm FP16 only** — No BF16 support on gfx1151. Always `dtype=torch.float16`.
3. **Qwen chat template** — Uses custom Hermes-compatible template (from main repo).
4. **Sprite matching for caught_fish** — Fish names cropped due to enlarged UI. Use template matching against `datasets/assets/sprites/`.

---

## External Dependencies

- **HuggingFace Hub**: Base models, dataset artifacts
- **Sprite sheet**: Stardew Valley `springobjects.png` (CC-BY-NC-SA)
- **Item manifest**: Community-maintained JSON (`datasets/assets/item_manifest.json`)
- **No external APIs** for training (everything runs locally or on OpenShift AI)

---

## Testing Strategy

```bash
# Run all tests
pytest tests/

# Test synthetic data generation
pytest tests/test_synthetic_data.py

# Test fine-tuning script (dry run)
pytest tests/test_fine_tuning.py
```

---

## Relationship to Main Application Repo

**Main repo (`stardew-vision`)**: Production application
- FastAPI coordinator service
- KServe serving on OpenShift AI
- Extraction tools (crop_pierres_detail_panel, etc.)
- Deployment manifests

**This repo (`stardew-vision-training`)**: Model development
- Dataset collection and annotation
- Fine-tuning scripts
- Evaluation metrics
- Synthetic data generation

**Handoff**: Trained LoRA adapters uploaded to HuggingFace Hub → main repo pulls for serving.

---

## Documentation

Key docs:
- **docs/phase2-collection-plan.md** — Screenshot collection guidelines
- **docs/screen-ocr-analysis.md** — OCR difficulty analysis, tool requirements
- **docs/adr/001-vlm-selection.md** — Why Qwen2.5-VL
- **docs/adr/010-screen-region-extraction.md** — OCR strategy

---

## Overall Instructions

- This is a **training-focused repository** — no production serving code here
- When creating scripts, assume they run in devcontainer with ROCm GPU access
- All training artifacts (checkpoints, logs) go to `experiments/`
- Dataset files stay in `datasets/` (not committed to git, managed via host volume)
- Use MLFlow for experiment tracking (already configured in devcontainer)
- LoRA adapters are the primary output artifact (upload to HF Hub)

---

## Development Environment

- All work happens in devcontainer (ROCm 7.2 base image)
- Python 3.12
- Virtual env at `.venv/` (managed by uv)
- HuggingFace cache at `.cache/huggingface/`
- MLFlow tracking at `experiments/mlruns/`

---

## Important Notes

- This repo is used for conference talks/workshops — keep code readable and well-documented
- Synthetic data generation is critical due to small real dataset (15-20 examples per screen)
- Training target: ~450 total examples (pierre_shop + tv_dialog + caught_fish)
- Evaluation metrics: tool selection accuracy, field extraction F1, narration quality
