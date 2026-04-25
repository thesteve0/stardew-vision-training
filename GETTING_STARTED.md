# Getting Started — Stardew Vision Training

Welcome to the training repository! Follow these steps to get up and running.

---

## Prerequisites

- AMD GPU with ROCm 7.2 support (tested on Strix Halo gfx1151)
- Docker + VS Code with Dev Containers extension
- Git, GitHub CLI
- HuggingFace account (for downloading models and uploading fine-tuned adapters)

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/stardew-vision-training.git
cd stardew-vision-training
```

### 2. Open in Dev Container

**VS Code**:
1. Open folder in VS Code
2. Click "Reopen in Container" when prompted
3. Wait for devcontainer build (~5-10 minutes first time)

**Manual**:
```bash
# Build and run devcontainer
docker build -t stardew-training .devcontainer/
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video --ipc=host \
  -v $(pwd):/workspaces/stardew-vision-training \
  stardew-training
```

### 3. Install Dependencies

```bash
# Inside devcontainer
uv sync
source .venv/bin/activate
```

### 4. Verify GPU Access

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected output:
```
True
AMD Radeon Graphics (or similar)
```

### 5. Download Base Models

```bash
# Login to HuggingFace
huggingface-cli login

# Download Qwen2.5-VL-7B-Instruct
python -c "from transformers import Qwen2VLForConditionalGeneration; \
  Qwen2VLForConditionalGeneration.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')"
```

Models cache to `.cache/huggingface/`.

---

## Quick Workflow

### Phase 1: Collect Screenshots

See [`docs/phase2-collection-plan.md`](docs/phase2-collection-plan.md) for guidelines.

```bash
# Organize a screenshot
python scripts/annotation/organize_screenshot.py \
  ~/Desktop/screenshot.jpg \
  --screen-type tv_dialog \
  --name tv_weather_01
```

Goal: 15-20 screenshots for tv_dialog and caught_fish.

### Phase 2: Annotate Screenshots

(Once extraction tools are built)

```bash
python scripts/annotation/annotate_tv_dialog.py \
  datasets/tv_dialog/images/tv_weather_01.jpg
```

Saves annotation to `datasets/tv_dialog/annotations.jsonl`.

### Phase 3: Generate Synthetic Data

```bash
python synthetic_data/generate_variations.py \
  --screen-type tv_dialog \
  --num-variations 100 \
  --output datasets/tv_dialog/conversations_synthetic.jsonl
```

Creates 100 variations from real examples.

### Phase 4: Fine-Tune Qwen

```bash
python fine_tuning/qwen/train.py \
  --config fine_tuning/qwen/lora_config_local.yaml \
  --output-dir experiments/qwen-tv-fish-v1
```

Trains LoRA adapter on combined dataset (real + synthetic).

### Phase 5: Evaluate

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

### Phase 6: Upload to HuggingFace Hub

```bash
python scripts/upload_to_hub.py \
  --model-path experiments/qwen-tv-fish-v1 \
  --repo-id yourusername/stardew-vision-vlm \
  --version v1
```

---

## Directory Overview

**datasets/** — Screenshots, annotations, sprites
- Each screen type has: `images/`, `annotations.jsonl`, `README.md`
- `assets/` contains sprites and item manifest

**fine_tuning/** — Training scripts
- `qwen/train.py` — LoRA fine-tuning script
- `qwen/lora_config_local.yaml` — Hyperparameters (standalone training)
- `qwen/lora_config_ray_local.yaml` — Hyperparameters (local Ray training)

**evaluation/** — Metrics
- `eval_tool_calling.py` — Tool selection accuracy
- `eval_narration.py` — Narration quality

**synthetic_data/** — Data augmentation
- `generate_variations.py` — LLM-based synthetic generation

**experiments/** — MLFlow tracking
- `mlruns/` — Experiment logs
- `checkpoints/` — Model checkpoints

---

## Common Tasks

### View Collection Progress

```bash
cat datasets/COLLECTION_CHECKLIST.md
```

### List Collected Screenshots

```bash
ls -lh datasets/tv_dialog/images/
ls -lh datasets/caught_fish/images/
```

### View Annotations

```bash
# View latest annotation
tail -n 1 datasets/tv_dialog/annotations.jsonl | jq .

# Count annotations
wc -l datasets/tv_dialog/annotations.jsonl
```

### Monitor Training

```bash
# View MLFlow UI
mlflow ui --backend-store-uri file://experiments/mlruns

# Open browser to http://localhost:5000
```

### Run Tests

```bash
pytest tests/ -v
```

---

## Troubleshooting

### GPU Not Detected

```bash
# Check ROCm installation
rocm-smi

# Check PyTorch CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

If `False`, rebuild devcontainer.

### PaddlePaddle SIGTERM

Error: `ConvertPirAttribute2RuntimeAttribute not support`

**Fix**: Ensure `paddlepaddle==3.2.0` (NOT 3.3.0).

```bash
uv remove paddlepaddle
uv add "paddlepaddle==3.2.0"
```

### Package Conflicts

**Problem**: `pip install` overwrites ROCm packages.

**Fix**: Use `uv` exclusively.

```bash
# Remove pip-installed packages
pip freeze | xargs pip uninstall -y

# Reinstall with uv
uv sync
```

### Out of Memory During Training

**Fix**: Reduce batch size or gradient accumulation steps in `lora_config_local.yaml`.

```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
```

---

## Next Steps

1. **Read documentation**:
   - `docs/phase2-collection-plan.md` — Collection guidelines
   - `docs/screen-ocr-analysis.md` — OCR tool requirements
   - `CLAUDE.md` — Full repository context

2. **Collect screenshots**:
   - tv_dialog: 15-20 examples
   - caught_fish: 15-20 examples

3. **Build extraction tools** (or use simplified OCR wrappers)

4. **Generate synthetic data** to augment real examples

5. **Fine-tune Qwen** on combined dataset

6. **Evaluate and iterate**

---

## Resources

- **Main Application**: [stardew-vision](https://github.com/yourusername/stardew-vision)
- **HuggingFace Hub**: [yourusername/stardew-vision-vlm](https://huggingface.co/yourusername/stardew-vision-vlm)
- **Qwen2.5-VL Docs**: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- **PEFT Docs**: https://huggingface.co/docs/peft

---

## Questions?

See `README.md` and `CLAUDE.md` for full context, or open an issue in the main [stardew-vision](https://github.com/yourusername/stardew-vision) repository.