# Fine-Tuning Scripts

This directory contains fine-tuning scripts for VLM and OCR models.

---

## qwen/

**Purpose**: Fine-tune Qwen2.5-VL-7B with LoRA for screen classification and narration generation.

**Key files**:
- `train.py` — Main training script
- `lora_config.yaml` — Hyperparameters and configuration
- `data_prep.py` — (TODO) Prepare ChatML conversations from annotations

**Usage**:
```bash
python fine_tuning/qwen/train.py --config fine_tuning/qwen/lora_config.yaml
```

**Training data**: `datasets/splits/train.jsonl` (ChatML format)

**Output**: LoRA adapters in `experiments/{run_name}/`

---

## paddleocr/

**Purpose**: (Future) Fine-tune PaddleOCR for Stardew Valley pixel art text recognition.

**Status**: Not implemented yet. Current PaddleOCR (PP-OCRv5) performance is sufficient for MVP.

**Future use cases**:
- Low-contrast text (game letters)
- Pixel art fonts
- Specialty UI elements

---

## Training Workflow

1. **Prepare datasets**: `datasets/splits/{train,eval,test}.jsonl`
2. **Configure**: Edit `qwen/lora_config.yaml`
3. **Train**: Run `qwen/train.py`
4. **Monitor**: MLFlow UI at `http://localhost:5000`
5. **Evaluate**: Use `evaluation/` scripts
6. **Upload**: Push to HuggingFace Hub

---

## LoRA Configuration

Default settings (see `qwen/lora_config.yaml`):
- **Rank**: 16
- **Alpha**: 32
- **Dropout**: 0.05
- **Target modules**: q_proj, k_proj, v_proj, o_proj
- **Precision**: FP16 (ROCm constraint)

---

## Monitoring Training

```bash
# Start MLFlow UI
mlflow ui --backend-store-uri file://experiments/mlruns

# Open http://localhost:5000
```

---

## Common Issues

**OOM during training**:
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps`
- Enable `gradient_checkpointing: true`

**ROCm FP16 precision**:
- Always use `fp16: true`, `bf16: false`
- No INT4 or INT8 quantization on gfx1151

**Chat template errors**:
- Ensure `chat_template_path` points to valid Jinja template
- Tool definitions must match extraction tool schemas
