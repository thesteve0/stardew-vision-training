# Evaluation Scripts

Evaluation metrics for fine-tuned Qwen2.5-VL model.

---

## Metrics

### 1. Tool Calling Accuracy

**Script**: `eval_tool_calling.py`

**Measures**:
- Screen classification accuracy (correct tool selected?)
- Tool argument correctness (correct image_b64 passed?)

**Target**: >95% accuracy on test set

**Usage**:
```bash
python evaluation/eval_tool_calling.py \
  --model experiments/qwen-tv-fish-v1 \
  --test-set datasets/splits/test.jsonl
```

**Output**: Confusion matrix, per-screen-type accuracy

---

### 2. Narration Quality

**Script**: `eval_narration.py`

**Measures**:
- Field extraction F1 score (are OCR fields correctly included in narration?)
- Fluency (perplexity score)
- Naturalness (human evaluation scale 1-5)

**Target**: F1 >90%, fluency perplexity <50

**Usage**:
```bash
python evaluation/eval_narration.py \
  --model experiments/qwen-tv-fish-v1 \
  --test-set datasets/splits/test.jsonl
```

**Output**: Field extraction F1, perplexity, sample narrations

---

### 3. OCR Field Extraction (Optional)

**Script**: `eval_ocr_accuracy.py`

**Measures**: How well OCR tools extract structured fields from screenshots

**Note**: This evaluates extraction tools, not the VLM. Separate from VLM fine-tuning evaluation.

---

## Test Set

Test set should be:
- Real screenshots (never seen during training)
- Balanced across screen types (pierre_shop, tv_dialog, caught_fish)
- ~20% of total dataset

Created via `scripts/data_prep/split_dataset.py`.

---

## Evaluation Workflow

1. **Split dataset**: Create train/eval/test splits
2. **Train model**: Fine-tune on train set
3. **Run evaluation**: `eval_tool_calling.py` + `eval_narration.py`
4. **Analyze results**: Review confusion matrix, sample outputs
5. **Iterate**: Adjust hyperparameters, collect more data if needed

---

## Output Format

Evaluation results saved to:
```
experiments/{run_name}/evaluation_results.json
```

Contains:
```json
{
  "tool_calling_accuracy": 0.96,
  "confusion_matrix": {...},
  "narration_f1": 0.92,
  "narration_perplexity": 42.3,
  "sample_outputs": [...]
}
```

Logged to MLFlow for tracking across experiments.
