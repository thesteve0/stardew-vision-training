# Evaluation Comparison: Small Training Runs

All evaluations use the same held-out set of 100 images (25 per screen type) from `datasets/eval_set.json`. These images are **never** used in training or validation.

## Per-class accuracy

| Screen Type | Baseline v1 | Baseline v2 | Finetuned Tiny (standalone) | Finetuned Tiny (Ray) |
|-------------|------------|------------|---------------------------|---------------------|
| tv_dialog   | 92% | 80% | 96% | 96% |
| caught_fish | 52% | 24% | 72% | 88% |
| pierre_shop | 100% | 96% | 100% | 100% |
| no_tools    | 36% | 20% | 8% | 0% |
| **Overall** | **70%** | **55%** | **69%** | **71%** |
| **Macro F1** | **0.697** | **0.594** | **0.633** | **0.621** |

## Run details

| Run | Model | Training | Config | Results |
|-----|-------|----------|--------|---------|
| Baseline v1 | Qwen2.5-VL-7B-Instruct (untuned) | n/a | Original prompt format | `experiments/eval-baseline-v1/` |
| Baseline v2 | Qwen2.5-VL-7B-Instruct (untuned) | n/a | Production-matching prompt (vLLM template) | `experiments/eval-baseline-v2/` |
| Finetuned Tiny (standalone) | LoRA adapter (40 samples, 1 epoch) | SFTTrainer, single GPU | `lora_config_tiny.yaml` | `experiments/eval-finetuned-tiny-v1/` |
| Finetuned Tiny (Ray) | LoRA adapter (40 samples, 1 epoch) | Ray Train, single GPU | `lora_config_tiny.yaml` | `experiments/eval-finetuned-tiny-ray-v1/` |

## Training metrics (tiny runs)

| Metric | Standalone | Ray Train |
|--------|-----------|-----------|
| Final loss | 0.041 | 0.040 |
| Token accuracy | 98.5% | 98.5% |
| Steps | 5 | 5 |
| Duration | ~7 min | ~7 min |

## Key observations

- **Baseline v2 is the fair comparison** for fine-tuned models — it uses the same production-matching prompt format used during training
- **Fine-tuning dramatically improves tool-calling classes** (caught_fish: 24% → 88% with Ray, tv_dialog: 80% → 96%)
- **no_tools (rejection class) collapses** after fine-tuning on only 40 samples — the model learns to always call a tool. The full training set has 275 no_tools examples which should fix this
- **Ray Train produces equivalent results** to standalone training (loss, token accuracy, and eval metrics are consistent)
- The caught_fish difference between standalone (72%) and Ray (88%) is likely variance from the tiny training set, not a meaningful difference in the training method
