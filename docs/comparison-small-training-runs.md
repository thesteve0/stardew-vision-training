# Evaluation Comparison: Training Runs

All evaluations use the held-out eval set from `datasets/eval_set.json`, which is **never** used in training or validation. The eval set was expanded from 100 to 125 images (50 no_tools, 25 each for tool classes) after the tiny runs.

## Per-class accuracy

| Screen Type | Baseline v1 | Baseline v2 | Tiny (standalone) | Tiny (Ray) | Full 3-epoch (Ray) |
|-------------|------------|------------|-------------------|-----------|-------------------|
| tv_dialog   | 92% | 80% | 96% | 96% | **96%** |
| caught_fish | 52% | 24% | 72% | 88% | **96%** |
| pierre_shop | 100% | 96% | 100% | 100% | **100%** |
| no_tools    | 36% | 20% | 8% | 0% | **98%** |
| **Overall** | **70%** | **55%** | **69%** | **71%** | **97.6%** |
| **Macro F1** | **0.697** | **0.594** | **0.633** | **0.621** | **0.977** |

## Run details

| Run | Model | Training | Eval Set | Results |
|-----|-------|----------|----------|---------|
| Baseline v1 | Qwen2.5-VL-7B-Instruct (untuned) | n/a | 100 images | `experiments/eval-baseline-v1/` |
| Baseline v2 | Qwen2.5-VL-7B-Instruct (untuned) | n/a | 100 images | `experiments/eval-baseline-v2/` |
| Tiny (standalone) | LoRA adapter (40 samples, 1 epoch) | SFTTrainer, single GPU | 100 images | `experiments/eval-finetuned-tiny-v1/` |
| Tiny (Ray) | LoRA adapter (40 samples, 1 epoch) | Ray Train, single GPU | 100 images | `experiments/eval-finetuned-tiny-ray-v1/` |
| Full 3-epoch (Ray) | LoRA adapter (775 samples, 3 epochs) | Ray Train, single GPU | 125 images | `experiments/eval-finetuned-full-ray-v1/` |

## Training metrics

| Metric | Tiny standalone | Tiny Ray | Full 3-epoch Ray |
|--------|----------------|----------|-----------------|
| Training samples | 40 | 40 | 775 |
| Steps | 5 | 5 | 291 |
| Final loss | 0.041 | 0.040 | 0.0005 |
| Token accuracy | 98.5% | 98.5% | 100% |
| Val loss (best) | n/a | n/a | 0.0038 (epoch 2) |
| Wall time | ~7 min | ~7 min | ~4h 40m |
| Seconds/step | ~84 | ~84 | ~48 (steady state) |

## Key observations

- **Baseline v2 is the fair comparison** for fine-tuned models — it uses the same production-matching prompt format used during training
- **Full training achieved 97.6% overall accuracy** (up from 55% baseline v2), with only 3 errors out of 125 eval images
- **no_tools (rejection class) fixed**: collapsed to 0% with 40 balanced samples, recovered to 98% after oversampling at 2:1 ratio with 310 training examples. Validates the Larson et al. / Gorilla guidance on negative class ratios
- **Val loss bottomed at epoch 2** (0.0038) and ticked up at epoch 3 (0.0044) — mild overfitting suggests 2 epochs may be the sweet spot for this dataset size
- **Ray Train produces equivalent results** to standalone training — no timing or quality difference on single GPU
- The 3 errors: 1 tv_dialog → no_tools, 1 caught_fish → no_tools, 1 no_tools → pierre_shop
