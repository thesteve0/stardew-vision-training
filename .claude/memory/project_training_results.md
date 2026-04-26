---
name: Training results and baseline eval
description: Complete training comparison — baseline 51.2%, Ray 96.0%, KubeFlow 97.6%; batch size and prompt impact documented
type: project
originSessionId: 0b1c1454-7379-436e-a94a-f747fbf758da
---
## Baseline eval (2026-04-26, untuned Qwen2.5-VL-7B-Instruct)

Using production-matched prompt (`evaluation/prompt.py`):
- tv_dialog: 80.0%
- caught_fish: 20.0%
- pierre_shop: 96.0%
- no_tools: 30.0%
- **Overall: 51.2%, Macro F1: 59.0%**

Results in `experiments/eval-baseline-v1/` and MLflow experiment `qwen-tool-selection-eval`.

## Prompt impact on baseline

First baseline run (commit `a0c3e78`, old prompt) scored **0% on no_tools**. After prompt was updated to match production (commit `f2ebebf`), no_tools jumped to **30%** — prompt engineering alone, no fine-tuning.

## Cluster training results (2x L40S, 775 samples, 3 epochs)

| Run | Effective batch | Steps | Wall time | s/step | Accuracy |
|-----|----------------|-------|-----------|--------|----------|
| Ray v3 (batch=16) | 16 | 147 | ~42 min | ~17 | 90.4% |
| Ray v4 (batch=8) | 8 | 291 | ~43 min | ~8.8 | 96.0% |
| KubeFlow v1 (batch=8) | 8 | 291 | ~42 min | ~8.6 | 97.6% |

Batch size doubling (8→16) caused a 7.2 point accuracy drop. The 1.6 point Ray-vs-KubeFlow difference is likely statistical noise (2 samples out of 125).

## Eval throughput

Local AMD iGPU (Strix Halo): ~63 min / 125 images (~30s/sample). Cluster L40S: ~5 min. 12x throughput difference.
