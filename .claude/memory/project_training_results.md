---
name: Training results and baseline eval
description: Complete training comparison — baseline 51.2%, local standalone 96.0%, Ray 96.0%, KubeFlow 97.6%; batch size and prompt impact documented
type: project
originSessionId: a1c61036-d7f5-4801-a415-0d618bd9f3e7
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

First baseline run (old prompt) scored **0% on no_tools**. After prompt was updated to match production, no_tools jumped to **30%** — prompt engineering alone, no fine-tuning.

## Local standalone results (AMD Strix Halo, 775 samples, 3 epochs)

| Metric | Value |
|--------|-------|
| Accuracy | 96.0% (120/125) |
| Macro F1 | 0.962 |
| Wall time | ~424 min (~7h) |
| s/step | ~87.4 |
| Best checkpoint | epoch 2 (checkpoint-200) |

Per-class: tv_dialog 96%, caught_fish 92%, pierre_shop 96%, no_tools 98%.
Results in `experiments/eval-local-standalone-v1/`.

## Cluster training results (2x L40S, 775 samples, 3 epochs)

| Run | Effective batch | Steps | Wall time | s/step | Accuracy |
|-----|----------------|-------|-----------|--------|----------|
| Ray v3 (batch=16) | 16 | 147 | ~42 min | ~17 | 90.4% |
| Ray v4 (batch=8) | 8 | 291 | ~43 min | ~8.8 | 96.0% |
| KubeFlow v1 (batch=8) | 8 | 291 | ~42 min | ~8.6 | 97.6% |

Batch size doubling (8→16) caused a 7.2 point accuracy drop. Local standalone matches Ray v4 exactly (96.0%), confirming Ray wrapper doesn't degrade quality.

## Eval throughput

Local AMD iGPU (Strix Halo): ~8 min / 125 images (~3.85s/sample with LoRA). Cluster L40S: ~5 min. Baseline (no LoRA) was ~63 min (~30s/sample).

## Remaining work

- Local Ray training run still TODO (skipped — moving to laptop/cluster)
- All comparison data in `docs/comparison-small-training-runs.md`
