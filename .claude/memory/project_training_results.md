---
name: Training results and data balance decisions
description: Full training achieved 97.6% accuracy; no_tools 2:1 ratio validated; 2 epochs likely optimal; val loss overfits at epoch 3
type: project
originSessionId: 1cac980d-2f8c-45f6-8704-b8fe85e81113
---
Full LoRA training run completed 2026-04-23. Key results:

- 97.6% overall accuracy, 97.7% macro F1 on 125-image eval set
- 775 training samples (310 no_tools, ~155 each tool class), 146 val samples
- 3 epochs, 291 steps, ~4h 40m on single AMD ROCm GPU

**no_tools data balance (validated):** 2:1 ratio of no_tools to each tool class in training. Overflow no_tools go to validation (~63 vs ~28 for tool classes). This fixed no_tools from 0% (tiny balanced run) to 98%. Based on Larson et al. 2019 (EMNLP) and Patil et al. 2023 (Gorilla).

**Overfitting signal:** Val loss bottomed at epoch 2 (0.0038), rose at epoch 3 (0.0044). Training loss hit near-zero. 2 epochs is likely the sweet spot for this dataset size. Do NOT increase LoRA rank — more parameters would make overfitting worse.

**How to apply:** When running full training, use 2 epochs. If adding more data, 3 may become viable. Check val loss curve in MLflow.
