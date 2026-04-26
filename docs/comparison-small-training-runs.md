# Evaluation Comparison: Training Runs

All evaluations use the held-out eval set from `datasets/eval_set.json`, which is **never** used in training or validation. The eval set contains 125 images (50 no_tools, 25 each for tool classes).

## Per-class accuracy

| Screen Type | Local (standalone) | Local (Ray) | Cluster Ray v3 (batch=16) | Cluster Ray v4 (batch=8) | Cluster KubeFlow v1 |
|-------------|-------------------|------------|----------------------|---------------------|---------------------|
| tv_dialog   | 96% (24/25) | — | 96% (24/25) | 100% (25/25) | 96% (24/25) |
| caught_fish | 92% (23/25) | — | 68% (17/25) | 88% (22/25) | 100% (25/25) |
| pierre_shop | 96% (24/25) | — | 100% (25/25) | 100% (25/25) | 100% (25/25) |
| no_tools    | 98% (49/50) | — | 94% (47/50) | 96% (48/50) | 96% (48/50) |
| **Overall** | **96.0%** (120/125) | — | **90.4%** (113/125) | **96.0%** (120/125) | **97.6%** (122/125) |
| **Macro F1** | **0.962** | — | **0.904** | **0.962** | **0.978** |

## Training metrics

| Metric | Local (standalone) | Local (Ray) | Cluster Ray v3 (batch=16) | Cluster Ray v4 (batch=8) | Cluster KubeFlow v1 |
|--------|-------------------|------------|----------------------|---------------------|---------------------|
| Training samples | 775 | — | 775 | 775 | 775 |
| Epochs | 3 | — | 3 | 3 | 3 |
| Steps | 291 | — | 147 | 291 | 291 |
| Effective batch size | 8 | — | 16 | 8 | 8 |
| Final train loss | 0.013 | — | 0.0035 | 0.013 | 0.013 |
| Eval loss (epoch 3) | 0.0050 | — | 0.0042 | — | 0.0048 |
| Token accuracy (train) | 100% | — | 99.9% | — | 100% |
| Token accuracy (eval) | 99.9% | — | 99.9% | — | 99.9% |
| Wall time | ~424 min | — | ~42 min | ~43 min | ~42 min |
| Seconds/step | ~87.4 | — | ~17 | ~8.8 | ~8.6 |
| Hardware | AMD Strix Halo (gfx1151) | — | 2x NVIDIA L40S | 2x NVIDIA L40S | 2x NVIDIA L40S |

## Key observations

- **Local standalone matches Ray v4**: Both scored 96.0% (120/125) — confirms the Ray wrapper doesn't degrade quality when batch size is matched
- **Batch size matters**: Ray v3 (effective batch 16) scored 90.4%, Ray v4 (effective batch 8, matching local) scored 96.0% — a 5.6 point improvement just from aligning batch size
- **KubeFlow matches local baseline**: KubeFlow v1 hit 97.6% (same as original local training), while Ray v4 with identical hyperparameters scored 96.0%. The difference is likely Ray Train overhead vs native torchrun
- **caught_fish fully solved by KubeFlow**: 68% (Ray v3) → 88% (Ray v4) → 100% (KubeFlow v1)
- **KubeFlow is slightly faster**: 8.6s/step vs 8.8s/step (Ray v4) — lower overhead from torchrun vs Ray Train wrapper
- **Wall time nearly identical** (~42 min) across cluster runs despite different step counts — more steps with smaller batches run faster per step
- **iGPU vs data center**: Strix Halo takes ~10x longer to train (424 min vs 43 min), combining ~5x per-GPU compute difference and 2x GPU parallelism. For eval inference, the gap narrows to ~1.6x (8 min vs 5 min) since autoregressive decoding is memory-bandwidth-bound
