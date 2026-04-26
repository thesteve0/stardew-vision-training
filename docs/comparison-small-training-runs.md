# Evaluation Comparison: Training Runs

All evaluations use the held-out eval set from `datasets/eval_set.json`, which is **never** used in training or validation. The eval set contains 125 images (50 no_tools, 25 each for tool classes).

## Per-class accuracy

| Screen Type | Local (standalone) | Local (Ray) | Cluster v3 (batch=16) | Cluster v4 (batch=8) |
|-------------|-------------------|------------|----------------------|---------------------|
| tv_dialog   | — | — | 96% (24/25) | 100% (25/25) |
| caught_fish | — | — | 68% (17/25) | 88% (22/25) |
| pierre_shop | — | — | 100% (25/25) | 100% (25/25) |
| no_tools    | — | — | 94% (47/50) | 96% (48/50) |
| **Overall** | — | — | **90.4%** (113/125) | **96.0%** (120/125) |
| **Macro F1** | — | — | **0.904** | **0.962** |

## Training metrics

| Metric | Local (standalone) | Local (Ray) | Cluster v3 (batch=16) | Cluster v4 (batch=8) |
|--------|-------------------|------------|----------------------|---------------------|
| Training samples | — | — | 775 | 775 |
| Epochs | — | — | 3 | 3 |
| Steps | — | — | 147 | 291 |
| Effective batch size | — | — | 16 | 8 |
| Final train loss | — | — | 0.0035 | 0.013 |
| Eval loss (epoch 3) | — | — | 0.0042 | — |
| Token accuracy (train) | — | — | 99.9% | — |
| Token accuracy (eval) | — | — | 99.9% | — |
| Wall time | — | — | ~42 min | ~43 min |
| Seconds/step | — | — | ~17 | ~8.8 |
| Hardware | — | — | 2x NVIDIA L40S | 2x NVIDIA L40S |

## Key observations

- **Batch size matters**: Cluster v3 (effective batch 16) scored 90.4%, cluster v4 (effective batch 8, matching local) scored 96.0% — a 5.6 point improvement just from aligning batch size
- **caught_fish is the most sensitive class**: 68% → 88% with the batch fix, but still the weakest. 3 caught_fish still misclassified as no_tools
- **Cluster v4 is close to prior local result (97.6%)** but not quite there — remaining gap may be dtype (BF16 cluster vs FP16 local) or noise. Local baseline run with `use_reentrant=False` still needed to confirm
- **Wall time nearly identical** (~42-43 min) between v3 and v4 despite 2x more steps — v4 steps are 2x faster (8.8s vs 17s) since there's no gradient accumulation wait
