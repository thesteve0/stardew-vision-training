# Evaluation Comparison: Training Runs

All evaluations use the held-out eval set from `datasets/eval_set.json`, which is **never** used in training or validation. The eval set contains 125 images (50 no_tools, 25 each for tool classes).

## Per-class accuracy

| Screen Type | Local (standalone) | Local (Ray) | Cluster (Ray, 2-GPU) |
|-------------|-------------------|------------|---------------------|
| tv_dialog   | — | — | 96% (24/25) |
| caught_fish | — | — | 68% (17/25) |
| pierre_shop | — | — | 100% (25/25) |
| no_tools    | — | — | 94% (47/50) |
| **Overall** | — | — | **90.4%** (113/125) |
| **Macro F1** | — | — | **0.904** |

## Training metrics

| Metric | Local (standalone) | Local (Ray) | Cluster (Ray, 2-GPU) |
|--------|-------------------|------------|---------------------|
| Training samples | — | — | 775 |
| Epochs | — | — | 3 |
| Steps | — | — | 147 |
| Final train loss | — | — | 0.0035 |
| Eval loss (epoch 3) | — | — | 0.0042 |
| Token accuracy (train) | — | — | 99.9% |
| Token accuracy (eval) | — | — | 99.9% |
| Wall time | — | — | ~42 min |
| Seconds/step | — | — | ~17 |
| Hardware | — | — | 2x NVIDIA L40S |

## Key observations

- **Cluster run (90.4%) underperforms vs prior local run (97.6%)** — the biggest gap is caught_fish (68% vs 96% previously), with 8 caught_fish misclassified as no_tools
- **Effective batch size doubled**: local was 8 (2 × 4 grad_accum), cluster is 16 (4 × 2 workers × 2 grad_accum), with the same learning rate. This likely undertrained the model — fewer optimization steps per epoch
- **Next step**: rerun cluster with `gradient_accumulation_steps: 1` to match local effective batch size of 8, or run local baseline first to confirm the `use_reentrant=False` change didn't regress
