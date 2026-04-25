# Evaluation Comparison: Training Runs

All evaluations use the held-out eval set from `datasets/eval_set.json`, which is **never** used in training or validation. The eval set contains 125 images (50 no_tools, 25 each for tool classes).

## Per-class accuracy

| Screen Type | Local (standalone) | Local (Ray) | Cluster (Ray, 2-GPU) |
|-------------|-------------------|------------|---------------------|
| tv_dialog   | — | — | — |
| caught_fish | — | — | — |
| pierre_shop | — | — | — |
| no_tools    | — | — | — |
| **Overall** | — | — | — |
| **Macro F1** | — | — | — |

## Training metrics

| Metric | Local (standalone) | Local (Ray) | Cluster (Ray, 2-GPU) |
|--------|-------------------|------------|---------------------|
| Training samples | — | — | — |
| Steps | — | — | — |
| Final loss | — | — | — |
| Val loss (best) | — | — | — |
| Wall time | — | — | — |
| Seconds/step | — | — | — |

## Key observations

(To be filled after runs complete)
