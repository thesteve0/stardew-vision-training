---
name: Training results — all runs complete
description: Full training comparison across all paths — local 96.0%, Ray 96.0%, KubeFlow 2-GPU 97.6%, KubeFlow 1-GPU 96.8%; training phase is done
type: project
---
Training is complete as of 2026-04-26. All orchestration paths validated.

## Summary of all runs (775 samples, 3 epochs, effective batch 8)

| Run | Hardware | Steps | Wall time | s/step | Accuracy |
|-----|----------|-------|-----------|--------|----------|
| Local standalone | 1x AMD Strix Halo | 291 | ~424 min | ~87.4 | 96.0% |
| Cluster KubeFlow 1-GPU | 1x NVIDIA L40S | 291 | ~82 min | ~16.9 | 96.8% |
| Cluster Ray v4 | 2x NVIDIA L40S | 291 | ~43 min | ~8.8 | 96.0% |
| Cluster KubeFlow v1 | 2x NVIDIA L40S | 291 | ~42 min | ~8.6 | 97.6% |

Baseline (untuned): 51.2%. Prompt engineering alone lifted no_tools from 0% to 30%.

## Key findings

- Effective batch size mismatch (8→16) caused 7.2 point accuracy drop (Ray v3: 90.4%)
- L40S is 5.2x faster than Strix Halo per GPU; 2-GPU scaling is near-linear (1.96x)
- All effective-batch-8 runs are within noise (96.0–97.6%)

**Why:** Training phase is done. Next focus is MLflow dashboard and adapter deployment.

**How to apply:** Best adapter is from KubeFlow 2-GPU run. Files on cluster PVC and locally at `experiments/qwen-tool-select-cluster-kubeflow-v1/`. Full comparison in `docs/comparison-small-training-runs.md`.
