# Next Steps — MLflow and Adapter Deployment

Training is complete. All orchestration paths validated (local, Ray, KubeFlow). Best result: 97.6% accuracy (KubeFlow 2-GPU). Full comparison in `docs/comparison-small-training-runs.md`.

## Immediate Tasks

### 1. Get MLflow Working on Cluster

MLflow is deployed via the `mlflows.mlflow.opendatahub.io` CRD in `redhat-ods-applications`. Training logs metrics there but we haven't verified the dashboard or queried runs programmatically.

- Verify MLflow UI is accessible (port-forward or route)
- Confirm training runs from KubeFlow v1 and 1-GPU runs are visible
- Fix any auth/TLS issues between training pods and MLflow service

### 2. Transition Adapter to Production App

The trained LoRA adapter needs to move from this training repo to the `stardew-vision` production app. The handoff path:

1. Upload adapter to HuggingFace Hub (`scripts/upload_to_hub.py`)
2. Production app pulls adapter for KServe serving
3. Verify adapter loads correctly in the serving pipeline

### Key Decision

Which adapter to ship? The KubeFlow 2-GPU run (97.6%) is the best, but all runs with effective batch 8 are within noise (96.0–97.6%). The adapter files are on the cluster PVC at `/checkpoints/model-output/` and locally at `experiments/qwen-tool-select-cluster-kubeflow-v1/`.

## Completed Work (for reference)

| Run | Hardware | Accuracy | Wall time |
|-----|----------|----------|-----------|
| Local standalone | 1x AMD Strix Halo | 96.0% | ~424 min |
| Cluster KubeFlow 1-GPU | 1x NVIDIA L40S | 96.8% | ~82 min |
| Cluster Ray v4 | 2x NVIDIA L40S | 96.0% | ~43 min |
| Cluster KubeFlow v1 | 2x NVIDIA L40S | 97.6% | ~42 min |

## Key Files

| File | What it does |
|------|-------------|
| `fine_tuning/qwen/train_kubeflow.py` | KubeFlow/torchrun training script |
| `deploy/pytorchjob.yaml` | 2-GPU KubeFlow PyTorchJob manifest |
| `deploy/pytorchjob-1gpu.yaml` | 1-GPU baseline (config via ConfigMap) |
| `deploy/eval-job.yaml` | Evaluation job manifest |
| `RUNNING_KUBEFLOW_DISTRIBUTED.md` | Full KubeFlow runbook |
| `RUNNING_RAY_DISTRIBUTED.md` | Ray runbook (for reference) |
| `docs/comparison-small-training-runs.md` | All training results |
| `POINTS_FOR_TALK.md` | Talk points with findings |
