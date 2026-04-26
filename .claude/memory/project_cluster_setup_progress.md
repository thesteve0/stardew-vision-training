---
name: Cluster setup — operational
description: OpenShift AI cluster fully operational; Ray, KubeFlow 2-GPU, and KubeFlow 1-GPU training paths all validated
type: project
---
## Cluster infrastructure (operational as of 2026-04-26)

- Namespace: `stardew-vision-training`
- GPU: 1x L40S (g6e.2xlarge), autoscales to 4
- MLflow: running in `redhat-ods-applications` (dashboard access TBD)
- Secrets: `s3-credentials`, `mlflow-credentials`, `hf-credentials`
- PVCs: `training-data`, `training-checkpoints`
- Training data mounted via PVCs (not S3)
- Code baked into container images

## Validated training paths

- **KubeFlow PyTorchJob (2-GPU)**: `deploy/pytorchjob.yaml` + `stardew-training-kf` image — 97.6%
- **KubeFlow PyTorchJob (1-GPU)**: `deploy/pytorchjob-1gpu.yaml` + same image, config via ConfigMap — 96.8%
- **Ray Train**: `deploy/rayjob.yaml` + `stardew-training` image — 96.0%

## Known issues

- Eval job needs `MLFLOW_TRACKING_URI=file:///tmp/mlruns` and `workingDir=/data`
- KubeFlow 2-GPU post-training eval hangs (fixed in train_kubeflow.py by removing it)
- GPU nodes take ~5-10 min to provision from zero (MachineSet → Machine → Node → NVIDIA driver)
