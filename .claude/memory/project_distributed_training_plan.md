---
name: Distributed training plan
description: Ray Train validated locally; next step is KubeRay on OpenShift AI 3.3 with 2 NVIDIA L40S GPUs
type: project
originSessionId: 1cac980d-2f8c-45f6-8704-b8fe85e81113
---
Ray Train integration is complete and validated (2026-04-23). Full training run achieved 97.6% eval accuracy.

Progression:
1. ~~Ray Train locally (devcontainer)~~ — DONE
2. KubeRay on OpenShift AI — NEXT
3. Kubeflow on OpenShift AI — future

**Why:** Scale beyond single-GPU and enable reproducible training on the cluster.

**How to apply:**
- `train_ray.py` is ready for multi-GPU — uses `prepare_trainer()` + `RayTrainReportCallback`
- `lora_config_cluster.yaml` has BF16, batch 4, S3 data paths (CHANGEME placeholders)
- `deploy/rayjob.yaml` has the KubeRay manifest (CHANGEME placeholders for namespace, S3 bucket, MLflow URI, container image)
- Container image TBD: candidate `quay.io/opendatahub/odh-pipeline-runtime-pytorch-cuda-py312-ubi9`
- Need to: upload datasets to S3, confirm container image, coordinate MLflow URI with James Harmison
- Key code difference vs local: no `device_map="auto"` for multi-worker, BF16 instead of FP16
