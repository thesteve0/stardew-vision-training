---
name: Cluster hardware and ops
description: OpenShift AI cluster — 1x L40S autoscaling to 4; MLflow deployed; base image identified
type: project
originSessionId: 08a9a0d5-1cf4-4bcc-a246-d057ff1cac92
---
OpenShift AI cluster at api.stardew-vision.sandbox5291.opentlc.com, namespace `stardew-vision-training`.

- GPU: 1x NVIDIA L40S (g6e.2xlarge, 48GB VRAM, Ada Lovelace) with autoscaling to 4 GPU nodes
- MLflow: deployed via mlflows.mlflow.opendatahub.io CRD, running in redhat-ods-applications
- S3: NooBaa via OpenShift Data Foundation (OBC in our namespace)
- Base training image: `registry.redhat.io/rhoai/odh-training-cuda128-torch29-py312-rhel9:v3.4.0-ea.2` (PyTorch 2.9, Python 3.12, CUDA 12.8)

**Why:** Need multi-GPU training at scale beyond single-GPU devcontainer.

**How to apply:** Training code for cluster uses BF16 (not FP16 like local ROCm). DDP with 2 workers planned. Custom image needed on top of base to add transformers/peft/trl/ray/etc. User wants model output to both HuggingFace Hub and S3 (for vLLM in stardew-vision namespace).
