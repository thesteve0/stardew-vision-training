---
name: Distributed training plan
description: Next phase — Ray local → KubeRay → Kubeflow on OpenShift AI, with MLflow, S3 data, and model serving
type: project
originSessionId: c728f63d-97fb-4427-a5c5-1ec5406dc659
---
Distributed training is the next phase after baseline eval (completed 2026-04-22, 70% baseline accuracy).

Planned progression:
1. Ray Train locally (devcontainer)
2. KubeRay on OpenShift AI
3. Kubeflow on OpenShift AI

**Why:** Need to scale beyond single-GPU LoRA training for the full dataset.

**How to apply:** When working on training scripts, design for this progression. Keep training code Ray-compatible from the start.

Open questions to resolve:
- How to get training data (datasets/) into OpenShift — S3 on AWS is available
- MLflow instance on the same OpenShift cluster for training metrics
- Model storage after training — needs to be accessible by the stardew-vision production app (currently pulls from HuggingFace Hub)
- LoRA adapter serving workflow: train → store → serve in KServe
