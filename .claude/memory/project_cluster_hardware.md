---
name: Cluster hardware, dashboards, and ops
description: OpenShift AI cluster details — GPUs, namespaces, dashboard URLs, image registry workflow
type: project
originSessionId: f41ae552-4148-41b3-b324-d052d1ac722d
---
OpenShift AI cluster at api.stardew-vision.sandbox5291.opentlc.com, namespace `stardew-vision-training`.

- GPU: NVIDIA L40S (g6e.2xlarge, 48GB VRAM, Ada Lovelace) with autoscaling up to 4 GPU nodes
- MLflow: deployed via mlflows.mlflow.opendatahub.io CRD, running in `redhat-ods-applications`
- S3: NooBaa via OpenShift Data Foundation
- Base training image: `registry.redhat.io/rhoai/odh-training-cuda128-torch29-py312-rhel9:v3.4.0-ea.2`

## Dashboard URLs

- **MLflow**: https://data-science-gateway.apps.stardew-vision.sandbox5291.opentlc.com/mlflow/#/workspaces/stardew-vision-training
- **Ray Dashboard**: no external route; use `oc port-forward svc/<head-svc-name> 8265:8265 -n stardew-vision-training` then open http://localhost:8265

## Internal service URLs (pod-to-pod)

- MLflow tracking: `https://mlflow.redhat-ods-applications.svc:8443` (requires `MLFLOW_TRACKING_INSECURE_TLS=true`)

**Why:** Need multi-GPU training at scale beyond single-GPU devcontainer.

**How to apply:** Cluster uses BF16 (not FP16 like local ROCm). Demo instructions in DEMO-INSTRUCTIONS.md at repo root.
