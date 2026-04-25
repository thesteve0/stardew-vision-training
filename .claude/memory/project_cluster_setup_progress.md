---
name: Cluster setup progress
description: OpenShift AI infrastructure is ready; code reverted to clean state; need fresh plan with PVCs
type: project
originSessionId: 08a9a0d5-1cf4-4bcc-a246-d057ff1cac92
---
## Infrastructure still in place on the cluster (2026-04-24)

- Namespace: `stardew-vision-training`
- OBC: `stardew-training-bucket` → bucket `stardew-training-de41fcf6-ae74-43cc-8d20-64be0dc8c9a4`
- S3 endpoints: internal `https://s3.openshift-storage.svc:443`, external `https://s3-openshift-storage.apps.stardew-vision.sandbox5291.opentlc.com`
- MLflow: running in `redhat-ods-applications`, UI at `https://data-science-gateway.apps.stardew-vision.sandbox5291.opentlc.com/mlflow`
- Secrets in `stardew-vision-training`: `s3-credentials`, `mlflow-credentials`, `hf-credentials`
- Secret `hf-credentials` also in `stardew-vision` namespace
- GPU: 1x L40S (g6e.2xlarge), autoscales to 4
- BuildConfig `stardew-training` exists with ImageStream — can rebuild images with `oc start-build`
- Training data already uploaded to S3 (may still be useful for final model export, not for training data access)
- Base image: `registry.redhat.io/rhoai/odh-training-cuda128-torch29-py312-rhel9:v3.4.0-ea.2` — has most deps pre-installed

## Code state

All code reverted to pre-session state (commit c72b453). No today's changes committed. Config files still have CHANGEME placeholders.

## Known bugs to fix (minimal, targeted changes)

1. `train_ray.py` `_resolve()` — doesn't handle `https://` URLs, mangles MLflow tracking URI
2. `train_ray.py` — needs `gradient_checkpointing_kwargs={"use_reentrant": False}` for DDP + LoRA
3. `lora_config_ray_cluster.yaml` — set to 2 epochs (user preference, validated by val loss curve)

## Plan for next session

Start with `EnterPlanMode`. Design the architecture around:
- PVCs for training data (mount datasets into pods)
- Training code baked into container image
- PVC or alternative for Ray checkpoint storage
- MLflow tracking to the cluster server
- S3 only for final model export to HF Hub and for vLLM access

**Why:** PVC-based approach matches how OpenShift AI training jobs work natively and avoids all the S3 SSL/CA issues.
