---
name: Cluster setup — operational
description: OpenShift AI cluster fully operational; both Ray and KubeFlow training paths validated
type: project
originSessionId: 0b1c1454-7379-436e-a94a-f747fbf758da
---
## Cluster infrastructure (operational as of 2026-04-26)

- Namespace: `stardew-vision-training`
- GPU: 1x L40S (g6e.2xlarge), autoscales to 4
- MLflow: running in `redhat-ods-applications`
- Secrets: `s3-credentials`, `mlflow-credentials`, `hf-credentials`
- PVCs: `training-data`, `training-checkpoints`
- Training data mounted via PVCs (not S3)
- Code baked into container images

## Validated training paths

- **Ray Train**: `deploy/rayjob.yaml` + `stardew-training` image
- **KubeFlow PyTorchJob**: `deploy/pytorchjob.yaml` + `stardew-training-kf` image

Both produce comparable results (~96-97.6% accuracy).
