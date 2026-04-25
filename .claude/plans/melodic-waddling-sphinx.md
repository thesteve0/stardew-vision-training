# Plan: Local Container Build for OpenShift

## Context

Training ran successfully on the cluster. Now we want to enable local container builds that push to the internal OpenShift registry via `oc port-forward`. User will build and push manually from the host machine.

Merging into a single Dockerfile — Docker layer caching means dependency installs stay cached and only code layers rebuild on changes.

## Changes

### 1. Rewrite `deploy/Dockerfile` — single-stage with layer caching

Merge base + app into one file. Dependencies before code for layer caching.

```dockerfile
FROM registry.redhat.io/rhoai/odh-training-cuda128-torch29-py312-rhel9:v3.4.0-ea.2

WORKDIR /app

RUN pip install --no-cache-dir \
    "ray[default,train]>=2.40.0" \
    "qwen-vl-utils>=0.0.8"

COPY fine_tuning/ fine_tuning/
COPY evaluation/ evaluation/
```

### 2. Delete `deploy/Dockerfile.base`

No longer needed.

### 3. Fix `.dockerignore` — remove `evaluation/` exclusion

Needed for COPY and eval-job.yaml.

## Verification

From host machine:
```bash
# Build
podman build -f deploy/Dockerfile -t localhost:5000/stardew-vision-training/stardew-training:latest .

# Port-forward and push
oc port-forward svc/image-registry -n openshift-image-registry 5000:5000 &
podman push --tls-verify=false localhost:5000/stardew-vision-training/stardew-training:latest

# Verify
oc get istag stardew-training:latest -n stardew-vision-training
```
