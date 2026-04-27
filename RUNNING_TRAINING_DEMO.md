# Running the Conference Training Demo

This demo runs a LoRA fine-tuning job on 15% of the training data (114 samples, 3 epochs) using KubeRay on OpenShift AI. The audience watches both the **Ray Dashboard** (job progress, worker GPU utilization) and the **MLflow Dashboard** (experiment metrics accumulating in real time). Total runtime is approximately 14 minutes.

Training parameters (learning rate, batch size, epochs, etc.) are stored in a **ConfigMap**, not baked into the container image. To experiment with different parameters, edit the ConfigMap and resubmit the job -- no image rebuild needed.

## Prerequisites

- OpenShift AI cluster with KubeRay operator installed
- 2 GPU workers (NVIDIA L40S) available
- Kubernetes secrets: `s3-credentials`, `mlflow-credentials`
- PVCs: `training-data` (with `train_demo.jsonl`), `training-checkpoints`
- Demo image built and pushed (see below)

## One-Time Setup

### 1. Generate the demo training split

From the devcontainer:

```bash
python scripts/make_demo_split.py
```

This creates `datasets/splits/train_demo.jsonl` (114 samples, 15% of the full training data, class-balanced).

### 2. Copy demo data to the cluster PVC

```bash
oc run pvc-copy --image=registry.access.redhat.com/ubi9/ubi:latest --restart=Never \
  --overrides='{
    "spec": {
      "containers": [{"name": "pvc-copy", "image": "registry.access.redhat.com/ubi9/ubi:latest",
        "command": ["sleep", "300"],
        "volumeMounts": [{"name": "training-data", "mountPath": "/data"}]}],
      "volumes": [{"name": "training-data",
        "persistentVolumeClaim": {"claimName": "training-data"}}]
    }
  }'

oc wait --for=condition=Ready pod/pvc-copy --timeout=120s
oc cp datasets/splits/train_demo.jsonl pvc-copy:/data/datasets/splits/train_demo.jsonl
oc delete pod pvc-copy
```

### 3. Build and push the demo image

Run these on the **host** (not the devcontainer). The image only contains Python code -- training parameters come from a ConfigMap at runtime, so you only need to rebuild when the training *code* changes.

```bash
podman build --no-cache -t stardew-training-demo:latest -f deploy/Dockerfile .

oc port-forward svc/image-registry -n openshift-image-registry 5000:5000 &
podman login localhost:5000 -u $(oc whoami) -p $(oc whoami -t) --tls-verify=false

podman tag stardew-training-demo:latest \
  localhost:5000/stardew-vision-training/stardew-training-demo:latest
podman push localhost:5000/stardew-vision-training/stardew-training-demo:latest \
  --tls-verify=false

kill %1
```

This creates a separate image from the production `stardew-training:latest` so you can rebuild without affecting real training runs.

## Running the Demo

### 1. Apply the training config and submit the job

```bash
oc apply -f deploy/training-config-demo.yaml
oc apply -f deploy/rayjob-demo.yaml
```

### 2. Open the MLflow UI

Open in your browser:

```
https://data-science-gateway.apps.stardew-vision.sandbox5291.opentlc.com/mlflow/
```

Sign in with your OpenShift credentials. Navigate to:
- Workspace: **stardew-vision-training**
- Experiment: **qwen-tool-selection-demo**
- Run: **demo-ray-v1**

### 3. Open the Ray Dashboard

Run this on the **host** (not the devcontainer):

```bash
# Find the Ray head service (created automatically by KubeRay)
oc get svc | grep head

# Port-forward to it
oc port-forward svc/<head-service-name> 8265:8265
```

Then open in your browser:

```
http://localhost:8265
```

The Ray Dashboard shows the job status, worker nodes, GPU utilization, and task progress.

## What the Audience Sees

| Time | What's Happening |
|------|-----------------|
| 0-5 min | Model loading (Qwen2.5-VL-7B) + data mapping (114 train, 146 val samples) |
| 5-7 min | Epoch 1 training (~15 steps, metrics appearing in MLflow every step) |
| 7-8 min | Epoch 1 eval (eval_loss logged to MLflow) |
| 8-10 min | Epoch 2 training + eval |
| 10-12 min | Epoch 3 training + eval |
| 12-14 min | Training complete, config + model artifacts logged to MLflow |

### MLflow Tabs to Show

The demo config enables extra MLflow logging that populates tabs beyond the default metrics:

| MLflow Tab | What Appears | When |
|------------|-------------|------|
| **Metrics** | `loss`, `learning_rate`, `eval_loss` (from HuggingFace), plus `step_wall_seconds` (per-step timing) | During training, every step |
| **System Metrics** | GPU utilization %, GPU memory MB, CPU utilization %, disk/network I/O | Throughout the run (sampled automatically) |
| **Artifacts** | `config/training-config.yaml` (the ConfigMap config) and `model/` (saved LoRA adapter) | After training completes |

These features are controlled by three flags in the ConfigMap (`deploy/training-config-demo.yaml`) under the `mlflow:` section:

```yaml
mlflow:
  log_system_metrics: true   # GPU/CPU metrics in System Metrics tab
  log_artifacts: true         # Config + model files in Artifacts tab
  log_step_timing: true       # step_wall_seconds in Metrics tab
```

Production training configs omit these flags (they default to off).

## Clean Up

```bash
oc delete rayjob stardew-lora-train-demo
```

## Changing Training Parameters

Training parameters live in the ConfigMap, not the image. To change them:

1. Edit `deploy/training-config-demo.yaml` (learning rate, batch size, epochs, run name, etc.)
2. Apply the updated config: `oc apply -f deploy/training-config-demo.yaml`
3. Delete and resubmit the job:
   ```bash
   oc delete rayjob stardew-lora-train-demo
   oc apply -f deploy/rayjob-demo.yaml
   ```

No image rebuild required for parameter changes. Only rebuild the image when the Python training code itself changes.

## Re-Running the Demo

To run it again (e.g., for a second talk or rehearsal):

1. Delete the previous job: `oc delete rayjob stardew-lora-train-demo`
2. Optionally bump `run_name` in `deploy/training-config-demo.yaml` (e.g., `demo-ray-v2`) so you get a separate run in MLflow
3. Apply and resubmit:
   ```bash
   oc apply -f deploy/training-config-demo.yaml
   oc apply -f deploy/rayjob-demo.yaml
   ```

## Key Files

| File | Purpose |
|------|---------|
| `scripts/make_demo_split.py` | Generates the 15% demo training split |
| `deploy/training-config-demo.yaml` | ConfigMap with training parameters (edit this to change experiments) |
| `deploy/rayjob-demo.yaml` | KubeRay job manifest (2 GPU workers, demo image, mounts ConfigMap) |
| `fine_tuning/qwen/lora_config_ray_demo.yaml` | Local reference copy of the training config |
| `datasets/splits/train_demo.jsonl` | Demo training data (114 samples) |
