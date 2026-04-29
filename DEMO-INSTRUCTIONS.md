# Demo Training Job Instructions

Run a 3-epoch LoRA fine-tuning job on 15% of training data using 2x L40S GPUs via KubeRay on OpenShift AI. Completes in ~14 minutes.

## Prerequisites

Verify everything is in place before submitting:

```bash
# PVCs (training-data must contain datasets/splits/train_demo.jsonl)
oc get pvc -n stardew-vision-training

# Secrets
oc get secret -n stardew-vision-training | grep -E 's3-credentials|mlflow-credentials'

# ConfigMap
oc get configmap training-config-demo -n stardew-vision-training

# Demo image
oc get istag stardew-training-demo:latest -n stardew-vision-training
```

If the image needs rebuilding (only for code changes, not parameter changes):

```bash
podman build --no-cache -t stardew-training-demo:latest -f deploy/Dockerfile .

# Port-forward to internal registry
oc port-forward svc/image-registry -n openshift-image-registry 5000:5000 &
podman login localhost:5000 -u $(oc whoami) -p $(oc whoami -t) --tls-verify=false

podman tag stardew-training-demo:latest \
  localhost:5000/stardew-vision-training/stardew-training-demo:latest
podman push localhost:5000/stardew-vision-training/stardew-training-demo:latest \
  --tls-verify=false
kill %1
```

## Running the Demo

```bash
# 1. Delete any previous run
oc delete rayjob stardew-lora-train-demo -n stardew-vision-training --ignore-not-found

# 2. Apply config and submit the job
oc apply -f deploy/training-config-demo.yaml
oc apply -f deploy/rayjob-demo.yaml

# 3. Watch job status
oc get rayjob stardew-lora-train-demo -n stardew-vision-training -w
```

## Changing Training Parameters

Training parameters (learning rate, batch size, epochs, etc.) live in `deploy/training-config-demo.yaml`, not in the image. To change them:

1. Edit `deploy/training-config-demo.yaml`
2. `oc apply -f deploy/training-config-demo.yaml`
3. Delete and resubmit the RayJob (see above)

No image rebuild needed.

## Dashboards

### Ray Dashboard

The Ray Dashboard is available via port-forward while the job is running:

```bash
# Find the head service (created when the RayCluster starts)
oc get svc -n stardew-vision-training | grep head

# Port-forward (replace <head-svc-name> with the actual service name)
oc port-forward svc/<head-svc-name> 8265:8265 -n stardew-vision-training
```

Open <http://localhost:8265> in your browser.

### MLflow Dashboard

Open <https://data-science-gateway.apps.stardew-vision.sandbox5291.opentlc.com/mlflow/#/workspaces/stardew-vision-training> in your browser.

Experiment name: `qwen-tool-selection-demo`

## Monitoring Logs

```bash
# Follow the head pod logs (training output)
oc logs -f -l ray.io/cluster=$(oc get rayjob stardew-lora-train-demo -o jsonpath='{.status.rayClusterName}') -c ray-head -n stardew-vision-training
```

## Cleanup

```bash
oc delete rayjob stardew-lora-train-demo -n stardew-vision-training
```

The RayCluster is automatically deleted 1 hour after the job finishes (`ttlSecondsAfterFinished: 3600`).
