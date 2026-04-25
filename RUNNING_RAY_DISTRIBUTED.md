# Running Distributed Training on OpenShift AI

Step-by-step runbook for submitting a Ray distributed training job, monitoring it, running evaluation, and copying results locally.

---

## 1. Build and Push the Training Image

Run these on the **host** (not in the devcontainer), from the repo root:

```bash
# Build
podman build -t stardew-training:latest -f deploy/Dockerfile .

# Port-forward to the OpenShift internal registry
oc port-forward svc/image-registry -n openshift-image-registry 5000:5000 &

# Auth, tag, push
podman login localhost:5000 -u $(oc whoami) -p $(oc whoami -t) --tls-verify=false
podman tag stardew-training:latest localhost:5000/stardew-vision-training/stardew-training:latest
podman push localhost:5000/stardew-vision-training/stardew-training:latest --tls-verify=false

# Kill the port-forward
kill %1
```

---

## 2. Submit the Training Job

```bash
oc apply -f deploy/rayjob.yaml
```

---

## 3. Monitor GPU Node Provisioning

The GPU node (L40S on g6e.2xlarge) may need to scale up from zero. Track the full provisioning sequence:

```bash
# Watch MachineSet scale-up
oc get machineset -n openshift-machine-api -w

# Watch Machines being created and transitioning to Running
oc get machines -n openshift-machine-api -w

# Watch the node join the cluster
oc get nodes -w
```

---

## 4. Monitor NVIDIA Driver and Toolkit Installation

Once the node joins, the GPU operator installs drivers and the container toolkit:

```bash
# Watch GPU operator pods (driver, toolkit, device-plugin, etc.)
oc get pods -n nvidia-gpu-operator -w

# Tail driver installation logs
oc logs -f -n nvidia-gpu-operator -l app=nvidia-driver-daemonset

# Tail container toolkit logs
oc logs -f -n nvidia-gpu-operator -l app=nvidia-container-toolkit-daemonset

# Verify GPU is advertised as allocatable
oc describe node -l nvidia.com/gpu.present=true | grep -A5 "Allocatable"
```

---

## 5. Monitor Training

Once GPU nodes are ready and worker pods are scheduled:

```bash
# Watch RayJob status (provisioning → running → succeeded/failed)
oc get rayjob stardew-lora-train -w

# Watch Ray worker pods scheduling and running
oc get pods -l ray.io/cluster=stardew-lora-train-raycluster -w

# Tail training logs (all pods, prefixed)
oc logs -f -l job-name=stardew-lora-train --prefix

# Ray dashboard (open http://localhost:8265 in browser)
oc port-forward svc/stardew-lora-train-head-svc 8265:8265
```

---

## 6. Run Evaluation on the Cluster

After training completes, the model adapter is saved to `/checkpoints/model-output` on the PVC. Run eval against it:

```bash
# Delete any previous eval job
oc delete job stardew-eval -n stardew-vision-training --ignore-not-found

# Submit the eval job
oc apply -f deploy/eval-job.yaml

# Watch it run
oc get pods -l job-name=stardew-eval -w
oc logs -f job/stardew-eval
```

Eval results are written to `/checkpoints/eval-results/` on the PVC (results.json + sample_outputs.jsonl).

---

## 7. Copy Results Locally

Use a temporary pod to access PVC contents, or exec into an existing pod. The examples below use a one-shot pod:

```bash
# Start a helper pod that mounts the checkpoints PVC
oc run pvc-reader --rm -it \
  --image=image-registry.openshift-image-registry.svc:5000/stardew-vision-training/stardew-training:latest \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "pvc-reader",
        "image": "image-registry.openshift-image-registry.svc:5000/stardew-vision-training/stardew-training:latest",
        "command": ["sleep", "3600"],
        "volumeMounts": [{"name": "ckpt", "mountPath": "/checkpoints"}]
      }],
      "volumes": [{
        "name": "ckpt",
        "persistentVolumeClaim": {"claimName": "training-checkpoints"}
      }]
    }
  }' \
  -- sleep 3600 &

# Wait for it to be running
oc wait --for=condition=Ready pod/pvc-reader --timeout=120s
```

### Copy the LoRA adapter

```bash
mkdir -p experiments/qwen-tool-select-cluster-ray-v3
oc cp pvc-reader:/checkpoints/model-output/ experiments/qwen-tool-select-cluster-ray-v3/
```

### Copy eval results

```bash
mkdir -p experiments/eval-cluster-ray-v3
oc cp pvc-reader:/checkpoints/eval-results/ experiments/eval-cluster-ray-v3/
```

### Copy training timing from trainer_state.json (if checkpoints were saved)

```bash
# List what's on the PVC
oc exec pvc-reader -- find /checkpoints -name "trainer_state.json" -o -name "*.json" | head -20

# Copy any checkpoint trainer states
oc cp pvc-reader:/checkpoints/ray-results/ experiments/qwen-tool-select-cluster-ray-v3/ray-results/
```

### Clean up the helper pod

```bash
oc delete pod pvc-reader
```

---

## 8. Clean Up the RayJob

```bash
# Delete the RayJob (also cleans up the RayCluster)
oc delete rayjob stardew-lora-train -n stardew-vision-training

# Delete the eval job
oc delete job stardew-eval -n stardew-vision-training
```

---

## Quick Reference

| What | Where |
|------|-------|
| Training config | `fine_tuning/qwen/lora_config_ray_cluster.yaml` |
| RayJob manifest | `deploy/rayjob.yaml` |
| Eval job manifest | `deploy/eval-job.yaml` |
| Training data PVC | `training-data` (mounted at `/data`) |
| Checkpoints PVC | `training-checkpoints` (mounted at `/checkpoints`) |
| Model output | `/checkpoints/model-output/` on PVC |
| Eval results | `/checkpoints/eval-results/` on PVC |
| Namespace | `stardew-vision-training` |
